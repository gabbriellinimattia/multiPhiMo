"""
agents.py — 8 agenti IA (inverse model), sez.5 pipeline_multiPhiMo.txt / sez.4 pipeline.md.

Ogni agente e' una Mixture Density Network (MLP piccola + mistura di gaussiane in
uscita): mappa un vettore di descrittori target (vedi analyzer.descriptors) sui
parametri di sintesi di UN eccitatore (7 agenti, param range in exciters.PARAM_RANGES)
oppure di UNA forma del risonatore condiviso (4 agenti, uno per topologia — vedi sotto,
param range in resonator.PARAM_RANGES). 11 agenti istanziabili in totale, mai piu' di 2
in memoria insieme (sez. Selezione).

MDN e' la scelta esplicita del doc (sez.5): a differenza di una MLP semplice non fa
mode-averaging, quindi descrittori target identici possono restituire configurazioni
di parametri diverse — voluto dal progetto (matching solo sui descrittori, non
timbrico: "configurazioni simili possono dare suoni diversi").

Risonatore (agente 8): resonator.py esiste ed e' un generatore ANALITICO (formule
fisiche standard, Fletcher & Rossing) dei parametri modali per 4 topologie (bar,
plate_rect, plate_circ, membrane), ciascuna col proprio spazio parametri
(resonator.PARAM_RANGES) — diverso dalla rappresentazione unificata "scala/materiale/
forma" ipotizzata in claude/risonatore_modello_scelto.md (quella resta l'obiettivo per
un futuro regressore forma->modale che sostituira' resonator_params(), non ancora
scritto). La topologia e' una scelta discreta hard-coded in resonator.py (formule e
persino il numero di parametri cambiano da forma a forma, es. "aspect" esiste solo per
plate_rect) e non e' interpolabile con continuita': qui quindi rispecchia l'interfaccia
reale, un agente per topologia (stesso pattern dei 7 eccitatori), non un'unica testa
categorica.

Selezione: SOLO UN eccitatore e SOLA UNA forma di risonatore alla volta sono attivi
(scelta manuale dell'utente, non routing IA — vale per entrambi, non solo per gli
eccitatori: resonator.py stesso richiede di fissare "shape" per generare qualunque
parametro). AgentManager carica solo l'agente eccitatore selezionato + l'agente forma-
risonatore selezionato: mai piu' di 2 modelli in memoria contemporaneamente, gli altri
6 agenti eccitatore e le altre 3 forme non vengono ne' istanziati ne' eseguiti.

Training offline (sez.4): CSV con colonne = DESCRIPTOR_KEYS + nomi parametri
dell'agente — il dataset (parametri->descrittori) invertito, un CSV per agente (quindi
anche uno per forma di risonatore, coi soli parametri di quella forma). Inferenza
diretta O(1) via forward pass, nessuna ricerca iterativa (vincolo runtime <=25ms,
sez.1): coerente col loop realtime di pipeline.md sez.5.

Early stopping (2026-09-14): il primo training (bow, 200 epoche) ha mostrato overfitting
netto — val_nll minimo ~0.12 verso l'epoca 80-110, poi risalito a 0.45 all'epoca 199
mentre train_nll continuava a scendere. train() ora traccia il miglior val_nll epoca per
epoca e, a fine training, ripristina i pesi del checkpoint migliore prima di restituire
il controllo a save() — niente piu' bisogno di indovinare --epochs per evitare overfit.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from exciters import PARAM_RANGES as EXCITER_PARAM_RANGES
from resonator import PARAM_RANGES as RESONATOR_PARAM_RANGES

EPS = 1e-8
DEFAULT_N_COMPONENTS = 5
DEFAULT_HIDDEN = 64

# Vettore descrittori in ingresso — stesso ordine/nomi di analyzer.descriptors
# (esclusi "decay_capped", che e' un flag non un descrittore, e i metadati CSV).
# 2026-09-16f: +spectral_rolloff (accanto a centroid/spread, stesso gruppo "forma
# spettrale") e +inharmonicity (accanto a harmonic_tension, stesso gruppo "carattere
# armonico") -- vedi analyzer/descriptors.py per le formule e le motivazioni.
DESCRIPTOR_KEYS = [
    "spectral_centroid", "spectral_spread", "spectral_rolloff", "spectral_flatness",
    "roughness", "harmonic_tension", "inharmonicity", "formant_f1", "formant_f2",
    "formant_f3", "mod_rate", "mod_depth", "attack_time", "decay_time", "pitch",
]

# decay_time ha senso fisico per eccitazioni impulsive (pluck, strike) e per i 4
# resonator_* (eccitati da impulso in eval/inferenza: il decadimento e' proprieta' del
# risonatore stesso, damping/loss). Per gli eccitatori sostenuti (bow, blow, chaos,
# noise, shaker) e' risultato empiricamente una funzione quasi deterministica di
# attack_time (decay ~= durata_render - attack, corr>0.999 su bow/blow/chaos/noise,
# 2026-09-14) -> zero informazione indipendente, escluso dal vettore target per quegli
# agenti (resta nel CSV, solo non e' usato per training/predict/eval). Nota: sui
# resonator_* e' risultato ugualmente quasi-degenere con l'attuale durata di render
# (1.5s troppo corta per farli decadere sotto floor_db coi range di damping/loss
# attuali, decay mediano 1.42-1.45s) -- tenerlo comunque per loro e' una scelta di
# progetto, ma finche' la durata di render non cambia il modello non avra' molto da
# imparare oltre alla stessa relazione con attack_time.
AGENTS_WITH_DECAY = {"pluck", "strike"} | {f"resonator_{s}" for s in RESONATOR_PARAM_RANGES}


def descriptor_keys_for(agent_name):
    """Sottoinsieme di DESCRIPTOR_KEYS usato in input/target da un dato agente."""
    keys = list(DESCRIPTOR_KEYS)
    if agent_name not in AGENTS_WITH_DECAY:
        keys.remove("decay_time")
    if agent_name.startswith("resonator_"):  # 2026-09-24: pitch = freq eccitatore, non piu' input risonatore
        keys.remove("pitch")
    return keys


# 2026-09-14: stesso trattamento log10 di LOG_PARAMS (sez. parametri, sopra) ma sul lato
# descrittori, mai fatto finora. attack_time/decay_time hanno un range dinamico enorme
# (es. attack_time 5ms-1.5s, ~300x su bow) -- normalizzati linearmente (z-score) come gli
# altri descrittori, la coda destra schiaccia la rappresentazione e il modello risolve
# male gli attacchi/decadimenti brevi. Applicato SOLO come feature di input (_desc_to_x/
# train X), mai ai parametri predetti in uscita -- nessun bisogno di trasformazione inversa.
LOG_DESCRIPTORS = {"attack_time", "decay_time", "pitch"}  # pitch: log come freq (NaN propagato -> 0 dopo z-score)


def _log_transform_descriptors(arr, keys):
    """arr: array con ultima dimensione = len(keys) (vettore singolo o batch). Log10 in-
    place (su copia) sulle colonne in LOG_DESCRIPTORS."""
    arr = np.array(arr, dtype=np.float32, copy=True)
    for i, k in enumerate(keys):
        if k in LOG_DESCRIPTORS:
            arr[..., i] = np.log10(np.maximum(arr[..., i], EPS))
    return arr

# 2026-09-16h (fix del fix, vedi criticita_modelli.md): il tentativo precedente
# (2026-09-16g, sentinel -1.0 per inharmonicity) ha PEGGIORATO l'errore quasi
# ovunque nel retrain di conferma -- np.nan_to_num()/_fill_nan_descriptors()
# iniettavano il sentinel PRIMA di calcolare mean/std, quindi le statistiche di
# normalizzazione restavano contaminate dal sentinel (0.0 o -1.0 non importa): con
# il 40-65% delle righe a un singolo valore fisso, mean/std condivisi da TUTTE le
# righe (anche quelle valide) si spostano/si gonfiano, schiacciando la risoluzione
# disponibile per i valori VERI. Fix corretto: mean/std calcolati SOLO sui valori
# realmente misurati (np.nanmean/np.nanstd, NaN ignorati invece che sostituiti), il
# fill a 0.0 avviene DOPO la normalizzazione (z-score neutro, "assumi la media" --
# mai un valore di dominio, quindi nessuna collisione possibile con un valore reale
# per nessun descrittore). Sostituisce sia il vecchio np.nan_to_num() sia
# NAN_SENTINEL/_fill_nan_descriptors: non serve piu' un sentinel per-descrittore,
# la stessa funzione e' corretta per tutti.
def _nan_safe_stats(X):
    """Media/std lungo l'asse 0, ignorando i NaN (np.nanmean/np.nanstd) -- X puo'
    essere 1D o 2D (batch)."""
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0) + EPS
    return mean.astype(np.float32), std.astype(np.float32)

# Parametri con range che copre piu' ordini di grandezza: normalizzati in log10
# invece che linearmente, per non schiacciare la mistura su valori piccoli. Includono sia
# nomi eccitatore (exciters.PARAM_RANGES) sia nomi risonatore (resonator.PARAM_RANGES):
# nessuna collisione tra i due vocabolari, safe da tenere in un unico set.
LOG_PARAMS = {
    "freq", "hammer_stiffness", "coupling_rate", "n_particles", "rate", "gate_rate", "ps", "fold_q", "kc_scale",  # eccitatori
    "size", "thickness", "density", "stiffness", "loss",              # risonatore
    "radius", "ortho", "cavity", "hole", "speed", "beat", "beat",                      # risonatore tube/soundboard/chaotic (2026-09-21)
}

# Risonatore: un agente per topologia (resonator.PARAM_RANGES e' gia' un dict
# forma -> range parametri, stesso schema di exciters.PARAM_RANGES). Nomi agente
# prefissati "resonator_<forma>" per non confondersi con gli eccitatori.
RESONATOR_SHAPES = list(RESONATOR_PARAM_RANGES)  # ["bar", "plate_rect", "plate_circ", "membrane"]

# 2026-09-24: pitch lock. Negli eccitatori intonati `freq` = pitch target, imposto a runtime
# (non piu' predetto da MDN/KNN). Shaker/noise: pitch sempre NaN, freq resta centro spettrale.
PITCH_LOCKED_EXCITERS = {"bow", "blow", "strike", "pluck", "chaos", "mechanical", "bird", "vocal"}
AGENT_SPECS = {
    _n: ({p: r for p, r in _s.items() if p != "freq"} if _n in PITCH_LOCKED_EXCITERS else _s)
    for _n, _s in EXCITER_PARAM_RANGES.items()
}
for _shape in RESONATOR_SHAPES:
    AGENT_SPECS[f"resonator_{_shape}"] = RESONATOR_PARAM_RANGES[_shape]


# 2026-09-15: condizionamento opzionale dell'agente ECCITATORE sul risonatore accoppiato
# casuale (pair_resonator_shape/pair_res_* gia' loggati da dataset_gen.py). Diagnosi
# (analisi_pluck.py, vedi criticita_modelli.md sez. pluck): su pluck/chaos una quota
# consistente della varianza di formanti/harmonic_tension e' spiegata dal risonatore
# accoppiato, non dai parametri propri dell'eccitatore -- ma quel risonatore e' oggi
# invisibile all'agente (variabile confondente). Dandolo in input il confondimento
# sparisce per costruzione. Unione dei nomi parametro su tutte le forme: alcuni esistono
# solo per una forma (es. "aspect" solo plate_rect) -> 0.0 per le altre.
RESONATOR_PARAM_UNION = sorted({p for _shape in RESONATOR_SHAPES for p in RESONATOR_PARAM_RANGES[_shape]})
LOG_RESONATOR_PARAMS = {"size", "thickness", "density", "stiffness", "loss", "radius", "ortho", "cavity", "hole", "speed", "beat", "beat"}  # stessa scelta di LOG_PARAMS


def resonator_condition_keys():
    """Nomi delle feature di condizionamento: one-hot forma + parametri fisici (uniti al
    vettore descrittori quando Agent e' creato con condition_on_resonator=True)."""
    return [f"res_shape_{s}" for s in RESONATOR_SHAPES] + [f"res_{p}" for p in RESONATOR_PARAM_UNION]


def resonator_condition_features(shape, params):
    """shape: forma del risonatore accoppiato; params: dict nome->valore SOLO per i
    parametri di quella forma (es. {k: row[f'pair_res_{k}'] for k in
    RESONATOR_PARAM_RANGES[shape]}, niente celle vuote). Ritorna un dict da unire al
    dict descrittori passato a predict()/train() di un agente condizionato."""
    out = {f"res_shape_{s}": (1.0 if s == shape else 0.0) for s in RESONATOR_SHAPES}
    for p in RESONATOR_PARAM_UNION:
        if p in params and params[p] != "":
            v = float(params[p])
            out[f"res_{p}"] = float(np.log10(max(v, EPS))) if p in LOG_RESONATOR_PARAMS else v
        else:
            out[f"res_{p}"] = 0.0
    return out


# ---------------- Mixture Density Network ----------------

class MDN(nn.Module):
    """MLP (2 hidden layer, tanh) + mistura di K gaussiane diagonali su n_out parametri
    continui. Stessa architettura per tutti gli agenti (eccitatori e forme risonatore)."""

    def __init__(self, n_in, n_out, n_components=DEFAULT_N_COMPONENTS, hidden=DEFAULT_HIDDEN):
        super().__init__()
        self.n_out, self.k = n_out, n_components
        self.backbone = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi_head = nn.Linear(hidden, n_components)
        self.mu_head = nn.Linear(hidden, n_components * n_out)
        self.log_sigma_head = nn.Linear(hidden, n_components * n_out)

    def forward(self, x):
        h = self.backbone(x)
        pi = torch.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h).view(-1, self.k, self.n_out)
        sigma = torch.exp(self.log_sigma_head(h).clamp(-6, 2)).view(-1, self.k, self.n_out)
        return pi, mu, sigma


def mdn_loss(pi, mu, sigma, y):
    """Negative log-likelihood della mistura (y in [0,1]^n_out, spazio normalizzato)."""
    y = y.unsqueeze(1)  # (B,1,D) broadcast contro (B,K,D)
    var = sigma ** 2
    log_comp = -0.5 * (((y - mu) ** 2) / (var + EPS) + torch.log(2 * np.pi * var + EPS)).sum(-1)
    log_mix = torch.log(pi + EPS) + log_comp
    return -torch.logsumexp(log_mix, dim=-1).mean()


def _sample_mixture(pi, mu, sigma, rng, temperature=1.0):
    """Campiona UNA configurazione dalla mistura (batch size 1). temperature>1 = piu'
    varieta' tra chiamate identiche (coerente col non-timbral-matching del progetto).
    temperature<=0 = deterministico (media della componente piu' probabile, nessun
    rumore): serve per la valutazione (eval_agent.py). Nota: p**(1/temperature) con
    temperature vicino a 0 sottoflussa a 0 ovunque (NaN dopo la normalizzazione) --
    per questo e' un caso a parte, non il limite della formula generale."""
    p = pi.detach().numpy()[0]
    if temperature <= EPS:
        comp = int(np.argmax(p))
    else:
        if temperature != 1.0:
            p = p ** (1.0 / temperature)
            p = p / p.sum()
        comp = rng.choice(len(p), p=p)
    m = mu.detach().numpy()[0, comp]
    s = sigma.detach().numpy()[0, comp] * temperature
    return m + rng.standard_normal(len(m)) * s


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------- Agente ----------------

class Agent:
    """Un agente = un modello inverso descrittori -> parametri, per UN eccitatore
    (name in exciters.PARAM_RANGES) o per UNA forma di risonatore
    (name == "resonator_<bar|plate_rect|plate_circ|membrane>")."""

    def __init__(self, name, n_components=DEFAULT_N_COMPONENTS, hidden=DEFAULT_HIDDEN,
                 condition_on_resonator=False):
        if name not in AGENT_SPECS:
            raise ValueError(f"agente sconosciuto: {name} (validi: {list(AGENT_SPECS)})")
        self.name = name
        self.param_ranges = AGENT_SPECS[name]
        self.param_names = list(self.param_ranges)
        self.hidden = hidden
        self._eff_ranges = {
            p: ((np.log10(lo), np.log10(hi)) if p in LOG_PARAMS else (lo, hi))
            for p, (lo, hi) in self.param_ranges.items()
        }
        self.condition_on_resonator = condition_on_resonator
        self.descriptor_keys = descriptor_keys_for(name)
        if condition_on_resonator:
            self.descriptor_keys = self.descriptor_keys + resonator_condition_keys()
        self.model = MDN(len(self.descriptor_keys), len(self.param_names), n_components, hidden)
        self.desc_mean = np.zeros(len(self.descriptor_keys), dtype=np.float32)
        self.desc_std = np.ones(len(self.descriptor_keys), dtype=np.float32)
        self.rng = np.random.default_rng()

    # ---- normalizzazione ----

    def _desc_to_x(self, descriptors):
        vec = np.array([descriptors.get(k, 0.0) for k in self.descriptor_keys], dtype=np.float32)
        vec = _log_transform_descriptors(vec, self.descriptor_keys)
        vec = (vec - self.desc_mean) / (self.desc_std + EPS)
        return np.nan_to_num(vec)  # NaN (non misurato) -> 0.0 in spazio z-score, DOPO la normalizzazione

    def _params_to_y(self, params):
        y = []
        for name in self.param_names:
            lo, hi = self._eff_ranges[name]
            v = float(params[name])
            v = np.log10(max(v, EPS)) if name in LOG_PARAMS else v
            y.append((v - lo) / (hi - lo + EPS))
        return np.array(y, dtype=np.float32)

    def _y_to_params(self, y01):
        params = {}
        for name, val in zip(self.param_names, y01):
            lo, hi = self._eff_ranges[name]
            v = float(np.clip(val, 0.0, 1.0)) * (hi - lo) + lo
            params[name] = float(10 ** v) if name in LOG_PARAMS else float(v)
        return params

    # ---- inferenza (runtime, O(1)) ----

    def predict(self, descriptors, temperature=1.0, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        x = torch.from_numpy(self._desc_to_x(descriptors)).float().unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            pi, mu, sigma = self.model(x)
            y = _sample_mixture(pi, mu, sigma, rng, temperature)
            params = self._y_to_params(y)
        return params

    # ---- training (offline, sul dataset sintetico invertito) ----

    def _input_row(self, row):
        """row: dict CSV (stringhe). Ritorna dict nome->float sui descriptor_keys BASE
        (senza condizionamento), piu' le feature risonatore se condition_on_resonator."""
        base_keys = descriptor_keys_for(self.name)
        out = {k: float(row[k]) for k in base_keys}
        if self.condition_on_resonator:
            shape = row["pair_resonator_shape"]
            res_params = {k: row[f"pair_res_{k}"] for k in RESONATOR_PARAM_RANGES[shape]}
            out.update(resonator_condition_features(shape, res_params))
        return out

    def train(self, csv_path, epochs=200, batch_size=64, lr=1e-3, val_split=0.1, seed=0):
        rows = _read_csv(csv_path)
        if not rows:
            raise ValueError(f"CSV vuoto: {csv_path}")
        rows_in = [self._input_row(r) for r in rows]
        X = np.array([[ri[k] for k in self.descriptor_keys] for ri in rows_in], dtype=np.float32)
        X = _log_transform_descriptors(X, self.descriptor_keys)
        Y = np.array([self._params_to_y(r) for r in rows], dtype=np.float32)

        self.desc_mean, self.desc_std = _nan_safe_stats(X)
        Xn = np.nan_to_num((X - self.desc_mean) / self.desc_std)

        n = len(Xn)
        idx = np.random.default_rng(seed).permutation(n)
        n_val = max(1, int(n * val_split))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        Xt, Yt = torch.from_numpy(Xn[tr_idx]), torch.from_numpy(Y[tr_idx])
        Xv, Yv = torch.from_numpy(Xn[val_idx]), torch.from_numpy(Y[val_idx])

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        n_tr = len(Xt)
        log_every = max(1, epochs // 20)
        # early stopping: traccia il checkpoint al miglior val_nll, non l'ultimo epoch —
        # vedi nota nel docstring del modulo (overfitting osservato sul primo training).
        best_val = float("inf")
        best_state = None
        best_epoch = -1
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(n_tr)
            total = 0.0
            for i in range(0, n_tr, batch_size):
                b = perm[i:i + batch_size]
                pi, mu, sigma = self.model(Xt[b])
                loss = mdn_loss(pi, mu, sigma, Yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            self.model.eval()
            with torch.no_grad():
                pi, mu, sigma = self.model(Xv)
                vloss = mdn_loss(pi, mu, sigma, Yv).item()
            is_best = vloss < best_val
            if is_best:
                best_val = vloss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            if epoch % log_every == 0 or epoch == epochs - 1:
                print(f"[{self.name}] epoch {epoch:4d}  train_nll={total / n_tr:.4f}  val_nll={vloss:.4f}"
                      f"{'  *best*' if is_best else ''}")
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[{self.name}] pesi ripristinati al miglior val_nll={best_val:.4f} (epoch {best_epoch})")

    # ---- persistenza ----

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(
            name=self.name, state=self.model.state_dict(),
            desc_mean=self.desc_mean, desc_std=self.desc_std,
            n_components=self.model.k, hidden=self.hidden,
            condition_on_resonator=self.condition_on_resonator,
        ), path)

    @classmethod
    def from_checkpoint(cls, path):
        # weights_only=False: PyTorch >=2.6 di default rifiuta i numpy array (desc_mean/
        # desc_std) nel checkpoint. Fonte fidata (i checkpoint li generiamo noi stessi).
        ck = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(ck["name"], n_components=ck["n_components"], hidden=ck["hidden"],
                    condition_on_resonator=ck.get("condition_on_resonator", False))
        agent.model.load_state_dict(ck["state"])
        agent.desc_mean, agent.desc_std = ck["desc_mean"], ck["desc_std"]
        return agent


# ---------------- Manager: un eccitatore + una forma risonatore alla volta ----------------

class AgentManager:
    """Tiene in memoria al massimo 2 agenti: UN eccitatore e UNA forma di risonatore,
    entrambi selezionati manualmente (resonator.py richiede "shape" fisso per generare
    qualunque parametro, quindi la forma si comporta come l'eccitatore, non e' "sempre
    attiva" in blocco). Cambiare selezione scarica il precedente e carica il nuovo —
    non c'e' routing/IA sulla scelta stessa, ne' per l'eccitatore ne' per la forma."""

    def __init__(self, weights_dir="weights"):
        self.weights_dir = Path(weights_dir)
        self.exciter_name = None
        self.exciter = None
        self.resonator_shape = None
        self.resonator = None

    def _load_or_new(self, name):
        ckpt = self.weights_dir / f"{name}.pt"
        return Agent.from_checkpoint(ckpt) if ckpt.exists() else Agent(name)

    def select_exciter(self, name):
        if name not in EXCITER_PARAM_RANGES:
            raise ValueError(f"eccitatore sconosciuto: {name} (validi: {list(EXCITER_PARAM_RANGES)})")
        if name != self.exciter_name:
            self.exciter = self._load_or_new(name)  # sostituisce il precedente, non lo affianca
            self.exciter_name = name

    def select_resonator(self, shape):
        if shape not in RESONATOR_SHAPES:
            raise ValueError(f"forma risonatore sconosciuta: {shape} (valide: {RESONATOR_SHAPES})")
        if shape != self.resonator_shape:
            self.resonator = self._load_or_new(f"resonator_{shape}")
            self.resonator_shape = shape

    def infer(self, descriptors, temperature=1.0, seed=None):
        if self.exciter is None or self.resonator is None:
            raise RuntimeError(
                "seleziona prima eccitatore e forma risonatore: select_exciter(nome), select_resonator(forma)")
        return {
            "exciter": self.exciter_name,
            "exciter_params": self.exciter.predict(descriptors, temperature, seed),
            "resonator_shape": self.resonator_shape,
            "resonator_params": self.resonator.predict(descriptors, temperature, seed),
        }


# ---------------- CLI ----------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Training/inferenza offline degli 8 agenti IA (sez.5 pipeline).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="allena un agente su un CSV (descrittori + parametri dataset invertito)")
    pt.add_argument("agent", choices=list(AGENT_SPECS))
    pt.add_argument("csv", help="colonne: DESCRIPTOR_KEYS + parametri dell'agente (specifici della forma, se resonator_*)")
    pt.add_argument("--epochs", type=int, default=200)
    pt.add_argument("--components", type=int, default=DEFAULT_N_COMPONENTS)
    pt.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    pt.add_argument("--out", default=None, help="checkpoint .pt (default weights/<agent>.pt)")
    pt.add_argument("--condition-on-resonator", action="store_true",
                     help="aggiunge shape+parametri del risonatore accoppiato come input extra "
                          "(solo eccitatori, vedi resonator_condition_features)")

    pp = sub.add_parser("predict", help="inferenza: descrittori JSON -> parametri")
    pp.add_argument("agent", choices=list(AGENT_SPECS))
    pp.add_argument("descriptors_json", help="file JSON con i descrittori target")
    pp.add_argument("--checkpoint", default=None)
    pp.add_argument("--temperature", type=float, default=1.0)
    pp.add_argument("--seed", type=int, default=None)

    args = p.parse_args()

    if args.cmd == "train":
        if args.condition_on_resonator and args.agent.startswith("resonator_"):
            p.error("--condition-on-resonator si applica solo agli eccitatori, non ai resonator_*")
        agent = Agent(args.agent, n_components=args.components, hidden=args.hidden,
                      condition_on_resonator=args.condition_on_resonator)
        agent.train(args.csv, epochs=args.epochs)
        out = Path(args.out or f"weights/{args.agent}.pt")
        agent.save(out)
        print(f"salvato: {out}")

    elif args.cmd == "predict":
        ckpt = Path(args.checkpoint or f"weights/{args.agent}.pt")
        agent = Agent.from_checkpoint(ckpt) if ckpt.exists() else Agent(args.agent)
        with open(args.descriptors_json) as f:
            descriptors = json.load(f)
        params = agent.predict(descriptors, temperature=args.temperature, seed=args.seed)
        print(json.dumps(params, indent=2))
