"""surrogate.py (2026-09-26) -- modello DIRETTO parametri -> descrittori + inversione per ottimizzazione.

Sostituisce gli agenti MDN/KNN (diag_routing.py: correlazione target/ottenuto ~0 sugli MDN, KNN valido solo su target
gia' presenti nel corpus). Il problema diretto e' ben posto (una configurazione -> un suono), quello inverso no.

  train  : per eccitatore, MLP (config codificata -> 15 descrittori trasformati + logit "intonato") su dataset_v5/.
  invert : dato un target (dict descrittori), forma e freq, cerca i parametri che minimizzano l'errore pesato:
           32 punti di partenza (24 vicini KNN nel dataset con la stessa forma + soluzione precedente + casuali),
           Adam sugli ingressi (sigmoide -> sempre dentro i range), 150 passi in batch; vince il minimo.
           Ogni descrittore pesa uguale (errore in unita' di deviazione standard del dataset) -> ogni slider conta.
  eval   : come diag_routing.py ma col surrogato: target slider/raggiungibili -> invert -> render -> correlazioni.

Uso:
    python3 surrogate.py train [eccitatori...] [--epochs 150]      -> weights_v5/<eccitatore>.pt
    python3 surrogate.py eval  [eccitatori...] [--shape bar]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import agents
import exciters as E
import resonator as R

DKEYS = list(agents.DESCRIPTOR_KEYS)
SHAPES = list(R.PARAM_RANGES)
RES_NAMES = sorted({k for r in R.PARAM_RANGES.values() for k in r})
COUP = list(R.COUPLING_RANGES)
# parametri d'accoppiamento di dataset precedenti (v3/v4), per poter rileggerli (es. curva di apprendimento su dataset_v4)
LEGACY_COUP = dict(env_attack=((0.001, 0.5), True), env_decay=((0.05, 8.0), True))


def _crange(k):
    if k in R.COUPLING_RANGES:
        return R.COUPLING_RANGES[k], k in R.COUPLING_LOG
    return LEGACY_COUP[k]
LOG_X = set(agents.LOG_PARAMS) | {"freq"}
LOG_R = set(agents.LOG_RESONATOR_PARAMS)
FREE_FREQ = {"noise", "shaker"}  # freq = timbro (non pitch-lock): la ottimizza il solutore
PITCH_TOL_C = 50.0
# trasformazione dei descrittori: log10(v + offset) (None = lineare)
OFFSET = dict(spectral_centroid=1.0, spectral_spread=1.0, spectral_rolloff=1.0, spectral_flatness=1e-4,
              roughness=1e-4, harmonic_tension=None, inharmonicity=5e-3, formant_f1=1.0, formant_f2=1.0,
              formant_f3=1.0, mod_rate=1e-2, mod_depth=1e-4, attack_time=1e-3, decay_time=1e-3, pitch=1.0)
DATA_DIR = Path("dataset_v5")  # v5 = attacco/tenuta all'eccitatore, decadimento al risonatore
WEIGHTS_DIR = Path("weights_v5")


def tdesc(v, k):
    v = np.asarray(v, dtype=np.float64)
    o = OFFSET[k]
    return v if o is None else np.log10(np.maximum(v, 0.0) + o)


# ---------------- codifica della configurazione ----------------

class Codec:
    """Ordine ingressi: parametri eccitatore (freq compresa) | one-hot forma | parametri risonatore (unione, 0 se
    assenti, normalizzati col range DELLA forma) | accoppiamento. Tutto in [0,1]."""

    def __init__(self, exc, coup=None):
        self.coup = list(coup) if coup else list(COUP)
        self.exc = exc
        self.xnames = list(E.PARAM_RANGES[exc])
        self.n_x, self.n_s, self.n_r = len(self.xnames), len(SHAPES), len(RES_NAMES)
        self.dim = self.n_x + self.n_s + self.n_r + len(self.coup)

    @staticmethod
    def _to01(v, lo, hi, log):
        if log and lo > 0:
            v, lo, hi = np.log10(max(v, 1e-12)), np.log10(lo), np.log10(hi)
        return float(np.clip((v - lo) / (hi - lo + 1e-12), 0.0, 1.0))

    @staticmethod
    def _from01(u, lo, hi, log):
        if log and lo > 0:
            return float(10 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo))))
        return float(lo + u * (hi - lo))

    def encode(self, xp, shape, rp, cp):
        v = np.zeros(self.dim, dtype=np.float32)
        for i, k in enumerate(self.xnames):
            lo, hi = E.PARAM_RANGES[self.exc][k]
            v[i] = self._to01(xp[k], lo, hi, k in LOG_X)
        v[self.n_x + SHAPES.index(shape)] = 1.0
        for k, (lo, hi) in R.PARAM_RANGES[shape].items():
            v[self.n_x + self.n_s + RES_NAMES.index(k)] = self._to01(rp[k], lo, hi, k in LOG_R)
        for i, k in enumerate(self.coup):
            (lo, hi), lg = _crange(k)
            v[self.n_x + self.n_s + self.n_r + i] = self._to01(cp[k], lo, hi, lg)
        return v

    def decode(self, v, shape):
        xp = {k: self._from01(float(v[i]), *E.PARAM_RANGES[self.exc][k], k in LOG_X) for i, k in enumerate(self.xnames)}
        if "n_particles" in xp:
            xp["n_particles"] = int(round(xp["n_particles"]))
        rp = {k: self._from01(float(v[self.n_x + self.n_s + RES_NAMES.index(k)]), lo, hi, k in LOG_R)
              for k, (lo, hi) in R.PARAM_RANGES[shape].items()}
        cp = {k: self._from01(float(v[self.n_x + self.n_s + self.n_r + i]), *_crange(k)[0], _crange(k)[1])
              for i, k in enumerate(self.coup)}
        return xp, rp, cp

    def free_mask(self, shape, freq_free):
        """1 = variabile ottimizzata. Fissi: one-hot forma, parametri di altre forme, freq (se pitch-lock)."""
        m = np.zeros(self.dim, dtype=np.float32)
        for i, k in enumerate(self.xnames):
            m[i] = 1.0 if (k != "freq" or freq_free) else 0.0
        for k in R.PARAM_RANGES[shape]:
            m[self.n_x + self.n_s + RES_NAMES.index(k)] = 1.0
        m[self.n_x + self.n_s + self.n_r:] = 1.0
        return m


PITCH_FEAT = True  # --no-pitch-feat per confronto
PC_HARM = 3  # armoniche delle feature di classe d'altezza


class Net(nn.Module):
    """pitch_feat = (indice di freq nell'ingresso, log2 lo, log2 hi): aggiunge sin/cos(2*pi*k*log2 f), k=1..PC_HARM
    (classe d'altezza: harmonic_tension/tonal focus e inharmonicity dipendono dall'altezza in modo periodico per
    ottava, difficile da imparare da log f lineare). Calcolate dentro la rete -> differenziabili anche con freq libera."""

    def __init__(self, n_in, n_out, hidden=256, pitch_feat=None):
        super().__init__()
        self.pf = tuple(pitch_feat) if pitch_feat else None
        n_in = n_in + (2 * PC_HARM if self.pf else 0)
        self.f = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                               nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        if self.pf:
            i, lo, hi = self.pf
            l2 = lo + x[:, int(i)] * (hi - lo)
            k = torch.arange(1, PC_HARM + 1, dtype=x.dtype)
            a = 2 * np.pi * l2[:, None] * k[None, :]
            x = torch.cat([x, torch.sin(a), torch.cos(a)], dim=1)
        return self.f(x)


def pitch_feat_of(codec):
    if "freq" not in codec.xnames:
        return None
    lo, hi = E.PARAM_RANGES[codec.exc]["freq"]
    return (codec.xnames.index("freq"), float(np.log2(lo)), float(np.log2(hi)))


# ---------------- dati ----------------

def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def data_files(exc, data_dir=None):
    d = Path(data_dir or DATA_DIR)
    return [d / f"{exc}.csv"] if (d / f"{exc}.csv").exists() else sorted(d.glob(f"{exc}.part*.csv"))


def read_rows(exc, data_dir=None):
    rows = []
    for f in data_files(exc, data_dir):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def load_rows(exc, data_dir=None):
    X, Y, P, S = [], [], [], []
    rows = read_rows(exc, data_dir)
    codec = Codec(exc, [k for k in rows[0] if k in R.COUPLING_RANGES or k in LEGACY_COUP])
    if True:
        for r in rows:
            shape = r["shape"]
            xp = {k: _f(r[f"x_{k}"]) for k in codec.xnames}
            rp = {k: _f(r[f"r_{k}"]) for k in R.PARAM_RANGES[shape]}
            cp = {k: _f(r[k]) for k in codec.coup}
            X.append(codec.encode(xp, shape, rp, cp))
            d = [tdesc(_f(r[k]), k) for k in DKEYS]
            Y.append(d)
            err = _f(r["pitch_err_c"])
            P.append(1.0 if np.isfinite(err) and abs(err) <= PITCH_TOL_C else 0.0)
            S.append(SHAPES.index(shape))
    Y = np.array(Y, dtype=np.float64)
    pi = DKEYS.index("pitch")
    Y[np.array(P) < 0.5, pi] = np.nan  # pitch come descrittore solo dove il suono e' intonato
    return codec, np.array(X, np.float32), Y, np.array(P, np.float32), np.array(S)


# ---------------- training ----------------

def _fit_one(Xt, Zt, Mt, Pt, tr, va, dim, epochs, hidden, seed, exc, verbose, pf=None):
    torch.manual_seed(seed)
    net = Net(dim, len(DKEYS) + 1, hidden, pf)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((len(tr) + 255) // 256))
    bce = nn.BCEWithLogitsLoss()

    def loss_fn(b):
        out = net(Xt[b])
        mse = ((out[:, :-1] - Zt[b]) ** 2 * Mt[b]).sum() / Mt[b].sum().clamp(min=1)
        return mse + 0.3 * bce(out[:, -1], Pt[b])

    best, best_state = 1e9, None
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(tr))
        for i in range(0, len(perm), 256):
            l = loss_fn(perm[i:i + 256])
            opt.zero_grad()
            l.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(torch.from_numpy(va)).item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    if verbose:
        print(f"[{exc}] rete {seed}: val={best:.4f}", file=sys.stderr, flush=True)
    net.load_state_dict(best_state)
    net.eval()
    return net, best, best_state


def train(exc, epochs=80, hidden=256, seed=0, frac=1.0, save=True, ens=1):
    """ens > 1: media di `ens` reti con inizializzazione/ordine dei dati diversi (stessa validazione).
    Stampa R2 e ERRORE per descrittore = errore medio assoluto / (p95-p5) del descrittore, in %, sulla validazione."""
    codec, X, Y, P, S = load_rows(exc)
    mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0) + 1e-9
    Z = (Y - mu) / sd
    M = np.isfinite(Z).astype(np.float32)
    Z = np.nan_to_num(Z).astype(np.float32)
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    nv = max(1, n // 10)
    va, tr = idx[:nv], idx[nv:]
    tr = tr[:max(256, int(len(tr) * frac))]  # curva di apprendimento: frazione del training (validazione fissa)
    T = lambda a: torch.from_numpy(a)
    Xt, Zt, Mt, Pt = T(X), T(Z), T(M), T(P)
    t0 = time.time()
    nets, states, bests = [], [], []
    for e in range(ens):
        net, b, st = _fit_one(Xt, Zt, Mt, Pt, tr, va, codec.dim, epochs, hidden, seed + e, exc, save and ens > 1,
                              pitch_feat_of(codec) if PITCH_FEAT else None)
        nets.append(net)
        states.append(st)
        bests.append(b)
    with torch.no_grad():
        out = torch.stack([nt(Xt[va]) for nt in nets]).mean(0).numpy()
    r2, err = [], []
    for j, k in enumerate(DKEYS):
        m = M[va, j] > 0
        zt, zo = Z[va][m, j], out[m, j]
        ss = np.sum((zt - zt.mean()) ** 2) + 1e-12
        r2.append(1.0 - np.sum((zo - zt) ** 2) / ss)
        p5, p95 = np.percentile(zt, [5, 95])
        err.append(100.0 * np.mean(np.abs(zo - zt)) / (p95 - p5 + 1e-12))
    acc = np.mean((out[:, -1] > 0) == (P[va] > 0.5))
    tag = f"ens={ens}" if ens > 1 else "singola"
    print(f"[{exc}] {tag} R2:     " + " ".join(f"{k[:10]}={v:.2f}" for k, v in zip(DKEYS, r2)) + f"  intonato_acc={acc:.2f}",
          file=sys.stderr, flush=True)
    print(f"[{exc}] {tag} ERRORE: " + " ".join(f"{k[:10]}={v:4.1f}%" for k, v in zip(DKEYS, err))
          + f"   sotto 10%: {sum(e < 10 for e in err)}/{len(err)}  ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    if not save:
        return float(np.mean(bests)), float(np.mean(r2))
    WEIGHTS_DIR.mkdir(exist_ok=True)
    torch.save(dict(exc=exc, coup=codec.coup, pf=pitch_feat_of(codec) if PITCH_FEAT else None, hidden=hidden, state=states[0], states=states, mu=mu, sd=sd,
                    X=X.astype(np.float16), Z=Z.astype(np.float16), M=M.astype(np.uint8), S=S.astype(np.uint8),
                    r2=r2, err=err), WEIGHTS_DIR / f"{exc}.pt")


def curve(exc, epochs=80):
    """Curva di apprendimento: stessa validazione, training su 1/8, 1/4, 1/2, tutto. Se val/R2 migliorano ancora
    raddoppiando i dati, il dataset e' troppo piccolo; la pendenza dell'ultimo raddoppio stima il guadagno del prossimo."""
    out = []
    for f in (0.125, 0.25, 0.5, 1.0):
        v, r = train(exc, epochs, frac=f, save=False)
        out.append((f, v, r))
        print(f"[{exc}] fatto frazione {f:5.3f}: val={v:.3f}  R2 medio={r:.3f}", file=sys.stderr, flush=True)
    print(f"CURVA {exc:11s} " + "  ".join(f"{f:5.3f}: val={v:.3f} R2={r:.3f}" for f, v, r in out), file=sys.stderr, flush=True)


# ---------------- inversione ----------------

class Inverter:
    def __init__(self, exc, weights_dir=None):
        # None -> WEIGHTS_DIR letto ORA (prima il default era fissato all'import e --weights non aveva effetto)
        ck = torch.load(Path(weights_dir or WEIGHTS_DIR) / f"{exc}.pt", weights_only=False)
        self.exc, self.codec = exc, Codec(exc, ck.get("coup"))
        self.nets = []
        for st in ck.get("states", [ck["state"]]):  # ensemble: uscita = media delle reti
            nt = Net(self.codec.dim, len(DKEYS) + 1, ck["hidden"], ck.get("pf"))
            nt.load_state_dict(st)
            nt.eval()
            for p in nt.parameters():
                p.requires_grad_(False)
            self.nets.append(nt)
        self.mu, self.sd = ck["mu"], ck["sd"]
        self.X, self.Z, self.M, self.S = ck["X"].astype(np.float32), ck["Z"].astype(np.float32), ck["M"], ck["S"]
        self.prev = {}  # forma -> ultimo vettore soluzione (continuita' tra target vicini)
        self.prev_z = None  # ultimo target (z): i descrittori appena mossi pesano CHANGED_GAIN volte gli altri
        self.moved = None
        self.bias = np.zeros(len(DKEYS), np.float32)  # correzione a circuito chiuso (z), vedi correct()

    def _fwd(self, x):
        return torch.stack([nt(x) for nt in self.nets]).mean(0) if len(self.nets) > 1 else self.nets[0](x)

    def target_z(self, target):
        z, w = np.zeros(len(DKEYS), np.float32), np.zeros(len(DKEYS), np.float32)
        for j, k in enumerate(DKEYS):
            v = target.get(k)
            if v is not None and np.isfinite(v):
                z[j] = (tdesc(v, k) - self.mu[j]) / self.sd[j]
                w[j] = 1.0
        return z, w

    def correct(self, target, measured, rate=0.7, clip=2.0):
        """Circuito chiuso (in background): misurato il render, sposta il target interno dell'errore residuo del
        surrogato (bias per descrittore, in z). Il bias si dimezza sui descrittori il cui target cambia (solve)."""
        zt, wt = self.target_z(target)
        zm, wm = self.target_z(measured)
        m = (wt > 0) & (wm > 0)
        self.bias = np.clip(self.bias + rate * (zt - zm) * m, -clip, clip).astype(np.float32)

    def solve(self, target, shape, freq=None, n_knn=24, n_rand=7, steps=150, lr=0.05, prev_w=0.02, seed=None,
              changed_gain=4.0):
        c = self.codec
        freq_free = self.exc in FREE_FREQ or freq is None
        mask = c.free_mask(shape, freq_free)
        z, w = self.target_z(target)
        # 2026-09-26: priorita' allo slider appena mosso -- con 15 target indipendenti spesso incompatibili il minimo
        # e' un compromesso; il descrittore cambiato rispetto al target precedente pesa di piu', cosi' ogni
        # modifica si sente (gli altri restano vincoli piu' morbidi).
        if self.prev_z is not None:
            moved = (np.abs(z - self.prev_z) > 0.05) & (w > 0)
            if moved.any():  # la priorita' resta sull'ultimo slider mosso finche' non se ne muove un altro
                self.moved = moved
                self.bias = np.where(moved, 0.5 * self.bias, self.bias).astype(np.float32)
        if self.moved is not None:
            w = np.where(self.moved & (w > 0), w * changed_gain, w).astype(np.float32)
        self.prev_z = z.copy()
        z_goal = z.copy()  # target vero (senza bias): lo usa refine() sulle misure reali
        z = z + self.bias * (w > 0)  # il target che il surrogato deve colpire per ottenere quello vero
        if not freq_free:
            w[DKEYS.index("pitch")] = 0.0  # pitch = freq bloccata; conta la probabilita' "intonato"
        want_pitched = target.get("pitch") is not None and np.isfinite(target.get("pitch", np.nan))
        # punti di partenza
        rows = np.where(self.S == SHAPES.index(shape))[0]
        d = (((self.Z[rows] - z) ** 2) * w * self.M[rows]).sum(1)
        starts = [self.X[rows[i]] for i in np.argsort(d)[:n_knn]]
        rng = np.random.default_rng(seed)
        if shape in self.prev:
            starts.append(self.prev[shape])
        starts += [rng.uniform(0.05, 0.95, c.dim).astype(np.float32) for _ in range(n_rand)]
        base = np.stack(starts)
        fixed = np.zeros_like(base)  # tutto cio' che non e' ottimizzato: one-hot forma, freq bloccata, altre forme = 0
        fixed[:, c.n_x + SHAPES.index(shape)] = 1.0
        if not freq_free:
            lo, hi = E.PARAM_RANGES[self.exc]["freq"]
            fixed[:, c.xnames.index("freq")] = Codec._to01(float(np.clip(freq, lo, hi)), lo, hi, True)
        u = torch.logit(torch.from_numpy(np.clip(base, 0.02, 0.98)), eps=1e-4).requires_grad_(True)
        Mt, Ft = torch.from_numpy(mask), torch.from_numpy(fixed)
        zt, wt = torch.from_numpy(z), torch.from_numpy(w)
        pv = torch.from_numpy(self.prev.get(shape, base[0]))
        opt = torch.optim.Adam([u], lr=lr)
        for _ in range(steps):
            x = torch.sigmoid(u) * Mt + Ft
            out = self._fwd(x)
            loss = ((out[:, :-1] - zt) ** 2 * wt).sum(1) / wt.sum().clamp(min=1)
            if want_pitched and not freq_free:
                loss = loss + 0.5 * nn.functional.softplus(-out[:, -1])
            loss = loss + prev_w * (((x - pv) * Mt) ** 2).mean(1)
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
        with torch.no_grad():
            x = torch.sigmoid(u) * Mt + Ft
            out = self._fwd(x)
            loss = ((out[:, :-1] - zt) ** 2 * wt).sum(1) / wt.sum().clamp(min=1)
            b = int(torch.argmin(loss))
            best = x[b].numpy().copy()
            pred = out[b].numpy()
        self.prev[shape] = best
        self.last = dict(z=z_goal, w=w.copy(), mask=mask, fixed=fixed[0].copy(), shape=shape, x=best,
                         want_pitched=want_pitched, freq_free=freq_free, freq=freq)
        xp, rp, cp = c.decode(best, shape)
        pred_d = {k: float(pred[j] * self.sd[j] + self.mu[j]) for j, k in enumerate(DKEYS)}
        return xp, rp, cp, float(loss[b]), pred_d

    # ---------- analisi-per-sintesi ibrida (2026-09-27) ----------

    def meas_err(self, meas, L=None):
        """Errore MISURATO rispetto al target di solve() (L = self.last se None): RMS pesato in z (stessi pesi
        dell'ottimizzazione, slider mosso x changed_gain). Descrittore chiesto ma non misurabile (nan) = errore 1;
        pitch-lock: nessuna altezza misurata = +1, scarto oltre PITCH_TOL_C = fino a +1."""
        L = L or self.last
        zm, wm = self.target_z(meas)
        W = max(float(L["w"].sum()), 1e-9)
        e = float((((zm - L["z"]) ** 2) * L["w"] * (wm > 0)).sum() + (L["w"] * (wm == 0)).sum()) / W
        if L["want_pitched"] and not L["freq_free"] and L["freq"]:
            p = meas.get("pitch")
            if p is None or not np.isfinite(p) or p <= 0:
                e += 1.0
            else:
                cents = abs(1200.0 * np.log2(p / L["freq"]))
                e += float(np.clip(cents / PITCH_TOL_C - 1.0, 0.0, 1.0))
        return float(np.sqrt(e))

    def refine(self, render, n_eval=20, seed=None, on_step=None, **kw):
        """Ricerca sul synth vero dall'ultimo solve (vedi RealSearch) fino a n_eval render in tutto.
        on_step(n, err_migliore, misure_migliore) dopo ogni render. Ritorna xp, rp, cp, errore misurato, misure."""
        s = RealSearch(self, render, seed=seed, **kw)
        while s.n < n_eval:
            s.step()
            if on_step:
                on_step(s.n, s.e_best, s.m_best)
        return s.result()


class RealSearch:
    """Analisi-per-sintesi ibrida (2026-09-27), avanzabile UN render alla volta (thread di background).
    Parte dal punto della rete (inv.last, ultimo solve). A gruppi di `group`: screen*group perturbazioni
    gaussiane (sigma, spazio [0,1]) di una frazione `frac` dei parametri liberi, preselezionate con l'ensemble
    corretto del residuo misurato nel migliore; si rendono le `group` migliori, si tiene il migliore MISURATO.
    Passo x1.5 se il gruppo migliora, x0.6 se no. render(xp, rp, cp) -> descrittori misurati."""

    def __init__(self, inv, render, group=5, sigma=0.08, frac=0.35, screen=8, seed=None):
        self.inv, self.render, self.L = inv, render, dict(inv.last)
        self.group, self.sigma, self.frac, self.screen = group, sigma, frac, screen
        self.rng = np.random.default_rng(seed)
        self.free = np.where(self.L["mask"] > 0)[0]
        self.best, self.e_best, self.m_best = self.L["x"].copy(), None, None
        self.n, self.groups, self.improved, self.queue = 0, 0, False, []

    def _propose(self):
        L, inv = self.L, self.inv
        zm, wm = inv.target_z(self.m_best)
        with torch.no_grad():
            pb = inv._fwd(torch.from_numpy(self.best[None].astype(np.float32)))[0, :-1].numpy()
        resid = np.where(wm > 0, zm - pb, 0.0).astype(np.float32)  # scarto rete/synth nel punto migliore
        cand = np.repeat(self.best[None], self.group * self.screen, 0)
        for i in range(len(cand)):
            sub = self.free[self.rng.random(len(self.free)) < self.frac]
            if len(sub) == 0:
                sub = self.rng.choice(self.free, 1)
            cand[i, sub] += self.rng.normal(0.0, self.sigma, len(sub))
        cand = (np.clip(cand, 0.0, 1.0) * L["mask"] + L["fixed"]).astype(np.float32)
        with torch.no_grad():
            out = inv._fwd(torch.from_numpy(cand))[:, :-1].numpy() + resid
        s = (((out - L["z"]) ** 2) * L["w"]).sum(1)
        self.queue = list(cand[np.argsort(s)[:self.group]])

    def step(self):
        """Un render reale. True se il migliore misurato e' cambiato (escluso il primo = punto della rete)."""
        first = self.e_best is None
        if first:
            x = self.best
        else:
            if not self.queue:
                if self.groups:
                    self.sigma = min(self.sigma * 1.5, 0.3) if self.improved else max(self.sigma * 0.6, 0.01)
                self.improved = False
                self._propose()
                self.groups += 1
            x = self.queue.pop(0)
        m = self.render(*self.inv.codec.decode(x, self.L["shape"]))
        e = self.inv.meas_err(m, self.L)
        self.n += 1
        if first or e < self.e_best:
            self.best, self.e_best, self.m_best = np.array(x, np.float32), e, m
            if not first:
                self.improved = True
                return True
        return False

    def result(self):
        xp, rp, cp = self.inv.codec.decode(self.best, self.L["shape"])
        return xp, rp, cp, self.e_best, self.m_best


# ---------------- valutazione ----------------

def _dur(exc, shape, rp, cp):
    import inspect
    return R.note_duration(inspect.signature(E.EXCITERS[exc]).parameters["duration"].default, shape, rp, cp.get("exc_hold"))


SOLVE_KW = {}  # sweep/eval: --starts/--steps (budget ridotto del porting C++: 8 partenze x 80 passi)


def render_measure(exc, shape, xp, rp, cp):
    """Synth vero + analizzatore (come il dataset)."""
    from analyzer.descriptors import analyze_signal
    raw, sr = E.generate(exc, duration=_dur(exc, shape, rp, cp), **xp)
    y = R.apply_resonator(raw, shape=shape, f0=xp.get("freq"), **rp, **cp)
    return analyze_signal(y, sr, extract_pitch=True, normalize=False)


def sweep(exc, shape="bar", n_base=4, steps=5, seed=5, feedback=0, real=()):
    """Test "ogni slider conta": da un target raggiungibile, muove UN descrittore alla volta su 5 valori (p10-p90
    del dataset) tenendo fermi gli altri; risolve in sequenza (partenza a caldo come dal vivo), rende, misura.
    rho = Spearman tra valori chiesti e ottenuti per quel descrittore (1 = segue in modo monotono);
    span = escursione ottenuta / escursione chiesta (in unita' trasformate).
    real = budget di render reali (es. 10,20,30): per ogni target la rete risolve, poi refine() fino al budget
    massimo; il migliore-finora e' registrato a 1 (= rete sola) e a ogni budget -> confronto a parita' di target.
    err = RMS pesato z misurato (meas_err), err_sl = |z| dello slider mosso, t = secondi per target."""
    from scipy.stats import spearmanr
    inv = Inverter(exc)
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1)  # separato: la scelta dei target resta uguale con o senza --real
    rows = [r for r in read_rows(exc) if r["shape"] == shape]
    D = np.array([[_f(r[k]) for k in DKEYS] for r in rows])
    cps = [1] + sorted({int(b) for b in real if int(b) > 1})
    res = {n: {k: ([], []) for k in DKEYS if k != "pitch"} for n in cps}
    ERR, ERK, TT = ({n: [] for n in cps} for _ in range(3))

    def render(xp, rp, cp):
        return render_measure(exc, shape, xp, rp, cp)

    for b in range(n_base):
        base = {k: _f(v) for k, v in zip(DKEYS, D[int(rng.integers(len(D)))])}
        freq = base["pitch"] if np.isfinite(base["pitch"]) else None
        for j, k in enumerate(DKEYS):
            if k == "pitch":
                continue
            col = D[:, j][np.isfinite(D[:, j])]
            vals = np.percentile(col, np.linspace(10, 90, steps))
            inv.prev, inv.prev_z = {}, None
            inv.bias[:] = 0.0
            inv.moved = None
            inv.solve(base, shape, freq=freq, **SOLVE_KW)
            got = {n: [] for n in cps}
            for v in vals:
                t = dict(base)
                t[k] = float(v)
                t0 = time.time()
                if len(cps) == 1:
                    for it in range(1 + feedback):
                        xp, rp, cp, _, _ = inv.solve(t, shape, freq=freq, **SOLVE_KW)
                        meas = render(xp, rp, cp)
                        if it < feedback:
                            inv.correct(t, meas)
                    snap = {1: (inv.meas_err(meas), meas, time.time() - t0)}
                else:
                    inv.solve(t, shape, freq=freq, **SOLVE_KW)
                    snap = {}

                    def on_step(n, e, m):
                        if n in cps:
                            snap[n] = (e, m, time.time() - t0)
                    inv.refine(render, n_eval=cps[-1], seed=int(rng2.integers(1 << 30)), on_step=on_step)
                for n, (e, m, dt) in snap.items():
                    g = m.get(k, np.nan)
                    got[n].append(g)
                    ERR[n].append(e)
                    TT[n].append(dt)
                    ERK[n].append(abs(float(tdesc(g, k) - tdesc(v, k))) / inv.sd[j] if np.isfinite(g) else np.nan)
            for n in cps:
                g = np.array(got[n], float)
                m = np.isfinite(g)
                if m.sum() >= 3 and np.std(g[m]) > 0:
                    res[n][k][0].append(spearmanr(vals[m], g[m])[0])
                    tv, tg = tdesc(vals[m], k), tdesc(g[m], k)
                    res[n][k][1].append((tg.max() - tg.min()) / (tv.max() - tv.min() + 1e-12))
    for n in cps:
        line = " ".join(f"{k[:8]}={np.nanmedian(r):+.2f}/{np.nanmedian(s):.2f}" if r else f"{k[:8]}=  nan"
                        for k, (r, s) in res[n].items())
        ok = np.mean([np.nanmedian(r) > 0.7 for r, _ in res[n].values() if r])
        lab = "rete" if n == 1 else f"rete+{n}"
        print(f"{exc:11s} {shape} fb={feedback} {lab:8s} slider-ok={100 * ok:3.0f}%  err={np.nanmedian(ERR[n]):.3f}"
              f"  err_sl={np.nanmedian(ERK[n]):.3f}  t={np.median(TT[n]):.1f}s  (rho/span)\n    {line}",
              file=sys.stderr, flush=True)


def evaluate(exc, shape="bar", n=30, seed=3):
    from scipy.stats import spearmanr
    from analyzer.descriptors import analyze_signal
    inv = Inverter(exc)
    rng = np.random.default_rng(seed)
    rows = read_rows(exc)
    D = np.array([[_f(r[k]) for k in DKEYS] for r in rows])
    p5, p95 = np.nanpercentile(D, 5, 0), np.nanpercentile(D, 95, 0)
    pit = D[:, DKEYS.index("pitch")]
    pit = pit[np.isfinite(pit)]
    sets = {"slider": [], "reach": []}
    for _ in range(n):
        t = {k: float(rng.uniform(a, b)) for k, a, b in zip(DKEYS, p5, p95) if k != "pitch"}
        t["pitch"] = float(rng.choice(pit))
        sets["slider"].append(t)
        r = rows[int(rng.integers(len(rows)))]
        sets["reach"].append({k: _f(r[k]) for k in DKEYS})
    for name, targets in sets.items():
        got, t_used, ms = [], [], []
        t0 = time.time()
        for t in targets:
            freq = t.get("pitch") if np.isfinite(t.get("pitch", np.nan)) else None
            xp, rp, cp, loss, _ = inv.solve(t, shape, freq=freq, **SOLVE_KW)
            ms.append(time.time() - t0)
            t0 = time.time()
            raw, sr = E.generate(exc, duration=_dur(exc, shape, rp, cp), **xp)
            y = R.apply_resonator(raw, shape=shape, f0=xp.get("freq"), **rp, **cp)
            g = analyze_signal(y, sr, extract_pitch=True, normalize=False)
            got.append([g.get(k, np.nan) for k in DKEYS])
            t_used.append([t.get(k, np.nan) for k in DKEYS])
        T, G = np.array(t_used, float), np.array(got, float)
        corr = []
        for j in range(len(DKEYS)):
            m = np.isfinite(T[:, j]) & np.isfinite(G[:, j])
            c = spearmanr(T[m, j], G[m, j])[0] if m.sum() > 5 and np.std(G[m, j]) > 0 else np.nan
            corr.append(c)
        print(f"{exc:11s} {name:6s} corr_med={np.nanmedian(corr):5.2f}  solve={1000 * np.median(ms):.0f} ms", file=sys.stderr)
        print("    " + " ".join(f"{k[:8]}={c:+.2f}" for k, c in zip(DKEYS, corr)), file=sys.stderr, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "sweep", "curve"])
    ap.add_argument("exciters", nargs="*")
    ap.add_argument("--epochs", type=int, default=80)  # v2: minimo di val a ~45-60 su 150, poi overfit
    ap.add_argument("--shape", default="bar")
    ap.add_argument("--no-pitch-feat", action="store_true")
    ap.add_argument("--ens", type=int, default=1, help="train: numero di reti mediate")
    ap.add_argument("--weights", default=None, help="cartella pesi (default WEIGHTS_DIR)")
    ap.add_argument("--data", default=None, help="cartella dataset (default DATA_DIR)")
    ap.add_argument("--feedback", type=int, default=0, help="sweep: giri di correzione a circuito chiuso")
    ap.add_argument("--real", default="", help="sweep: budget di render reali, es. 10,20,30 (0 prove = rete sola)")
    ap.add_argument("--n-base", type=int, default=4, help="sweep: target di base per descrittore")
    ap.add_argument("--starts", type=int, default=0, help="sweep/eval: partenze del solutore (0 = default 24 knn + 7 casuali)")
    ap.add_argument("--steps", type=int, default=0, help="sweep/eval: passi Adam del solutore (0 = default 150)")
    a = ap.parse_args()
    real = [int(s) for s in a.real.split(",") if s.strip()]
    if a.starts:
        SOLVE_KW.update(n_knn=max(a.starts - 1, 1), n_rand=1)
    if a.steps:
        SOLVE_KW.update(steps=a.steps)
    if a.data:
        DATA_DIR = Path(a.data)
    if a.no_pitch_feat:
        PITCH_FEAT = False
    if a.weights:
        WEIGHTS_DIR = Path(a.weights)
    excs = a.exciters or sorted({p.name.split('.')[0] for p in DATA_DIR.glob('*.csv')})
    for e in excs:
        {"train": lambda: train(e, a.epochs, ens=a.ens), "eval": lambda: evaluate(e, a.shape),
         "sweep": lambda: sweep(e, a.shape, n_base=a.n_base, feedback=a.feedback, real=real), "curve": lambda: curve(e, a.epochs)}[a.cmd]()
