"""
param_candidate.py -- routing one-shot/ibrido (punto 4) + raffinamento SPSA-live in
background (punto 5), thread separato dal thread "play" (in play_engine.py). Il
candidato raffinato sostituisce quello one-shot con una singola assegnazione atomica di
riferimento (`self.candidate = Candidate(...)`, atomica in CPython) -- il thread "play"
legge sempre e solo l'ultimo riferimento disponibile, mai in attesa.

Routing per agente (da criticita_modelli.md, "Routing finale inferenza per agente"):
  - bow, noise, strike, shaker, blow: MDN one-shot (agent.predict) su tutti i
    descrittori/parametri.
  - pluck, chaos: KNN CONGIUNTO VINCOLATO ALLA FORMA (2026-09-15, risolve il
    confondimento da risonatore accoppiato casuale -- vedi
    knn_corpus.KnnJointLockedCorpus/knn_joint_locked.py per la validazione).
    Sostituisce interamente MDN eccitatore + ibrido risonatore: il vicino piu' vicino,
    cercato solo tra le righe con la forma GIA' selezionata in AgentManager, fornisce
    sia i parametri eccitatore sia i parametri continui del risonatore -- la forma
    resta sempre la scelta manuale, mai il routing. NON usato per noise (peggiora
    formanti/harmonic_tension senza risolvere il pitch, limite dell'analizzatore su
    materiale aperiodico, non del routing -- vedi criticita_modelli.md).
  - resonator_bar/plate_rect/plate_circ/membrane: ibrido -- `loss` dalla MDN, tutti gli
    altri parametri dal lookup KNN sul proprio corpus (knn_corpus.py). Non applicabile
    a pluck/chaos, che non passano piu' da qui (vedi sopra).

SPSA-live (punto 5, ammesso SOLO in questi casi -- runtime_architettura_realtime.md):
  - Eccitatore in {bow, noise, strike, shaker}: raffina TUTTI i parametri di
    quell'eccitatore + i parametri del risonatore attivo non presi dalla MDN (cioe' non
    `loss`), rendendo la catena completa exciters.generate -> apply_resonator e
    confrontando via analyze_signal solo sui descrittori spettrali (centroid/spread/
    flatness/harmonic_tension/formanti) contro il target.
  - Eccitatore in {blow, pluck, chaos}: NESSUN raffinamento SPSA (blow: render troppo
    lento, consuma da solo il budget; pluck/chaos: eccitatori patologici/instabili) --
    resta valido solo il candidato one-shot/ibrido, ri-generato ad ogni ciclo.
  - mod_rate/decay_time: mai nell'obiettivo SPSA, su nessun agente (non si stabilizzano
    nel budget 0.2-0.5s indipendentemente dal metodo, vedi doc architettura) -- il loro
    valore resta quello prodotto dal routing one-shot/ibrido sopra.
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import agents
import knn_corpus
import pair_selector
from analyzer import analyze_signal
from exciters import generate as exciter_generate
from resonator import apply_resonator

SPSA_ELIGIBLE_EXCITERS = {"bow", "noise", "strike", "shaker"}

# KNN congiunto vincolato alla forma (2026-09-15): sostituisce MDN+ibrido per
# questi due, vedi docstring del modulo e knn_corpus.KnnJointLockedCorpus.
# 2026-09-21: +bird (KNN congiunto vincolato: NMAE medio 0.242 vs 0.395 del MDN su 14 descrittori,
# vedi claude/nuovi_eccitatori_stato.md); MDN resta come ripiego in caso di eccezione.
# 2026-09-21: +mechanical (KNN congiunto: NMAE medio 0.130 vs 0.162 MDN; temporali 0.15-0.20 vs 0.41-0.73).
JOINT_LOCKED_EXCITERS = {"pluck", "chaos", "bird", "mechanical"}

# Selettore auto (fase 2, priorita' 2): tempo minimo tra due switch di coppia
# auto-selezionata, per non "sfarfallare" tra coppie vicine in punteggio quando il
# target oscilla di poco -- la selezione manuale (CLI/GUI) non e' soggetta a questo
# vincolo, e' sempre immediata.
AUTO_PAIR_MIN_HOLD = 2.0

# Solo i descrittori spettrali (punto 5) -- MAI mod_rate/decay_time/attack_time/pitch:
# quelli restano quelli del one-shot/ibrido, l'obiettivo SPSA non li tocca.
SPSA_DESCRIPTORS = (
    "spectral_centroid", "spectral_spread", "spectral_flatness",
    "harmonic_tension", "formant_f1", "formant_f2", "formant_f3",
)

# Durata usata SOLO per i render-sonda di SPSA (2026-09-16, diagnosi crackle --
# test_audio_pipeline.py sez.7: spsa_refine su bow costa 1064ms/chiamata contro un
# refine_period di 300ms, 354% -- il worker non e' mai idle, CPU/GIL quasi sempre
# occupati mentre bow e' selezionato, indipendentemente da Play/auto-trigger).
# SPSA_DESCRIPTORS sopra sono tutti stime spettrali (centroid/spread/flatness/
# harmonic_tension/formanti) che non hanno bisogno dell'intera durata della nota per
# stabilizzarsi -- analyze_signal_windowed (descriptors.py) usa gia' win_s=0.5 di
# default come finestra minima ragionevole per queste stime, stesso valore riusato
# qui. NON influenza la durata del render finale (quella resta _estimate_duration in
# play_engine.py, sempre alla durata piena) -- solo le 8 valutazioni interne per
# ciclo di raffinamento.
SPSA_PROBE_DURATION = 0.5

# Auto-trigger su cambio sostanziale del candidato (fase 2, priorita' 3 punto 2:
# "sintesi che segue la curva" via una sequenza di note ravvicinate, zero modifiche a
# exciters.py/resonator.py/play_engine.py -- vedi ParamCandidateWorker._maybe_auto_trigger
# e prompt_fase2_range_router_curve.md). Distanza euclidea in spazio 0-1 per-parametro
# (stesso schema di _param_to01) tra candidato precedente e nuovo; sopra questa soglia
# il cambio e' "sostanziale". Un cambio di eccitatore/risonatore e' SEMPRE sostanziale,
# indipendentemente dalla soglia.
AUTO_TRIGGER_THRESHOLD = 0.15

# Hold minimo tra due auto-trigger (stesso schema di AUTO_PAIR_MIN_HOLD sopra): SENZA,
# il rumore stocastico di SPSA (eccitatori in SPSA_ELIGIBLE_EXCITERS ripartono dal
# candidato one-shot con perturbazioni casuali ad ogni ciclo, nessun seed fisso -- vedi
# spsa_refine) puo' da solo superare AUTO_TRIGGER_THRESHOLD quasi ogni refine_period
# anche a target fermo (osservato dal vivo 2026-09-16: trigger quasi ogni 0.3s, non solo
# sui cambi reali del target). Il hold NON sostituisce la soglia di distanza: la
# combina, limitando quanto spesso un cambio "sostanziale" puo' tradursi in una nota.
AUTO_TRIGGER_MIN_HOLD = 1.0


@dataclass(frozen=True)
class Candidate:
    exciter_name: str
    exciter_params: dict
    resonator_shape: str
    resonator_params: dict
    timestamp: float
    refined: bool = False  # True se e' passato anche dallo SPSA-live, non solo one-shot/ibrido

    @property
    def f0(self):
        """2026-09-24: f0 per apply_resonator (cap tau) = freq eccitatore se intonato, altrimenti None."""
        return _res_f0(self.exciter_name, self.exciter_params)


def _res_f0(exciter_name, ex_params):
    if exciter_name in agents.PITCH_LOCKED_EXCITERS and ex_params.get("freq", 0) > 0:
        return float(ex_params["freq"])
    return None


def _lock_freq(exciter_name, exciter_params, target):
    """2026-09-24: freq = pitch target (clip al range dell'eccitatore); pitch NaN/assente ->
    media geometrica del range. No-op per shaker/noise (freq resta parametro predetto)."""
    if exciter_name not in agents.PITCH_LOCKED_EXCITERS:
        return exciter_params
    lo, hi = agents.EXCITER_PARAM_RANGES[exciter_name]["freq"]
    p = target.get("pitch")
    p = float(p) if p is not None and np.isfinite(p) and p > 0 else float(np.sqrt(lo * hi))
    out = dict(exciter_params)
    out["freq"] = float(np.clip(p, lo, hi))
    return out


# ---------------- normalizzazione 0-1 per-parametro (stesso schema di Agent) ----------------

def _param_to01(agent, name, value):
    lo, hi = agent._eff_ranges[name]
    v = np.log10(max(value, agents.EPS)) if name in agents.LOG_PARAMS else value
    return float(np.clip((v - lo) / (hi - lo + agents.EPS), 0.0, 1.0))


def _param_from01(agent, name, y):
    lo, hi = agent._eff_ranges[name]
    v = float(np.clip(y, 0.0, 1.0)) * (hi - lo) + lo
    return float(10 ** v) if name in agents.LOG_PARAMS else float(v)


def _spsa_loss(exciter_name, ex_params, resonator_shape, res_params, target):
    raw, sr = exciter_generate(exciter_name, duration=SPSA_PROBE_DURATION, **ex_params)
    audio = apply_resonator(raw, shape=resonator_shape, f0=_res_f0(exciter_name, ex_params), **res_params)
    got = analyze_signal(audio, sr, extract_pitch=False, normalize=False)
    err = 0.0
    for k in SPSA_DESCRIPTORS:
        t = target.get(k)
        g = got.get(k)
        if t is None or g is None or not np.isfinite(t) or not np.isfinite(g):
            continue
        denom = max(abs(t), 1e-6)
        err += ((g - t) / denom) ** 2
    return err


def spsa_refine(exciter_agent, exciter_params, resonator_agent, resonator_params,
                 resonator_shape, exciter_name, target, iterations=4, a=0.15, c=0.08, rng=None):
    """Poche iterazioni di SPSA (2 render+analisi per passo) sopra il render-sonda
    (exciters.generate + apply_resonator + analyze_signal), round-trip economico entro
    il budget 0.2-0.5s (vedi runtime_architettura_realtime.md, validazione empirica).
    Dimensioni ottimizzate: TUTTI i parametri dell'eccitatore + i parametri del
    risonatore ECCETTO `loss` (che resta quello della MDN, coerente con l'ibrido)."""
    rng = rng or np.random.default_rng()
    ex_names = exciter_agent.param_names
    res_names = [p for p in resonator_agent.param_names if p != "loss"]
    dims = [("ex", n) for n in ex_names] + [("res", n) for n in res_names]
    if not dims:
        return exciter_params, resonator_params

    theta = np.array([
        _param_to01(exciter_agent, n, exciter_params[n]) if grp == "ex"
        else _param_to01(resonator_agent, n, resonator_params[n])
        for grp, n in dims
    ])

    def decode(vec):
        ex_p, res_p = dict(exciter_params), dict(resonator_params)
        for (grp, n), y in zip(dims, vec):
            if grp == "ex":
                ex_p[n] = _param_from01(exciter_agent, n, y)
            else:
                res_p[n] = _param_from01(resonator_agent, n, y)
        return ex_p, res_p

    def loss(vec):
        ex_p, res_p = decode(vec)
        return _spsa_loss(exciter_name, ex_p, resonator_shape, res_p, target)

    for it in range(iterations):
        delta = rng.choice([-1.0, 1.0], size=len(dims))
        ck = c / ((it + 1) ** 0.101)
        ak = a / ((it + 1) ** 0.602)
        lp = loss(np.clip(theta + ck * delta, 0.0, 1.0))
        lm = loss(np.clip(theta - ck * delta, 0.0, 1.0))
        ghat = (lp - lm) / (2 * ck) * delta
        theta = np.clip(theta - ak * ghat, 0.0, 1.0)

    return decode(theta)


class ParamCandidateWorker:
    """Thread di background (punto 5): ad ogni ciclo (refine_period, 0.2-0.5s) legge
    l'ultimo target da descriptor_input, ricalcola il candidato one-shot/ibrido (punto
    4) e, se l'eccitatore attivo lo ammette, lo raffina via SPSA. Pubblica il risultato
    con un'unica assegnazione di riferimento a `self.candidate`."""

    def __init__(self, descriptor_input, agent_manager, dataset_dir="dataset",
                 spsa_iterations=4, refine_period=0.3, auto_pair=False,
                 selector_csv="selector_dataset.csv", selector_k=30, selector_bias_alpha=0.3,
                 auto_pair_min_hold=AUTO_PAIR_MIN_HOLD,
                 auto_trigger=False, auto_trigger_threshold=AUTO_TRIGGER_THRESHOLD,
                 auto_trigger_min_hold=AUTO_TRIGGER_MIN_HOLD, trigger_callback=None):
        self.descriptor_input = descriptor_input
        self.manager = agent_manager
        self.dataset_dir = Path(dataset_dir)
        self.spsa_iterations = spsa_iterations
        self.refine_period = refine_period
        self.auto_pair = auto_pair
        self.selector_csv = selector_csv
        self.selector_k = selector_k
        self.selector_bias_alpha = selector_bias_alpha
        self.auto_pair_min_hold = auto_pair_min_hold
        self._last_pair_switch = 0.0
        # auto-trigger (punto 2, priorita' 3): auto_trigger=False by default, mai al
        # posto del trigger manuale (Invio/bottone Play) -- trigger_callback e'
        # tipicamente play_engine.PlayEngine.trigger, impostato dal chiamante
        # (main.py/gui.py) DOPO aver creato il PlayEngine (che ha bisogno del worker).
        self.auto_trigger = auto_trigger
        self.auto_trigger_threshold = auto_trigger_threshold
        self.auto_trigger_min_hold = auto_trigger_min_hold
        self.trigger_callback = trigger_callback
        self._prev_candidate: Optional[Candidate] = None
        self._last_auto_trigger = 0.0
        self.candidate: Optional[Candidate] = None
        self._stop_evt = None
        self._thread = None

    def start(self):
        import threading
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._stop_evt is not None:
            self._stop_evt.set()

    def _run(self):
        while not self._stop_evt.is_set():
            target_obj = self.descriptor_input.latest
            if target_obj is not None:
                try:
                    self._update_candidate(dict(target_obj.values))
                except Exception as e:
                    print(f"[param_candidate] update fallito, candidato precedente invariato: {e}",
                          file=sys.stderr)
            self._stop_evt.wait(self.refine_period)

    def _mdn_hybrid_route(self, target, resonator_agent_name):
        """Routing invariato pre-2026-09-15: MDN per l'eccitatore, ibrido (loss dalla
        MDN, resto dal KNN sul corpus del risonatore) per il risonatore. Usato per
        bow/noise/strike/shaker/blow, e come fallback se il KNN congiunto vincolato
        fallisce su pluck/chaos."""
        exciter_params = self.manager.exciter.predict(target)
        mdn_res_params = self.manager.resonator.predict(target)
        csv_path = self.dataset_dir / f"{resonator_agent_name}.csv"
        corpus = knn_corpus.build_corpus(csv_path, resonator_agent_name)
        resonator_params = corpus.query(target)
        resonator_params["loss"] = mdn_res_params["loss"]
        return exciter_params, resonator_params

    def _maybe_select_pair(self, target):
        """Se auto_pair e' attivo (punto 3 del prompt fase 2/priorita' 2): interroga
        il selettore O(1) (pair_selector.py) sul target corrente e, se la coppia scelta
        e' diversa da quella attiva e il tempo minimo di hold e' trascorso, la applica
        con la STESSA API della selezione manuale (manager.select_exciter/
        select_resonator) -- il contratto di AgentManager non cambia (coppia sempre
        esplicita una volta scelta), cambia solo la fonte della scelta. Fallback
        silenzioso (coppia invariata) se il CSV del selettore manca o la query fallisce
        -- mai un'eccezione che interrompe il ciclo di raffinamento."""
        if not self.auto_pair:
            return
        now = time.monotonic()
        if now - self._last_pair_switch < self.auto_pair_min_hold:
            return
        try:
            selector = pair_selector.build_selector(
                self.selector_csv, k=self.selector_k, bias_alpha=self.selector_bias_alpha)
            exciter_name, resonator_shape, _ = selector.query(target)
        except Exception as e:
            print(f"[param_candidate] selettore auto-pair fallito, coppia invariata: {e}",
                  file=sys.stderr)
            return
        if (exciter_name != self.manager.exciter_name
                or resonator_shape != self.manager.resonator_shape):
            self.manager.select_exciter(exciter_name)
            self.manager.select_resonator(resonator_shape)
            self._last_pair_switch = now

    def _candidate_param_distance(self, prev, curr):
        """Distanza euclidea in spazio 0-1 per-parametro (stesso schema di _param_to01)
        tra due candidati con STESSO eccitatore+risonatore -- un cambio di coppia e'
        gestito a parte dal chiamante (sempre "sostanziale", non serve questa distanza).
        Confronta solo le chiavi presenti in entrambi e nello spazio parametri
        dell'agente (es. 'loss' non c'e' per pluck/chaos, che non passano dall'ibrido --
        ignorata senza errori)."""
        ex_agent, res_agent = self.manager.exciter, self.manager.resonator
        sq = 0.0
        for name, val in curr.exciter_params.items():
            if name in prev.exciter_params and name in ex_agent.param_names:
                a = _param_to01(ex_agent, name, prev.exciter_params[name])
                b = _param_to01(ex_agent, name, val)
                sq += (a - b) ** 2
        for name, val in curr.resonator_params.items():
            if name in prev.resonator_params and name in res_agent.param_names:
                a = _param_to01(res_agent, name, prev.resonator_params[name])
                b = _param_to01(res_agent, name, val)
                sq += (a - b) ** 2
        return float(np.sqrt(sq))

    def _maybe_auto_trigger(self, prev, curr):
        """Punto 2, priorita' 3 (fase 2): approssima "sintesi che segue la curva" con
        una sequenza di note ravvicinate -- fa scattare trigger_callback (tipicamente
        play_engine.PlayEngine.trigger) quando il candidato cambia in modo sostanziale,
        SENZA aspettare Invio/bottone Play. Aggiunta esplicita (auto_trigger=False di
        default), mai al posto del trigger manuale, che resta sempre disponibile.
        "Sostanziale" = primo candidato disponibile, o cambio di eccitatore/risonatore,
        o distanza normalizzata sopra auto_trigger_threshold -- MA anche un cambio
        sostanziale scatta solo se e' passato almeno auto_trigger_min_hold dall'ultimo
        auto-trigger (necessario: il rumore stocastico di SPSA da solo supera la soglia
        di distanza quasi ogni ciclo anche a target fermo, osservato dal vivo
        2026-09-16 -- vedi AUTO_TRIGGER_MIN_HOLD). Nessuna eccezione puo' interrompere
        il ciclo di raffinamento (stesso stile difensivo del resto del worker)."""
        if not self.auto_trigger or self.trigger_callback is None:
            return
        substantial = (
            prev is None
            or prev.exciter_name != curr.exciter_name
            or prev.resonator_shape != curr.resonator_shape
            or self._candidate_param_distance(prev, curr) >= self.auto_trigger_threshold
        )
        if not substantial:
            return
        now = time.monotonic()
        if now - self._last_auto_trigger < self.auto_trigger_min_hold:
            return
        self._last_auto_trigger = now
        try:
            self.trigger_callback()
        except Exception as e:
            print(f"[param_candidate] auto-trigger fallito: {e}", file=sys.stderr)

    def _update_candidate(self, target):
        self._maybe_select_pair(target)
        exciter_name = self.manager.exciter_name
        resonator_shape = self.manager.resonator_shape
        if exciter_name is None or resonator_shape is None:
            return

        resonator_agent_name = f"resonator_{resonator_shape}"

        if exciter_name in JOINT_LOCKED_EXCITERS:
            try:
                joint_corpus = knn_corpus.build_joint_locked_corpus(
                    self.dataset_dir / f"{exciter_name}.csv", exciter_name, resonator_shape)
                exciter_params, resonator_params = joint_corpus.query(target)
            except Exception as e:
                print(f"[param_candidate] KNN congiunto vincolato fallito, ripiego su MDN: {e}",
                      file=sys.stderr)
                exciter_params, resonator_params = self._mdn_hybrid_route(
                    target, resonator_agent_name)
        else:
            exciter_params, resonator_params = self._mdn_hybrid_route(target, resonator_agent_name)

        exciter_params = _lock_freq(exciter_name, exciter_params, target)
        refined = False
        if exciter_name in SPSA_ELIGIBLE_EXCITERS:
            try:
                exciter_params, resonator_params = spsa_refine(
                    self.manager.exciter, exciter_params,
                    self.manager.resonator, resonator_params,
                    resonator_shape, exciter_name, target,
                    iterations=self.spsa_iterations)
                refined = True
            except Exception as e:
                print(f"[param_candidate] SPSA fallita, uso il candidato one-shot/ibrido: {e}",
                      file=sys.stderr)

        prev = self._prev_candidate
        new_candidate = Candidate(
            exciter_name=exciter_name, exciter_params=exciter_params,
            resonator_shape=resonator_shape, resonator_params=resonator_params,
            timestamp=time.monotonic(), refined=refined)
        self.candidate = new_candidate
        self._prev_candidate = new_candidate
        self._maybe_auto_trigger(prev, new_candidate)
