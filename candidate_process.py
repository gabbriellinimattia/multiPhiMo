"""
candidate_process.py -- motore candidato+render in un PROCESSO separato (2026-09-16,
diagnosi crackle: vedi test_audio_pipeline.py sez.1/3/7 e criticita_modelli.md).

Perche' un processo e non solo un altro thread: il GIL e' per-processo. Nella
versione precedente (thread in-process, ParamCandidateWorker.start()), il render
sincrono di alcuni eccitatori costava fino a 1.1s (blow, sez.1) e spsa_refine() per
bow saturava il 354% del suo stesso refine_period di 300ms (sez.7) -- il thread di
raffinamento restava quasi sempre occupato, competendo per lo stesso GIL del thread
audio (sounddevice._callback) nello stesso processo. Spostare quel lavoro in un
processo figlio elimina questa contesa alla radice: il processo principale (GUI +
PlayEngine + stream audio) non fa mai calcoli DSP pesanti, riceve solo target in
ingresso e (candidato, audio gia' renderizzato) in uscita via multiprocessing.Queue.

Riusa param_candidate.ParamCandidateWorker COSI' COM'E' (routing, SPSA, auto-pair,
auto-trigger -- nessuna logica duplicata) dentro il processo figlio, con un
DescriptorInput "ombra" (_ShadowDescriptorInput) alimentato dai target che arrivano
dal processo principale invece che da OSC/GUI direttamente. Riusa anche
play_engine._estimate_duration/_apply_fade + exciters.generate/resonator.apply_resonator
per il render (stessa logica di PlayEngine.trigger(), spostata qui).

Interfaccia lato processo principale: CandidateProcessHandle espone la stessa
superficie usata da gui.py/main.py sul vecchio ParamCandidateWorker (.candidate,
.auto_pair, .auto_trigger, .auto_trigger_threshold, .auto_trigger_min_hold, .start(),
.stop()) piu' .exciter_name/.resonator_shape (mirror, sostituiscono AgentManager lato
GUI -- l'AgentManager vero vive solo nel processo figlio, mai piu' di un'istanza),
.select_exciter(nome)/.select_resonator(forma), .request_trigger() (sostituisce
engine.trigger() per Play manuale/Play continuo: il render avviene nel figlio) e
.audio_sink (callable, tipicamente play_engine.PlayEngine.push_audio, impostato dal
chiamante DOPO aver costruito sia l'handle sia il PlayEngine).
"""
import queue
import sys
import threading
import time
from multiprocessing import get_context
from types import MappingProxyType, SimpleNamespace

import numpy as np

# sustain "seamless" sul tasto Play tenuto premuto (2026-09-16, sostituisce il morph-
# crossfade: l'utente ha segnalato che due attacchi indipendenti in crossfade suonano
# come "due strumenti", non uno che continua -- tecnica standard dei sample-loop
# sustain, si taglia via l'attacco e si innesta nel corpo gia' a regime, vedi
# https://samplestack.app/news/seamless-loops-for-sustained-samples/). Ogni retrigger
# DOPO il primo di una nota tenuta (vedi gui.py: sustain_continuation) scarta i primi
# SUSTAIN_TRIM_S secondi del render prima di accodarlo.
SUSTAIN_TRIM_S = 0.22

# Morphing B/A (2026-09-18, vedi claude/morphing_spettrale_ricerca_proposta.md): solo
# intra-coppia (stesso eccitatore+risonatore del render precedente), mai per cambi di
# coppia -- bow/blow/chaos/pluck hanno stato per-campione che puo' scivolare nel tempo
# (interpolazione lineare dei parametri su morph_ms, poi fermi); noise usa overlap-add
# a blocchi (vedi exciters.noise_morph). Esclusi: strike (evento impulsivo, niente da
# far scivolare), shaker (lasciato a una eventuale Opzione A futura, non qui). Esclusa
# anche sustain_continuation (retrigger di nota tenuta gia' validato -- non toccarlo).
MORPH_EXCITERS_B = {"bow", "blow", "chaos", "pluck"}
MORPH_EXCITERS_A = {"noise"}
MORPH_B_TRAJ_PARAMS = {
    "bow": ("bow_force", "bow_velocity", "brightness", "damping"),
    "blow": ("mouth_pressure", "reed_stiffness", "breath_noise", "brightness", "damping"),
    "chaos": ("bifurcation", "brightness", "damping"),
    "pluck": ("decay_time", "dispersion"),
}


# ---------------- processo figlio ----------------

def _child_main(target_queue, control_queue, candidate_queue, audio_queue, config):
    # import qui dentro (non in testa al modulo): questo codice gira SOLO nel
    # processo figlio (spawn su macOS re-importa il modulo, ma _child_main viene
    # chiamata esplicitamente solo li') -- tiene torch/agents fuori dal processo
    # principale, che non ne ha bisogno.
    from agents import AgentManager
    from descriptor_input import DescriptorTarget
    from exciters import generate as exciter_generate, SR, noise_morph
    from param_candidate import ParamCandidateWorker
    from play_engine import _apply_fade, _estimate_duration
    from resonator import apply_resonator

    manager = AgentManager(weights_dir=config["weights_dir"])
    if config.get("initial_exciter"):
        manager.select_exciter(config["initial_exciter"])
    if config.get("initial_resonator"):
        manager.select_resonator(config["initial_resonator"])

    shadow_input = SimpleNamespace(latest=None)

    # Stato del morphing B/A (2026-09-18): _last_render tiene i parametri dell'ULTIMO
    # render effettivamente accodato (non del candidato di background, che cambia ogni
    # refine_period anche senza note) -- e' il punto di partenza della traiettoria per
    # il prossimo render sulla stessa coppia. runtime_state["morph"] e' aggiornabile a
    # caldo dal control_queue (comando set_morph), stesso schema di auto_pair/auto_trigger.
    _last_render = {"exciter_name": None, "resonator_shape": None, "exciter_params": None}
    runtime_state = {"morph": bool(config.get("morph", False))}

    def _build_morph_params(cand, duration, sr):
        """Se morph e' attivo e l'ultimo render era sulla STESSA coppia, restituisce i
        parametri eccitatore con una traiettoria (array) che scivola dai valori
        dell'ultimo render a quelli del candidato attuale su morph_ms, poi resta ferma --
        altrimenti i parametri fissi di sempre (nessun cambiamento se morph e' spento o
        la coppia e' cambiata, vedi MORPH_B_TRAJ_PARAMS per i parametri coinvolti per
        eccitatore)."""
        same_pair = (runtime_state["morph"]
                     and _last_render["exciter_name"] == cand.exciter_name
                     and _last_render["resonator_shape"] == cand.resonator_shape
                     and _last_render["exciter_params"] is not None)
        if not same_pair:
            return dict(cand.exciter_params)
        n = int(duration * sr)
        morph_n = max(1, min(int(sr * config["morph_ms"] / 1000.0), n))
        prev_p = _last_render["exciter_params"]
        params = dict(cand.exciter_params)
        for k in MORPH_B_TRAJ_PARAMS.get(cand.exciter_name, ()):
            if k in prev_p and k in cand.exciter_params:
                arr = np.full(n, cand.exciter_params[k], dtype=np.float64)
                arr[:morph_n] = np.linspace(prev_p[k], cand.exciter_params[k], morph_n)
                params[k] = arr
        return params

    def _render_and_push(note_id=None, gain=1.0, sustain_continuation=False):
        # note_id/gain/sustain_continuation (2026-09-16, hold-to-sustain sul tasto Play
        # in gui.py): default a vuoto per ogni chiamata pre-esistente (auto_trigger via
        # trigger_callback qui sotto, "trigger" senza payload dal control_queue) --
        # nessun cambiamento di comportamento li'.
        cand = worker.candidate
        if cand is None:
            return
        try:
            duration = _estimate_duration(cand)
            morph_ready = (not sustain_continuation and runtime_state["morph"]
                           and _last_render["exciter_name"] == cand.exciter_name
                           and _last_render["resonator_shape"] == cand.resonator_shape
                           and _last_render["exciter_params"] is not None)
            if sustain_continuation or cand.exciter_name not in (MORPH_EXCITERS_B | MORPH_EXCITERS_A):
                raw, sr = exciter_generate(cand.exciter_name, duration=duration, **cand.exciter_params)
            elif cand.exciter_name in MORPH_EXCITERS_B:
                ex_params = _build_morph_params(cand, duration, SR)
                raw, sr = exciter_generate(cand.exciter_name, duration=duration, **ex_params)
            elif morph_ready:  # noise, morph attivo e coppia invariata (noise_morph
                # ritorna solo l'array audio, non una tupla (raw, sr) come exciter_generate)
                raw = noise_morph(_last_render["exciter_params"], cand.exciter_params,
                                   duration=duration, sr=SR, morph_ms=config["morph_ms"])
                sr = SR
            else:
                raw, sr = exciter_generate(cand.exciter_name, duration=duration, **cand.exciter_params)
            audio = apply_resonator(raw, shape=cand.resonator_shape, **cand.resonator_params)
            if sustain_continuation:
                trim_n = min(int(SUSTAIN_TRIM_S * sr), len(audio) // 2)
                audio = audio[trim_n:]
            audio = _apply_fade(audio, sr, fade_in_ms=config["fade_ms"], fade_out_ms=config["fade_ms"])
            tag = "raffinato" if cand.refined else "one-shot/ibrido"
            suffix = " [continua]" if sustain_continuation else ""
            label = f"{cand.exciter_name}+{cand.resonator_shape} ({tag}, {len(audio) / sr:.2f}s){suffix}"
            audio_queue.put((np.asarray(audio, dtype=np.float32), label, note_id, gain))
            if not sustain_continuation:
                _last_render["exciter_name"] = cand.exciter_name
                _last_render["resonator_shape"] = cand.resonator_shape
                _last_render["exciter_params"] = dict(cand.exciter_params)
        except Exception as e:
            print(f"[candidate_process] render fallito: {e}", file=sys.stderr)

    worker = ParamCandidateWorker(
        shadow_input, manager, dataset_dir=config["dataset_dir"],
        spsa_iterations=config["spsa_iterations"], refine_period=config["refine_period"],
        auto_pair=config["auto_pair"], selector_csv=config["selector_csv"],
        selector_k=config["selector_k"], selector_bias_alpha=config["selector_bias_alpha"],
        auto_pair_min_hold=config["auto_pair_min_hold"],
        auto_trigger=config["auto_trigger"], auto_trigger_threshold=config["auto_trigger_threshold"],
        auto_trigger_min_hold=config["auto_trigger_min_hold"], trigger_callback=_render_and_push)
    worker.start()

    last_sent = None
    stopping = False
    try:
        while not stopping:
            # target: teniamo solo l'ultimo arrivato, il merge/filtro per relevant_keys
            # e' gia' stato fatto lato processo principale (descriptor_input.py)
            new_target = None
            try:
                while True:
                    new_target = target_queue.get_nowait()
            except queue.Empty:
                pass
            if new_target is not None:
                shadow_input.latest = DescriptorTarget(
                    values=MappingProxyType(dict(new_target)), timestamp=time.monotonic())

            try:
                while True:
                    cmd, payload = control_queue.get_nowait()
                    if cmd == "select_exciter":
                        worker.stop()
                        manager.select_exciter(payload)
                        worker.start()
                    elif cmd == "select_resonator":
                        worker.stop()
                        manager.select_resonator(payload)
                        worker.start()
                    elif cmd == "set_auto_pair":
                        worker.auto_pair = payload
                    elif cmd == "set_morph":
                        runtime_state["morph"] = bool(payload)
                    elif cmd == "set_auto_trigger":
                        worker.auto_trigger = payload
                    elif cmd == "set_auto_trigger_threshold":
                        worker.auto_trigger_threshold = payload
                    elif cmd == "set_auto_trigger_min_hold":
                        worker.auto_trigger_min_hold = payload
                    elif cmd == "trigger":
                        note_id, gain, sustain_cont = payload if payload is not None else (None, 1.0, False)
                        _render_and_push(note_id=note_id, gain=gain, sustain_continuation=sustain_cont)
                    elif cmd == "stop":
                        stopping = True
            except queue.Empty:
                pass

            cand = worker.candidate
            if cand is not None and cand is not last_sent:
                try:
                    candidate_queue.put(cand)
                except Exception:
                    pass
                last_sent = cand

            time.sleep(0.03)
    finally:
        worker.stop()


# ---------------- handle lato processo principale ----------------

class CandidateProcessHandle:
    def __init__(self, descriptor_input, weights_dir="weights", dataset_dir="dataset",
                 initial_exciter=None, initial_resonator=None,
                 spsa_iterations=4, refine_period=0.3, fade_ms=8.0, auto_pair=False,
                 selector_csv="selector_dataset.csv", selector_k=30, selector_bias_alpha=0.3,
                 auto_pair_min_hold=2.0, auto_trigger=False, auto_trigger_threshold=0.15,
                 auto_trigger_min_hold=1.0, morph=False, morph_ms=150.0):
        self.descriptor_input = descriptor_input
        self.exciter_name = initial_exciter
        self.resonator_shape = initial_resonator
        self.candidate = None
        self.audio_sink = None  # tipicamente play_engine.PlayEngine.push_audio, assegnato dal chiamante

        self._config = dict(
            weights_dir=weights_dir, dataset_dir=dataset_dir,
            initial_exciter=initial_exciter, initial_resonator=initial_resonator,
            spsa_iterations=spsa_iterations, refine_period=refine_period, fade_ms=fade_ms,
            auto_pair=auto_pair, selector_csv=selector_csv, selector_k=selector_k,
            selector_bias_alpha=selector_bias_alpha, auto_pair_min_hold=auto_pair_min_hold,
            auto_trigger=auto_trigger, auto_trigger_threshold=auto_trigger_threshold,
            auto_trigger_min_hold=auto_trigger_min_hold, morph=morph, morph_ms=morph_ms)

        ctx = get_context("spawn")
        self._target_queue = ctx.Queue()
        self._control_queue = ctx.Queue()
        self._candidate_queue = ctx.Queue()
        self._audio_queue = ctx.Queue()
        self._process = ctx.Process(
            target=_child_main,
            args=(self._target_queue, self._control_queue, self._candidate_queue,
                  self._audio_queue, self._config),
            daemon=True)

        self._stop_evt = threading.Event()
        self._last_sent_ts = None
        self._forwarder = threading.Thread(target=self._forward_targets, daemon=True)
        self._receiver = threading.Thread(target=self._receive_loop, daemon=True)

    # ---- ciclo di vita ----

    def start(self):
        self._process.start()
        self._forwarder.start()
        self._receiver.start()

    def stop(self):
        self._stop_evt.set()
        try:
            self._control_queue.put(("stop", None))
        except Exception:
            pass
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()

    # ---- target: inoltra al figlio ogni volta che descriptor_input.latest cambia ----

    def _forward_targets(self):
        while not self._stop_evt.is_set():
            latest = self.descriptor_input.latest
            if latest is not None and latest.timestamp != self._last_sent_ts:
                self._last_sent_ts = latest.timestamp
                try:
                    self._target_queue.put(dict(latest.values))
                except Exception as e:
                    print(f"[candidate_process] inoltro target fallito: {e}", file=sys.stderr)
            time.sleep(0.02)

    # ---- candidato/audio: ricevuti dal figlio ----

    def _receive_loop(self):
        while not self._stop_evt.is_set():
            got_any = False
            try:
                while True:
                    cand = self._candidate_queue.get_nowait()
                    self.candidate = cand
                    self.exciter_name = cand.exciter_name
                    self.resonator_shape = cand.resonator_shape
                    got_any = True
            except queue.Empty:
                pass
            try:
                while True:
                    audio, label, note_id, gain = self._audio_queue.get_nowait()
                    if self.audio_sink is not None:
                        self.audio_sink(audio, label=label, note_id=note_id, gain=gain)
                    got_any = True
            except queue.Empty:
                pass
            if not got_any:
                time.sleep(0.01)

    # ---- comandi verso il figlio (stessa granularita' di gui.py: un comando per
    # eccitatore, uno per risonatore -- vedi App._on_exciter_change/_on_resonator_change) ----

    def select_exciter(self, name):
        self.exciter_name = name  # aggiornamento ottimistico, per la UI immediata
        self._control_queue.put(("select_exciter", name))

    def select_resonator(self, shape):
        self.resonator_shape = shape
        self._control_queue.put(("select_resonator", shape))

    def set_auto_pair(self, value):
        self._control_queue.put(("set_auto_pair", bool(value)))

    def set_morph(self, value):
        """On/off del morphing B/A (2026-09-18) -- il processo figlio applica il nuovo
        valore al prossimo render, coppia per coppia (vedi runtime_state in _child_main)."""
        self._control_queue.put(("set_morph", bool(value)))

    def set_auto_trigger(self, value):
        self._control_queue.put(("set_auto_trigger", bool(value)))

    def set_auto_trigger_threshold(self, value):
        self._control_queue.put(("set_auto_trigger_threshold", float(value)))

    def set_auto_trigger_min_hold(self, value):
        self._control_queue.put(("set_auto_trigger_min_hold", float(value)))

    def request_trigger(self, note_id=None, gain=1.0, sustain_continuation=False):
        """Sostituisce play_engine.PlayEngine.trigger() per Play manuale/Play continuo:
        il render avviene nel processo figlio (_render_and_push), il buffer arriva a
        self.audio_sink via _receive_loop -- mai un render sincrono nel processo
        principale (che ha anche lo stream audio). note_id/gain (2026-09-16,
        hold-to-sustain in gui.py): inoltrati fino a PlayEngine.push_audio senza essere
        interpretati qui -- vedi play_engine.py per il loro uso. sustain_continuation:
        vedi SUSTAIN_TRIM_S sopra -- True per ogni retrigger DOPO il primo di una nota
        tenuta (salta l'attacco, innesta nel corpo)."""
        self._control_queue.put(("trigger", (note_id, gain, sustain_continuation)))
