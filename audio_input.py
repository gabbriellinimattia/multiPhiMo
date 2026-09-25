"""
audio_input.py -- acquisizione descrittori da MICROFONO (2026-09-16, prossimo step
dopo il debug del motore realtime -- vedi runtime_architettura_realtime.md: "un
modulo separato... fornira' i descrittori target da microfono o OSC, campionati ogni
0.2-0.5s").

Pipeline: microfono (sounddevice.InputStream, letto in modo BLOCCANTE dal thread di
cattura -- niente callback/lock, piu' semplice e comunque corretto: la sola voce che
scrive lo stato condiviso e' questo stesso thread) -> finestra scorrevole di win_s
secondi -> analyzer.analyze_signal ogni hop_s -> un secondo thread, leggero, fa
FADE ESPONENZIALE continuo (non un salto discreto ad ogni hop, come richiesto) verso
l'ultimo set analizzato e scrive su descriptor_input.set_target() -- stesso ingresso
gia' usato da OSC e dagli slider manuali, nessuna logica duplicata li'.

Costo misurato (test_audio_pipeline.py sez.8, 2026-09-16): analyze_signal su una
finestra di 0.5s costa ~14-16ms, il 5% circa di un hop da 300ms -- un thread nel
processo principale e' sicuro (a differenza del render in exciters.py, qui non c'e'
alcun ciclo Python per-campione: analyze_signal e' FFT/numpy vettoriale). Non serve
un processo separato come candidate_process.py.

Scambio dati fra i due thread interni: solo riassegnazione di riferimento
(self._raw = {...}), lock-free per lo stesso motivo di descriptor_input.py/
param_candidate.py (atomico sotto il GIL di CPython).
"""
import sys
import threading

import numpy as np
import sounddevice as sd

from analyzer import analyze_signal

SR = 44100


class AudioDescriptorSource:
    def __init__(self, descriptor_input, win_s=0.5, hop_s=0.3, smoothing_ms=200.0,
                 device=None, extract_pitch=True, interp_period=0.03):
        self.descriptor_input = descriptor_input
        self.win_s = win_s
        self.hop_s = hop_s
        self.smoothing_ms = smoothing_ms  # regolabile a runtime (gui.py: slider "Smoothing")
        self.device = device
        self.extract_pitch = extract_pitch
        self.interp_period = interp_period

        self._raw = None       # ultimo dict da analyze_signal (thread di cattura)
        self._smoothed = None  # dict correntemente inviato (thread di interpolazione)
        self._stop_evt = None
        self._capture_thread = None
        self._interp_thread = None

    # ---- cattura + analisi (un ciclo = hop_s, bloccante su stream.read) ----

    def _capture_loop(self):
        win_samples = max(1, int(self.win_s * SR))
        hop_samples = max(1, int(self.hop_s * SR))
        buf = np.zeros(win_samples, dtype=np.float64)
        try:
            stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", device=self.device)
            stream.start()
        except Exception as e:
            print(f"[audio_input] apertura microfono fallita: {e}", file=sys.stderr)
            return
        try:
            while not self._stop_evt.is_set():
                try:
                    chunk, overflowed = stream.read(hop_samples)
                except Exception as e:
                    print(f"[audio_input] lettura microfono fallita: {e}", file=sys.stderr)
                    break
                if overflowed:
                    print("[audio_input] overflow input (finestra persa, non bloccante)",
                          file=sys.stderr)
                chunk = np.asarray(chunk[:, 0], dtype=np.float64)
                if len(chunk) >= len(buf):
                    buf = chunk[-len(buf):].copy()
                else:
                    buf = np.concatenate([buf[len(chunk):], chunk])
                try:
                    self._raw = analyze_signal(buf, SR, extract_pitch=self.extract_pitch,
                                                normalize=False)
                except Exception as e:
                    print(f"[audio_input] analisi fallita: {e}", file=sys.stderr)
        finally:
            stream.stop()
            stream.close()

    # ---- fade continuo verso l'ultimo set analizzato, a un ritmo indipendente
    # (interp_period, piu' rapido di hop_s) -- questo e' il "valori interpolati nel
    # tempo" richiesto, non un salto discreto ogni hop_s ----

    def _interp_loop(self):
        while not self._stop_evt.is_set():
            raw = self._raw
            if raw is not None:
                if self._smoothed is None:
                    self._smoothed = dict(raw)
                else:
                    tau = max(self.smoothing_ms, 1.0) / 1000.0
                    coef = 1.0 - np.exp(-self.interp_period / tau)
                    for k, v in raw.items():
                        if k == "decay_capped":
                            continue
                        if not np.isfinite(v):
                            # analyze_signal puo' restituire NaN (es. pitch: nessuna
                            # periodicita' rilevata in quella finestra, silenzio o
                            # rumore) -- MAI propagarlo come target: se non c'e' ancora
                            # uno smoothed valido per questa chiave la si salta del
                            # tutto (niente valore fino al primo campione buono), non
                            # scriverla in set_target -- self._smoothed resta il dict
                            # inviato all'ultimo giro valido, filtrato sotto.
                            continue
                        prev = self._smoothed.get(k, v)
                        self._smoothed[k] = prev + (v - prev) * coef if np.isfinite(prev) else v
                self.descriptor_input.set_target(
                    {k: v for k, v in self._smoothed.items() if k == "decay_capped" or np.isfinite(v)})
            self._stop_evt.wait(self.interp_period)

    # ---- ciclo di vita ----

    def start(self):
        self._raw = None
        self._smoothed = None
        self._stop_evt = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._interp_thread = threading.Thread(target=self._interp_loop, daemon=True)
        self._capture_thread.start()
        self._interp_thread.start()

    def stop(self):
        if self._stop_evt is not None:
            self._stop_evt.set()
