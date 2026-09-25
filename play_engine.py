"""
play_engine.py -- trigger "play", render sincrono, coda di riproduzione, uscita audio
(punto 6). Il trigger legge il candidato corrente prodotto da
param_candidate.ParamCandidateWorker (assegnazione atomica di riferimento, nessun lock),
renderizza SINCRONAMENTE (exciters.generate -> apply_resonator) e mette il buffer in
coda per il mixer polifonico nel callback di sounddevice.OutputStream.

Latenza del render (decine di ms in Python, vedi runtime_architettura_realtime.md)
accettata in questo prototipo: qui si valida l'ARCHITETTURA (thread separati, routing,
handoff), non la latenza finale sull'attacco -- demandata al porting nativo (fase
successiva, non in questo prompt).
"""
import sys
import threading
import time

import numpy as np
import sounddevice as sd

import inspect

from exciters import generate as exciter_generate, SR, EXCITERS
from resonator import apply_resonator, resonator_params


def _apply_fade(audio, sr, fade_in_ms=8.0, fade_out_ms=None):
    """Fade in/out raised-cosine in testa/coda al buffer -- senza, un render che non
    parte/finisce a zero produce un click udibile all'attacco o al troncamento
    (segnalato all'ascolto, 2026-09-15). Applicato qui (non dentro exciters/resonator,
    riusati cosi' come sono) sull'intero buffer prima di metterlo in coda: costo
    trascurabile, click eliminato indipendentemente da eccitatore/forma.

    fade_in_ms e fade_out_ms separati (2026-09-16, morph -- vedi PlayEngine.morph):
    in modalita' morph l'attacco della nuova voce usa un fade-in lungo quanto
    morph_ms per confondersi con la dissolvenza in uscita della voce precedente,
    mentre la coda naturale della nota resta sul fade_ms breve di sempre."""
    if fade_out_ms is None:
        fade_out_ms = fade_in_ms
    n = len(audio)
    fade_in_n = min(int(sr * fade_in_ms / 1000.0), n // 2)
    fade_out_n = min(int(sr * fade_out_ms / 1000.0), n // 2)
    audio = audio.copy()
    if fade_in_n > 1:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade_in_n, dtype=np.float32))).astype(np.float32)
        audio[:fade_in_n] *= ramp
    if fade_out_n > 1:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade_out_n, dtype=np.float32))).astype(np.float32)
        audio[-fade_out_n:] *= ramp[::-1]
    return audio


_STEAL_RELEASE_MS = 5.0  # dissolvenza breve quando una voce viene "rubata" oltre max_voices

_DURATION_MARGIN = 3.0  # multiplo del tempo caratteristico di decadimento del risonatore
_MAX_DURATION = 6.0     # tetto (s) per evitare render/buffer eccessivi

# Limiter (2026-09-16, richiesto dall'utente dopo aver sentito saturazione vera con
# 11 voci ravvicinate -- vedi test_audio_pipeline.py sez.5, clipping reale gia'
# misurato a gain=0.4/10 voci). A differenza del clip duro di sicurezza gia' presente
# (istantaneo, distorce se superato), questo e' un feedforward limiter senza
# lookahead: guadagno che si abbassa in fretta (attack) quando il picco del buffer
# supera limiter_ceiling e risale piano (release, per non "pompare" udibilmente) --
# stato (_limiter_gain) persistente SOLO nel thread audio (_callback), nessun lock
# necessario. Costo: un np.max in piu' per callback, trascurabile (vedi sez.4: l'intero
# mixer costa 0.023ms su un budget di 23.2ms).
LIMITER_CEILING = 0.85
_LIMITER_ATTACK_MS = 5.0
_LIMITER_RELEASE_MS = 60.0


def _estimate_duration(cand):
    """La durata di default di ogni eccitatore (in exciters.py) e' un valore fisso
    (1.0-2.0s) indipendente dal target: apply_resonator produce un buffer della STESSA
    lunghezza dell'eccitazione (vedi resonator.py), quindi qualunque decadimento --
    del risonatore (damping_times) o dell'eccitatore stesso (decay_time, pluck/shaker) --
    piu' lungo del default veniva sempre tagliato di netto (segnalato dall'utente
    2026-09-16: suono troncato ad ogni trigger di 'Play continuo'). Qui si stima la
    durata minima necessaria a contenere il decadimento reale e si estende il render
    di conseguenza, senza mai accorciarlo sotto il default originale."""
    default_dur = inspect.signature(EXCITERS[cand.exciter_name]).parameters["duration"].default
    try:
        _, damping_times, _ = resonator_params(shape=cand.resonator_shape, **cand.resonator_params)
        reso_decay = max(damping_times) * _DURATION_MARGIN
    except Exception:
        reso_decay = 0.0
    exc_decay = cand.exciter_params.get("decay_time", 0.0)
    return float(np.clip(max(default_dur, reso_decay, exc_decay), default_dur, _MAX_DURATION))


class _Voice:
    __slots__ = ("audio", "pos", "born", "stolen", "note_id", "gain")

    def __init__(self, audio, note_id=None, gain=1.0):
        self.audio = audio
        self.pos = 0
        self.born = time.monotonic()
        self.stolen = False  # gia' in dissolvenza forzata (voice-stealing o note-off
        # esplicito sotto, stesso meccanismo _steal_release) -- non ricontare/rifadare
        # note_id (2026-09-16, hold-to-sustain sul tasto Play in gui.py): None per ogni
        # voce che non fa parte di una nota tenuta (OSC, audio-in, Play continuo, Auto
        # play, click singolo) -- mai un bersaglio di release_note in quel caso.
        self.note_id = note_id
        # gain per-voce (2026-09-16, shaker granulare): default 1.0, usato per dare a
        # ogni "grano" un'intensita' pseudo-random invece di una texture troppo regolare.
        self.gain = gain


def _steal_release(voice, sr, release_ms=_STEAL_RELEASE_MS):
    """Chiamata quando una voce va rimossa per voice-stealing (oltre max_voices, vedi
    _callback): invece di un troncamento istantaneo (click udibile, segnalato
    dall'utente 2026-09-16), applica una dissolvenza raised-cosine di release_ms dal
    punto di riproduzione attuale e accorcia li' il buffer -- la voce finisce comunque
    "in modo naturale" per il resto del mixing loop (nessun'altra modifica li'
    necessaria), solo anticipata invece che tagliata di scatto."""
    remaining = len(voice.audio) - voice.pos
    release_n = min(int(sr * release_ms / 1000.0), remaining)
    if release_n <= 1:
        return  # gia' quasi finita comunque, nessun beneficio da una dissolvenza qui
    end = voice.pos + release_n
    ramp = (0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, release_n, dtype=np.float32))).astype(np.float32)
    voice.audio = voice.audio.copy()
    voice.audio[voice.pos:end] *= ramp
    voice.audio = voice.audio[:end]


class PlayEngine:
    def __init__(self, candidate_source, max_voices=10, sr=SR, fade_ms=8.0,
                 morph=False, morph_ms=150.0, gain=0.4, latency="high",
                 limiter_ceiling=LIMITER_CEILING):
        """candidate_source: oggetto con attributo `.candidate` (param_candidate.Candidate
        o None) -- tipicamente un param_candidate.ParamCandidateWorker gia' avviato.

        morph (2026-09-16, segnalato dall'utente durante il test dell'auto-trigger,
        priorita' 3 punto 2: "il suono gratta ad ogni movimento di slider"). Prima
        versione: le voci vecchie venivano messe in dissolvenza forzata all'arrivo di
        una nuova (riusando _steal_release) -- l'utente ha chiesto di TOGLIERE questo
        fadeout (10 voci in overlap vanno bene) e ha ipotizzato che il "grattare" fosse
        in realta' CLIPPING (somma di piu' voci a piena ampiezza senza headroom), non
        lo scontro timbrico -- vedi `gain` sotto. morph ora controlla SOLO il fade-in
        della nuova voce (piu' lungo, morph_ms invece del fade_ms breve di default): le
        voci precedenti non vengono piu' toccate, continuano a suonare fino alla fine
        naturale o al voice-stealing oltre max_voices (invariato, vedi _callback).

        gain (2026-09-16, stesso motivo): fattore fisso applicato al mix finale prima
        dell'uscita PRIMA di un clip di sicurezza a [-1, 1] -- senza, sommare piu' voci
        vicine in ampiezza puo' superare 1.0 e il backend audio taglia in modo duro
        (suono "gritty"/gratta, indistinguibile a orecchio da un crossfade mancante).
        Default 0.4: con fino a max_voices=10 voci a piena ampiezza si resta entro
        [-1, 1] anche nel caso peggiore (10*0.4=4.0 di picco teorico, il clip di
        sicurezza copre comunque il residuo); nell'uso tipico (1-3 voci sovrapposte) da
        headroom ampio senza clip udibile. Regolabile se troppo debole/forte a orecchio."""
        self.candidate_source = candidate_source
        self.max_voices = max_voices
        self.sr = sr
        self.fade_ms = fade_ms
        self.morph = morph
        self.morph_ms = morph_ms
        self.gain = gain
        # latency (2026-09-16, diagnosi "click/grattare" segnalati dall'utente,
        # persistenti anche dopo morph+gain): sospetto principale non e' il mixer ma il
        # thread audio real-time in affanno -- ParamCandidateWorker fa girare SPSA in
        # background ogni refine_period (0.3s) e, per bow, spsa_refine() chiama
        # analyze_signal->formants() 8 volte (4 iterazioni x 2 valutazioni), un loop
        # Python per-frame che non rilascia il GIL a lungo. Col GIL condiviso, questo
        # puo' impedire al callback di sounddevice di rispettare la scadenza real-time
        # -> underrun del buffer -> click/crepitio, indipendente da mixing/gain (che
        # infatti non hanno risolto nulla). sd.OutputStream(latency="high") chiede a
        # PortAudio un buffer piu' grande (piu' margine, un po' piu' di latenza
        # sull'attacco) senza toccare mixing/render -- zero modifiche a
        # exciters.py/resonator.py. Se non bastasse, il passo successivo (non ancora
        # fatto qui) sarebbe ridurre il carico SPSA stesso (spsa_iterations/refine_period).
        self.latency = latency
        self.limiter_ceiling = limiter_ceiling
        self._limiter_gain = 1.0   # stato del limiter, letto/scritto SOLO da _callback
        self._voices = []          # letta/scritta solo dal thread audio (callback)
        self._pending = []         # buffer nuovi in attesa, protetti da _pending_lock
        self._release_requests = []  # richieste di note-off in attesa (note_id, fade_ms)
        self._pending_lock = threading.Lock()
        self._stream = None

    def start(self):
        self._stream = sd.OutputStream(
            samplerate=self.sr, channels=1, dtype="float32", callback=self._callback,
            latency=self.latency)
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def push_audio(self, audio, label=None, note_id=None, gain=1.0):
        """Accoda un buffer GIA' renderizzato per la riproduzione (2026-09-16, motore a
        processo separato -- vedi candidate_process.py): stessa coda/lock di trigger()
        sotto, solo senza fare il render qui. label opzionale solo per il log. note_id/
        gain (2026-09-16, hold-to-sustain): vedi _Voice sopra e release_note sotto."""
        with self._pending_lock:
            self._pending.append((np.asarray(audio, dtype=np.float32), note_id, gain))
        if label:
            ts = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
            print(f"[play_engine] {ts} play: {label}", file=sys.stderr)

    def trigger(self):
        """Percorso SINCRONO legacy (render nel thread chiamante) -- tenuto per
        compatibilita'/fallback, ma main.py/gui.py con motore a processo separato
        chiamano invece candidate_source.request_trigger() (il render avviene nel
        processo figlio, vedi candidate_process.py, cosi' non contende mai la GIL del
        thread audio -- diagnosi 2026-09-16, test_audio_pipeline.py sez.1/3/7).
        Legge il candidato corrente e renderizza SINCRONAMENTE (bloccante, puo' costare
        centinaia di ms in Python per alcuni eccitatori, vedi sez.1) -- un trigger per
        nota, non un flusso continuo (punto 6)."""
        cand = self.candidate_source.candidate
        if cand is None:
            print("[play_engine] nessun candidato disponibile ancora (nessun target ricevuto)",
                  file=sys.stderr)
            return
        try:
            duration = _estimate_duration(cand)
            raw, sr = exciter_generate(cand.exciter_name, duration=duration, **cand.exciter_params)
            audio = apply_resonator(raw, shape=cand.resonator_shape, f0=cand.f0, **cand.resonator_params)
            # in morph, l'attacco della nuova voce e' lungo quanto la dissolvenza in
            # uscita applicata alle voci precedenti in _callback (vedi PlayEngine.morph)
            # -- la coda naturale della nota resta invece sul fade_ms breve di sempre.
            fade_in = self.morph_ms if self.morph else self.fade_ms
            audio = _apply_fade(audio, sr, fade_in_ms=fade_in, fade_out_ms=self.fade_ms)
        except Exception as e:
            print(f"[play_engine] render fallito: {e}", file=sys.stderr)
            return
        tag = "raffinato" if cand.refined else "one-shot/ibrido"
        self.push_audio(audio, label=f"{cand.exciter_name}+{cand.resonator_shape} "
                                      f"({tag}, {len(audio) / sr:.2f}s)")

    def release_note(self, note_id, fade_ms=120.0):
        """Note-off (2026-09-16, hold-to-sustain sul tasto Play in gui.py): dissolvenza
        rapida (_steal_release, stesso meccanismo del voice-stealing) di tutte le voci
        vive con questo note_id. Chiamabile da un thread qualunque (tipicamente la GUI)
        -- accoda solo sotto lock, la mutazione vera di _voices resta confinata al
        thread audio (_callback), stesso pattern di push_audio/_pending sopra.
        note_id=None e' sempre un no-op (nessuna voce lo usa come tag reale, vedi
        _Voice.note_id -- una voce non tenuta non e' mai rilasciabile)."""
        if note_id is None:
            return
        with self._pending_lock:
            self._release_requests.append((note_id, fade_ms))

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print(f"[play_engine] stream status: {status}", file=sys.stderr)

        with self._pending_lock:
            if self._pending:
                # 2026-09-16: NON si tocca piu' qui la voci gia' attive (vedi
                # PlayEngine.morph -- l'utente ha chiesto di togliere il fadeout
                # forzato). Restano semplicemente in overlap fino alla fine naturale o
                # al voice-stealing oltre max_voices sotto, invariato.
                self._voices.extend(_Voice(a, note_id=nid, gain=g) for a, nid, g in self._pending)
                self._pending.clear()
            releases = self._release_requests
            self._release_requests = []

        # note-off esplicito (2026-09-16, hold-to-sustain): stessa dissolvenza breve del
        # voice-stealing, mirata pero' alle sole voci di questo note_id -- riusa .stolen
        # come flag "gia' in fade forzato" (il voice-stealing sotto la rispetta gia':
        # `live = [v for v in self._voices if not v.stolen]`, una voce appena rilasciata
        # non conta piu' nemmeno li', comportamento corretto).
        for note_id, fade_ms in releases:
            for v in self._voices:
                if v.note_id == note_id and not v.stolen:
                    _steal_release(v, self.sr, release_ms=fade_ms)
                    v.stolen = True

        # voice-stealing: oltre max_voices, le voci piu' vecchie NON attive gia' in
        # dissolvenza vengono avviate in dissolvenza (non ricontate qui finche' non
        # escono da sole a fine buffer accorciato, vedi _steal_release) invece di
        # essere scartate di scatto.
        live = [v for v in self._voices if not v.stolen]
        if len(live) > self.max_voices:
            live.sort(key=lambda v: v.born)
            for v in live[:-self.max_voices]:
                _steal_release(v, self.sr)
                v.stolen = True

        mix = np.zeros(frames, dtype=np.float32)
        alive = []
        for v in self._voices:
            end = min(v.pos + frames, len(v.audio))
            n = end - v.pos
            if n > 0:
                mix[:n] += v.audio[v.pos:end] * v.gain
                v.pos = end
            if v.pos < len(v.audio):
                alive.append(v)
        self._voices = alive
        # gain manuale (vedi PlayEngine.gain) applicato per primo -- "quanto forte lo
        # voglio", scelta dell'utente. Il limiter sotto e' la rete di sicurezza finale,
        # indipendente dal gain: garantisce che il picco non superi limiter_ceiling
        # qualunque sia il gain scelto o il numero di voci in overlap.
        mix *= self.gain

        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        target_gain = (self.limiter_ceiling / peak) if peak > self.limiter_ceiling else 1.0
        buf_s = frames / self.sr
        tau_s = (_LIMITER_ATTACK_MS if target_gain < self._limiter_gain else _LIMITER_RELEASE_MS) / 1000.0
        coef = 1.0 - np.exp(-buf_s / tau_s) if tau_s > 0 else 1.0
        self._limiter_gain += (target_gain - self._limiter_gain) * coef
        mix *= self._limiter_gain

        np.clip(mix, -1.0, 1.0, out=mix)  # backstop finale, dovrebbe intervenire raramente ora
        outdata[:, 0] = mix
