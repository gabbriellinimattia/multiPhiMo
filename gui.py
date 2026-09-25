"""
gui.py -- GUI Tkinter per il prototipo (richiesta 2026-09-15): selettore eccitatore,
selettore risonatore, OSC receive on/off, uno slider per descrittore rilevante (auto se
OSC e' on, manuale se off), play singolo, play continuo/ripetuto ogni 1s.

Wiring identico a main.py -- stesso DescriptorInput + CandidateProcessHandle +
PlayEngine, qui guidati da un'interfaccia invece che da CLI/stdin, nessuna logica
duplicata. Tkinter: nella libreria standard, nessuna dipendenza aggiuntiva.

2026-09-16 (diagnosi crackle, vedi candidate_process.py): il candidato+render ora
girano in un processo separato, non piu' in AgentManager/ParamCandidateWorker
istanziati qui -- questo modulo non ha piu' un riferimento diretto all'agente, solo
un mirror locale (self._exciter_name/_resonator_shape) per la UI.

Non eseguito qui (vincolo di progetto): lancialo tu, riportami il log/l'esito.
"""
import argparse
import json
import math
import random
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import tuning
from agents import DESCRIPTOR_KEYS, EXCITER_PARAM_RANGES, RESONATOR_SHAPES, descriptor_keys_for
from audio_input import AudioDescriptorSource
from candidate_process import CandidateProcessHandle
from descriptor_input import DescriptorInput
from param_candidate import AUTO_TRIGGER_MIN_HOLD, AUTO_TRIGGER_THRESHOLD
from play_engine import LIMITER_CEILING, PlayEngine

EXCITERS = sorted(EXCITER_PARAM_RANGES)
RESONATORS = sorted(RESONATOR_SHAPES)

# Range di partenza (usati solo per i descrittori assenti da slider_ranges.json, o se
# il file manca del tutto -- Hz per gli spettrali/formanti/pitch, 0-1 per i normalizzati,
# secondi per attack/decay).
DEFAULT_SLIDER_RANGES = {
    "spectral_centroid": (20.0, 8000.0),
    "spectral_spread": (0.0, 4000.0),
    "spectral_rolloff": (20.0, 8000.0),  # 2026-09-16f, nuovo descrittore
    "spectral_flatness": (0.0, 1.0),
    "roughness": (0.0, 1.0),
    "harmonic_tension": (0.0, 1.0),
    "inharmonicity": (0.0, 1.0),  # 2026-09-16f, nuovo descrittore
    "formant_f1": (50.0, 2000.0),
    "formant_f2": (50.0, 4000.0),
    "formant_f3": (50.0, 6000.0),
    "mod_rate": (0.0, 20.0),
    "mod_depth": (0.0, 1.0),
    "attack_time": (0.0, 2.0),
    "decay_time": (0.0, 2.0),
    "pitch": (25.0, 4500.0),
}


def _load_slider_ranges():
    """Range reali (p1-p99 sul corpus ~/Desktop/sample, vedi compute_slider_ranges.py)
    se slider_ranges.json e' presente accanto a questo file, altrimenti i default sopra
    -- per descrittore, mai bloccante: chiave assente dal JSON usa comunque il default."""
    ranges = dict(DEFAULT_SLIDER_RANGES)
    try:
        with open(Path(__file__).parent / "slider_ranges.json") as f:
            real = json.load(f)
        for k, v in real.items():
            ranges[k] = (float(v[0]), float(v[1]))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError, IndexError):
        pass
    return ranges


SLIDER_RANGES = _load_slider_ranges()


def _load_agent_slider_ranges():
    """Range p1-p99 PER ECCITATORE (vedi compute_agent_ranges.py, 2026-09-24): a
    differenza di SLIDER_RANGES (corpus di suoni reali, un range unico per tutti gli
    agenti), qui il range riflette cosa il singolo eccitatore selezionato puo'
    REALMENTE produrre nel proprio dataset di training -- elimina le zone 'inerti'
    quando il suo registro e' piu' stretto del corpus reale globale. Assente/mancante
    -> dict vuoto, _agent_range ricade su SLIDER_RANGES (comportamento pre-esistente)."""
    try:
        with open(Path(__file__).parent / "agent_slider_ranges.json") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return {}


AGENT_SLIDER_RANGES = _load_agent_slider_ranges()

# Descrittori di frequenza -- slider su scala logaritmica (l'orecchio percepisce il
# pitch/formanti/centroid logaritmicamente, non linearmente; mod_rate/attack_time/
# decay_time restano lineari, non richiesti dal punto 2 del prompt 2026-09-16).
LOG_SCALE_KEYS = {"spectral_centroid", "spectral_spread", "spectral_rolloff",
                  "formant_f1", "formant_f2", "formant_f3", "pitch"}

# 2026-09-16f: descrittori esclusi dagli slider manuali (restano nel training/routing
# tramite agents.DESCRIPTOR_KEYS, invariato). inharmonicity e' diagnostico -- deriva da
# f0 e spesso NaN quando il pitch non e' rilevato -- non un parametro pensato per essere
# impostato a mano.
GUI_HIDDEN_KEYS = {"inharmonicity"}

# Hold-to-sustain sul tasto Play (punto 1, 2026-09-16): press=note-on, unpress=note-off.
SUSTAIN_CROSSFADE_EXCITERS = {"bow", "blow", "chaos", "noise"}
GRANULAR_EXCITERS = {"shaker"}
CLICK_THRESHOLD_S = 0.18       # sotto: click, nessun release forzato (inviluppo naturale)
SUSTAIN_RETRIGGER_MS = 1000    # stesso ritmo di "Play continuo", gia' validato dal vivo
GRANULAR_RETRIGGER_MS = 130    # shaker e' vettoriale (no loop per-campione), regge un ritmo fitto
RELEASE_FADE_MS = 120.0        # dissolvenza al note-off (PlayEngine.release_note)
GRANULAR_GAIN_RANGE = (0.4, 1.0)  # intensita' pseudo-random per grano (shaker)


def _agent_range(k, exciter_name):
    """Range REALE (non-log) per il descrittore k. Se exciter_name e' dato (2026-09-24,
    'niente zone inerti'): pitch usa il range ESATTO di freq dell'eccitatore selezionato
    (EXCITER_PARAM_RANGES, lo stesso che param_candidate._lock_freq usa per il clip a
    runtime -- ogni valore dello slider e' quindi un target raggiungibile); gli altri
    descrittori usano il p1-p99 per-eccitatore in AGENT_SLIDER_RANGES quando disponibile.
    Altrimenti (exciter_name assente, o descrittore non coperto) ricade su SLIDER_RANGES
    (corpus reale globale, comportamento pre-esistente)."""
    if exciter_name is not None:
        if k == "pitch" and exciter_name in EXCITER_PARAM_RANGES:
            return tuple(EXCITER_PARAM_RANGES[exciter_name]["freq"])
        per_agent = AGENT_SLIDER_RANGES.get(exciter_name, {}).get(k)
        if per_agent is not None:
            return tuple(per_agent)
    return SLIDER_RANGES.get(k, (0.0, 1.0))


def _slider_bounds(k, exciter_name=None):
    """Estremi dello slider nel suo 'dominio' (lineare, o log10 per LOG_SCALE_KEYS)."""
    lo, hi = _agent_range(k, exciter_name)
    if k in LOG_SCALE_KEYS:
        return math.log10(max(lo, 1e-3)), math.log10(max(hi, 1e-3))
    return lo, hi


def _to_slider_domain(k, value):
    return math.log10(max(value, 1e-3)) if k in LOG_SCALE_KEYS else value


def _from_slider_domain(k, pos):
    return 10 ** pos if k in LOG_SCALE_KEYS else pos


class App:
    def __init__(self, root, args):
        self.root = root
        self.args = args

        # mirror locale eccitatore/risonatore attivi -- l'AgentManager vero vive solo
        # nel processo figlio (candidate_process.py), qui serve solo per calcolare le
        # chiavi rilevanti degli slider (_relevant_keys) e per il testo della UI.
        self._exciter_name = args.exciter
        self._resonator_shape = args.resonator

        self.descriptor_input = DescriptorInput(
            self._relevant_keys(), ip=args.osc_ip, port=args.osc_port, address=args.osc_address)
        # audio-in dal microfono (2026-09-16): terza sorgente di target, simmetrica a
        # OSC/slider manuali -- scrive sullo stesso descriptor_input.set_target(),
        # nessuna logica duplicata. Costruito ma non avviato (vedi checkbox "Audio in").
        self.audio_source = AudioDescriptorSource(self.descriptor_input, smoothing_ms=args.smoothing_ms)

        self.candidate_proc = CandidateProcessHandle(
            self.descriptor_input, weights_dir=args.weights_dir, dataset_dir=args.dataset_dir,
            initial_exciter=args.exciter, initial_resonator=args.resonator,
            spsa_iterations=args.spsa_iterations, refine_period=args.refine_period,
            fade_ms=args.fade_ms, selector_csv=args.selector_csv, selector_k=args.selector_k,
            selector_bias_alpha=args.selector_bias_alpha, auto_pair_min_hold=args.auto_pair_min_hold,
            auto_trigger_threshold=args.auto_trigger_threshold,
            auto_trigger_min_hold=args.auto_trigger_min_hold, morph_ms=args.morph_ms)

        self.engine = PlayEngine(self.candidate_proc, max_voices=args.max_voices, fade_ms=args.fade_ms,
                                  morph_ms=args.morph_ms, gain=args.gain, latency=args.latency,
                                  limiter_ceiling=args.limiter_ceiling)
        self.engine.start()
        # audio_sink impostato PRIMA di candidate_proc.start() (stesso ordine di prima
        # con trigger_callback/worker.start(), vedi main.py): evita che il thread
        # ricevitore del processo figlio scarti in silenzio un buffer arrivato prima
        # che il sink sia agganciato -- render+trigger vivono ora nel processo figlio
        # (2026-09-16, diagnosi crackle), vedi candidate_process.py.
        self.candidate_proc.audio_sink = self.engine.push_audio
        self.candidate_proc.start()

        self._osc_on = False
        self._repeat_job = None
        self._slider_vars = {}
        self._sliders = {}
        self._suppress_slider_cb = False

        # toggle "Scale" (punto 3, 2026-09-16): arrotonda il pitch target alla nota
        # temperata piu' vicina -- vedi tuning.py per il meccanismo (ancora = "La" scelto,
        # nessuna mappatura a nomi di nota/tonalita').
        self.scale_var = tk.BooleanVar(value=False)
        self.a4_var = tk.StringVar(value="440")
        self.scale_name_var = tk.StringVar(value="edo12")
        self._custom_scale = None
        self._custom_scale_label = "(nessun file)"

        # hold-to-sustain (2026-09-16): note_id incrementale per distinguere una nota
        # tenuta dalla successiva (vedi PlayEngine.release_note/_Voice.note_id);
        # _held_job e' l'after() del retrigger periodico in corso. Il morph forzato e'
        # stato tolto (2026-09-16b, "morph non e' un granche'"): i retrigger di sustain
        # ora tagliano l'attacco (sustain_continuation, vedi candidate_process.py) invece
        # di crossfadare due render indipendenti -- morph resta quello che l'utente imposta.
        self._note_id_counter = 0
        self._held_note_id = None
        self._held_job = None
        self._press_time = None

        self._build_ui()
        self._poll_osc()

    # ---- chiavi rilevanti (dipendono da eccitatore+risonatore attivi, punto 1) ----

    def _relevant_keys(self):
        if getattr(self, "_auto_on", False):
            return [k for k in DESCRIPTOR_KEYS if k not in GUI_HIDDEN_KEYS]
        # ordine canonico (agents.DESCRIPTOR_KEYS), non alfabetico: gli slider restano
        # sempre nella stessa posizione al cambiare di eccitatore/risonatore (punto 1
        # del prompt 2026-09-16) -- pitch e' sempre l'ultimo per costruzione della lista.
        keys = set(descriptor_keys_for(self._exciter_name)) | \
            set(descriptor_keys_for(f"resonator_{self._resonator_shape}"))
        return [k for k in DESCRIPTOR_KEYS if k in keys and k not in GUI_HIDDEN_KEYS]

    # ---- costruzione UI ----

    def _build_ui(self):
        self.root.title("multiPhiMo -- prototipo")

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Eccitatore").grid(row=0, column=0, sticky="w")
        self.exciter_var = tk.StringVar(value=self.args.exciter)
        exciter_menu = ttk.Combobox(top, textvariable=self.exciter_var, values=EXCITERS,
                                     state="readonly", width=12)
        exciter_menu.grid(row=0, column=1, padx=4)
        exciter_menu.bind("<<ComboboxSelected>>", self._on_exciter_change)

        ttk.Label(top, text="Risonatore").grid(row=0, column=2, sticky="w")
        self.resonator_var = tk.StringVar(value=self.args.resonator)
        resonator_menu = ttk.Combobox(top, textvariable=self.resonator_var, values=RESONATORS,
                                       state="readonly", width=12)
        resonator_menu.grid(row=0, column=3, padx=4)
        resonator_menu.bind("<<ComboboxSelected>>", self._on_resonator_change)

        self.osc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="OSC receive", variable=self.osc_var,
                        command=self._on_osc_toggle).grid(row=0, column=4, padx=12)

        self._audio_in_on = False
        self.audio_in_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Audio in (microfono)", variable=self.audio_in_var,
                        command=self._on_audio_in_toggle).grid(row=0, column=6, padx=12)

        self._auto_on = False
        self.auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Auto coppia", variable=self.auto_var,
                        command=self._on_auto_toggle).grid(row=0, column=5, padx=12)
        self.exciter_menu = exciter_menu
        self.resonator_menu = resonator_menu

        self.sliders_frame = ttk.Frame(self.root, padding=8)
        self.sliders_frame.pack(fill="both", expand=True)
        self._build_sliders()

        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(fill="x")
        play_btn = ttk.Button(bottom, text="Play")
        play_btn.pack(side="left")
        # hold-to-sustain (2026-09-16): niente command=, press/release espliciti --
        # vedi _on_play_press/_on_play_release sotto.
        play_btn.bind("<ButtonPress-1>", self._on_play_press)
        play_btn.bind("<ButtonRelease-1>", self._on_play_release)
        self.repeat_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bottom, text="Play continuo (ogni 1s)", variable=self.repeat_var,
                        command=self._on_repeat_toggle).pack(side="left", padx=12)
        self.autotrigger_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bottom, text="Auto play (segue variazioni)", variable=self.autotrigger_var,
                        command=self._on_autotrigger_toggle).pack(side="left", padx=12)
        self.morph_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bottom, text="Morph (continuita' timbrica)", variable=self.morph_var,
                        command=self._on_morph_toggle).pack(side="left", padx=12)
        # gain regolabile a runtime (2026-09-16, diagnosi crackle): prima solo --gain in
        # avvio (valore fisso per tutta la sessione), ora anche uno slider live -- utile
        # per cercare a orecchio il punto di clipping mentre si ascolta, senza riavviare.
        ttk.Label(bottom, text="Gain").pack(side="left", padx=(12, 2))
        self.gain_var = tk.DoubleVar(value=self.engine.gain)
        gain_slider = ttk.Scale(bottom, from_=0.0, to=1.5, variable=self.gain_var,
                                 orient="horizontal", length=100, command=self._on_gain_change)
        gain_slider.pack(side="left")
        self.gain_lbl = ttk.Label(bottom, text=f"{self.engine.gain:.2f}", width=5)
        self.gain_lbl.pack(side="left", padx=(2, 0))
        # smoothing dell'audio-in (2026-09-16): costante di tempo (ms) del fade tra un
        # set di descrittori analizzati dal microfono e il successivo -- regolabile a
        # runtime, stesso schema di gain sopra.
        ttk.Label(bottom, text="Smoothing").pack(side="left", padx=(12, 2))
        self.smoothing_var = tk.DoubleVar(value=self.audio_source.smoothing_ms)
        smoothing_slider = ttk.Scale(bottom, from_=20.0, to=1000.0, variable=self.smoothing_var,
                                      orient="horizontal", length=100, command=self._on_smoothing_change)
        smoothing_slider.pack(side="left")
        self.smoothing_lbl = ttk.Label(bottom, text=f"{self.audio_source.smoothing_ms:.0f}ms", width=7)
        self.smoothing_lbl.pack(side="left", padx=(2, 0))
        self.status_var = tk.StringVar(value="pronto")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="right")

    def _build_sliders(self):
        for w in self.sliders_frame.winfo_children():
            w.destroy()
        self._slider_vars.clear()
        self._sliders.clear()

        for i, k in enumerate(self._relevant_keys()):
            lo, hi = _slider_bounds(k, self._exciter_name)
            lo_real, hi_real = _agent_range(k, self._exciter_name)
            ttk.Label(self.sliders_frame, text=f"{k}  [{lo_real:.4g}-{hi_real:.4g}]",
                      width=32).grid(row=i, column=0, sticky="w")
            var = tk.DoubleVar(value=(lo + hi) / 2.0)
            slider = ttk.Scale(self.sliders_frame, from_=lo, to=hi, variable=var, orient="horizontal",
                                length=280, command=lambda _v, k=k: self._on_slider_move(k))
            slider.grid(row=i, column=1, sticky="ew", padx=6)
            value_lbl = ttk.Label(self.sliders_frame, width=14)
            value_lbl.grid(row=i, column=2, sticky="w")
            self._slider_vars[k] = var
            self._sliders[k] = (slider, value_lbl)
            if k == "pitch":
                self._build_scale_controls(row=i + 1)

        self.sliders_frame.columnconfigure(1, weight=1)
        self._refresh_slider_labels()
        self._set_sliders_enabled(not (self._osc_on or self._audio_in_on))
        if not self._osc_on and not self._audio_in_on:
            self._push_manual_target()

    # ---- controlli "Scale" (punto 3, 2026-09-16): checkbox + riferimento "La" (A4) +
    # scala (built-in o importata da file .scl) -- affiancati allo slider pitch, mai un
    # secondo slider: la quantizzazione agisce sul VALORE inviato (descriptor_input.
    # quantize_pitch), non sulla posizione dello slider, coerente con "matching sui
    # descrittori" del progetto (il target diventa la nota piu' vicina, il resto invariato) ----

    def _build_scale_controls(self, row):
        # riga propria sotto pitch (2026-09-16b), non piu' affiancata: pitch e'
        # sempre l'ultimo slider per costruzione di _relevant_keys, quindi row+1 e'
        # sempre libera.
        frame = ttk.Frame(self.sliders_frame)
        frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=(2, 6))
        ttk.Checkbutton(frame, text="Scale", variable=self.scale_var,
                        command=self._on_scale_change).pack(side="left")
        ttk.Label(frame, text="La").pack(side="left", padx=(8, 2))
        a4_menu = ttk.Combobox(frame, textvariable=self.a4_var,
                                values=["415", "432", "440", "442", "443", "444"], width=5)
        a4_menu.pack(side="left")
        a4_menu.bind("<<ComboboxSelected>>", self._on_scale_change)
        a4_menu.bind("<Return>", self._on_scale_change)
        scale_menu = ttk.Combobox(
            frame, textvariable=self.scale_name_var,
            values=["edo12", "edo24", "edo31", "perfect", "harmonic", "custom"],
            state="readonly", width=9)
        scale_menu.pack(side="left", padx=(8, 2))
        scale_menu.bind("<<ComboboxSelected>>", self._on_scale_change)
        ttk.Button(frame, text="Importa...", command=self._import_tuning).pack(side="left", padx=(4, 2))
        self.custom_scale_lbl = ttk.Label(frame, text=self._custom_scale_label)
        self.custom_scale_lbl.pack(side="left")

    def _import_tuning(self):
        path = filedialog.askopenfilename(
            title="Importa file di tuning (formato Scala .scl)",
            filetypes=[("Scala tuning", "*.scl"), ("Tutti i file", "*.*")])
        if not path:
            return
        try:
            degrees, period = tuning.load_scl(path)
        except Exception as e:
            self.status_var.set(f"import tuning fallito: {e}")
            return
        self._custom_scale = (degrees, period)
        self._custom_scale_label = Path(path).name
        self.custom_scale_lbl.config(text=self._custom_scale_label)
        self.scale_name_var.set("custom")
        self.status_var.set(f"tuning importato: {self._custom_scale_label} "
                             f"({len(degrees)} gradi, periodo {period:.4g})")
        self._on_scale_change()

    def _on_scale_change(self, _evt=None):
        if not self.scale_var.get():
            self.descriptor_input.quantize_pitch = None
            self._refresh_slider_labels()
            return
        name = self.scale_name_var.get()
        if name == "custom" and self._custom_scale is None:
            self.status_var.set("Scale: importa prima un file di tuning per 'custom'")
            self.scale_var.set(False)
            self.descriptor_input.quantize_pitch = None
            return
        try:
            a4 = float(self.a4_var.get())
        except ValueError:
            a4 = 440.0
        self.descriptor_input.quantize_pitch = tuning.make_quantizer(
            name, a4=a4, custom=self._custom_scale)
        if not self._osc_on and not self._audio_in_on:
            self._push_manual_target()  # riapplica subito la quantizzazione al target corrente
        self._refresh_slider_labels()

    def _refresh_slider_labels(self):
        for k, (_, lbl) in self._sliders.items():
            actual = _from_slider_domain(k, self._slider_vars[k].get())
            if k == "pitch" and self.descriptor_input.quantize_pitch is not None:
                q = self.descriptor_input.quantize_pitch(actual)
                lbl.config(text=f"{actual:.4g}\u2192{q:.4g}")
            else:
                lbl.config(text=f"{actual:.3g}")

    def _set_sliders_enabled(self, enabled):
        for slider, _ in self._sliders.values():
            slider.state(["!disabled"] if enabled else ["disabled"])

    # ---- selettori: cambiare eccitatore/risonatore ricrea gli slider e inoltra la
    # selezione al processo figlio (candidate_process.py fa stop/select/start sul suo
    # worker interno, stessa cautela di prima contro una coppia nome/agente
    # disallineata -- solo spostata li', non piu' qui) ----

    def _on_exciter_change(self, _evt=None):
        self._exciter_name = self.exciter_var.get()
        self.candidate_proc.select_exciter(self._exciter_name)
        self.descriptor_input.relevant_keys = set(self._relevant_keys())
        self._build_sliders()
        self.status_var.set(f"eccitatore -> {self._exciter_name}")

    def _on_resonator_change(self, _evt=None):
        self._resonator_shape = self.resonator_var.get()
        self.candidate_proc.select_resonator(self._resonator_shape)
        self.descriptor_input.relevant_keys = set(self._relevant_keys())
        self._build_sliders()
        self.status_var.set(f"risonatore -> {self._resonator_shape}")

    def _on_auto_toggle(self):
        self._auto_on = self.auto_var.get()
        self.candidate_proc.set_auto_pair(self._auto_on)
        state = "disabled" if self._auto_on else "readonly"
        self.exciter_menu.state([state] if state == "disabled" else ["!disabled", "readonly"])
        self.resonator_menu.state([state] if state == "disabled" else ["!disabled", "readonly"])
        if not self._auto_on:
            # ripristina la coppia mostrata nei combobox (override manuale esplicito)
            self._exciter_name = self.exciter_var.get()
            self._resonator_shape = self.resonator_var.get()
            self.candidate_proc.select_exciter(self._exciter_name)
            self.candidate_proc.select_resonator(self._resonator_shape)
        self.descriptor_input.relevant_keys = set(self._relevant_keys())
        self._build_sliders()

    def _on_osc_toggle(self):
        self._osc_on = self.osc_var.get()
        if self._osc_on:
            if not self.descriptor_input.start_osc():
                self.osc_var.set(False)
                self._osc_on = False
                self.status_var.set("python-osc non installato: OSC non disponibile")
        else:
            self.descriptor_input.stop()
            if not self._audio_in_on:
                self._push_manual_target()  # riprende dal valore corrente degli slider
        self._set_sliders_enabled(not (self._osc_on or self._audio_in_on))

    # ---- audio-in dal microfono (2026-09-16): stesso schema di _on_osc_toggle sopra
    # -- indipendente da OSC (entrambi possono restare attivi insieme, come Play
    # continuo + Auto play altrove: ridondante ma non escluso), scrive sullo stesso
    # descriptor_input ----

    def _on_audio_in_toggle(self):
        self._audio_in_on = self.audio_in_var.get()
        if self._audio_in_on:
            self.audio_source.start()
        else:
            self.audio_source.stop()
            if not self._osc_on:
                self._push_manual_target()
        self._set_sliders_enabled(not (self._osc_on or self._audio_in_on))

    def _on_smoothing_change(self, _v):
        self.audio_source.smoothing_ms = self.smoothing_var.get()
        self.smoothing_lbl.config(text=f"{self.audio_source.smoothing_ms:.0f}ms")

    def _on_slider_move(self, _k):
        if self._suppress_slider_cb:
            return
        self._refresh_slider_labels()
        if not self._osc_on:
            self._push_manual_target()

    def _push_manual_target(self):
        self.descriptor_input.set_target(
            {k: _from_slider_domain(k, var.get()) for k, var in self._slider_vars.items()})

    # ---- hold-to-sustain sul tasto Play (punto 1, 2026-09-16): press=note-on,
    # unpress=note-off. Un click breve (sotto CLICK_THRESHOLD_S) lascia il buffer gia'
    # innescato suonare fino alla fine naturale (attacco/decay), esattamente come
    # prima -- release_note interviene SOLO se il tasto resta premuto oltre la soglia. ----

    def _on_play_press(self, _evt=None):
        self._note_id_counter += 1
        self._held_note_id = self._note_id_counter
        self._press_time = time.monotonic()
        exc = self._exciter_name
        if exc in GRANULAR_EXCITERS:
            self._fire_granular_grain()
            self._held_job = self.root.after(GRANULAR_RETRIGGER_MS, self._schedule_granular)
        elif exc in SUSTAIN_CROSSFADE_EXCITERS:
            # primo trigger = attacco vero (sustain_continuation=False di default);
            # niente piu' morph forzato (2026-09-16b): vedi _schedule_sustain sotto.
            self.candidate_proc.request_trigger(note_id=self._held_note_id)
            self._held_job = self.root.after(SUSTAIN_RETRIGGER_MS, self._schedule_sustain)
        else:
            # pluck/strike (impulsivi): hold ignorato, comportamento invariato
            self.candidate_proc.request_trigger()

    def _schedule_sustain(self):
        # retrigger di continuazione (2026-09-16b): sustain_continuation=True fa
        # tagliare l'attacco al processo figlio (SUSTAIN_TRIM_S, candidate_process.py)
        # e lo splice diventa il corpo del suono che continua a evolvere, non un
        # secondo attacco crossfadato sopra al primo.
        self.candidate_proc.request_trigger(note_id=self._held_note_id, sustain_continuation=True)
        self._held_job = self.root.after(SUSTAIN_RETRIGGER_MS, self._schedule_sustain)

    def _fire_granular_grain(self):
        gain = random.uniform(*GRANULAR_GAIN_RANGE)
        self.candidate_proc.request_trigger(note_id=self._held_note_id, gain=gain)

    def _schedule_granular(self):
        self._fire_granular_grain()
        self._held_job = self.root.after(GRANULAR_RETRIGGER_MS, self._schedule_granular)

    def _on_play_release(self, _evt=None):
        if self._held_job is not None:
            self.root.after_cancel(self._held_job)
            self._held_job = None
        held_s = time.monotonic() - (self._press_time or 0.0)
        exc = self._exciter_name
        if exc in SUSTAIN_CROSSFADE_EXCITERS:
            if held_s >= CLICK_THRESHOLD_S:
                self.engine.release_note(self._held_note_id, fade_ms=RELEASE_FADE_MS)
            # click (held_s < soglia): nessun release forzato, il buffer gia' innescato
            # suona fino alla fine naturale, esattamente come un click prima di oggi.
        elif exc in GRANULAR_EXCITERS and held_s >= CLICK_THRESHOLD_S:
            self.engine.release_note(self._held_note_id, fade_ms=RELEASE_FADE_MS)
        self._held_note_id = None

    # ---- play continuo/ripetuto ----

    def _on_repeat_toggle(self):
        if self.repeat_var.get():
            self._schedule_repeat()
        elif self._repeat_job is not None:
            self.root.after_cancel(self._repeat_job)
            self._repeat_job = None

    def _schedule_repeat(self):
        self.candidate_proc.request_trigger()
        if self.repeat_var.get():
            self._repeat_job = self.root.after(1000, self._schedule_repeat)

    # ---- auto play (priorita' 3 punto 2): suona da solo ad ogni cambio sostanziale del
    # candidato invece di attendere Invio/Play -- indipendente da "Play continuo" sopra
    # (che suona ad intervallo fisso indipendentemente dal contenuto del candidato),
    # possono anche essere entrambi attivi ma e' ridondante, nessuna esclusione forzata ----

    def _on_autotrigger_toggle(self):
        self.candidate_proc.set_auto_trigger(self.autotrigger_var.get())

    # ---- morph (segnalato dall'utente 2026-09-16: "il suono gratta ad ogni movimento
    # di slider" con Auto play attivo -- crossfade tra voce precedente e nuova invece di
    # farle suonare in overlap indipendente, vedi PlayEngine.morph) ----

    def _on_morph_toggle(self):
        # 2026-09-18: il vecchio self.engine.morph pilotava solo play_engine.trigger()
        # (percorso sync legacy, mai usato da questa GUI -- il render vero passa dal
        # processo figlio, vedi candidate_process.py) quindi la checkbox non aveva
        # alcun effetto sul suono qui. Ora comanda candidate_proc.set_morph (B/A,
        # continuita' dei parametri intra-coppia) -- vedi
        # claude/morphing_spettrale_ricerca_proposta.md. self.engine.morph resta
        # assegnato per non rompere il percorso legacy, se mai riusato altrove.
        self.engine.morph = self.morph_var.get()
        self.candidate_proc.set_morph(self.morph_var.get())

    def _on_gain_change(self, _v):
        self.engine.gain = self.gain_var.get()
        self.gain_lbl.config(text=f"{self.engine.gain:.2f}")

    # ---- polling: quando OSC e' on riflette il target sugli slider; quando "Auto
    # coppia" e' on riflette la coppia scelta dal selettore sui combobox (sempre
    # visibili, anche se disabilitati -- vedi _on_auto_toggle) ----

    def _poll_osc(self):
        if self._auto_on:
            exc, res = self.candidate_proc.exciter_name, self.candidate_proc.resonator_shape
            if exc is not None and self.exciter_var.get() != exc:
                self.exciter_var.set(exc)
                self._exciter_name = exc
            if res is not None and self.resonator_var.get() != res:
                self.resonator_var.set(res)
                self._resonator_shape = res
        if (self._osc_on or self._audio_in_on) and self.descriptor_input.latest is not None:
            self._suppress_slider_cb = True
            for k, v in self.descriptor_input.latest.values.items():
                if k in self._slider_vars:
                    lo, hi = _agent_range(k, self._exciter_name)
                    clipped = max(lo, min(hi, v))
                    self._slider_vars[k].set(_to_slider_domain(k, clipped))
            self._refresh_slider_labels()
            self._suppress_slider_cb = False
        self.root.after(100, self._poll_osc)

    def shutdown(self):
        if self._repeat_job is not None:
            self.root.after_cancel(self._repeat_job)
        self.descriptor_input.stop()
        self.audio_source.stop()
        self.candidate_proc.stop()
        self.engine.stop()


def main():
    p = argparse.ArgumentParser(description="GUI del prototipo multiPhiMo.")
    p.add_argument("--exciter", default="bow", choices=EXCITERS)
    p.add_argument("--resonator", default="bar", choices=RESONATORS)
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--dataset-dir", default="dataset")
    p.add_argument("--osc-ip", default="0.0.0.0")
    p.add_argument("--osc-port", type=int, default=9000)
    p.add_argument("--osc-address", default="/multiphimo/target")
    p.add_argument("--spsa-iterations", type=int, default=4)
    p.add_argument("--refine-period", type=float, default=0.3)
    p.add_argument("--max-voices", type=int, default=10)
    p.add_argument("--fade-ms", type=float, default=8.0)
    p.add_argument("--gain", type=float, default=0.4,
                    help="attenuazione fissa sul mix finale prima del clip di sicurezza (evita clipping con piu' voci in overlap)")
    p.add_argument("--limiter-ceiling", type=float, default=LIMITER_CEILING,
                    help="picco massimo assoluto dopo il gain (limiter, indipendente dal gain scelto)")
    p.add_argument("--latency", default="high",
                    help="latency di sounddevice.OutputStream ('low'/'high' o secondi): piu' alta = piu' margine "
                         "contro click/underrun causati dal thread SPSA in background")
    p.add_argument("--selector-csv", default="selector_dataset.csv",
                    help="dataset del selettore (gen_selector_dataset.py), usato solo con Auto coppia")
    p.add_argument("--selector-k", type=int, default=30)
    p.add_argument("--selector-bias-alpha", type=float, default=0.3,
                    help="correzione anti-sbilanciamento per eccitatore ""(0=nessuna, vedi pair_selector.PairSelectorKnn)")
    p.add_argument("--auto-pair-min-hold", type=float, default=2.0)
    p.add_argument("--auto-trigger-threshold", type=float, default=AUTO_TRIGGER_THRESHOLD,
                    help="soglia di distanza normalizzata (0-1) per l'Auto play (priorita' 3 punto 2)")
    p.add_argument("--auto-trigger-min-hold", type=float, default=AUTO_TRIGGER_MIN_HOLD,
                    help="secondi minimi tra due auto-trigger (limita il rumore di SPSA)")
    p.add_argument("--morph-ms", type=float, default=150.0,
                    help="durata (ms) del crossfade quando la checkbox Morph e' attiva")
    p.add_argument("--smoothing-ms", type=float, default=200.0,
                    help="costante di tempo (ms) del fade tra un set di descrittori "
                         "audio-in e il successivo (vedi audio_input.py)")
    args = p.parse_args()

    root = tk.Tk()
    app = App(root, args)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.shutdown(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
