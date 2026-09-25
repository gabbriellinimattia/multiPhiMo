#pragma once
// Tabelle/indici condivisi tra DSP (PluginMultiPhiMo.cpp) e UI (UIMultiPhiMo.cpp) --
// spostate qui dal punto 8A (GUI) per evitare di duplicare le stesse stringhe/range in
// due translation unit separate (rischio di disallineamento). Va incluso DENTRO
// START_NAMESPACE_DISTRHO/END_NAMESPACE_DISTRHO in entrambi i file (come le tabelle
// stavano gia' inline in PluginMultiPhiMo.cpp prima di questo spostamento) -- qualifica
// "static" mantenuta apposta: ogni .cpp che include questo header ottiene la propria
// copia locale (linkage interno), nessun simbolo condiviso da definire una sola volta.
// Contenuto verificato contro il sorgente reale (agents.py/exciters.py/resonator.py)
// al punto 2/fase 3, invariato qui -- vedi claude/vst3_native_stato.md.

// agents.py:DESCRIPTOR_KEYS
static const char* const kDescriptorNames[15] = {
    "spectral_centroid", "spectral_spread", "spectral_rolloff", "spectral_flatness",
    "roughness", "harmonic_tension", "inharmonicity", "formant_f1", "formant_f2",
    "formant_f3", "mod_rate", "mod_depth", "attack_time", "decay_time", "pitch"
};

static const char* const kDescriptorSymbols[15] = {
    "target_spectral_centroid", "target_spectral_spread", "target_spectral_rolloff",
    "target_spectral_flatness", "target_roughness", "target_harmonic_tension",
    "target_inharmonicity", "target_formant_f1", "target_formant_f2", "target_formant_f3",
    "target_mod_rate", "target_mod_depth", "target_attack_time", "target_decay_time",
    "target_pitch"
};

// Range HOST per descrittore. 2026-09-25 (deciso con l'utente): lo/hi = UNIONE del range
// globale originale (slider_ranges.json, p1-p99 corpus reale) e dei 10 range per
// eccitatore di kExciterDescriptorRanges sotto -- un parametro VST3 non puo' cambiare
// range a runtime, quindi deve contenerli tutti. def invariato (centro del range globale
// ORIGINALE, gia' dentro l'unione).
// def = centro range (media geometrica per i 7 log-scale = LOG_SCALE_KEYS di gui.py,
// media aritmetica per gli altri 8). Ordine = kDescriptorKeys/kDescriptorNames.
struct DescriptorRange { float lo, hi, def; bool logScale; };
static const DescriptorRange kDescriptorRanges[15] = {
    /* spectral_centroid */ {7.19823f, 22002.5f,       447.21f,  true},
    /* spectral_spread   */ {13.7207f, 9562.43f, 1877.69f,  true},
    /* spectral_rolloff  */ {21.5332f, 22050.0f,  776.00f,  true},
    /* spectral_flatness */ {0.06636f, 0.942809f,    0.248702f, false},
    /* roughness         */ {0.0f, 4.33331f,    0.468598f, false},
    /* harmonic_tension  */ {0.065575f, 1.49999f,    0.827034f, false},
    /* inharmonicity     */ {0.000298f, 0.359183f,    0.066719f, false},
    /* formant_f1        */ {94.3818f, 2304.04f,  456.79f,  true},
    /* formant_f2        */ {199.055f, 3160.67f,  1146.05f,  true},
    /* formant_f3        */ {898.061f, 4092.25f, 2179.63f,  true},
    /* mod_rate          */ {0.0f, 28.1489f,   6.924367f, false},
    /* mod_depth         */ {0.0f, 0.957121f,    0.326592f, false},
    /* attack_time       */ {0.005f, 6.76464f,    3.384819f, false},
    /* decay_time        */ {0.0f, 15.41f,   7.814739f, false},
    /* pitch             */ {25.0f, 4500.0f,       331.66f,  true},
};

// Range per ECCITATORE dei 15 descrittori (2026-09-25, porting di gui.py _agent_range):
// agent_slider_ranges.json (p1-p99 dal dataset di training di ogni eccitatore,
// _archive/scripts_oneoff/compute_agent_ranges.py) + pitch = range ESATTO di freq
// dell'eccitatore (exciters.py PARAM_RANGES). Log-scale: lo >= 1e-3 come gui.py.
// Indice [eccitatore = ordine kExciterEnum][descrittore = ordine kDescriptorNames].
// Usato da GUI (range slider), CC 102-116 e clamp al cambio eccitatore (run()).
// Da RIGENERARE se agent_slider_ranges.json cambia.
struct DescriptorLoHi { float lo, hi; };
static const DescriptorLoHi kExciterDescriptorRanges[10][15] = {
    /* bow        */ {{15.1735f, 4959.11f}, {83.6353f, 4899.16f}, {21.5332f, 5103.37f}, {0.068965f, 0.252515f}, {0.0f, 0.671518f}, {0.290533f, 1.4955f}, {0.000424f, 0.172338f}, {122.873f, 1701.37f}, {564.287f, 2507.91f}, {1299.23f, 3837.25f}, {0.0f, 21.9767f}, {0.0f, 0.440013f}, {0.005f, 1.49162f}, {0.0f, 1.48662f}, {65.0f, 2000.0f}},
    /* blow       */ {{72.8462f, 6181.1f}, {34.0594f, 6627.06f}, {64.5996f, 3811.38f}, {0.06636f, 0.403017f}, {0.0f, 0.89376f}, {0.329634f, 1.49388f}, {0.000376f, 0.173124f}, {97.2393f, 1431.08f}, {490.254f, 2513.46f}, {1396.18f, 4031.41f}, {0.0f, 28.1489f}, {0.0f, 0.532751f}, {0.005f, 1.49162f}, {0.0f, 1.48662f}, {55.0f, 1200.0f}},
    /* strike     */ {{32.0735f, 2375.5f}, {24.1046f, 859.418f}, {43.0664f, 2304.05f}, {0.068808f, 0.344925f}, {0.0f, 0.018136f}, {0.604172f, 1.49947f}, {0.000425f, 0.359183f}, {94.3818f, 1962.83f}, {199.055f, 2336.05f}, {898.061f, 4092.25f}, {0.502507f, 1.08821f}, {0.070146f, 0.564742f}, {0.005f, 0.054887f}, {0.009977f, 0.987755f}, {65.0f, 2400.0f}},
    /* pluck      */ {{7.19823f, 1446.96f}, {13.7207f, 895.184f}, {21.5332f, 1970.29f}, {0.102475f, 0.942809f}, {0.0f, 1.04589f}, {0.173003f, 1.49878f}, {0.000298f, 0.118712f}, {127.37f, 1701.45f}, {580.641f, 1906.05f}, {1159.25f, 3225.45f}, {0.250625f, 1.03696f}, {0.001468f, 0.498279f}, {0.005f, 0.269399f}, {0.238027f, 3.9619f}, {55.0f, 1800.0f}},
    /* shaker     */ {{30.1143f, 21676.6f}, {48.0914f, 9562.43f}, {21.5332f, 22050.0f}, {0.072052f, 0.281715f}, {0.0f, 0.185512f}, {0.361366f, 1.49996f}, {0.00302f, 0.293746f}, {116.865f, 1793.15f}, {366.63f, 2825.48f}, {1099.82f, 4065.56f}, {0.34012f, 13.4664f}, {0.038863f, 0.572084f}, {0.019966f, 1.25217f}, {0.119078f, 1.9809f}, {200.0f, 4000.0f}},
    /* noise      */ {{18.5896f, 22002.5f}, {22.9263f, 8710.07f}, {21.5332f, 22050.0f}, {0.073415f, 0.298151f}, {0.0f, 0.256388f}, {0.410919f, 1.49999f}, {0.001645f, 0.210675f}, {100.522f, 1736.98f}, {295.055f, 2686.31f}, {1151.25f, 3870.17f}, {0.0f, 20.4439f}, {0.0f, 0.55354f}, {0.029943f, 0.992755f}, {0.0f, 0.962812f}, {50.0f, 2000.0f}},
    /* chaos      */ {{170.038f, 7231.67f}, {328.911f, 6798.65f}, {86.1328f, 6524.56f}, {0.122302f, 0.48248f}, {1e-06f, 1.72588f}, {0.130732f, 1.38291f}, {0.0008f, 0.114363f}, {135.612f, 1122.29f}, {690.692f, 2020.97f}, {1576.51f, 3110.61f}, {0.0f, 20.2371f}, {0.0f, 0.326882f}, {0.089807f, 1.49162f}, {0.0f, 1.40181f}, {60.0f, 1200.0f}},
    /* mechanical */ {{95.3405f, 10891.0f}, {285.077f, 8011.49f}, {43.0664f, 21253.3f}, {0.105528f, 0.727672f}, {0.0f, 4.33331f}, {0.065575f, 1.31581f}, {0.001397f, 0.321054f}, {118.464f, 1260.06f}, {719.349f, 2298.27f}, {1632.38f, 3561.06f}, {0.0f, 20.3957f}, {0.0f, 0.558897f}, {0.019966f, 1.48663f}, {0.004989f, 1.47165f}, {80.0f, 4500.0f}},
    /* bird       */ {{470.439f, 7681.21f}, {227.509f, 5987.94f}, {430.664f, 6223.1f}, {0.097759f, 0.346816f}, {0.0f, 0.821763f}, {0.301896f, 1.49739f}, {0.000356f, 0.083332f}, {399.328f, 2304.04f}, {775.978f, 3160.67f}, {1354.37f, 4052.75f}, {0.0f, 19.7323f}, {0.0f, 0.957121f}, {0.005f, 1.4567f}, {0.014966f, 1.48662f}, {400.0f, 3500.0f}},
    /* vocal      */ {{133.504f, 6517.34f}, {130.095f, 4721.32f}, {86.1328f, 6599.93f}, {0.084935f, 0.482963f}, {0.0f, 3.33826f}, {0.094417f, 1.40379f}, {0.000946f, 0.169743f}, {105.458f, 1428.55f}, {381.439f, 2526.9f}, {1156.41f, 3613.63f}, {0.0f, 20.498f}, {0.0f, 0.151745f}, {0.005f, 1.48164f}, {0.009977f, 1.48662f}, {70.0f, 900.0f}},
};

static inline DescriptorLoHi exciterDescriptorRange(int excIdx, uint32_t d)
{
    if (excIdx < 0) excIdx = 0;
    if (excIdx > 9) excIdx = 9;
    return kExciterDescriptorRanges[excIdx][d];
}

// Preset iniziali reali (task separato dal punto 8, deciso con l'utente 2026-09-22/23) --
// vedi claude/vst3_native_stato.md per la provenienza (selector_dataset.csv, righe
// source=='real', 1 a caso per categoria). Usati SOLO dal costruttore del DSP (non
// serve alla UI), ma tenuti qui insieme al resto per coerenza del file spostato.
static const float kInitialPresets[7][15] = {
    // Inharmonic (target_id=10, Inharmonic/Multiphonics-Cl-vo/MulClBb-mulvocl-N-N-mph156_v1.wav)
    {1836.630692f, 1784.253358f, 1851.855469f, 0.151472f, 0.048478f, 0.894632f, 0.020382f,
     829.675294f, 1822.554361f, 2542.223364f, 0.637061f, 0.378542f, 1.297063f, 1.446712f, 263.474299f},
    // blow (target_id=40, blow/Accordion/ordinario/Acc-ord-B4-ff-alt2-N.wav)
    {3527.917910f, 3338.230349f, 3962.109375f, 0.158495f, 0.094060f, 0.967936f, 0.004760f,
     951.978023f, 1501.515991f, 2462.663216f, 0.623749f, 0.113918f, 0.424048f, 5.048526f, 494.678522f},
    // bow (target_id=98, bow/Viola/sul_tasto_tremolo/Va-tasto_trem-G#3-mf-4c-N.wav)
    {1517.098098f, 2562.069687f, 430.664062f, 0.249445f, 0.083638f, 0.437279f, 0.035067f,
     417.607076f, 1386.168668f, 2131.576397f, 0.653986f, 0.162832f, 4.554660f, 2.075283f, 208.174977f},
    // noise (target_id=129, noise/indigenousFlute_scream/D Indigenous Flute Scream 01.wav)
    {1561.878859f, 2416.079210f, 1171.875000f, 0.239257f, 0.116832f, 0.579687f, 0.056015f,
     740.622232f, 1194.232005f, 2365.095714f, 2.018544f, 0.038215f, 2.360000f, 4.345000f, 145.654307f},
    // pluck (target_id=160, pluck/Harp/bisbigliando/Hp-bisb-F6-mf-N-N.wav)
    {2719.793442f, 4161.437981f, 1399.658203f, 0.259480f, 0.017498f, 1.344003f, 0.007872f,
     1062.698129f, 1705.269941f, 2823.811569f, 2.993288f, 0.099051f, 0.588673f, 4.160544f, 1409.193941f},
    // shaker (target_id=193, shaker/conga_slide/00_CongaLoSlide_SP_01.wav)
    {403.699815f, 1305.631530f, 236.865234f, 0.103126f, 0.000182f, 0.845973f, 0.003043f,
     227.272985f, 788.734939f, 2187.031856f, 1.071791f, 0.201993f, 0.254433f, 0.982766f, 214.729880f},
    // strike (target_id=237, strike/Conga/conga_mute/00_CongaHiMute_SP_02.wav)
    {518.162597f, 1305.395984f, 322.998047f, 0.139061f, 0.024755f, 0.627489f, 0.057152f,
     291.425751f, 965.320626f, 1667.401207f, 1.654039f, 0.531990f, 0.009989f, 0.294331f, 264.079697f},
};
static constexpr uint32_t kInitialPresetCount = 7;

static const char* const kExciterParamNames[9] = {
    "Exciter Param 1", "Exciter Param 2", "Exciter Param 3", "Exciter Param 4",
    "Exciter Param 5", "Exciter Param 6", "Exciter Param 7", "Exciter Param 8",
    "Exciter Param 9"
};
static const char* const kExciterParamSymbols[9] = {
    "exciter_param_1", "exciter_param_2", "exciter_param_3", "exciter_param_4",
    "exciter_param_5", "exciter_param_6", "exciter_param_7", "exciter_param_8",
    "exciter_param_9"
};

static const char* const kResonatorParamNames[10] = {
    "Resonator Param 1", "Resonator Param 2", "Resonator Param 3", "Resonator Param 4",
    "Resonator Param 5", "Resonator Param 6", "Resonator Param 7", "Resonator Param 8",
    "Resonator Param 9", "Resonator Param 10"
};
static const char* const kResonatorParamSymbols[10] = {
    "resonator_param_1", "resonator_param_2", "resonator_param_3", "resonator_param_4",
    "resonator_param_5", "resonator_param_6", "resonator_param_7", "resonator_param_8",
    "resonator_param_9", "resonator_param_10"
};

static ParameterEnumerationValue kModeEnum[2] = {
    { 0.0f, "Agent" }, { 1.0f, "Manual" }
};

static ParameterEnumerationValue kExciterEnum[10] = {
    { 0.0f, "bow" }, { 1.0f, "blow" }, { 2.0f, "strike" }, { 3.0f, "pluck" },
    { 4.0f, "shaker" }, { 5.0f, "noise" }, { 6.0f, "chaos" }, { 7.0f, "mechanical" },
    { 8.0f, "bird" }, { 9.0f, "vocal" }
};

static ParameterEnumerationValue kResonatorEnum[7] = {
    { 0.0f, "bar" }, { 1.0f, "plate_rect" }, { 2.0f, "plate_circ" }, { 3.0f, "membrane" },
    { 4.0f, "tube" }, { 5.0f, "soundboard" }, { 6.0f, "chaotic" }
};

// ---------------------------------------------------------------------------------
// Indici parametro (38 totali, superset fisso agent+manual + gain master -- vedi stato doc)

enum Parameters {
    kParameterMode = 0,
    kParameterExciterSelect,
    kParameterResonatorSelect,
    kParameterDescriptorFirst,
    kParameterDescriptorLast = kParameterDescriptorFirst + 14,      // 15 descrittori
    kParameterExciterParamFirst,
    kParameterExciterParamLast = kParameterExciterParamFirst + 8,   // 9 slot
    kParameterResonatorParamFirst,
    kParameterResonatorParamLast = kParameterResonatorParamFirst + 9, // 10 slot
    kParameterGain,  // 2026-09-25: gain master del mix (play_engine.py PlayEngine.gain), IN CODA per non spostare gli indici esistenti
    kParameterAudioIn,    // 2026-09-25: audio-in on/off (audio_input.py), in coda
    kParameterSmoothing,  // 2026-09-25: smoothing audio-in 20-1000 ms (gui.py slider "Smoothing"), in coda
    kParameterMorph,      // 2026-09-25: morph spettrale on/off (SpectralMorph.hpp), in coda
    kParameterAutoPair,   // 2026-09-25: selettore coppia automatico (PairSelector.hpp), solo Mode=Agent, in coda
    kParameterCount
};

// ---------------------------------------------------------------------------------
// State plugin (round 3.2, 2026-09-23) -- SOLO informativo, nessun nuovo parametro host
// (decisione presa con l'utente, vedi claude/vst3_native_stato.md/Round 3). Chiave
// condivisa DSP/UI qui per evitare due literal string che possono disallinearsi (stesso
// principio delle tabelle sopra). hints=0 in initState() -- "by default states are
// completely internal to the plugin and not visible by hosts" (DistrhoDetails.hpp) --
// non deve ne' essere automatizzabile ne' comparire come parametro nell'editor generico.
static const char* const kStateKeyOscActive = "osc_active";

// Round 3.3a (2026-09-23): Scale/tuning -- SOLO state, nessun nuovo parametro host
// (stessa decisione del round 3, vedi vst3_native_stato.md/Round 3 e Round 3.3a). A
// differenza di kStateKeyOscActive queste chiavi SONO scrivibili dalla UI (l'utente le
// imposta dai controlli Scale/round 3.3a) e vengono lette dal DSP in run()/VoiceEngine
// (mai una lookup a stringa nel thread audio o nella coda SPSC: run() copia solo i
// valori gia' convertiti in fScaleActive/fScaleIndex/fA4, vedi PluginMultiPhiMo.cpp).
static const char* const kStateKeyScaleActive = "scale_active";
static const char* const kStateKeyScaleName   = "scale_name";
static const char* const kStateKeyA4          = "a4";

// Round 3.3b (2026-09-24): import .scl custom. scl_import = chiave TRANSITORIA passata a
// UI::requestStateFile (il DSP la ignora; DPF vi invia il path scelto, il DSP non la
// persiste in modo significativo); scl_file = path/etichetta (solo per la GUI);
// scl_data = scala in formato canonico su una riga (TuningQuantizer.hpp::
// serializeCustomScale), FONTE DI VERITA' persistita nel progetto host.
static const char* const kStateKeySclImport = "scl_import";
static const char* const kStateKeySclFile   = "scl_file";
static const char* const kStateKeySclData   = "scl_data";

enum States {
    kStateOscActive = 0,
    kStateScaleActive,
    kStateScaleName,
    kStateA4,
    kStateSclImport,
    kStateSclFile,
    kStateSclData,
    kStateCount
};

// Canale MIDI (0-based, quindi canale "16" in notazione 1-based) riservato al pulsante
// Play (UIMultiPhiMo.cpp): su questo canale la nota NON sovrascrive mai pitch/freq anche
// se Scale e' attivo (decisione esplicita dell'utente, round 3.3a) -- qualunque altro
// canale e' trattato come MIDI reale. Condiviso qui DSP/UI per non avere lo stesso
// numero magico duplicato in due file (PluginMultiPhiMo.cpp::run(),
// UIMultiPhiMo.cpp::fPlayButton->onPress/onRelease).
static constexpr uint8_t kPlayMidiChannel = 15;
