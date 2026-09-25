#ifndef DISTRHO_PLUGIN_INFO_H_INCLUDED
#define DISTRHO_PLUGIN_INFO_H_INCLUDED

// MultiPhiMo -- 37 parametri host + MIDI CC fissi + motore reale (fase 2 chiusa) + GUI
// DGL/NanoVG (punto 8A). Round 1 (2026-09-23): mode/selettori/15 descrittori, validato
// dal vivo, poi abbandonato per GUI generica host e ripristinato lo stesso giorno (DPF
// non onora kParameterIsHidden nel wrapper VST3, verificato in DistrhoPluginVST3.cpp --
// solo l'export LV2 lo fa -- quindi nessun modo di nascondere un sottoinsieme dei 37
// parametri nella GUI generica). Round 2 (2026-09-23): vista Manual coi controlli
// diretti del synth (slot eccitatore/risonatore reali), altezza finestra aumentata da
// 420 a 460 per ospitare i 10 slot del risonatore piu' esteso (chaotic). Round 3.1
// (2026-09-23): pulsante Play (UI::sendNote), nessuna nuova define richiesta (MIDI e RT
// gia' abilitati sopra). Round 3.2 (2026-09-23): state plugin (WANT_STATE+WANT_FULL_STATE)
// per l'indicatore "OSC in ascolto". Round 3.3a (2026-09-23): Scale/tuning built-in
// (3 nuovi state sulla stessa infrastruttura WANT_STATE/WANT_FULL_STATE gia' presente,
// nessuna nuova define qui) -- vedi PluginMultiPhiMo.cpp/UIMultiPhiMo.cpp/
// VoiceEngine.hpp/TuningQuantizer.hpp e claude/vst3_native_stato.md.
#define DISTRHO_PLUGIN_BRAND   "gabbriellini"
#define DISTRHO_PLUGIN_NAME    "MultiPhiMo"
#define DISTRHO_PLUGIN_URI     "https://github.com/gabbriellinimattia/multiPhiMo"

#define DISTRHO_PLUGIN_HAS_UI          1
#define DISTRHO_UI_USE_NANOVG          1
#define DISTRHO_UI_DEFAULT_WIDTH       760
#define DISTRHO_UI_DEFAULT_HEIGHT      500  // 2026-09-25: +40 per la riga Morph
#define DISTRHO_UI_USER_RESIZABLE      1
// Round 3.3b: browser file DGL (NSOpenPanel su macOS) per UI::requestStateFile -- il default
// DPF e' 0 (DistrhoPluginChecks.h); richiede DGL_USE_FILE_BROWSER, gia' attivo di default
// (Makefile.base.mk: USE_FILE_BROWSER ?= true). Il backend VST3 non ha un proprio file
// request (nullptr, "TODO file request"), usa questo fallback.
#define DISTRHO_UI_FILE_BROWSER        1
#define DISTRHO_PLUGIN_IS_RT_SAFE      1
#define DISTRHO_PLUGIN_IS_SYNTH        1
#define DISTRHO_PLUGIN_NUM_INPUTS      2  // 2026-09-25: audio-in stereo (AudioInAnalyzer.hpp, analizza (L+R)/2)
#define DISTRHO_PLUGIN_NUM_OUTPUTS     2  // 2026-09-25: stereo (stesso segnale su L/R). Con 1 uscita Reaper
                                          // lasciava passare il canale 2 della traccia (audio in ingresso
                                          // mescolato al synth)
#define DISTRHO_PLUGIN_WANT_MIDI_INPUT 1
// 2026-09-25: strumento (VST3i) con ingresso audio SOLO sidechain (initAudioPort in
// PluginMultiPhiMo.cpp): come strumento Reaper sommava l'ingresso principale all'uscita; il
// sidechain va sui canali 3-4 della traccia e non esce. Ripiego gia' provato: categoria
// "Fx|Generator" (#define DISTRHO_PLUGIN_VST3_CATEGORIES), funzionante.

// Serve a Plugin::requestParameterValueChange() usato in run() per notificare
// all'host i cambi di parametro avviati da MIDI CC (default DPF: disabilitato).
#define DISTRHO_PLUGIN_WANT_PARAMETER_VALUE_CHANGE_REQUEST 1

// Round 3.2 (2026-09-23): state plugin, per ora solo l'indicatore "OSC in ascolto"
// (nessun nuovo parametro host, vedi PluginMultiPhiMo.cpp/vst3_native_stato.md).
// WANT_FULL_STATE e' necessario anche da solo (non solo con WANT_PROGRAMS, verificato
// in DistrhoPluginVST3.cpp) perche' e' cio' che fa rileggere lo stato vero dal DSP
// (getState()) quando la UI si (ri)connette, invece di restare bloccati al defaultValue
// di initState() -- updateStateValue() (push live) non e' cablato per il backend VST3
// in questa versione di DPF, ma non serve: il bind OSC e' fisso dopo il costruttore.
#define DISTRHO_PLUGIN_WANT_STATE      1
#define DISTRHO_PLUGIN_WANT_FULL_STATE 1

#endif
