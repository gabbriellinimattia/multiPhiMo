/*
 * MultiPhiMo -- sintetizzatore a modelli fisici pilotato da agenti IA.
 * Round 2: superficie parametri host (37, superset fisso agent+manual) + MIDI 1.0 CC
 * fissi -> requestParameterValueChange (automazione DAW). Motore/render: fase 2, non
 * ancora qui -- run() produce ancora silenzio. OSC: round 3.
 */

#include "DistrhoPlugin.hpp"
#include "DistrhoPluginUtils.hpp"  // getResourcePath (dati nel bundle)

#ifndef MPM_VERSION_MAJOR  // definite dal Makefile; valori di ripiego se si compila altrimenti
# define MPM_VERSION_MAJOR 0
# define MPM_VERSION_MINOR 1
# define MPM_VERSION_PATCH 0
#endif

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <thread>

#include "VoiceEngine.hpp"
#include "TuningQuantizer.hpp"
#include "AudioInAnalyzer.hpp"
#include "Resampler.hpp"
#include "PairSelector.hpp"

#include "../../oscpack/ip/IpEndpointName.h"
#include "../../oscpack/ip/UdpSocket.h"
#include "../../oscpack/osc/OscPacketListener.h"
#include "../../oscpack/osc/OscReceivedElements.h"

START_NAMESPACE_DISTRHO

// Tabelle/indici parametro condivisi con la UI (punto 8A) -- spostati in un header
// comune 2026-09-23 per evitare di duplicarli in due translation unit (rischio di
// disallineamento tra DSP e UI). Verificato contro il sorgente reale al punto 2/fase 3,
// contenuto invariato -- vedi ParamLayout.hpp e claude/vst3_native_stato.md.
#include "ParamLayout.hpp"

// ---------------------------------------------------------------------------------
// Listener OSC (round 3): un messaggio per descrittore, indirizzo
// "/multiphimo/target/<nome>" con un solo argomento float -- stesso schema del
// prototipo Python descriptor_input.py. Gira sul proprio thread (UdpListeningReceiveSocket
// e' bloccante), scrive SOLO in un array atomico: nessuna chiamata a host/DPF da qui,
// requestParameterValueChange() viene fatta da run() (thread audio) che drena i valori
// pendenti -- stesso principio del routing MIDI CC sopra, niente lock nel thread audio.

class MultiPhiMoOscListener : public osc::OscPacketListener
{
public:
    MultiPhiMoOscListener(std::atomic<float>* pending, std::atomic<bool>* hasPending,
                           const char* addressPrefix)
        : fPending(pending), fHasPending(hasPending), fPrefix(addressPrefix) {}

protected:
    void ProcessMessage(const osc::ReceivedMessage& m, const IpEndpointName&) override
    {
        try
        {
            const char* const addr = m.AddressPattern();
            const size_t prefixLen = std::strlen(fPrefix);
            if (std::strncmp(addr, fPrefix, prefixLen) != 0)
                return;
            const char* const name = addr + prefixLen;

            for (uint32_t d = 0; d < 15; ++d)
            {
                if (std::strcmp(name, kDescriptorNames[d]) == 0)
                {
                    float value = 0.0f;
                    m.ArgumentStream() >> value >> osc::EndMessage;
                    // TODO fase 2: i descrittori hanno ancora range placeholder 0-1
                    // (vedi initParameter) -- il valore fisico reale (es. Hz per il
                    // centroid) va rimappato quando i range veri saranno definiti da
                    // analyzer/descriptors.py. Per ora memorizzato cosi' com'e'.
                    fPending[kParameterDescriptorFirst + d].store(value, std::memory_order_relaxed);
                    fHasPending[kParameterDescriptorFirst + d].store(true, std::memory_order_release);
                    return;
                }
            }
        }
        catch (const osc::Exception&)
        {
            // pacchetto OSC malformato: ignorato, il thread OSC continua a girare.
        }
    }

private:
    std::atomic<float>* const fPending;
    std::atomic<bool>* const fHasPending;
    const char* const fPrefix;
};

// Mappatura CC MIDI fissa (rimappata 2026-09-25, decisa con l'utente: nessun CC standard
// riassegnato, eccetto CC7 = volume usato per il suo significato standard):
// CC7 gain, CC16 mode, CC17 exciter_select, CC18 resonator_select (general purpose),
// CC20-28 slot eccitatore (undefined), CC80-83 + CC85-90 slot risonatore 1-4/5-10
// (general purpose + undefined, CC84=portamento saltato), CC102-116 descrittori (undefined),
// CC117 Audio In, CC118 Smoothing, CC119 Morph, CC9 Auto pair (undefined, 2026-09-25;
// i toggle: valore >= 64 = on).
// Altri CC ignorati. Tabella gemella: native/dpf/distrho/src/DistrhoPluginVST3.cpp.

static inline float ccToRange(uint8_t cc, float lo, float hi)
{
    return lo + (hi - lo) * (static_cast<float>(cc) / 127.0f);
}

// ---------------------------------------------------------------------------------

class MultiPhiMoPlugin : public Plugin
{
public:
    MultiPhiMoPlugin()
        : Plugin(kParameterCount, 0, kStateCount)
    {
        for (uint32_t i = 0; i < kParameterCount; ++i)
            fParams[i] = 0.5f;
        fParams[kParameterMode]             = 0.0f; // default: agent
        fParams[kParameterExciterSelect]    = 0.0f; // default: bow
        fParams[kParameterResonatorSelect]  = 0.0f; // default: bar
        fParams[kParameterGain]             = phimo::kMixGain; // 0.4, default PlayEngine.gain
        fParams[kParameterAudioIn]          = 0.0f;   // audio-in spento
        fParams[kParameterSmoothing]        = 200.0f; // audio_input.py smoothing_ms default
        fParams[kParameterMorph]            = 0.0f;   // morph spettrale spento
        fParams[kParameterAutoPair]         = 0.0f;   // auto pair spento

        // Round 3.3a (2026-09-23): Scale/tuning -- default Scale=off, edo12, A4=440 (stessi
        // default di gui.py). Sovrascritti da setState() se l'host ripristina un progetto
        // salvato (vedi initState/getState/setState sotto).
        fScaleActive.store(false, std::memory_order_relaxed);
        fScaleIndex.store(phimo::kScaleEdo12, std::memory_order_relaxed);
        fA4.store(440.0f, std::memory_order_relaxed);
        // Preset iniziale a caso tra le ~7 righe reali del corpus (vedi kInitialPresets
        // sopra) invece del centro-range fisso -- task separato dal punto 8, deciso con
        // l'utente 2026-09-22/23. Il "reset to default" dell'host resta il centro-range
        // (parameter.ranges.def in initParameter, invariato): solo il valore INIZIALE
        // all'istanziazione del plugin cambia.
        {
            std::mt19937 presetRng(std::random_device{}());
            std::uniform_int_distribution<uint32_t> presetDist(0, kInitialPresetCount - 1);
            const uint32_t presetIdx = presetDist(presetRng);
            for (uint32_t d = 0; d < 15; ++d)
                fParams[kParameterDescriptorFirst + d] = kInitialPresets[presetIdx][d];
        }
        for (uint32_t d = 0; d < 15; ++d)
        {
            fSelTarget[d] = fParams[kParameterDescriptorFirst + d];
            fReqValid[d] = false;
        }

        fOscListener = nullptr;
        fOscSocket = nullptr;
        startOsc();

        fAudioIn.setSampleRate(getSampleRate());
        setupOutputResampler(getSampleRate());
        fAudioIn.start();  // thread d'analisi, inattivo finche' Audio In e' spento

        // Percorso dati (pesi MDN, corpus KNN, corpus del selettore) -- 2026-09-25: DENTRO il
        // bundle, <bundle>/Contents/Resources/data (copiati da `make vst3`, vedi Makefile),
        // trovati con getResourcePath di DPF. Nessun ripiego sulla cartella di sviluppo
        // (deciso con l'utente: il .vst3 deve essere autosufficiente). MULTIPHIMO_DATA_DIR
        // resta solo come override esplicito per il debug.
        std::string dataDir;
        if (const char* env = std::getenv("MULTIPHIMO_DATA_DIR"))
            dataDir = env;
        else if (const char* bundle = getBundlePath())
            if (const char* res = getResourcePath(bundle))
                dataDir = std::string(res) + "/data";
        std::string engineErr;
        if (!dataDir.empty() &&
            fEngine.loadAll(dataDir + "/weights", dataDir + "/knn_corpus.bin", &engineErr))
        {
            fVoiceEngine.reset(new phimo::VoiceEngine(fEngine));
            fVoiceEngine->start();
        }
        // Auto pair (2026-09-25): corpus del selettore nella stessa cartella dati; se manca
        // il parametro resta inerte (coppia invariata), nessun effetto sul resto del motore.
        if (!dataDir.empty())
        {
            const char* excNames[10];
            const char* resNames[7];
            for (int i = 0; i < 10; ++i) excNames[i] = static_cast<const char*>(kExciterEnum[i].label);
            for (int i = 0; i < 7; ++i) resNames[i] = static_cast<const char*>(kResonatorEnum[i].label);
            fAutoPair.start(dataDir + "/selector_corpus.bin", excNames, 10, resNames, 7);
        }
        // se il caricamento fallisce, fVoiceEngine resta nullptr: run() produce silenzio
        // invece di crashare (stesso principio di degrado morbido gia' usato per l'OSC
        // quando il bind della porta fallisce).
    }

    ~MultiPhiMoPlugin() override
    {
        fAudioIn.stop();
        fAutoPair.stop();
        fVoiceEngine.reset();  // ferma e joina il worker prima di distruggere fEngine
        stopOsc();
    }

protected:
    // -------------------------------------------------------------------
    // Informazioni plugin

    const char* getLabel() const override { return "MultiPhiMo"; }
    const char* getDescription() const override
    {
        return "IA-driven physical modeling synth (round 2: parametri/MIDI, motore in fase 2).";
    }
    const char* getMaker() const override { return "gabbriellini"; }
    const char* getHomePage() const override { return DISTRHO_PLUGIN_URI; }
    const char* getLicense() const override { return "ISC"; }
    // Versione: unica fonte = Makefile (VERSION_MAJOR/MINOR/PATCH -> -DMPM_VERSION_*).
    uint32_t getVersion() const override { return d_version(MPM_VERSION_MAJOR, MPM_VERSION_MINOR, MPM_VERSION_PATCH); }
    int64_t getUniqueId() const override { return d_cconst('M', 'P', 'h', 'M'); }

    // -------------------------------------------------------------------
    // Parametri

    // 2026-09-25: i 2 ingressi audio sono SIDECHAIN (bus ausiliario VST3, canali 3-4 della
    // traccia in Reaper), non ingresso principale: l'audio analizzato da Audio In non si somma
    // mai all'uscita dello strumento. Le uscite restano principali (L/R).
    void initAudioPort(bool input, uint32_t index, AudioPort& port) override
    {
        if (input)
            port.hints |= kAudioPortIsSidechain;
        Plugin::initAudioPort(input, index, port);
        if (input)
        {
            port.name   = index == 0 ? "Sidechain L" : "Sidechain R";
            port.symbol = index == 0 ? "sidechain_l" : "sidechain_r";
        }
    }

    void initParameter(uint32_t index, Parameter& parameter) override
    {
        parameter.hints = kParameterIsAutomatable;

        if (index == kParameterMode)
        {
            parameter.hints |= kParameterIsInteger | kParameterIsBoolean;
            parameter.name = "Mode";
            parameter.symbol = "mode";
            parameter.ranges.def = 0.0f;
            parameter.ranges.min = 0.0f;
            parameter.ranges.max = 1.0f;
            parameter.enumValues.count = 2;
            parameter.enumValues.restrictedMode = true;
            parameter.enumValues.values = kModeEnum;
            parameter.enumValues.deleteLater = false;
            return;
        }

        if (index == kParameterExciterSelect)
        {
            parameter.hints |= kParameterIsInteger;
            parameter.name = "Exciter";
            parameter.symbol = "exciter_select";
            parameter.ranges.def = 0.0f;
            parameter.ranges.min = 0.0f;
            parameter.ranges.max = 9.0f;
            parameter.enumValues.count = 10;
            parameter.enumValues.restrictedMode = true;
            parameter.enumValues.values = kExciterEnum;
            parameter.enumValues.deleteLater = false;
            return;
        }

        if (index == kParameterResonatorSelect)
        {
            parameter.hints |= kParameterIsInteger;
            parameter.name = "Resonator";
            parameter.symbol = "resonator_select";
            parameter.ranges.def = 0.0f;
            parameter.ranges.min = 0.0f;
            parameter.ranges.max = 6.0f;
            parameter.enumValues.count = 7;
            parameter.enumValues.restrictedMode = true;
            parameter.enumValues.values = kResonatorEnum;
            parameter.enumValues.deleteLater = false;
            return;
        }

        if (index >= kParameterDescriptorFirst && index <= kParameterDescriptorLast)
        {
            const uint32_t d = index - kParameterDescriptorFirst;
            parameter.name = kDescriptorNames[d];
            parameter.symbol = kDescriptorSymbols[d];
            parameter.ranges.def = kDescriptorRanges[d].def;
            parameter.ranges.min = kDescriptorRanges[d].lo;
            parameter.ranges.max = kDescriptorRanges[d].hi;
            if (kDescriptorRanges[d].logScale)
                parameter.hints |= kParameterIsLogarithmic;
            return;
        }

        if (index >= kParameterExciterParamFirst && index <= kParameterExciterParamLast)
        {
            const uint32_t s = index - kParameterExciterParamFirst;
            parameter.name = kExciterParamNames[s];
            parameter.symbol = kExciterParamSymbols[s];
            parameter.ranges.def = 0.5f;
            parameter.ranges.min = 0.0f;
            parameter.ranges.max = 1.0f;
            // TODO fase 2: reinterpretato secondo PARAM_RANGES[eccitatore attivo]
            // (exciters.py), nome vero mostrato in GUI quando esiste.
            return;
        }

        if (index >= kParameterResonatorParamFirst && index <= kParameterResonatorParamLast)
        {
            const uint32_t s = index - kParameterResonatorParamFirst;
            parameter.name = kResonatorParamNames[s];
            parameter.symbol = kResonatorParamSymbols[s];
            parameter.ranges.def = 0.5f;
            parameter.ranges.min = 0.0f;
            parameter.ranges.max = 1.0f;
            // TODO fase 2: reinterpretato secondo PARAM_RANGES[risonatore attivo]
            // (resonator.py), nome vero mostrato in GUI quando esiste.
            return;
        }

        if (index == kParameterGain)
        {
            // 2026-09-25: porting di gui.py slider "Gain" (0-1.5, default PlayEngine.gain=0.4),
            // applicato al mix prima del limiter (VoiceEngine::mixBlock).
            parameter.name = "Gain";
            parameter.symbol = "gain";
            parameter.ranges.def = phimo::kMixGain;
            parameter.ranges.min = 0.0f;
            parameter.ranges.max = 1.5f;
            return;
        }

        if (index == kParameterAudioIn)
        {
            // 2026-09-25: porting di audio_input.py -- descrittori target analizzati
            // dall'ingresso audio della traccia (AudioInAnalyzer.hpp).
            parameter.hints |= kParameterIsInteger | kParameterIsBoolean;
            parameter.name = "Audio In";
            parameter.symbol = "audio_in";
            parameter.ranges.def = 0.0f;
            parameter.ranges.min = 0.0f;
            parameter.ranges.max = 1.0f;
            return;
        }

        if (index == kParameterSmoothing)
        {
            parameter.name = "Smoothing";
            parameter.symbol = "smoothing_ms";
            parameter.unit = "ms";
            parameter.ranges.def = 200.0f;
            parameter.ranges.min = 20.0f;
            parameter.ranges.max = 1000.0f;
            return;
        }

        if (index == kParameterMorph)
        {
            // 2026-09-25: morph spettrale continuo (SpectralMorph.hpp), voce mono.
            parameter.hints |= kParameterIsInteger | kParameterIsBoolean;
            parameter.name = "Morph";
            parameter.symbol = "morph";
            parameter.ranges.def = 0.0f;
            parameter.ranges.min = 0.0f;
            parameter.ranges.max = 1.0f;
            return;
        }

        if (index == kParameterAutoPair)
        {
            // 2026-09-25: porting di pair_selector.py/param_candidate._maybe_select_pair --
            // sceglie Exciter+Resonator dai descrittori target, solo in Mode=Agent.
            parameter.hints |= kParameterIsInteger | kParameterIsBoolean;
            parameter.name = "Auto pair";
            parameter.symbol = "auto_pair";
            parameter.ranges.def = 0.0f;
            parameter.ranges.min = 0.0f;
            parameter.ranges.max = 1.0f;
            return;
        }
    }

    float getParameterValue(uint32_t index) const override
    {
        return fParams[index];
    }

    void setParameterValue(uint32_t index, float value) override
    {
        fParams[index] = value;
        // Auto pair: un valore descrittore che arriva dall'host (GUI/automazione) aggiorna il
        // target del selettore, TRANNE l'eco di un valore chiesto da noi stessi
        // (clamp/audio-in/OSC gia' clippati): cosi' il selettore vede il valore grezzo.
        if (index >= kParameterDescriptorFirst && index <= kParameterDescriptorLast)
        {
            const uint32_t d = index - kParameterDescriptorFirst;
            const float q = fReqDesc[d];
            if (!fReqValid[d] || std::fabs(value - q) > 1e-4f * std::max(1.0f, std::fabs(q)))
                fSelTarget[d] = value;
        }
    }

    // Aggiorna lo stato interno e chiede all'host di registrarlo come automazione
    // (best-effort: alcuni host non supportano canRequestParameterValueChanges(),
    // in quel caso lo stato interno resta comunque aggiornato).
    void applyParameterChangeFromMidi(uint32_t index, float value)
    {
        fParams[index] = value;
        if (index >= kParameterDescriptorFirst && index <= kParameterDescriptorLast)
        {
            fReqDesc[index - kParameterDescriptorFirst] = value;
            fReqValid[index - kParameterDescriptorFirst] = true;
        }
        requestParameterValueChange(index, value);
    }

    int currentExciterIndex() const
    {
        const int i = static_cast<int>(std::lround(fParams[kParameterExciterSelect]));
        return std::min(std::max(i, 0), 9);
    }

    // 2026-09-25 (deciso con l'utente, "clamp al range"): al cambio di eccitatore i 15
    // descrittori vengono portati dentro il range dell'eccitatore nuovo
    // (kExciterDescriptorRanges, ParamLayout.hpp) e scritti all'host; i valori gia' dentro
    // restano invariati. Chiamata SOLO da run() (thread audio): nessuna allocazione.
    // fLastExciterIdx=-1 all'avvio -> anche preset iniziale/progetto ripristinato vengono
    // portati nel range al primo blocco.
    void syncDescriptorsToExciter()
    {
        const int exc = currentExciterIndex();
        if (exc == fLastExciterIdx)
            return;
        fLastExciterIdx = exc;
        for (uint32_t d = 0; d < 15; ++d)
        {
            const DescriptorLoHi r = exciterDescriptorRange(exc, d);
            const uint32_t idx = kParameterDescriptorFirst + d;
            const float v = fParams[idx];
            const float c = std::min(std::max(v, r.lo), r.hi);
            if (c != v)
                applyParameterChangeFromMidi(idx, c);
        }
    }

    // -------------------------------------------------------------------
    // State (round 3.2, 2026-09-23) -- SOLO l'indicatore "OSC in ascolto" per ora,
    // Scale/tuning arriva al round 3.3. Verificato nel sorgente DPF reale
    // (distrho/src/DistrhoPluginVST3.cpp) che Plugin::updateStateValue() -- il push
    // "live" DSP->UI documentato in DistrhoPlugin.hpp -- NON e' cablato per il backend
    // VST3 in questa versione di DPF (assente da fData->updateStateValueCallbackFunc,
    // a differenza di LV2/CLAP/AU/Carla). Non e' un problema per questo indicatore: il
    // bind OSC avviene UNA SOLA VOLTA nel costruttore, mai piu' dopo, e VST3 risincronizza
    // gli state correnti (chiamando la nostra getState()) ogni volta che la UI si connette
    // (ctrl2view_connect -> sendStateSetToUI, MA solo se DISTRHO_PLUGIN_WANT_FULL_STATE e'
    // definito -- altrimenti resta bloccato al defaultValue di initState()) -- quindi
    // riaprire la finestra del plugin basta a leggere il valore vero. DISTRHO_PLUGIN_
    // WANT_FULL_STATE aggiunto in DistrhoPluginInfo.h per questo. Nessun nuovo parametro
    // host (deciso con l'utente, vedi claude/vst3_native_stato.md/Round 3).
    void initState(uint32_t index, State& state) override
    {
        if (index == kStateOscActive)
        {
            state.hints = 0; // interno, non salvato/automatizzabile -- vedi nota sopra
            state.key = kStateKeyOscActive;
            state.defaultValue = "0";
            state.label = "OSC attivo";
            state.description = "1 se il bind della porta OSC (9000) e' riuscito al costruttore, altrimenti 0.";
        }
        else if (index == kStateScaleActive)
        {
            // Round 3.3a (2026-09-23): a differenza di osc_active queste 3 chiavi SONO
            // scritte dalla UI (controlli Scale) e vanno persistite/ripristinate col
            // progetto host -- hints=0 basta comunque (il salvataggio/ripristino dello state
            // chunk DPF non dipende dagli hint, verificato in DistrhoPluginVST3.cpp::getState/
            // setState: tutte le chiavi registrate qui vengono salvate a prescindere; gli
            // hint riguardano solo l'esposizione come "parametro stringa" lato host, che qui
            // non vogliamo, vedi Round 3).
            state.hints = 0;
            state.key = kStateKeyScaleActive;
            state.defaultValue = "0";
            state.label = "Scale attivo";
            state.description = "1 se la quantizzazione a scala musicale (round 3.3a) e' attiva, altrimenti 0.";
        }
        else if (index == kStateScaleName)
        {
            state.hints = 0;
            state.key = kStateKeyScaleName;
            state.defaultValue = "edo12";
            state.label = "Scala";
            state.description = "Nome della scala built-in attiva (edo12/edo24/edo31/perfect/harmonic).";
        }
        else if (index == kStateA4)
        {
            state.hints = 0;
            state.key = kStateKeyA4;
            state.defaultValue = "440";
            state.label = "A4";
            state.description = "Frequenza di riferimento (Hz) usata come ancora della scala attiva.";
        }
        else if (index == kStateSclImport)
        {
            // Round 3.3b: chiave transitoria per UI::requestStateFile, ignorata dal DSP.
            state.hints = 0;
            state.key = kStateKeySclImport;
            state.defaultValue = "";
            state.label = "Import .scl";
            state.description = "Path del file .scl scelto dall'utente (transitorio, elaborato dalla UI).";
        }
        else if (index == kStateSclFile)
        {
            state.hints = 0;
            state.key = kStateKeySclFile;
            state.defaultValue = "";
            state.label = "File .scl";
            state.description = "Path del .scl importato (solo etichetta GUI).";
        }
        else if (index == kStateSclData)
        {
            state.hints = 0;
            state.key = kStateKeySclData;
            state.defaultValue = "";
            state.label = "Scala custom";
            state.description = "Scala custom in formato canonico: n periodo grado0 grado1 ... (persistita col progetto).";
        }
    }

    String getState(const char* key) const override
    {
        if (std::strcmp(key, kStateKeyOscActive) == 0)
            return String(fOscSocket != nullptr ? "1" : "0");
        if (std::strcmp(key, kStateKeyScaleActive) == 0)
            return String(fScaleActive.load(std::memory_order_relaxed) ? "1" : "0");
        if (std::strcmp(key, kStateKeyScaleName) == 0)
            return String(phimo::scaleIndexToName(fScaleIndex.load(std::memory_order_relaxed)));
        if (std::strcmp(key, kStateKeyA4) == 0)
            return String(fA4.load(std::memory_order_relaxed));
        if (std::strcmp(key, kStateKeySclFile) == 0)
            return fSclFile;
        if (std::strcmp(key, kStateKeySclData) == 0)
            return fSclData;
        return String();
    }

    // Round 3.3a: le 3 chiavi Scale/tuning sono scritte dalla UI (vedi UIMultiPhiMo.cpp,
    // toggle/selettore/slider A4) e qui semplicemente applicate ai membri corrispondenti --
    // niente di piu' pesante di un parsing di stringa, chiamato dal thread messaggi
    // dell'host, mai dal thread audio (osc_active resta non scrivibile, invariato).
    void setState(const char* key, const char* value) override
    {
        if (std::strcmp(key, kStateKeyScaleActive) == 0)
            fScaleActive.store(value != nullptr && value[0] == '1', std::memory_order_relaxed);
        else if (std::strcmp(key, kStateKeyScaleName) == 0)
            fScaleIndex.store(phimo::scaleNameToIndex(value), std::memory_order_relaxed);
        else if (std::strcmp(key, kStateKeyA4) == 0)
            fA4.store(static_cast<float>(std::atof(value)), std::memory_order_relaxed);
        else if (std::strcmp(key, kStateKeySclFile) == 0)
            fSclFile = (value != nullptr) ? value : "";
        else if (std::strcmp(key, kStateKeySclData) == 0)
        {
            // Round 3.3b: thread messaggi host (mai run()). Parsing/validazione e
            // pubblicazione dello snapshot immutabile letto poi dal worker via ScaleContext.
            fSclData = (value != nullptr) ? value : "";
            phimo::CustomScale cs;
            if (value != nullptr && value[0] != '\0' && phimo::deserializeCustomScale(value, cs))
                fCustomScales.publish(cs);
            else
                fCustomScales.clear();
        }
        // kStateKeySclImport: ignorato (gestito dalla UI).
    }

    // -------------------------------------------------------------------
    // Audio/MIDI processing

    // Costruisce e accoda un trigger al worker con i parametri CORRENTI (target descrittori in
    // Agent, slot grezzi in Manual) -- estratto dal note-on il 2026-09-25 per riusarlo anche
    // dai render periodici del morph spettrale. Solo thread audio (run()).
    void sendTrigger(uint8_t note, uint8_t vel, uint8_t channel, bool morph)
    {
                        const int selExc = static_cast<int>(std::lround(fParams[kParameterExciterSelect]));
                        const int selRes = static_cast<int>(std::lround(fParams[kParameterResonatorSelect]));
                        const char* excName = kExciterEnum[std::min(std::max(selExc, 0), 9)].label;
                        const char* resName = kResonatorEnum[std::min(std::max(selRes, 0), 6)].label;
                        // note: il numero di nota MIDI e' usato SOLO come note_id per
                        // l'hold-to-sustain/voice-stealing (deciso con l'utente
                        // 2026-09-22) -- l'altezza del suono resta governata dal
                        // descrittore target 'pitch' (Mode=Agent) o dal parametro 'freq'
                        // dell'eccitatore (Mode=Manual), mai dalla nota premuta -- ECCETTO
                        // quando Scale e' attivo E il canale non e' quello riservato al
                        // pulsante Play (round 3.3a, deciso con l'utente 2026-09-23): in tal
                        // caso la nota sovrascrive pitch/freq con la sua altezza cromatica
                        // quantizzata alla scala attiva. Qui solo una copia grezza degli
                        // ingredienti (nessuna lookup/quantizzazione: quella resta nel worker
                        // thread, vedi VoiceEngine.hpp/TuningQuantizer.hpp).
                        phimo::ScaleContext scaleCtx;
                        scaleCtx.active = fScaleActive.load(std::memory_order_relaxed);
                        scaleCtx.scaleIndex = fScaleIndex.load(std::memory_order_relaxed);
                        scaleCtx.a4 = fA4.load(std::memory_order_relaxed);
                        scaleCtx.custom = fCustomScales.current();  // solo un load atomico di puntatore
                        scaleCtx.overrideNoteMidi = (scaleCtx.active && channel != kPlayMidiChannel)
                                                        ? static_cast<int32_t>(note) : -1;

                        if (fParams[kParameterMode] >= 0.5f)
                        {
                            // Mode=Manual (punto 8A round 2, 2026-09-23, deciso con
                            // l'utente): bypassa gli agenti MDN/KNN, controllo diretto
                            // della generazione. Copia SOLO i valori grezzi 0-1 dei 19
                            // slot cosi' come li tiene l'host -- la denormalizzazione
                            // (richiede una lookup su agentSpecs()) resta nel worker
                            // thread (VoiceEngine.hpp), mai qui: run() e' il thread
                            // audio, nessuna lookup su std::map in piu' rispetto a
                            // quelle gia' esistenti.
                            float excRaw[9];
                            for (uint32_t i = 0; i < 9; ++i)
                                excRaw[i] = fParams[kParameterExciterParamFirst + i];
                            float resRaw[10];
                            for (uint32_t i = 0; i < 10; ++i)
                                resRaw[i] = fParams[kParameterResonatorParamFirst + i];
                            fVoiceEngine->triggerNoteOnManual(excName, resName, static_cast<int32_t>(note),
                                                               static_cast<float>(vel), excRaw, resRaw, scaleCtx, morph);
                        }
                        else
                        {
                            float target[15];
                            for (uint32_t d = 0; d < 15; ++d)
                                target[d] = fParams[kParameterDescriptorFirst + d];
                            fVoiceEngine->triggerNoteOn(excName, resName, static_cast<int32_t>(note),
                                                         static_cast<float>(vel), target, scaleCtx, morph);
                        }
    }

    void sampleRateChanged(double newSampleRate) override
    {
        fAudioIn.setSampleRate(newSampleRate);
        setupOutputResampler(newSampleRate);
    }

    // 2026-09-25: il motore rende sempre a 44.1 kHz (kResonatorSR); a un rate host diverso
    // l'uscita passa per un resampler in streaming (Resampler.hpp). Alloca: chiamata solo
    // da costruttore/sampleRateChanged (processing fermo), mai da run().
    void setupOutputResampler(double hostSr)
    {
        fUseOutRs = hostSr > 1000.0 && std::fabs(hostSr - (double)phimo::kResonatorSR) > 0.5;
        if (!fUseOutRs)
            return;
        fOutRs.init((double)phimo::kResonatorSR, hostSr, kOutRsChunk);
        fRsIn.assign(fOutRs.maxInput(), 0.0f);
    }

    void run(const float** inputs, float** outputs, uint32_t frames,
             const MidiEvent* midiEvents, uint32_t midiEventCount) override
    {
        // Valori arrivati via OSC dal thread di ascolto UDP (round 3): drenati qui,
        // mai nel thread OSC stesso -- stesso principio del routing MIDI sotto.
        for (uint32_t i = 0; i < kParameterCount; ++i)
        {
            if (fOscHasPending[i].exchange(false, std::memory_order_acq_rel))
            {
                float v = fOscPending[i].load(std::memory_order_relaxed);
                if (i >= kParameterDescriptorFirst && i <= kParameterDescriptorLast)
                {
                    fSelTarget[i - kParameterDescriptorFirst] = v;  // grezzo, per Auto pair
                    // 2026-09-25: clip al range dell'eccitatore attivo, come gui.py _poll_osc
                    const DescriptorLoHi r = exciterDescriptorRange(currentExciterIndex(), i - kParameterDescriptorFirst);
                    v = std::min(std::max(v, r.lo), r.hi);
                }
                applyParameterChangeFromMidi(i, v);
            }
        }

        syncDescriptorsToExciter();  // cambio eccitatore da host/GUI/automazione

        // Audio-in (2026-09-25): campioni in ingresso -> ring SPSC del thread d'analisi;
        // valori smussati pronti -> descrittori host, clip al range dell'eccitatore attivo
        // (come gui.py _poll_osc fa per audio-in/OSC).
        {
            const bool audioInOn = fParams[kParameterAudioIn] >= 0.5f;
            fAudioIn.setEnabled(audioInOn);
            fAudioIn.setSmoothingMs(fParams[kParameterSmoothing]);
            if (audioInOn)
            {
                fAudioIn.pushAudioStereo(inputs[0], inputs[1], frames);
                for (uint32_t d = 0; d < 15; ++d)
                {
                    float v;
                    if (!fAudioIn.takePending(static_cast<int>(d), v))
                        continue;
                    fSelTarget[d] = v;  // grezzo (non clippato al range dell'eccitatore), per Auto pair
                    const DescriptorLoHi r = exciterDescriptorRange(currentExciterIndex(), d);
                    applyParameterChangeFromMidi(kParameterDescriptorFirst + d, std::min(std::max(v, r.lo), r.hi));
                }
            }
        }

        for (uint32_t e = 0; e < midiEventCount; ++e)
        {
            const MidiEvent& ev = midiEvents[e];
            if (ev.size < 3)
                continue;

            const uint8_t status  = static_cast<uint8_t>(ev.data[0] & 0xF0);
            const uint8_t channel = static_cast<uint8_t>(ev.data[0] & 0x0F);

            if (status == 0x90 || status == 0x80) // note-on / note-off
            {
                const uint8_t note = ev.data[1];
                const uint8_t vel  = ev.data[2];
                if (fVoiceEngine)
                {
                    if (status == 0x90 && vel > 0)
                    {
                        if (fParams[kParameterMorph] >= 0.5f)
                        {
                            // Morph spettrale (2026-09-25): voce mono continua, il render
                            // diventa un'istantanea spettrale (SpectralMorph.hpp).
                            fVoiceEngine->morphNoteOn(static_cast<int32_t>(note), static_cast<float>(vel));
                            fMorphVel = vel;
                            fMorphChannel = channel;
                            sendTrigger(note, vel, channel, true);
                        }
                        else
                        {
                            sendTrigger(note, vel, channel, false);
                        }
                    }
                    else
                    {
                        // note-off esplicito, o note-on con velocity 0 (convenzione MIDI
                        // standard, equivalente a note-off).
                        fVoiceEngine->triggerNoteOff(static_cast<int32_t>(note));
                        fVoiceEngine->morphNoteOff(static_cast<int32_t>(note));
                    }
                }
                continue;
            }

            if (status != 0xB0) // solo Control Change, canale ignorato
                continue;

            const uint8_t cc  = ev.data[1];
            const uint8_t val = ev.data[2];

            int paramIndex = -1;
            float mapped = 0.0f;

            if (cc >= 102 && cc <= 116)
            {
                const uint8_t d = static_cast<uint8_t>(cc - 102);
                paramIndex = static_cast<int>(kParameterDescriptorFirst) + d;
                // 2026-09-25 (deciso con l'utente): CC 0-127 copre il range dell'ECCITATORE
                // attivo (come lo slider GUI), non il range host (unione).
                const DescriptorLoHi r = exciterDescriptorRange(currentExciterIndex(), d);
                mapped = ccToRange(val, r.lo, r.hi);
                fSelTarget[d] = mapped;  // Auto pair
            }
            else if (cc == 7)
            {
                paramIndex = kParameterGain;
                mapped = ccToRange(val, 0.0f, 1.5f);
            }
            else if (cc == 16)
            {
                paramIndex = kParameterMode;
                mapped = ccToRange(val, 0.0f, 1.0f);
            }
            else if (cc == 17)
            {
                paramIndex = kParameterExciterSelect;
                mapped = ccToRange(val, 0.0f, 9.0f);
            }
            else if (cc == 18)
            {
                paramIndex = kParameterResonatorSelect;
                mapped = ccToRange(val, 0.0f, 6.0f);
            }
            else if (cc >= 20 && cc <= 28)
            {
                paramIndex = static_cast<int>(kParameterExciterParamFirst) + (cc - 20);
                mapped = ccToRange(val, 0.0f, 1.0f);
            }
            else if (cc >= 80 && cc <= 83)
            {
                paramIndex = static_cast<int>(kParameterResonatorParamFirst) + (cc - 80);  // slot 1-4
                mapped = ccToRange(val, 0.0f, 1.0f);
            }
            else if (cc >= 85 && cc <= 90)
            {
                paramIndex = static_cast<int>(kParameterResonatorParamFirst) + 4 + (cc - 85);  // slot 5-10
                mapped = ccToRange(val, 0.0f, 1.0f);
            }
            else if (cc == 117 || cc == 119 || cc == 9)
            {
                paramIndex = (cc == 117) ? kParameterAudioIn : (cc == 119) ? kParameterMorph : kParameterAutoPair;
                mapped = (val >= 64) ? 1.0f : 0.0f;
            }
            else if (cc == 118)
            {
                paramIndex = kParameterSmoothing;
                mapped = ccToRange(val, 20.0f, 1000.0f);
            }

            if (paramIndex >= 0)
            {
                applyParameterChangeFromMidi(static_cast<uint32_t>(paramIndex), mapped);
                if (paramIndex == static_cast<int>(kParameterExciterSelect))
                    syncDescriptorsToExciter();  // cambio eccitatore via CC17
            }
        }

        // Auto pair (2026-09-25, porting di param_candidate._maybe_select_pair): il thread
        // del selettore propone una coppia ogni 0.3 s sul target corrente; qui la si applica
        // ai parametri host Exciter/Resonator (come select_exciter/select_resonator) se e'
        // diversa da quella attiva e sono passati almeno 2 s dall'ultimo cambio (le
        // proposte arrivate prima vengono scartate, come in Python). Solo Mode=Agent.
        {
            const bool on = fParams[kParameterAutoPair] >= 0.5f && fParams[kParameterMode] < 0.5f;
            fAutoPair.setEnabled(on);
            fAutoPairHold += frames;
            int e, r;
            if (!on)
            {
                fAutoPair.takePending(e, r);  // scarta proposte vecchie
            }
            else
            {
                float t[15];
                for (uint32_t d = 0; d < 15; ++d)
                    t[d] = fSelTarget[d];  // grezzo: col target clippato al range dell'eccitatore
                                           // attivo il selettore restava bloccato su quell'eccitatore
                fAutoPair.setTarget(t);
                const int curRes = std::min(std::max(static_cast<int>(std::lround(fParams[kParameterResonatorSelect])), 0), 6);
                if (fAutoPair.takePending(e, r) && (e != currentExciterIndex() || r != curRes) &&
                    fAutoPairHold >= phimo::kAutoPairMinHoldSeconds * getSampleRate())
                {
                    applyParameterChangeFromMidi(kParameterExciterSelect, static_cast<float>(e));
                    applyParameterChangeFromMidi(kParameterResonatorSelect, static_cast<float>(r));
                    syncDescriptorsToExciter();
                    fAutoPairHold = 0.0;
                }
            }
        }

        // Morph spettrale (2026-09-25): spento -> rilascia la voce; tenuto -> un nuovo render
        // con i parametri correnti ogni kMorphRequestSeconds (uno alla volta).
        if (fVoiceEngine)
        {
            if (fParams[kParameterMorph] < 0.5f)
                fVoiceEngine->morphRelease();
            else if (fVoiceEngine->morphDue(fUseOutRs
                         ? (uint32_t)std::lround(frames * (double)phimo::kResonatorSR / getSampleRate())
                         : frames))
                sendTrigger(static_cast<uint8_t>(fVoiceEngine->morphNoteId() & 0x7F), fMorphVel, fMorphChannel, true);
        }

        float* const out = outputs[0];
        if (fVoiceEngine && fUseOutRs)
        {
            // pull: il mixer (44.1 kHz) genera esattamente i campioni interni richiesti
            for (uint32_t done = 0; done < frames;)
            {
                const uint32_t n = std::min<uint32_t>(frames - done, kOutRsChunk);
                const uint32_t need = fOutRs.inputNeeded(n);
                if (need > 0)
                {
                    fVoiceEngine->mixBlock(fRsIn.data(), need, fParams[kParameterGain], fParams[kParameterSmoothing]);
                    fOutRs.push(fRsIn.data(), need);
                }
                fOutRs.process(out + done, n);
                done += n;
            }
            for (uint32_t i = 0; i < frames; ++i)
                out[i] = std::min(1.0f, std::max(-1.0f, out[i]));  // backstop dopo l'interpolazione
        }
        else if (fVoiceEngine)
            fVoiceEngine->mixBlock(out, frames, fParams[kParameterGain], fParams[kParameterSmoothing]);
        else
            for (uint32_t i = 0; i < frames; ++i)
                out[i] = 0.0f;
        // 2026-09-25: uscita stereo = stesso segnale su L/R (l'ingresso non passa mai in uscita)
        if (outputs[1] != out)
            std::memcpy(outputs[1], out, sizeof(float) * frames);

    }

private:
    float fParams[kParameterCount];
    int fLastExciterIdx = -1;  // syncDescriptorsToExciter, solo thread audio
    phimo::AudioInAnalyzer fAudioIn;
    static constexpr uint32_t kOutRsChunk = 2048;  // blocco host massimo per giro del resampler
    phimo::StreamResampler fOutRs;
    std::vector<float> fRsIn;
    bool fUseOutRs = false;  // 2026-09-25, audio-in
    phimo::AutoPairThread fAutoPair;  // 2026-09-25, auto pair
    double fAutoPairHold = 1e12;      // campioni dall'ultimo cambio di coppia (grande = primo cambio subito)
    // Target del selettore = ultimo valore GREZZO di ogni descrittore (audio-in/OSC prima del
    // clip, CC, host). Il clip al range dell'eccitatore attivo (syncDescriptorsToExciter,
    // audio-in, OSC) altrimenti rinchiude il target nella regione di quell'eccitatore e il
    // selettore lo riconferma per sempre (osservato dal vivo: bloccato su pluck+membrane).
    float fSelTarget[15];
    float fReqDesc[15];   // ultimo valore descrittore chiesto da noi all'host (per riconoscere l'eco)
    bool fReqValid[15];
    uint8_t fMorphVel = 100;          // morph: velocity/canale della nota tenuta (render periodici)
    uint8_t fMorphChannel = 0;

    // Round 3.3a (2026-09-23): Scale/tuning. setState() (chiamata quando la UI cambia i
    // controlli Scale) puo' arrivare su un thread diverso da run() (thread audio) a seconda
    // dell'host -- atomic per lo stesso motivo di fOscPending/fOscHasPending sopra (mai un
    // dato non atomico condiviso tra thread in questo progetto), anche se qui basta un
    // semplice load/store (nessun bisogno di un flag "hasPending" separato: run() vuole
    // sempre il valore corrente, non un evento one-shot da drenare).
    std::atomic<bool> fScaleActive;
    std::atomic<int> fScaleIndex;
    std::atomic<float> fA4;

    // Round 3.3b: scala custom da .scl. fSclFile/fSclData sono toccati SOLO da
    // setState/getState (thread messaggi host); run()/worker vedono solo lo snapshot
    // immutabile pubblicato in fCustomScales (puntatore atomico).
    phimo::CustomScaleStore fCustomScales;
    String fSclFile;
    String fSclData;

    phimo::Engine fEngine;
    std::unique_ptr<phimo::VoiceEngine> fVoiceEngine;

    std::atomic<float> fOscPending[kParameterCount];
    std::atomic<bool> fOscHasPending[kParameterCount];
    MultiPhiMoOscListener* fOscListener;
    UdpListeningReceiveSocket* fOscSocket;
    std::thread fOscThread;

    static constexpr int kOscPort = 9000;

    void startOsc()
    {
        for (uint32_t i = 0; i < kParameterCount; ++i)
        {
            fOscPending[i].store(0.0f, std::memory_order_relaxed);
            fOscHasPending[i].store(false, std::memory_order_relaxed);
        }

        fOscListener = new MultiPhiMoOscListener(fOscPending, fOscHasPending, "/multiphimo/target/");

        try
        {
            fOscSocket = new UdpListeningReceiveSocket(IpEndpointName(kOscPort), fOscListener);
        }
        catch (const std::exception&)
        {
            // porta occupata o altro errore di bind: OSC resta disattivo per questa
            // istanza, il plugin funziona comunque via parametri host/MIDI.
            delete fOscListener;
            fOscListener = nullptr;
            fOscSocket = nullptr;
            return;
        }

        fOscThread = std::thread([this]() { fOscSocket->Run(); });
    }

    void stopOsc()
    {
        if (fOscSocket != nullptr)
        {
            fOscSocket->AsynchronousBreak();
            if (fOscThread.joinable())
                fOscThread.join();
            delete fOscSocket;
            fOscSocket = nullptr;
        }

        delete fOscListener;
        fOscListener = nullptr;
    }

    DISTRHO_DECLARE_NON_COPYABLE(MultiPhiMoPlugin)
};

Plugin* createPlugin()
{
    return new MultiPhiMoPlugin();
}

END_NAMESPACE_DISTRHO
