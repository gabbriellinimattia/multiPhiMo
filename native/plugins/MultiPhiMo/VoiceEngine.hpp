#pragma once
// Mixer polifonico + limiter + gestione voci (punto 7 fase 2, sotto-punto d) -- porting
// di play_engine.py: PlayEngine._callback/_Voice/_steal_release/release_note, MENO
// `morph` (fuori scope, specifico dello slider GUI continuo -- confermato con l'utente
// 2026-09-22) e MENO l'uso di `time.monotonic()` per l'ordine di anzianita' delle voci
// (qui un contatore monotono incrementato ad ogni voce creata: stesso ordine risultante,
// nessuna chiamata a orologio nel thread audio).
//
// Differenza deliberata dal porting 1:1 (non un bug, verificata come equivalente
// matematicamente -- vedi commento su Voice sotto): _steal_release in Python copia e
// accorcia l'array della voce quando parte una dissolvenza forzata (voice-stealing o
// note-off); qui, per non allocare/copiare mai nel thread audio, la dissolvenza e'
// applicata come inviluppo calcolato al momento del mixaggio (stessa formula raised-
// cosine, stesso punto di troncamento) senza mai toccare il buffer del pool.
//
// Architettura (decisa con l'utente 2026-09-22): un pool fisso pre-allocato di buffer
// (nessuna allocazione nel thread audio, vincolo esplicito del progetto) + tre code SPSC
// (Spsc.hpp) a dimensione fissa: TriggerQueue (audio->worker, richieste di render),
// ReadyQueue (worker->audio, buffer pronti), FreeQueue (audio->worker, slot da riciclare).
// Il worker fa anche il render vero e proprio (Render.hpp::renderTrigger) -- nessun altro
// thread tocca il pool oltre a lui (scrittura) e al thread audio (lettura). Note-off NON
// passa dal worker: agisce direttamente su fVoices, che e' stato privato del thread audio.
//
// Round 3.3a (2026-09-23): ScaleContext/TriggerRequest::scale -- quantizzazione Scale/
// tuning (TuningQuantizer.hpp) applicata QUI nel worker (mai nel thread audio), sia al
// target "pitch" (Agent, indice 14) sia allo slot "freq" (Manual, indice 0) -- vedi
// PluginMultiPhiMo.cpp::run() per la costruzione dello ScaleContext e
// claude/vst3_native_stato.md per il design completo.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "Engine.hpp"
#include "Render.hpp"
#include "Spsc.hpp"
#include "TuningQuantizer.hpp"
#include "SpectralMorph.hpp"

namespace phimo {

// ---- costanti (porting diretto dei default di PlayEngine.__init__, confermati 2026-09-22) ----
inline constexpr int kPoolSlots = 24;              // margine sopra max_voices per il render in volo
inline constexpr uint32_t kMaxDurationSamples = (uint32_t)(kMaxDurationSeconds * kResonatorSR);
inline constexpr int kSpscCapacity = 32;           // > kPoolSlots, margine sulle code

inline constexpr int kMaxVoices = 10;              // PlayEngine.max_voices
inline constexpr float kMixGain = 0.4f;            // PlayEngine.gain
inline constexpr double kFadeInMs = 8.0;           // PlayEngine.fade_ms (morph escluso)
inline constexpr double kFadeOutMs = 8.0;
inline constexpr double kNoteOffFadeMs = 120.0;    // release_note default
inline constexpr double kStealReleaseMs = 5.0;     // _STEAL_RELEASE_MS
inline constexpr float kLimiterCeiling = 0.85f;    // LIMITER_CEILING
inline constexpr float kLimiterAttackMs = 5.0f;
inline constexpr float kLimiterReleaseMs = 60.0f;
inline constexpr float kVelocityToGainDivisor = 116.0f;  // deciso con l'utente 2026-09-22

// Round 3.3a (2026-09-23): ingredienti GREZZI per la quantizzazione Scale/tuning, copiati
// da run() (thread audio, solo membri fScaleActive/fScaleIndex/fA4 gia' pronti, nessuna
// lookup) -- il calcolo vero (midiNoteToFreq/quantizeFreqToScale, TuningQuantizer.hpp)
// resta nel worker, come tutto il resto del percorso "pesante" di questa struct.
struct ScaleContext {
    bool active = false;
    int32_t overrideNoteMidi = -1;  // -1 = nessun override da nota reale (Play, o Scale off)
    int scaleIndex = kScaleNone;
    float a4 = 440.0f;
    // Round 3.3b: puntatore a uno snapshot IMMUTABILE (CustomScaleStore, TuningQuantizer.hpp),
    // nullptr se nessun .scl caricato -- POD, nessun refcount, copia lock-free da run().
    const CustomScale* custom = nullptr;
};

struct TriggerRequest {
    const char* exciterName = nullptr;    // puntatore a stringa statica (kExciterEnum[i].label)
    const char* resonatorName = nullptr;  // idem, kResonatorEnum[i].label
    int32_t noteId = -1;
    uint64_t triggerSeq = 0;  // fix drone, vedi VoiceEngine::releasedUpTo_
    float gain = 1.0f;
    float target[15] = {};
    // Mode=Manual (punto 8A round 2, 2026-09-23): quando manual=true il worker salta
    // routeCandidate (renderTriggerManual invece di renderTrigger, vedi Render.hpp).
    // Questi due array portano i valori GREZZI 0-1 cosi' come li tiene l'host (copia
    // diretta di fParams[kParameterExciterParamFirst/ResonatorParamFirst + i], nessun
    // calcolo in run()) -- la denormalizzazione in valore fisico (serve la ParamSpec
    // dell'agente, quindi una lookup su agentSpecs()) resta nel worker, come tutto il
    // resto del percorso agentSpecs()-dipendente: mai una nuova lookup su std::map nel
    // thread audio. Array a dimensione fissa (9/10, come i parametri host
    // kParameterExciterParamFirst/ResonatorParamFirst) -- nessuna allocazione nel thread
    // audio, stesso vincolo di 'target' sopra.
    bool manual = false;
    float excParams[9] = {};
    float resParams[10] = {};
    ScaleContext scale;  // round 3.3a -- default: nessun override, nessuna quantizzazione
    bool morph = false;  // 2026-09-25: render per il morph spettrale (-> istantanea, non voce)
};

struct ReadyAudio {
    int32_t slot = -1;
    uint32_t frames = 0;
    int32_t noteId = -1;
    float gain = 1.0f;
    uint64_t triggerSeq = 0;  // ultimo campo: ordine dell'aggregate init ReadyAudio{...}
};

// Stato di una voce -- SOLO il thread audio la legge/scrive (nessuna sincronizzazione
// necessaria, e' privata come in Python self._voices).
struct Voice {
    int32_t slot = -1;
    uint32_t frames = 0;      // lunghezza valida nel buffer del pool
    uint32_t pos = 0;         // cursore di lettura
    int32_t noteId = -1;
    float gain = 1.0f;
    bool stolen = false;
    uint32_t releaseStart = 0;  // valido solo se stolen && releaseLen>0
    uint32_t releaseLen = 0;    // 0 = nessuna dissolvenza forzata applicata (voce quasi finita comunque)
    uint64_t bornSeq = 0;       // ordine di creazione, sostituisce time.monotonic()
};

// applyFade usa la stessa formula alzata-a-coseno di play_engine._apply_fade; qui la
// stessa formula per _steal_release (0.5+0.5*cos, non 0.5-0.5*cos -- ramp che PARTE da 1
// e scende a 0, verificato nel sorgente Python).
inline float stealReleaseEnvelope(uint32_t posInRelease, uint32_t releaseLen) {
    if (releaseLen <= 1) return 1.0f;
    const double t = (double)posInRelease * M_PI / (double)(releaseLen - 1);
    return (float)(0.5 + 0.5 * std::cos(t));
}

class VoiceEngine {
public:
    explicit VoiceEngine(const Engine& engine) : engine_(engine) {
        pool_.assign((size_t)kPoolSlots * kMaxDurationSamples, 0.0f);
        voices_.reserve(kPoolSlots + 4);  // mai piu' di kPoolSlots voci vive (una per slot)
    }
    ~VoiceEngine() { stop(); }

    void start() {
        stopFlag_.store(false, std::memory_order_relaxed);
        worker_ = std::thread([this]() { workerLoop(); });
    }
    void stop() {
        if (worker_.joinable()) {
            stopFlag_.store(true, std::memory_order_relaxed);
            worker_.join();
        }
    }

    // ---- chiamato SOLO dal thread audio (run()) ----

    // MIDI note-on: costruisce e accoda una TriggerRequest (best-effort, scarta se la
    // coda e' piena -- mai un blocco nel thread audio). exciterName/resonatorName DEVONO
    // restare validi per tutta la vita del plugin (stringhe statiche, es. kExciterEnum).
    bool triggerNoteOn(const char* exciterName, const char* resonatorName, int32_t noteId,
                        float velocity01to127, const float target[15],
                        const ScaleContext& scale = ScaleContext(), bool morph = false) {
        TriggerRequest req;
        req.exciterName = exciterName;
        req.resonatorName = resonatorName;
        req.noteId = noteId;
        req.triggerSeq = ++triggerSeq_;
        req.gain = velocity01to127 / kVelocityToGainDivisor;
        for (int i = 0; i < 15; ++i) req.target[i] = target[i];
        req.scale = scale;
        req.morph = morph;
        const bool ok = triggerQueue_.push(req);
        if (!ok) ++triggerDropCount_;
        if (ok && morph) { morphInFlight_ = true; morphSinceReq_ = 0; }
        return ok;
    }

    // Mode=Manual (punto 8A round 2, 2026-09-23): stesso schema di triggerNoteOn ma con gli
    // slot manuali grezzi (0-1, stessi valori del parametro host) al posto del target[15]
    // -- il worker bypassa il routing Agent/KNN e denormalizza (vedi renderTriggerManual,
    // Render.hpp). excParams/resParams nell'ordine kParameterExciterParamFirst/
    // ResonatorParamFirst (== ordine di agentSpecs().at(...).params per l'agente attivo,
    // ma la corrispondenza puntuale e' responsabilita' del worker, non del chiamante).
    bool triggerNoteOnManual(const char* exciterName, const char* resonatorName, int32_t noteId,
                              float velocity01to127, const float excParams[9], const float resParams[10],
                              const ScaleContext& scale = ScaleContext(), bool morph = false) {
        TriggerRequest req;
        req.exciterName = exciterName;
        req.resonatorName = resonatorName;
        req.noteId = noteId;
        req.triggerSeq = ++triggerSeq_;
        req.gain = velocity01to127 / kVelocityToGainDivisor;
        req.manual = true;
        for (int i = 0; i < 9; ++i) req.excParams[i] = excParams[i];
        for (int i = 0; i < 10; ++i) req.resParams[i] = resParams[i];
        req.scale = scale;
        req.morph = morph;
        const bool ok = triggerQueue_.push(req);
        if (!ok) ++triggerDropCount_;
        if (ok && morph) { morphInFlight_ = true; morphSinceReq_ = 0; }
        return ok;
    }

    // ---- morph spettrale (2026-09-25, SpectralMorph.hpp) -- solo thread audio ----
    // Voce mono: il chiamante (run()) fa morphNoteOn + un trigger con morph=true; poi, finche'
    // la voce e' tenuta, morphDue() dice quando chiedere il render successivo (uno alla volta,
    // mai piu' di uno in volo: si auto-regola sulla velocita' del worker).
    void morphNoteOn(int32_t noteId, float velocity01to127) {
        morphVoice_.noteOn(noteId, velocity01to127 / kVelocityToGainDivisor);
    }
    void morphNoteOff(int32_t noteId) { morphVoice_.noteOff(noteId, kResonatorSR, kNoteOffFadeMs); }
    void morphRelease() { morphVoice_.forceRelease(kResonatorSR, kNoteOffFadeMs); }
    bool morphHeld() const { return morphVoice_.held(); }
    int32_t morphNoteId() const { return morphVoice_.noteId(); }
    bool morphDue(uint32_t frames) {
        if (!morphVoice_.held()) return false;
        morphSinceReq_ += frames;
        return !morphInFlight_ && morphSinceReq_ >= (uint32_t)(kMorphRequestSeconds * kResonatorSR);
    }

    // MIDI note-off / release_note: dissolvenza diretta sulle voci vive con questo
    // noteId -- nessuna coda, fVoices e' gia' privato del thread audio.
    void triggerNoteOff(int32_t noteId, double fadeMs = kNoteOffFadeMs) {
        if (noteId < 0) return;
        if (noteId < 128) releasedUpTo_[noteId] = triggerSeq_;  // copre anche i render ancora in volo
        for (Voice& v : voices_)
            if (v.noteId == noteId && !v.stolen)
                stealVoice(v, fadeMs);
    }

    // Un blocco audio: drena ReadyQueue, applica il tetto max_voices (voice-stealing
    // sulle piu' vecchie), mixa, applica gain+limiter+clip. out deve avere 'frames'
    // campioni (mono, come Python sd.OutputStream(channels=1)).
    void mixBlock(float* out, uint32_t frames, float mixGain = kMixGain, float morphSmoothingMs = 200.0f) {
        {   // istantanee spettrali pronte dal worker (morph): l'ultima vince
            int32_t ms;
            while (morphReady_.pop(ms)) {
                morphInFlight_ = false;
                if (ms < 0) continue;  // render/istantanea fallita: si riprova al prossimo morphDue
                morphVoice_.setTarget(morphPool_[ms]);
                morphFree_.push(ms);
            }
        }
        ReadyAudio ready;
        while (readyQueue_.pop(ready)) {
            if ((int)voices_.size() >= kPoolSlots) {
                // non dovrebbe mai accadere (un solo slot per voce, pool esaurito prima) --
                // per sicurezza scarta e rilascia subito lo slot invece di sforare.
                freeQueue_.push(ready.slot);
                continue;
            }
            Voice v;
            v.slot = ready.slot;
            v.frames = ready.frames;
            v.noteId = ready.noteId;
            v.gain = ready.gain;
            v.bornSeq = nextBornSeq_++;
            // fix drone: note-off gia' ricevuto per questo trigger mentre era in render
            if (v.noteId >= 0 && v.noteId < 128 && ready.triggerSeq <= releasedUpTo_[v.noteId])
                stealVoice(v, kNoteOffFadeMs);
            voices_.push_back(v);
        }

        // voice-stealing: oltre kMaxVoices, le piu' vecchie (bornSeq minore) non ancora
        // in dissolvenza vengono avviate in dissolvenza -- stesso schema di _callback.
        {
            std::vector<Voice*> live;
            live.reserve(voices_.size());
            for (Voice& v : voices_) if (!v.stolen) live.push_back(&v);
            if ((int)live.size() > kMaxVoices) {
                std::sort(live.begin(), live.end(),
                          [](const Voice* a, const Voice* b) { return a->bornSeq < b->bornSeq; });
                const size_t toSteal = live.size() - (size_t)kMaxVoices;
                for (size_t i = 0; i < toSteal; ++i) stealVoice(*live[i], kStealReleaseMs);
            }
        }

        std::fill(out, out + frames, 0.0f);
        for (size_t i = 0; i < voices_.size();) {
            Voice& v = voices_[i];
            const uint32_t effectiveEnd = (v.stolen && v.releaseLen > 0)
                                               ? std::min(v.frames, v.releaseStart + v.releaseLen)
                                               : v.frames;
            const uint32_t end = std::min(v.pos + frames, effectiveEnd);
            if (end > v.pos) {
                // out[] e' indicizzato dall'inizio del blocco (0..frames): il primo
                // campione letto (s = v.pos) va in out[0], non in out[v.pos].
                const float* src = pool_.data() + (size_t)v.slot * kMaxDurationSamples;
                uint32_t outIdx = 0;
                for (uint32_t s = v.pos; s < end; ++s, ++outIdx) {
                    float env = 1.0f;
                    if (v.stolen && v.releaseLen > 0 && s >= v.releaseStart)
                        env = stealReleaseEnvelope(s - v.releaseStart, v.releaseLen);
                    out[outIdx] += src[s] * v.gain * env;
                }
                v.pos = end;
            }
            if (v.pos >= effectiveEnd) {
                freeQueue_.push(v.slot);  // rende lo slot al worker per il riuso
                voices_[i] = voices_.back();
                voices_.pop_back();
                continue;  // non incrementare i, l'elemento swappato va ricontrollato
            }
            ++i;
        }

        // gain manuale fisso, poi limiter feedforward (stessa formula di _callback:
        // attacco/rilascio esponenziale sul guadagno del limiter, stato persistente).
        morphVoice_.render(out, frames, kResonatorSR, morphSmoothingMs);  // voce morph, prima di gain/limiter

        for (uint32_t s = 0; s < frames; ++s) out[s] *= mixGain;  // gain master (parametro host), prima del limiter come in Python

        float peak = 0.0f;
        for (uint32_t s = 0; s < frames; ++s) peak = std::max(peak, std::fabs(out[s]));
        const float targetGain = (peak > kLimiterCeiling) ? (kLimiterCeiling / peak) : 1.0f;
        const double bufS = (double)frames / (double)kResonatorSR;
        const double tauS = ((targetGain < limiterGain_) ? kLimiterAttackMs : kLimiterReleaseMs) / 1000.0;
        const double coef = (tauS > 0.0) ? (1.0 - std::exp(-bufS / tauS)) : 1.0;
        limiterGain_ += (float)((targetGain - limiterGain_) * coef);
        for (uint32_t s = 0; s < frames; ++s) {
            out[s] *= limiterGain_;
            out[s] = std::min(1.0f, std::max(-1.0f, out[s]));  // backstop finale
        }
    }

    // Diagnostica -- SICURA da leggere solo dopo stop()/join() (scritta da thread diversi
    // durante l'esecuzione, nessuna atomicita' necessaria: ogni contatore ha un solo
    // scrittore, vedi commenti sopra ogni push()).
    uint64_t triggerDropCount() const { return triggerDropCount_; }
    uint64_t readyDropCount() const { return readyDropCount_; }
    uint64_t renderFailCount() const { return renderFailCount_; }
    size_t liveVoiceCount() const { return voices_.size(); }

private:
    void stealVoice(Voice& v, double fadeMs) {
        v.stolen = true;  // sempre, anche se il ramo sotto non applica alcuna dissolvenza
        const uint32_t remaining = v.frames - v.pos;
        const uint32_t releaseN = std::min((uint32_t)(kResonatorSR * fadeMs / 1000.0), remaining);
        if (releaseN <= 1) { v.releaseLen = 0; return; }  // quasi finita comunque, nessun beneficio
        v.releaseStart = v.pos;
        v.releaseLen = releaseN;
    }

    void workerLoop() {
        std::vector<int32_t> freeSlots;
        freeSlots.reserve(kPoolSlots);
        for (int i = 0; i < kPoolSlots; ++i) freeSlots.push_back(i);
        std::vector<int32_t> morphFreeSlots;
        for (int i = 0; i < kMorphSlots; ++i) morphFreeSlots.push_back(i);
        std::mt19937 rng(std::random_device{}());

        while (!stopFlag_.load(std::memory_order_relaxed)) {
            int32_t freed;
            while (freeQueue_.pop(freed)) freeSlots.push_back(freed);
            while (morphFree_.pop(freed)) morphFreeSlots.push_back(freed);

            TriggerRequest req;
            if (!triggerQueue_.pop(req)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            if (req.morph) {
                if (morphFreeSlots.empty()) { morphReady_.push(-1); continue; }
            } else if (freeSlots.empty()) {
                continue;  // nessuno slot libero, scarta (best-effort)
            }
            // i render del morph non usano il pool delle voci (diventano un'istantanea)
            const int32_t slot = req.morph ? -1 : freeSlots.back();
            if (!req.morph) freeSlots.pop_back();
            const double maxDur = req.morph ? kMorphRenderSeconds : kMaxDurationSeconds;
            bool ok = false;
            try {
                RenderedAudio audio;
                if (req.manual) {
                    // Mode=Manual (punto 8A round 2): niente routeCandidate, i vettori
                    // vanno dimensionati esattamente a spec.params.size() -- generateExciter
                    // indicizza p[0..n-1] posizionalmente senza controllo bound (vedi
                    // Render.hpp), un vettore piu' corto/lungo del previsto e' un bug, non
                    // un caso limite da gestire silenziosamente.
                    // req.excParams/resParams sono grezzi 0-1 (copiati cosi' da run(), mai
                    // denormalizzati li' -- vedi commento su TriggerRequest sopra): la
                    // conversione in fisico serve la ParamSpec, quindi avviene qui, DOPO le
                    // lookup su agentSpecs() gia' necessarie per dimensionare i vettori.
                    const auto& excSpec = agentSpecs().at(req.exciterName);
                    const auto& resSpec = agentSpecs().at(std::string("resonator_") + req.resonatorName);
                    std::vector<float> excVec(excSpec.params.size());
                    for (size_t i = 0; i < excVec.size(); ++i)
                        excVec[i] = paramSpecDenormalize(req.excParams[i], excSpec.params[i]);
                    std::vector<float> resVec(resSpec.params.size());
                    for (size_t i = 0; i < resVec.size(); ++i)
                        resVec[i] = paramSpecDenormalize(req.resParams[i], resSpec.params[i]);
                    // Round 3.3a: "freq" e' SEMPRE lo slot 0 per tutti e 10 gli eccitatori
                    // (verificato in exciters.py PARAM_RANGES, vedi ParamRanges.hpp/nota al
                    // punto 4) -- unico punto di quantizzazione Scale in Mode=Manual.
                    if (req.scale.active && !excVec.empty()) {
                        const float base = (req.scale.overrideNoteMidi >= 0)
                            ? midiNoteToFreq(req.scale.overrideNoteMidi, req.scale.a4)
                            : excVec[0];
                        excVec[0] = quantizeFreqToScale(base, req.scale.scaleIndex, req.scale.a4, req.scale.custom);
                    }
                    audio = renderTriggerManual(req.exciterName, req.resonatorName, excVec, resVec,
                                                 rng, kFadeInMs, kFadeOutMs, maxDur);
                } else {
                    // Round 3.3a: target[14] == "pitch" (kDescriptorKeys[14], vedi
                    // ParamRanges.hpp) -- copia locale mutabile, req.target resta il valore
                    // grezzo cosi' come arrivato da run() (nessun bisogno di preservarlo oltre
                    // questo trigger, ma comunque piu' pulito non mutare la request).
                    float target[15];
                    std::memcpy(target, req.target, sizeof(target));
                    if (req.scale.active) {
                        constexpr int kPitchIdx = 14;
                        const float base = (req.scale.overrideNoteMidi >= 0)
                            ? midiNoteToFreq(req.scale.overrideNoteMidi, req.scale.a4)
                            : target[kPitchIdx];
                        target[kPitchIdx] = quantizeFreqToScale(base, req.scale.scaleIndex, req.scale.a4, req.scale.custom);
                    }
                    audio = renderTrigger(engine_, req.exciterName, req.resonatorName,
                                           target, rng, kFadeInMs, kFadeOutMs, maxDur);
                }
                if (req.morph) {
                    const int32_t ms = morphFreeSlots.back();
                    if (computeMorphSnapshot(audio.audio, audio.sr, morphPool_[ms])) {
                        morphFreeSlots.pop_back();
                        morphReady_.push(ms);
                    } else {
                        morphReady_.push(-1);
                    }
                    continue;  // nessuna voce da accodare
                }
                const uint32_t n = (uint32_t)std::min(audio.audio.size(), (size_t)kMaxDurationSamples);
                std::memcpy(pool_.data() + (size_t)slot * kMaxDurationSamples, audio.audio.data(),
                            n * sizeof(float));
                ReadyAudio ready{slot, n, req.noteId, req.gain, req.triggerSeq};
                ok = readyQueue_.push(ready);
                if (!ok) ++readyDropCount_;
            } catch (const std::exception&) {
                ++renderFailCount_;  // stesso spirito del try/except di play_engine.trigger()
                if (req.morph) morphReady_.push(-1);
            }
            if (!ok && slot >= 0) freeSlots.push_back(slot);  // render fallito o ReadyQueue piena: rilascia subito
        }
    }

    const Engine& engine_;
    std::vector<float> pool_;
    std::vector<Voice> voices_;  // SOLO thread audio
    uint64_t nextBornSeq_ = 0;
    // Fix drone (2026-09-25): un note-off arrivato PRIMA che la voce fosse pronta (render
    // ancora in coda/in corso nel worker) andava perso -> la voce suonava tutto il one-shot.
    // triggerSeq_ numera ogni note-on; releasedUpTo_[nota] = ultimo seq emesso al note-off:
    // una voce pronta con seq <= releasedUpTo_[nota] nasce gia' in rilascio. SOLO thread audio.
    uint64_t triggerSeq_ = 0;
    uint64_t releasedUpTo_[128] = {};

    // Morph spettrale (2026-09-25): 4 istantanee pre-allocate (worker scrive, audio legge
    // e restituisce via morphFree_), voce di risintesi mono (solo thread audio).
    static constexpr int kMorphSlots = 4;
    MorphSnapshot morphPool_[kMorphSlots];
    SpscQueue<int32_t, 8> morphReady_;   // worker -> audio (-1 = render fallito)
    SpscQueue<int32_t, 8> morphFree_;    // audio -> worker
    MorphVoice morphVoice_;
    bool morphInFlight_ = false;         // solo thread audio
    uint32_t morphSinceReq_ = 0;         // solo thread audio
    float limiterGain_ = 1.0f;

    SpscQueue<TriggerRequest, kSpscCapacity> triggerQueue_;
    SpscQueue<ReadyAudio, kSpscCapacity> readyQueue_;
    SpscQueue<int32_t, kSpscCapacity> freeQueue_;

    std::thread worker_;
    std::atomic<bool> stopFlag_{false};

    uint64_t triggerDropCount_ = 0;  // scritto solo dal thread audio (triggerNoteOn)
    uint64_t readyDropCount_ = 0;    // scritto solo dal worker
    uint64_t renderFailCount_ = 0;   // scritto solo dal worker
};

} // namespace phimo
