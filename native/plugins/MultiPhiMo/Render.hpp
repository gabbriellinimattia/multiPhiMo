#pragma once
// Render-on-trigger (punto 7 fase 2, sotto-punto c) -- porting 1:1 di play_engine.py:
// _estimate_duration, exciters.generate() (dispatch per nome), apply_resonator via
// Resonator.hpp, _apply_fade. Chiamato dal worker thread dopo routeCandidate (Routing.hpp):
// produce il buffer audio pronto da accodare al mixer (prossimo sotto-punto).
//
// NOTA: come gia' per Routing.hpp, la generazione stocastica (exciter con rngSeed) non e'
// riproducibile bit-a-bit contro Python: la' exciter_generate() e' chiamata senza seed
// esplicito (usa entropia fresca ad ogni chiamata via np.random.default_rng(None)), qui si
// pesca un seed da 'rng' (lo stesso generatore gia' passato a routeCandidate) -- stessa
// natura "sempre diverso a parita' di descrittori", solo la sorgente di entropia cambia.
#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ParamRanges.hpp"
#include "Engine.hpp"
#include "Routing.hpp"
#include "Exciters.hpp"
#include "Resonator.hpp"

namespace phimo {

// _MAX_DURATION di play_engine.py -- esposta qui (non piu' locale a estimateDuration)
// cosi' il pool di buffer del mixer (punto 7d) puo' dimensionare gli slot senza
// duplicare la costante.
inline constexpr double kMaxDurationSeconds = 6.0;

struct RenderedAudio {
    std::vector<float> audio;
    int sr = kResonatorSR;
};

namespace render_detail {

inline std::map<std::string, float> paramsToMap(const AgentSpec& spec, const std::vector<float>& values) {
    std::map<std::string, float> m;
    for (size_t i = 0; i < values.size() && i < spec.params.size(); ++i) m[spec.params[i].name] = values[i];
    return m;
}

// exciters.py EXCITERS[name] default 'duration' kwarg (inspect.signature(...).default in
// Python) -- valori letti direttamente dalle firme in exciters.py (2026-09-22).
inline double defaultDuration(const std::string& exciterName) {
    if (exciterName == "bow") return 1.5;
    if (exciterName == "blow") return 1.5;
    if (exciterName == "strike") return 1.0;
    if (exciterName == "pluck") return 2.0;
    if (exciterName == "shaker") return 1.5;
    if (exciterName == "noise") return 1.0;
    if (exciterName == "chaos") return 1.5;
    if (exciterName == "mechanical") return 1.5;
    if (exciterName == "bird") return 1.5;
    if (exciterName == "vocal") return 1.5;
    throw std::runtime_error("eccitatore sconosciuto (defaultDuration): " + exciterName);
}

} // namespace render_detail

// play_engine._estimate_duration: max(default_dur, decadimento risonatore * margine,
// decay_time dell'eccitatore se presente), clampato tra default_dur e _MAX_DURATION.
inline double estimateDuration(const std::string& exciterName, const std::string& resonatorShape,
                                const std::map<std::string, float>& exciterParamsMap,
                                const std::map<std::string, float>& resonatorParamsMap) {
    constexpr double kDurationMargin = 3.0;  // _DURATION_MARGIN
    const double defaultDur = render_detail::defaultDuration(exciterName);

    double resoDecay = 0.0;
    try {
        const ModalBank bank = computeModalBank(resonatorShape, resonatorParamsMap);
        double maxDt = 0.0;
        for (double dt : bank.dampingTimes) maxDt = std::max(maxDt, dt);
        resoDecay = maxDt * kDurationMargin;
    } catch (const std::exception&) {
        resoDecay = 0.0;  // stesso except Exception -> 0.0 di _estimate_duration
    }

    double excDecay = 0.0;
    auto it = exciterParamsMap.find("decay_time");
    if (it != exciterParamsMap.end()) excDecay = it->second;

    double dur = std::max({defaultDur, resoDecay, excDecay});
    return std::min(std::max(dur, defaultDur), kMaxDurationSeconds);
}

// exciters.generate(name, **params) / EXCITERS[name](**params): dispatch per nome,
// argomenti posizionali nello STESSO ordine di agentSpecs().at(name).params (verificato
// 1:1 contro le firme in Exciters.hpp per tutti e 10 gli eccitatori). Gli eccitatori
// deterministici (bow/strike) ignorano 'seed'.
inline std::vector<float> generateExciter(const std::string& name, double duration,
                                           const std::vector<float>& p, int sr, unsigned seed) {
    if (name == "bow")
        return bow(duration, p[0], p[1], p[2], p[3], p[4], p[5], sr);
    if (name == "blow")
        return blow(duration, p[0], p[1], p[2], p[3], p[4], p[5], sr, seed);
    if (name == "strike")
        return strike(duration, p[0], p[1], p[2], p[3], p[4], p[5], p[6], sr);
    if (name == "pluck")
        return pluck(duration, p[0], p[1], p[2], p[3], p[4], sr, seed);
    if (name == "shaker")
        return shaker(duration, p[0], p[1], p[2], p[3], p[4], sr, seed);
    if (name == "noise")
        return noise(duration, p[0], p[1], p[2], p[3], p[4], p[5], sr, seed);
    if (name == "chaos")
        return chaos(duration, p[0], p[1], p[2], p[3], p[4], p[5], sr, seed);
    if (name == "mechanical")
        return mechanical(duration, p[0], p[1], p[2], p[3], p[4], p[5], p[6], sr, seed);
    if (name == "bird")
        return bird(duration, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], sr, seed);
    if (name == "vocal")
        return vocal(duration, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], sr, seed);
    throw std::runtime_error("eccitatore sconosciuto (generateExciter): " + name);
}

// play_engine._apply_fade: fade in/out raised-cosine in testa/coda, sull'intero buffer.
// np.linspace(0, pi, n): ramp[i] = 0.5 - 0.5*cos(i * pi / (n - 1)) per i in [0, n).
inline void applyFade(std::vector<float>& audio, int sr, double fadeInMs, double fadeOutMs) {
    const int n = (int)audio.size();
    const int fadeInN = std::min((int)(sr * fadeInMs / 1000.0), n / 2);
    const int fadeOutN = std::min((int)(sr * fadeOutMs / 1000.0), n / 2);
    if (fadeInN > 1) {
        for (int i = 0; i < fadeInN; ++i) {
            double t = (double)i * M_PI / (double)(fadeInN - 1);
            audio[i] *= (float)(0.5 - 0.5 * std::cos(t));
        }
    }
    if (fadeOutN > 1) {
        for (int i = 0; i < fadeOutN; ++i) {
            // ramp[::-1][i] == ramp[fadeOutN - 1 - i]
            double t = (double)(fadeOutN - 1 - i) * M_PI / (double)(fadeOutN - 1);
            audio[n - fadeOutN + i] *= (float)(0.5 - 0.5 * std::cos(t));
        }
    }
}

// Pipeline completa di un trigger (play_engine.PlayEngine.trigger, ramo di render):
// routeCandidate -> estimateDuration -> generateExciter -> applyResonatorOriginal ->
// applyFade. fadeInMs/fadeOutMs passati dal chiamante (Python: self.fade_ms / self.morph_ms
// a seconda della modalita' -- qui lasciato al chiamante, non deciso in questo sotto-punto).
inline RenderedAudio renderTrigger(const Engine& engine, const std::string& exciterName,
                                    const std::string& resonatorShape, const float target[15],
                                    std::mt19937& rng, double fadeInMs, double fadeOutMs,
                                    double maxDuration = kMaxDurationSeconds) {
    const Candidate cand = routeCandidate(engine, exciterName, resonatorShape, target, rng);

    const auto& excSpec = agentSpecs().at(exciterName);
    const auto& resSpec = agentSpecs().at("resonator_" + resonatorShape);
    const auto excMap = render_detail::paramsToMap(excSpec, cand.exciterParams);
    const auto resMap = render_detail::paramsToMap(resSpec, cand.resonatorParams);

    // maxDuration (2026-09-25): tetto piu' basso per i render del morph spettrale
    // (SpectralMorph.hpp, kMorphRenderSeconds) -- serve solo la parte stabile.
    const double duration = std::min(estimateDuration(exciterName, resonatorShape, excMap, resMap), maxDuration);
    const unsigned seed = rng();

    // param_candidate._res_f0 (2026-09-24): f0 per il cap dei decadimenti del risonatore
    // (Resonator.hpp capDampingTimesForPitch) = freq dell'eccitatore SOLO se pitch-locked,
    // altrimenti nessun cap (0.0 = comportamento invariato).
    const double f0 = (excSpec.pitchLocked && !cand.exciterParams.empty() && cand.exciterParams[0] > 0.0f)
                           ? (double)cand.exciterParams[0] : 0.0;

    RenderedAudio out;
    out.sr = kResonatorSR;
    std::vector<float> raw = generateExciter(exciterName, duration, cand.exciterParams, out.sr, seed);
    out.audio = applyResonatorOriginal(raw, resonatorShape, resMap, f0);
    applyFade(out.audio, out.sr, fadeInMs, fadeOutMs);
    return out;
}

// Mode=Manual (punto 8A round 2, 2026-09-23): stessa pipeline di renderTrigger ma SENZA
// routeCandidate -- l'utente controlla direttamente i parametri di generazione invece
// delle caratteristiche del suono risultante (deciso con l'utente: bypassare gli agenti
// MDN/KNN in Manual, non solo il loro output). exciterParams/resonatorParams gia' in
// valore fisico (denormalizzati dal chiamante via paramSpecDenormalize, ParamRanges.hpp),
// stesso ordine di agentSpecs().at(...).params. 'rng' resta necessaria per il seed degli
// eccitatori stocastici (proprieta' dell'eccitatore stesso, indipendente dal routing).
inline RenderedAudio renderTriggerManual(const std::string& exciterName,
                                          const std::string& resonatorShape,
                                          const std::vector<float>& exciterParams,
                                          const std::vector<float>& resonatorParams,
                                          std::mt19937& rng, double fadeInMs, double fadeOutMs,
                                          double maxDuration = kMaxDurationSeconds) {
    const auto& excSpec = agentSpecs().at(exciterName);
    const auto& resSpec = agentSpecs().at("resonator_" + resonatorShape);
    const auto excMap = render_detail::paramsToMap(excSpec, exciterParams);
    const auto resMap = render_detail::paramsToMap(resSpec, resonatorParams);

    // maxDuration (2026-09-25): tetto piu' basso per i render del morph spettrale
    // (SpectralMorph.hpp, kMorphRenderSeconds) -- serve solo la parte stabile.
    const double duration = std::min(estimateDuration(exciterName, resonatorShape, excMap, resMap), maxDuration);
    const unsigned seed = rng();

    // Stessa logica f0/cap di renderTrigger sopra, estesa a Manual (nessuna controparte
    // Python: Manual e' un concetto solo-VST3, ma la stessa fisica -- il risonatore non
    // deve poter "suonare" una nota propria sopra il pitch scelto a mano -- vale anche qui).
    const double f0 = (excSpec.pitchLocked && !exciterParams.empty() && exciterParams[0] > 0.0f)
                           ? (double)exciterParams[0] : 0.0;

    RenderedAudio out;
    out.sr = kResonatorSR;
    std::vector<float> raw = generateExciter(exciterName, duration, exciterParams, out.sr, seed);
    out.audio = applyResonatorOriginal(raw, resonatorShape, resMap, f0);
    applyFade(out.audio, out.sr, fadeInMs, fadeOutMs);
    return out;
}

} // namespace phimo
