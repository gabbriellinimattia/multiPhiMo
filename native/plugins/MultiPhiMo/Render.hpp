#pragma once
// Render-on-trigger -- 2026-09-28 (v5): semantica di surrogate.render_measure / dataset_v2:
// exciters.generate(durata = resonator.note_duration) -> resonator.apply_resonator v5 (Coupling.hpp, stadio
// d'eccitazione + accoppiamento + formanti, f0 = freq dell'eccitatore per TUTTI gli eccitatori come
// param_candidate._res_f0) -> _apply_fade. Il routing Agent (surrogato + ricerca reale) vive nel worker
// (VoiceEngine.hpp); qui solo il render di una configurazione gia' decisa. MDN/KNN (Routing.hpp) non piu' usati.
// Generazione stocastica: seed pescato da rng (come prima), non riproducibile bit-a-bit contro Python.
#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ParamRanges.hpp"
#include "Exciters.hpp"
#include "Resonator.hpp"
#include "Coupling.hpp"

namespace phimo {

// Tetto del pool di buffer del mixer (VoiceEngine.hpp); note_duration resta sotto 3 s.
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

// Parametri eccitatore per nome -> vettore nell'ordine di agentSpecs().at(exc).params (ordine di generateExciter)
inline std::vector<float> exciterVector(const std::string& exc, const std::map<std::string, float>& m) {
    const auto& spec = agentSpecs().at(exc);
    std::vector<float> v(spec.params.size());
    for (size_t i = 0; i < v.size(); ++i) {
        auto it = m.find(spec.params[i].name);
        if (it == m.end()) throw std::runtime_error("parametro mancante: " + exc + "." + spec.params[i].name);
        v[i] = it->second;
    }
    return v;
}

// param_candidate._res_f0: f0 = freq dell'eccitatore (se > 0) per tutti gli eccitatori
inline double exciterF0(const std::map<std::string, float>& excMap) {
    auto it = excMap.find("freq");
    return (it != excMap.end() && it->second > 0.0f) ? (double)it->second : 0.0;
}

// surrogate.render_measure: render SENZA fade (anche per la ricerca reale)
inline std::vector<float> renderRaw(const std::string& exc, const std::string& shape,
                                    const std::map<std::string, float>& excMap,
                                    const std::map<std::string, float>& resMap,
                                    const std::map<std::string, float>& coupMap, unsigned seed,
                                    double maxDuration = kMaxDurationSeconds) {
    const double dur = std::min(noteDuration(render_detail::defaultDuration(exc), shape, resMap, coupMap), maxDuration);
    std::vector<float> raw = generateExciter(exc, dur, exciterVector(exc, excMap), kResonatorSR, seed);
    return applyResonatorV5(raw, shape, resMap, coupMap, exciterF0(excMap));
}

inline RenderedAudio renderConfig(const std::string& exc, const std::string& shape,
                                  const std::map<std::string, float>& excMap,
                                  const std::map<std::string, float>& resMap,
                                  const std::map<std::string, float>& coupMap, unsigned seed,
                                  double fadeInMs, double fadeOutMs, double maxDuration = kMaxDurationSeconds) {
    RenderedAudio out;
    out.sr = kResonatorSR;
    out.audio = renderRaw(exc, shape, excMap, resMap, coupMap, seed, maxDuration);
    applyFade(out.audio, out.sr, fadeInMs, fadeOutMs);
    return out;
}

// Mode=Manual: parametri gia' fisici (ordine agentSpecs), accoppiamento dai 10 slot normalizzati
inline RenderedAudio renderTriggerManual(const std::string& exciterName, const std::string& resonatorShape,
                                         const std::vector<float>& exciterParams,
                                         const std::vector<float>& resonatorParams, const float coupRaw[kCouplingCount],
                                         std::mt19937& rng, double fadeInMs, double fadeOutMs,
                                         double maxDuration = kMaxDurationSeconds) {
    const auto excMap = render_detail::paramsToMap(agentSpecs().at(exciterName), exciterParams);
    const auto resMap = render_detail::paramsToMap(agentSpecs().at("resonator_" + resonatorShape), resonatorParams);
    std::map<std::string, float> coupMap;
    for (int i = 0; i < kCouplingCount; ++i) coupMap[kCouplingSpecs[i].name] = couplingDenormalize(coupRaw[i], kCouplingSpecs[i]);
    return renderConfig(exciterName, resonatorShape, excMap, resMap, coupMap, rng(), fadeInMs, fadeOutMs, maxDuration);
}

} // namespace phimo
