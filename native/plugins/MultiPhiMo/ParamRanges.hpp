#pragma once
// Tabelle parametri/descrittori per il motore MDN nativo -- porting 1:1 di agents.py:
// DESCRIPTOR_KEYS, AGENTS_WITH_DECAY/descriptor_keys_for, LOG_PARAMS, exciters.PARAM_RANGES,
// resonator.PARAM_RANGES (compresi tube/soundboard/chaotic). Verificato per confronto diretto
// col sorgente Python (device_bash grep, 2026-09-22), non da documentazione/memoria.
// Richiede C++17.
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace phimo {

// Ordine canonico descrittori (agents.py DESCRIPTOR_KEYS). Gli indici qui sotto sono usati
// da AgentSpec::descriptorIndices per indicizzare questo array.
inline constexpr std::array<const char*, 15> kDescriptorKeys = {
    "spectral_centroid", "spectral_spread", "spectral_rolloff", "spectral_flatness",
    "roughness", "harmonic_tension", "inharmonicity", "formant_f1", "formant_f2",
    "formant_f3", "mod_rate", "mod_depth", "attack_time", "decay_time", "pitch",
};
inline constexpr int kAttackTimeIdx = 12;  // LOG_DESCRIPTORS: log10 solo su input, mai sui parametri
inline constexpr int kDecayTimeIdx = 13;
inline constexpr int kPitchIdx = 14;

struct ParamSpec {
    const char* name;
    float lo, hi;
    bool logScale;  // agents.py LOG_PARAMS: normalizzazione/denormalizzazione in log10 invece che lineare
};

// pitch-lock (agents.py PITCH_LOCKED_EXCITERS, 2026-09-24): per 8 eccitatori su 10 (tutti
// tranne shaker/noise) `freq` non e' piu' predetta da MDN/KNN -- e' imposta a runtime dal
// descrittore target `pitch` (vedi lockFreq() sotto). `params` resta la lista FISICA
// COMPLETA (freq sempre inclusa, in testa) -- serve invariata a Mode=Manual/GUI e a
// generateExciter() (dispatch posizionale in Render.hpp), che non cambiano. Solo il
// sottoinsieme "predetto dal modello" (MdnAgent/KnnCorpus, vedi modelParams() sotto) esclude
// freq per gli agenti pitchLocked -- mai i resonator_* (mai pitch-locked, freq non e' un loro
// parametro).
struct AgentSpec {
    std::string name;
    std::vector<ParamSpec> params;       // lista fisica completa (Manual/GUI/generateExciter), freq in testa
    std::vector<int> descriptorIndices;  // indici in kDescriptorKeys, ordine di ingresso (x)
    bool pitchLocked = false;
};

// Sottoinsieme di params realmente predetto da MDN/KNN: params[1:] se pitchLocked (freq
// esclusa), altrimenti l'intera lista -- porting di agents.py:
// AGENT_SPECS = {p:r for p,r in _s.items() if p!="freq"} if _n in PITCH_LOCKED_EXCITERS else _s.
inline std::vector<ParamSpec> modelParams(const AgentSpec& spec) {
    if (!spec.pitchLocked) return spec.params;
    return std::vector<ParamSpec>(spec.params.begin() + 1, spec.params.end());
}
inline size_t modelParamCount(const AgentSpec& spec) {
    return spec.params.size() - (spec.pitchLocked ? 1 : 0);
}

// Mode=Manual (punto 8A round 2, 2026-09-23): i 19 slot manuali generici restano
// esposti all'host come parametri normalizzati 0-1 fissi (DELIBERATO -- l'automazione
// host non deve dipendere da quale eccitatore/risonatore e' selezionato in quel momento,
// vedi claude/vst3_native_stato.md); il valore fisico dipende dalla ParamSpec
// dell'agente attivo. Formule identiche a Slider::fractionToValue/valueToFraction
// (UiWidgets.hpp) -- unica sorgente condivisa tra GUI (UIMultiPhiMo.cpp) e DSP
// (PluginMultiPhiMo.cpp) per evitare che le due copie divergano.
inline float paramSpecDenormalize(float t, const ParamSpec& spec) noexcept {
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    if (spec.logScale) {
        const float llo = std::log10(std::max(spec.lo, 1e-6f));
        const float lhi = std::log10(std::max(spec.hi, 1e-6f));
        return std::pow(10.0f, llo + t * (lhi - llo));
    }
    return spec.lo + t * (spec.hi - spec.lo);
}

inline float paramSpecNormalize(float v, const ParamSpec& spec) noexcept {
    if (spec.logScale) {
        const float lv  = std::log10(std::max(v, 1e-6f));
        const float llo = std::log10(std::max(spec.lo, 1e-6f));
        const float lhi = std::log10(std::max(spec.hi, 1e-6f));
        return (lhi > llo) ? (lv - llo) / (lhi - llo) : 0.0f;
    }
    return (spec.hi > spec.lo) ? (v - spec.lo) / (spec.hi - spec.lo) : 0.0f;
}

// Espande un vettore "modello" (uscita MDN/KNN, freq esclusa se pitchLocked) nella lista
// fisica COMPLETA attesa altrove (Manual/generateExciter): inserisce un placeholder in
// testa (indice 0 = freq), da riempire subito dopo con lockFreq(). No-op (nessuna copia
// concettuale, stessa dimensione) per gli agenti non pitchLocked.
inline std::vector<float> expandModelParams(const AgentSpec& spec, const std::vector<float>& modelVals) {
    if (!spec.pitchLocked) return modelVals;
    std::vector<float> full(spec.params.size(), 0.0f);
    for (size_t i = 0; i < modelVals.size(); ++i) full[i + 1] = modelVals[i];  // full.size() == modelVals.size()+1 per costruzione
    return full;
}

// param_candidate._lock_freq (2026-09-24), porting 1:1: freq = target pitch clippato al
// range fisico dell'eccitatore; pitch non valido (<=0) -> media geometrica del range.
// No-op per gli agenti non pitchLocked (freq resta quella gia' in fullParams[0]). NIENTE
// std::isnan/isfinite qui (-ffast-math li rende inaffidabili, vedi vst3_native_stato.md
// punto 7d bug B) -- range-check esplicito, stesso schema gia' usato in TuningQuantizer.hpp.
inline void lockFreq(const AgentSpec& spec, std::vector<float>& fullParams, float targetPitch) {
    if (!spec.pitchLocked || fullParams.empty()) return;
    const ParamSpec& fr = spec.params[0];  // "freq", sempre il primo fisico per costruzione
    const bool pitchValid = targetPitch > 0.0f && targetPitch < 1e6f;
    const float p = pitchValid ? targetPitch : std::sqrt(fr.lo * fr.hi);
    fullParams[0] = p < fr.lo ? fr.lo : (p > fr.hi ? fr.hi : p);
}

// agents.py descriptor_keys_for(): 15 chiavi se l'agente e' in AGENTS_WITH_DECAY
// ({"pluck","strike"} + tutti i resonator_*), altrimenti 14 (tutte tranne decay_time);
// 2026-09-24: i resonator_* escludono anche "pitch" dall'ingresso (freq eccitatore, non
// piu' un parametro del risonatore -- mai gli eccitatori, che tengono pitch in input anche
// se pitch-locked, lo usano come feature senza predire freq in uscita).
inline std::vector<int> descriptorIndicesFor(bool withDecay, bool excludePitch = false) {
    std::vector<int> idx;
    for (int i = 0; i < 15; ++i) {
        if (i == kDecayTimeIdx && !withDecay) continue;
        if (i == kPitchIdx && excludePitch) continue;
        idx.push_back(i);
    }
    return idx;
}

inline const std::map<std::string, AgentSpec>& agentSpecs() {
    static const std::map<std::string, AgentSpec> specs = [] {
        std::map<std::string, AgentSpec> m;
        auto add = [&](const char* name, bool withDecay, std::vector<ParamSpec> params) {
            const bool isResonator = std::string(name).rfind("resonator_", 0) == 0;
            m[name] = AgentSpec{name, std::move(params), descriptorIndicesFor(withDecay, isResonator), false};
        };

        // ---- eccitatori (exciters.py PARAM_RANGES, 10 agenti) ----
        add("bow", false, {
            {"freq", 65, 2000, true}, {"bow_force", 0.05f, 1.0f, false},
            {"bow_velocity", 0.05f, 1.0f, false}, {"bow_position", 0.02f, 0.45f, false},  // 0.5->0.45 (exciters.py 24/9: a 0.5 sub-armonica)
            {"brightness", 0.1f, 0.95f, false}, {"damping", 0.995f, 0.9999f, false},
        });
        add("blow", false, {
            {"freq", 55, 1200, true}, {"mouth_pressure", 0.1f, 1.0f, false},
            {"reed_stiffness", 0.0f, 1.0f, false}, {"breath_noise", 0.0f, 0.6f, false},
            {"brightness", 0.1f, 0.95f, false}, {"damping", 0.99f, 0.9999f, false},
        });
        add("strike", true, {
            {"freq", 65, 2400, true}, {"impact_velocity", 0.1f, 1.0f, false},
            {"hammer_mass", 0.001f, 0.1f, false}, {"hammer_stiffness", 1e5f, 1e9f, true},
            {"nonlinearity", 1.0f, 2.5f, false}, {"material", 0.0f, 1.0f, false},
            {"size_damping", 0.0f, 1.0f, false},
        });
        add("pluck", true, {
            {"freq", 55, 1800, true}, {"pluck_position", 0.01f, 0.5f, false},
            {"pluck_hardness", 0.0f, 1.0f, false}, {"decay_time", 0.2f, 4.0f, false},
            {"dispersion", 0.0f, 1.0f, false},
        });
        add("shaker", false, {
            {"freq", 200, 4000, true}, {"n_particles", 5, 500, true},
            {"energy", 0.1f, 1.0f, false}, {"decay_time", 0.1f, 2.0f, false},
            {"material", 0.0f, 1.0f, false},
        });
        add("noise", false, {
            {"color", -1.0f, 1.0f, false}, {"density", 0.1f, 1.0f, true},
            {"correlation", 0.0f, 0.95f, false}, {"freq", 50, 2000, true},
            {"tone_amount", 0.0f, 1.0f, false}, {"tone_q", 0.0f, 1.0f, false},
        });
        add("chaos", false, {
            {"freq", 60, 1200, true}, {"bifurcation", 0.0f, 1.0f, false},
            {"coupling_rate", 20, 2000, true}, {"x0", 0.05f, 0.95f, false},
            {"brightness", 0.1f, 0.95f, false}, {"damping", 0.99f, 0.9999f, false},
        });
        add("mechanical", false, {
            {"freq", 80, 4500, true}, {"rate", 2, 200, true},
            {"jitter", 0.0f, 1.0f, false}, {"air_mix", 0.0f, 1.0f, false},
            {"air_color", -0.9f, 0.9f, false}, {"load_mod", 0.0f, 0.5f, false},
            {"tone_mix", 0.0f, 1.0f, false},
        });
        add("bird", false, {
            {"freq", 400, 3500, true}, {"alpha0", 0.05f, 1.5f, false},
            {"beta", 0.2f, 1.8f, false}, {"eps", 0.0f, 0.6f, false},
            {"tau_d", 1.0f, 12.0f, false}, {"duty", 0.1f, 1.0f, false},
            {"rho_f", 1.0f, 2.5f, false}, {"gate_rate", 2.0f, 25.0f, true},
        });
        add("vocal", false, {
            {"freq", 70, 900, true}, {"ps", 100, 4000, true},
            {"gap", -0.1f, 1.5f, false}, {"fold_q", 2.0f, 20.0f, true},
            {"kc_scale", 0.3f, 3.0f, true}, {"jaw", 0.0f, 1.0f, false},
            {"tongue", 0.0f, 1.0f, false}, {"f3", 1500.0f, 3500.0f, false},
            {"tilt", 0.0f, 1.0f, false},
        });

        // ---- risonatore, un agente per topologia (resonator.py PARAM_RANGES, 7 forme) ----
        add("resonator_bar", true, {
            {"size", 0.05f, 1.5f, true}, {"thickness", 0.005f, 0.05f, true},
            {"density", 400, 8000, true}, {"stiffness", 5e9f, 2.1e11f, true},
            {"loss", 0.0005f, 0.05f, true}, {"mode_falloff", 0.0f, 1.0f, false},
        });
        add("resonator_plate_rect", true, {
            {"size", 0.05f, 1.0f, true}, {"aspect", 0.3f, 3.0f, false},
            {"thickness", 0.0005f, 0.02f, true}, {"density", 400, 8000, true},
            {"stiffness", 5e9f, 2.1e11f, true}, {"loss", 0.0005f, 0.05f, true},
            {"mode_falloff", 0.0f, 1.0f, false},
        });
        add("resonator_plate_circ", true, {
            {"size", 0.03f, 0.6f, true}, {"thickness", 0.0005f, 0.02f, true},
            {"density", 400, 8000, true}, {"stiffness", 5e9f, 2.1e11f, true},
            {"loss", 0.0005f, 0.05f, true}, {"mode_falloff", 0.0f, 1.0f, false},
        });
        add("resonator_membrane", true, {
            {"size", 0.05f, 0.6f, true}, {"thickness", 0.00005f, 0.002f, true},
            {"density", 200, 2000, true}, {"stiffness", 500, 8000, true},
            {"loss", 0.0005f, 0.05f, true}, {"mode_falloff", 0.0f, 1.0f, false},
        });
        add("resonator_tube", true, {
            {"size", 0.05f, 3.0f, true}, {"radius", 0.004f, 0.06f, true},
            {"closure", 0.0f, 1.0f, false}, {"flare", 0.0f, 1.0f, false},
            {"loss", 0.0005f, 0.05f, true}, {"mode_falloff", 0.0f, 1.0f, false},
        });
        add("resonator_soundboard", true, {
            {"size", 0.08f, 0.6f, true}, {"aspect", 0.3f, 1.0f, false},
            {"thickness", 0.001f, 0.02f, true}, {"stiffness", 5e9f, 8e10f, true},
            {"ortho", 1.0f, 30.0f, true}, {"cavity", 0.0005f, 0.01f, true},
            {"hole", 0.01f, 0.06f, true}, {"loss", 0.0005f, 0.05f, true},
            {"mode_falloff", 0.0f, 1.0f, false},
        });
        add("resonator_chaotic", true, {
            {"size", 0.04f, 0.25f, true}, {"thickness", 0.002f, 0.03f, true},
            {"stiffness", 5e9f, 2.1e11f, true}, {"loss", 0.0005f, 0.05f, true},
            {"mode_falloff", 0.0f, 1.0f, false}, {"nonlin", 0.0f, 0.5f, false},
            {"beat", 0.5f, 25.0f, true}, {"depth", 0.0f, 1.0f, false},
            {"chaos", 0.0f, 1.0f, false}, {"speed", 0.5f, 50.0f, true},
        });

        // agents.py PITCH_LOCKED_EXCITERS (2026-09-24): tutti gli eccitatori tranne
        // shaker/noise -- freq imposta a runtime (lockFreq), mai predetta dal modello.
        for (const char* n : {"bow", "blow", "strike", "pluck", "chaos", "mechanical", "bird", "vocal"})
            m.at(n).pitchLocked = true;

        return m;
    }();
    return specs;
}

} // namespace phimo
