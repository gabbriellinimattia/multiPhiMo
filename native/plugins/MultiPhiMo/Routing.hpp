#pragma once
// Routing one-shot/ibrido (punto 7 fase 2, sotto-punto b) -- porting 1:1 di
// param_candidate.py: JOINT_LOCKED_EXCITERS instrada su KNN congiunto vincolato (con
// fallback su MDN+ibrido se la combinazione manca/eccezione), tutti gli altri eccitatori
// SEMPRE su MDN+ibrido (_mdn_hybrid_route: eccitatore da MDN, risonatore ibrido = loss
// dalla MDN + resto dal KNN mono-agente).
//
// IMPORTANTE (voluto, non un bug): MdnAgent::predict campiona la mistura con
// temperature=1.0 di default (stocastico) -- stessa scelta di agents.py Agent.predict,
// coerente con le istruzioni di progetto ("i suoni generati non devono fare timbral
// matching... va bene che configurazioni simili diano suoni diversi"). Quindi
// exciter_params e il `loss` del risonatore NON sono riproducibili bit-a-bit contro
// Python nemmeno a parita' di input/seed equivalente -- solo la parte KNN (query
// congiunta, e la parte non-loss dell'ibrido) e' deterministica e validabile a match
// esatto, come gia' fatto al punto 6. Vedi tests/test_routing.cpp per cosa e come si
// valida qui.
//
// SPSA (raffinamento iterativo) volutamente NON incluso in questo punto -- rimandato,
// vedi claude/vst3_native_stato.md punto 7 (deciso con l'utente).
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "ParamRanges.hpp"
#include "Engine.hpp"

namespace phimo {

struct Candidate {
    std::string exciterName, resonatorShape;
    std::vector<float> exciterParams;    // ordine = agentSpecs()[exciterName].params
    std::vector<float> resonatorParams;  // ordine = agentSpecs()["resonator_"+shape].params (incl. loss)
    bool usedKnnJoint = false;           // solo diagnostica/log, non usato dal chiamante
};

// param_candidate.JOINT_LOCKED_EXCITERS (2026-09-21, vedi vst3_native_stato.md punto 6) --
// deve restare sincronizzato con param_candidate.py e con native/tools/build_knn_corpus.py.
inline const std::set<std::string>& jointLockedExciters() {
    static const std::set<std::string> s = {"bird", "chaos", "mechanical", "pluck"};
    return s;
}

namespace routing_detail {

inline std::map<std::string, float> target15ToMap(const float target[15]) {
    std::map<std::string, float> m;
    for (int i = 0; i < 15; ++i) m[kDescriptorKeys[i]] = target[i];
    return m;
}

inline int paramIndexOf(const AgentSpec& spec, const char* name) {
    for (size_t i = 0; i < spec.params.size(); ++i)
        if (spec.params[i].name == std::string(name)) return (int)i;
    return -1;
}

} // namespace routing_detail

// param_candidate._mdn_hybrid_route: eccitatore da MDN one-shot, risonatore ibrido
// (loss dalla MDN del risonatore, resto dal KNN mono-agente sul corpus della forma).
inline Candidate mdnHybridRoute(const Engine& engine, const std::string& exciterName,
                                 const std::string& resonatorShape, const float target[15],
                                 std::mt19937& rng) {
    const auto& specs = agentSpecs();
    const std::string resonatorAgentName = "resonator_" + resonatorShape;
    auto itExc = specs.find(exciterName);
    auto itRes = specs.find(resonatorAgentName);
    if (itExc == specs.end()) throw std::runtime_error("eccitatore sconosciuto: " + exciterName);
    if (itRes == specs.end()) throw std::runtime_error("forma risonatore sconosciuta: " + resonatorShape);
    const AgentSpec& excSpec = itExc->second;
    const AgentSpec& resSpec = itRes->second;

    const auto descMap = routing_detail::target15ToMap(target);

    Candidate c;
    c.exciterName = exciterName;
    c.resonatorShape = resonatorShape;

    // 2026-09-24 pitch-lock: la MDN predice SOLO i parametri modello (freq esclusa se
    // pitchLocked, vedi ParamRanges.hpp modelParams/expandModelParams/lockFreq) -- passare
    // excSpec.params interi qui denormalizzerebbe l'uscita con i ParamSpec sbagliati
    // (offset di uno, corruzione silenziosa, non un crash).
    const auto excX = buildDescriptorVector(excSpec, descMap);
    const std::vector<float> mdnExcParams =
        engine.agent(exciterName).predict(excX, modelParams(excSpec), rng, 1.0f);
    c.exciterParams = expandModelParams(excSpec, mdnExcParams);
    lockFreq(excSpec, c.exciterParams, target[kPitchIdx]);

    const auto resX = buildDescriptorVector(resSpec, descMap);
    const std::vector<float> mdnResParams =
        engine.agent(resonatorAgentName).predict(resX, resSpec.params, rng, 1.0f);

    std::vector<float> knnResParams;
    if (!engine.knn().queryResonator(resonatorShape, target, knnResParams))
        throw std::runtime_error("queryResonator fallita per " + resonatorShape);

    const int lossIdx = routing_detail::paramIndexOf(resSpec, "loss");
    if (lossIdx < 0) throw std::runtime_error("'loss' assente da resonator_" + resonatorShape);
    knnResParams[lossIdx] = mdnResParams[lossIdx];  // ibrido: loss dalla MDN, resto dal KNN

    c.resonatorParams = std::move(knnResParams);
    return c;
}

// param_candidate._update_candidate: JOINT_LOCKED_EXCITERS -> KNN congiunto vincolato
// alla forma, fallback su mdnHybridRoute se la combinazione manca dal corpus o la query
// solleva un'eccezione -- MAI un'eccezione che si propaga al chiamante (stesso principio
// difensivo del worker Python: il ciclo di raffinamento non si deve mai fermare).
inline Candidate routeCandidate(const Engine& engine, const std::string& exciterName,
                                 const std::string& resonatorShape, const float target[15],
                                 std::mt19937& rng) {
    if (jointLockedExciters().count(exciterName)) {
        try {
            std::vector<float> excParams, resParams;
            if (engine.knn().queryJoint(exciterName, resonatorShape, target, excParams, resParams)) {
                // 2026-09-24 pitch-lock: excParams dal corpus e' gia' "modello" (freq esclusa,
                // i 4 JOINT_LOCKED_EXCITERS sono tutti pitch-locked) -- stessa espansione+lock
                // del ramo MDN sopra.
                const AgentSpec& excSpec = agentSpecs().at(exciterName);
                Candidate c;
                c.exciterName = exciterName;
                c.resonatorShape = resonatorShape;
                c.exciterParams = expandModelParams(excSpec, excParams);
                lockFreq(excSpec, c.exciterParams, target[kPitchIdx]);
                c.resonatorParams = std::move(resParams);
                c.usedKnnJoint = true;
                return c;
            }
        } catch (const std::exception&) {
            // fallthrough sul fallback sotto, stesso schema del try/except Python
        }
    }
    return mdnHybridRoute(engine, exciterName, resonatorShape, target, rng);
}

} // namespace phimo
