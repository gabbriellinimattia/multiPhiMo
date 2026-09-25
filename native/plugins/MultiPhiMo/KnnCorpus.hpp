#pragma once
// Loader per native/data/knn_corpus.bin (punto 6 fase 2) -- porting 1:1 di knn_corpus.py:
// KnnJointLockedCorpus (routing JOINT_LOCKED_EXCITERS: pluck/chaos/bird/mechanical, righe
// filtrate per forma risonatore accoppiata) e KnnCorpus (ibrido resonator_*, mono-agente,
// tutte e 7 le forme). Formato binario: vedi docstring di native/tools/build_knn_corpus.py.
//
// Statistiche z-score (mean/std, NaN ignorati) calcolate QUI a runtime dai dati grezzi, MAI
// salvate nel binario (decisione punto 6, 2026-09-22) -- replica agents.py::_nan_safe_stats
// (nanmean/nanstd + EPS=1e-8), applicate poi a TUTTO il corpus prima della ricerca (non solo
// alla query), esattamente come knn_corpus.py fa con cKDTree.
//
// IMPORTANTE: a differenza di MdnAgent (vedi buildDescriptorVector in MdnAgent.hpp), QUI
// NON si applica log10 ad attack_time/decay_time -- knn_corpus.py costruisce il vettore per
// il KDTree direttamente da np.array([[float(r[k]) for k in desc_keys] ...]), senza il
// _log_transform_descriptors usato solo per l'input della MDN. Il chiamante deve passare
// target[] con i valori GREZZI dei descrittori (stessi che escono da Descriptors.hpp).
//
// Non ancora agganciato a run() (routing JOINT_LOCKED_EXCITERS/fallback MDN rimandato al
// punto 7/8, quando esistono gia' gli oggetti MdnAgent e il thread di refinement).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "ParamRanges.hpp"

namespace phimo {

struct KnnSection {
    uint8_t type = 0;  // 0 = joint (eccitatore+forma accoppiata), 1 = resonator ibrido
    std::string agentName, shapeName;
    std::vector<std::string> descNames, excParamNames, resParamNames;
    std::vector<int> descIndices;   // descNames[i] -> indice in kDescriptorKeys
    int rowCount = 0;
    std::vector<float> raw;         // rowCount x ncols(), grezzo, NaN preservato com'e' nel CSV
    std::vector<float> mean, stdv;  // solo colonne descrittore, len = descNames.size()
    std::vector<float> normDesc;    // rowCount x descNames.size(), z-score + nan_to_num(0)

    int descCount() const { return (int)descNames.size(); }
    int excCount() const { return (int)excParamNames.size(); }
    int resCount() const { return (int)resParamNames.size(); }
    int ncols() const { return descCount() + excCount() + resCount(); }
};

class KnnCorpusStore {
public:
    // Carica il blob. Ritorna false e valorizza *error su qualunque problema (file
    // mancante, magic/versione sbagliati, nomi che non combaciano con ParamRanges.hpp) --
    // mai un caricamento parziale silenzioso: un disallineamento futuro (es. un parametro
    // rinominato in agents.py/resonator.py senza rigenerare il binario) fa fallire load(),
    // non disallinea le colonne senza avviso.
    bool load(const std::string& path, std::string* error = nullptr);

    bool hasJoint(const std::string& exciterName, const std::string& shape) const {
        return joint_.count(exciterName + "+" + shape) != 0;
    }
    bool hasResonator(const std::string& shape) const {
        return resonator_.count(shape) != 0;
    }

    // target indicizzato come phimo::kDescriptorKeys (15 valori grezzi, quelli non
    // rilevanti per questa sezione sono ignorati). Ritorna false se (exciterName, shape)
    // non e' nel corpus (nessuna riga con quella forma nel dataset, o combinazione con
    // meno di 10 righe scartata a monte da build_knn_corpus.py).
    bool queryJoint(const std::string& exciterName, const std::string& shape,
                     const float target[15],
                     std::vector<float>& outExciterParams,
                     std::vector<float>& outResonatorParams) const;

    // Ritorna TUTTI i parametri propri della forma, inclusa `loss` GREZZA dal corpus KNN --
    // il chiamante a runtime deve sovrascrivere l'indice di `loss` col valore predetto da
    // MdnAgent prima di sintetizzare (stesso schema di param_candidate._mdn_hybrid_route:
    // "resonator_params['loss'] = mdn_res_params['loss']"), mai usare quello del KNN diretto.
    bool queryResonator(const std::string& shape, const float target[15],
                         std::vector<float>& outResonatorParams) const;

private:
    std::map<std::string, KnnSection> joint_;      // key = exciterName + "+" + shape
    std::map<std::string, KnnSection> resonator_;  // key = shape

    static int nearestRow(const KnnSection& s, const float target[15]);
};

namespace knn_detail {

inline bool readBytes(std::ifstream& f, void* dst, size_t n) {
    if (n == 0) return true;
    f.read(reinterpret_cast<char*>(dst), (std::streamsize)n);
    return (bool)f;
}
inline bool readU8(std::ifstream& f, uint8_t& v) { return readBytes(f, &v, 1); }
inline bool readU16(std::ifstream& f, uint16_t& v) { return readBytes(f, &v, 2); }
inline bool readU32(std::ifstream& f, uint32_t& v) { return readBytes(f, &v, 4); }

inline bool readString(std::ifstream& f, std::string& s) {
    uint16_t len = 0;
    if (!readU16(f, len)) return false;
    s.resize(len);
    if (len == 0) return true;
    return readBytes(f, &s[0], len);
}

inline bool readNames(std::ifstream& f, std::vector<std::string>& names) {
    uint16_t count = 0;
    if (!readU16(f, count)) return false;
    names.resize(count);
    for (auto& n : names)
        if (!readString(f, n)) return false;
    return true;
}

// Indice di `name` in phimo::kDescriptorKeys, -1 se assente.
inline int descriptorIndexOf(const std::string& name) {
    for (int i = 0; i < (int)kDescriptorKeys.size(); ++i)
        if (name == kDescriptorKeys[i]) return i;
    return -1;
}

// Confronta i nomi letti dal binario con i ParamSpec attesi (stesso ordine) --
// disallineamento = load() fallisce, non silenzioso.
inline bool namesMatch(const std::vector<std::string>& a, const std::vector<ParamSpec>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i].name) return false;
    return true;
}

} // namespace knn_detail

inline bool KnnCorpusStore::load(const std::string& path, std::string* error) {
    auto fail = [&](const std::string& msg) {
        if (error) *error = msg;
        return false;
    };

    std::ifstream f(path, std::ios::binary);
    if (!f) return fail("impossibile aprire " + path);

    char magic[4];
    if (!knn_detail::readBytes(f, magic, 4) || std::memcmp(magic, "PHKC", 4) != 0)
        return fail("magic non valido in " + path);

    uint32_t version = 0, sectionCount = 0;
    if (!knn_detail::readU32(f, version)) return fail("header troncato (version)");
    if (version != 1) return fail("versione non supportata: " + std::to_string(version));
    if (!knn_detail::readU32(f, sectionCount)) return fail("header troncato (section_count)");

    const auto& specs = agentSpecs();

    for (uint32_t s = 0; s < sectionCount; ++s) {
        KnnSection sec;
        uint8_t type = 0;
        if (!knn_detail::readU8(f, type)) return fail("sezione " + std::to_string(s) + ": type troncato");
        sec.type = type;
        if (!knn_detail::readString(f, sec.agentName)) return fail("sezione " + std::to_string(s) + ": agent_name troncato");
        if (!knn_detail::readString(f, sec.shapeName)) return fail("sezione " + std::to_string(s) + ": shape_name troncato");
        if (!knn_detail::readNames(f, sec.descNames)) return fail("sezione " + std::to_string(s) + ": desc_names troncato");
        if (!knn_detail::readNames(f, sec.excParamNames)) return fail("sezione " + std::to_string(s) + ": exc_param_names troncato");
        if (!knn_detail::readNames(f, sec.resParamNames)) return fail("sezione " + std::to_string(s) + ": res_param_names troncato");

        uint32_t rowCount = 0;
        if (!knn_detail::readU32(f, rowCount)) return fail("sezione " + std::to_string(s) + ": row_count troncato");
        sec.rowCount = (int)rowCount;

        for (auto& n : sec.descNames)
            if (knn_detail::descriptorIndexOf(n) < 0)
                return fail("sezione " + sec.agentName + "+" + sec.shapeName + ": descrittore sconosciuto '" + n + "'");
        sec.descIndices.reserve(sec.descNames.size());
        for (auto& n : sec.descNames) sec.descIndices.push_back(knn_detail::descriptorIndexOf(n));

        if (sec.type == 0) {
            auto itExc = specs.find(sec.agentName);
            if (itExc == specs.end()) return fail("eccitatore sconosciuto: " + sec.agentName);
            // 2026-09-24: pitch-lock -- il corpus non contiene piu' "freq" per i 4 eccitatori
            // joint-locked (tutti pitch-locked: bird/chaos/mechanical/pluck), vedi ParamRanges.hpp.
            if (!knn_detail::namesMatch(sec.excParamNames, modelParams(itExc->second)))
                return fail("parametri eccitatore disallineati per " + sec.agentName);
            auto itRes = specs.find("resonator_" + sec.shapeName);
            if (itRes == specs.end()) return fail("forma risonatore sconosciuta: " + sec.shapeName);
            if (!knn_detail::namesMatch(sec.resParamNames, itRes->second.params))
                return fail("parametri risonatore disallineati per resonator_" + sec.shapeName);
        } else {
            if (!sec.excParamNames.empty())
                return fail("sezione resonator con parametri eccitatore non vuoti: " + sec.agentName);
            auto itRes = specs.find(sec.agentName);
            if (itRes == specs.end()) return fail("agente risonatore sconosciuto: " + sec.agentName);
            if (!knn_detail::namesMatch(sec.resParamNames, itRes->second.params))
                return fail("parametri disallineati per " + sec.agentName);
        }

        const int ncols = sec.ncols();
        sec.raw.resize((size_t)sec.rowCount * ncols);
        if (!knn_detail::readBytes(f, sec.raw.data(), sec.raw.size() * sizeof(float)))
            return fail("sezione " + sec.agentName + "+" + sec.shapeName + ": dati troncati");

        // mean/std sulle sole colonne descrittore, NaN ignorati (agents.py _nan_safe_stats).
        // Accumulo in double (piu' preciso dell'accumulo float32 di numpy) -- se la
        // validazione mostrasse un vicino diverso da Python su una combinazione al
        // limite, e' il primo sospetto da controllare (vedi nota in vst3_native_stato.md).
        const int dc = sec.descCount();
        sec.mean.assign(dc, 0.0f);
        sec.stdv.assign(dc, 0.0f);
        for (int c = 0; c < dc; ++c) {
            double sum = 0.0;
            int n = 0;
            for (int r = 0; r < sec.rowCount; ++r) {
                float v = sec.raw[(size_t)r * ncols + c];
                if (!std::isnan(v)) { sum += v; ++n; }
            }
            double m = (n > 0) ? sum / n : 0.0;
            double var = 0.0;
            for (int r = 0; r < sec.rowCount; ++r) {
                float v = sec.raw[(size_t)r * ncols + c];
                if (!std::isnan(v)) { double d = v - m; var += d * d; }
            }
            double sd = (n > 0) ? std::sqrt(var / n) : 0.0;
            sec.mean[c] = (float)m;
            sec.stdv[c] = (float)sd + 1e-8f;  // EPS come agents.py
        }

        // Righe normalizzate (z-score + nan_to_num(0)) precalcolate una volta, stessa
        // trasformazione applicata all'INTERO corpus da knn_corpus.py prima di costruire
        // l'albero (non solo alla query) -- qui e' l'array su cui gira la ricerca lineare.
        sec.normDesc.resize((size_t)sec.rowCount * dc);
        for (int r = 0; r < sec.rowCount; ++r) {
            for (int c = 0; c < dc; ++c) {
                float v = sec.raw[(size_t)r * ncols + c];
                float z = (v - sec.mean[c]) / sec.stdv[c];
                sec.normDesc[(size_t)r * dc + c] = std::isnan(z) ? 0.0f : z;
            }
        }

        if (sec.type == 0)
            joint_[sec.agentName + "+" + sec.shapeName] = std::move(sec);
        else
            resonator_[sec.shapeName] = std::move(sec);
    }

    return true;
}

inline int KnnCorpusStore::nearestRow(const KnnSection& s, const float target[15]) {
    const int dc = s.descCount();
    std::vector<float> xn(dc);
    for (int i = 0; i < dc; ++i) {
        float v = target[s.descIndices[i]];
        float z = (v - s.mean[i]) / s.stdv[i];
        xn[i] = std::isnan(z) ? 0.0f : z;
    }
    int best = -1;
    double bestDist = std::numeric_limits<double>::infinity();
    for (int r = 0; r < s.rowCount; ++r) {
        double dist = 0.0;
        const float* row = &s.normDesc[(size_t)r * dc];
        for (int i = 0; i < dc; ++i) {
            double d = (double)row[i] - (double)xn[i];
            dist += d * d;
        }
        if (dist < bestDist) { bestDist = dist; best = r; }
    }
    return best;
}

inline bool KnnCorpusStore::queryJoint(const std::string& exciterName, const std::string& shape,
                                        const float target[15],
                                        std::vector<float>& outExciterParams,
                                        std::vector<float>& outResonatorParams) const {
    auto it = joint_.find(exciterName + "+" + shape);
    if (it == joint_.end()) return false;
    const KnnSection& s = it->second;
    int row = nearestRow(s, target);
    if (row < 0) return false;
    const int ncols = s.ncols();
    const float* r = &s.raw[(size_t)row * ncols];
    const int dc = s.descCount(), ec = s.excCount(), rc = s.resCount();
    outExciterParams.assign(r + dc, r + dc + ec);
    outResonatorParams.assign(r + dc + ec, r + dc + ec + rc);
    return true;
}

inline bool KnnCorpusStore::queryResonator(const std::string& shape, const float target[15],
                                            std::vector<float>& outResonatorParams) const {
    auto it = resonator_.find(shape);
    if (it == resonator_.end()) return false;
    const KnnSection& s = it->second;
    int row = nearestRow(s, target);
    if (row < 0) return false;
    const int ncols = s.ncols();
    const float* r = &s.raw[(size_t)row * ncols];
    const int dc = s.descCount();
    outResonatorParams.assign(r + dc, r + dc + s.resCount());
    return true;
}

} // namespace phimo
