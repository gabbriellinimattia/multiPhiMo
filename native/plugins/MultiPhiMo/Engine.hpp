#pragma once
// Stato del motore runtime (punto 7 fase 2, sotto-punto a): aggrega i pezzi validati ai
// punti 2-6 (MdnAgent per i 17 agenti, KnnCorpusStore per il corpus) in un unico oggetto
// caricato una volta all'avvio del plugin. Qui SOLO il caricamento -- worker thread,
// routing one-shot/ibrido, render, mixer e note-on/off arrivano nei prossimi round (vedi
// claude/vst3_native_stato.md, punto 7, per l'elenco concordato).
#include <map>
#include <string>
#include <utility>

#include "ParamRanges.hpp"
#include "MdnAgent.hpp"
#include "KnnCorpus.hpp"

namespace phimo {

class Engine {
public:
    // weightsDir: cartella con <agente>.bin (native/data/weights/, export_weights.py).
    // knnPath: native/data/knn_corpus.bin (build_knn_corpus.py). Carica TUTTI i 17
    // agenti elencati da agentSpecs() (10 eccitatori + 7 resonator_*) + il corpus KNN --
    // fallisce (ritorna false, *error valorizzato) se anche uno solo manca o le
    // dimensioni non combaciano con ParamRanges.hpp, mai un caricamento parziale
    // silenzioso (stesso principio gia' usato in KnnCorpusStore::load).
    bool loadAll(const std::string& weightsDir, const std::string& knnPath, std::string* error = nullptr) {
        for (const auto& kv : agentSpecs()) {
            const std::string& name = kv.first;
            const AgentSpec& spec = kv.second;
            MdnAgent agent;
            const std::string path = weightsDir + "/" + name + ".bin";
            if (!agent.loadFromFile(path)) {
                if (error) *error = "impossibile caricare " + path;
                return false;
            }
            if (agent.nIn() != spec.descriptorIndices.size() || agent.nOut() != modelParamCount(spec)) {
                if (error) *error = "mismatch dimensioni per " + name + ": bin nIn=" +
                                     std::to_string(agent.nIn()) + " nOut=" + std::to_string(agent.nOut()) +
                                     ", attesi in=" + std::to_string(spec.descriptorIndices.size()) +
                                     " out=" + std::to_string(modelParamCount(spec));
                return false;
            }
            agents_.emplace(name, std::move(agent));
        }
        return knn_.load(knnPath, error);
    }

    const MdnAgent& agent(const std::string& name) const { return agents_.at(name); }
    const KnnCorpusStore& knn() const { return knn_; }

private:
    std::map<std::string, MdnAgent> agents_;
    KnnCorpusStore knn_;
};

} // namespace phimo
