#pragma once
// Engine (2026-09-28, v5): carica i 10 surrogati (native/data/surrogate/<eccitatore>.bin, Surrogate.hpp) al posto
// dei 17 agenti MDN + corpus KNN (deciso con l'utente: solo rete v5 + ricerca reale, nessun ripiego).
// Verifica per nome che i parametri dell'eccitatore nel file esistano in agentSpecs() (ordine di generateExciter).
#include <map>
#include <string>

#include "ParamRanges.hpp"
#include "Surrogate.hpp"

namespace phimo {

class Engine {
public:
    bool loadAll(const std::string& surrogateDir, std::string* error = nullptr) {
        for (const auto& kv : agentSpecs()) {
            const std::string& name = kv.first;
            if (name.rfind("resonator_", 0) == 0) continue;
            SurrogateModel m;
            std::string err;
            if (!m.load(surrogateDir + "/" + name + ".bin", err)) {
                if (error) *error = err;
                return false;
            }
            for (const SgParam& p : m.xp) {
                bool found = false;
                for (const auto& ps : kv.second.params) found |= (p.name == ps.name);
                if (!found) {
                    if (error) *error = "parametro " + name + "." + p.name + " assente in agentSpecs";
                    return false;
                }
            }
            models_.emplace(name, std::move(m));
        }
        return true;
    }

    const SurrogateModel* surrogate(const std::string& exc) const {
        auto it = models_.find(exc);
        return it != models_.end() ? &it->second : nullptr;
    }

private:
    std::map<std::string, SurrogateModel> models_;
};

} // namespace phimo
