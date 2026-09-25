#pragma once
// Forward pass della MDN (porting di agents.py: MDN.forward, Agent._desc_to_x,
// Agent._y_to_params, _sample_mixture) da un binario esportato da
// native/tools/export_weights.py. "Stessi numeri" e' garantito per pi/mu/sigma
// (stessa matematica, stessi pesi) -- NON per il campione stocastico finale: qui si usa
// std::mt19937/std::normal_distribution invece del Generator numpy di agents.py, per
// design del progetto non serve un RNG identico (descrittori identici possono dare
// configurazioni diverse). Con temperature<=0 il campionamento e' deterministico
// (media della componente piu' probabile, nessun rumore) ed e' allora comparabile 1:1
// con `python agents.py predict <agent> ... --temperature 0`.
// Richiede C++17.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "ParamRanges.hpp"

namespace phimo {

class MdnAgent {
public:
    bool loadFromFile(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        char magic[4];
        bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "PHMD", 4) == 0;
        uint32_t version = 0, nIn = 0, nOut = 0, hidden = 0, k = 0;
        if (ok) ok = std::fread(&version, 4, 1, f) == 1 && version == 1;
        if (ok) ok = std::fread(&nIn, 4, 1, f) == 1 && std::fread(&nOut, 4, 1, f) == 1 &&
                     std::fread(&hidden, 4, 1, f) == 1 && std::fread(&k, 4, 1, f) == 1;
        if (!ok) { std::fclose(f); return false; }
        fNIn = nIn; fNOut = nOut; fHidden = hidden; fK = k;

        auto readArr = [&](std::vector<float>& v, size_t count) -> bool {
            v.resize(count);
            return std::fread(v.data(), sizeof(float), count, f) == count;
        };
        ok = readArr(fDescMean, nIn) && readArr(fDescStd, nIn)
          && readArr(fW0, (size_t)hidden * nIn) && readArr(fB0, hidden)
          && readArr(fW2, (size_t)hidden * hidden) && readArr(fB2, hidden)
          && readArr(fWPi, (size_t)k * hidden) && readArr(fBPi, k)
          && readArr(fWMu, (size_t)k * nOut * hidden) && readArr(fBMu, (size_t)k * nOut)
          && readArr(fWSigma, (size_t)k * nOut * hidden) && readArr(fBSigma, (size_t)k * nOut);
        std::fclose(f);
        fLoaded = ok;
        return ok;
    }

    bool isLoaded() const { return fLoaded; }
    uint32_t nIn() const { return fNIn; }
    uint32_t nOut() const { return fNOut; }
    uint32_t k() const { return fK; }

    // x: vettore di ingresso gia' log-trasformato (vedi buildDescriptorVector), NON ancora
    // normalizzato -- la normalizzazione z-score (desc_mean/desc_std) avviene qui dentro,
    // come in agents.py Agent._desc_to_x. paramSpecs: nell'ordine di AgentSpec::params
    // (= ordine di uscita del modello). Ritorna i parametri denormalizzati
    // (agents.py Agent._y_to_params).
    std::vector<float> predict(const std::vector<float>& x,
                                const std::vector<ParamSpec>& paramSpecs,
                                std::mt19937& rng, float temperature = 1.0f) const {
        std::vector<float> xn(fNIn);
        for (uint32_t i = 0; i < fNIn; ++i) {
            float d = (x[i] - fDescMean[i]) / (fDescStd[i] + kEps);
            xn[i] = std::isfinite(d) ? d : 0.0f;  // agents.py: np.nan_to_num DOPO la normalizzazione
        }

        std::vector<float> h0(fHidden), h1(fHidden);
        linear(fW0, fB0, xn.data(), fNIn, fHidden, h0.data());
        for (auto& v : h0) v = std::tanh(v);
        linear(fW2, fB2, h0.data(), fHidden, fHidden, h1.data());
        for (auto& v : h1) v = std::tanh(v);

        std::vector<float> piLogits(fK);
        linear(fWPi, fBPi, h1.data(), fHidden, fK, piLogits.data());
        std::vector<float> pi = softmax(piLogits);

        std::vector<float> mu(fK * fNOut), logSigma(fK * fNOut);
        linear(fWMu, fBMu, h1.data(), fHidden, fK * fNOut, mu.data());
        linear(fWSigma, fBSigma, h1.data(), fHidden, fK * fNOut, logSigma.data());
        std::vector<float> sigma(fK * fNOut);
        for (size_t i = 0; i < sigma.size(); ++i)
            sigma[i] = std::exp(std::min(std::max(logSigma[i], -6.0f), 2.0f));  // clamp(-6,2) poi exp

        std::vector<float> y01 = sampleMixture(pi, mu, sigma, rng, temperature);

        // agents.py Agent._eff_ranges: per i parametri LOG_PARAMS il range effettivo usato
        // per l'interpolazione e' gia' in log10 (log10(lo)..log10(hi)), poi si fa 10**v --
        // NON interpolare in spazio lineare e poi elevare a potenza (darebbe inf/valori assurdi).
        std::vector<float> params(fNOut);
        for (uint32_t i = 0; i < fNOut; ++i) {
            const ParamSpec& ps = paramSpecs[i];
            float lo = ps.logScale ? std::log10(ps.lo) : ps.lo;
            float hi = ps.logScale ? std::log10(ps.hi) : ps.hi;
            float v = std::min(std::max(y01[i], 0.0f), 1.0f) * (hi - lo) + lo;
            params[i] = ps.logScale ? std::pow(10.0f, v) : v;
        }
        return params;
    }

private:
    static constexpr float kEps = 1e-8f;

    // W: [outDim, inDim] row-major, layout nativo di PyTorch nn.Linear.weight. y = W*x + b.
    static void linear(const std::vector<float>& W, const std::vector<float>& b,
                        const float* x, uint32_t inDim, uint32_t outDim, float* y) {
        for (uint32_t o = 0; o < outDim; ++o) {
            float acc = b[o];
            const float* row = &W[(size_t)o * inDim];
            for (uint32_t i = 0; i < inDim; ++i) acc += row[i] * x[i];
            y[o] = acc;
        }
    }

    static std::vector<float> softmax(const std::vector<float>& logits) {
        float mx = logits[0];
        for (float v : logits) mx = std::max(mx, v);
        std::vector<float> out(logits.size());
        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) { out[i] = std::exp(logits[i] - mx); sum += out[i]; }
        for (auto& v : out) v /= sum;
        return out;
    }

    // agents.py _sample_mixture: componente scelta con prob (pi**(1/temperature),
    // rinormalizzata), poi campione gaussiano m + N(0,1)*s*temperature.
    // temperature<=EPS -> deterministico (argmax pi, media della componente, no rumore).
    std::vector<float> sampleMixture(const std::vector<float>& pi, const std::vector<float>& mu,
                                      const std::vector<float>& sigma, std::mt19937& rng,
                                      float temperature) const {
        uint32_t comp = 0;
        if (temperature <= kEps) {
            float best = pi[0];
            for (uint32_t i = 1; i < fK; ++i) if (pi[i] > best) { best = pi[i]; comp = i; }
        } else {
            std::vector<float> p(pi.begin(), pi.end());
            if (temperature != 1.0f) {
                float sum = 0.0f;
                for (auto& v : p) { v = std::pow(v, 1.0f / temperature); sum += v; }
                for (auto& v : p) v /= sum;
            }
            std::discrete_distribution<uint32_t> dist(p.begin(), p.end());
            comp = dist(rng);
        }
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::vector<float> y(fNOut);
        for (uint32_t i = 0; i < fNOut; ++i) {
            float m = mu[(size_t)comp * fNOut + i];
            float s = sigma[(size_t)comp * fNOut + i] * temperature;
            y[i] = (temperature <= kEps) ? m : m + gauss(rng) * s;
        }
        return y;
    }

    bool fLoaded = false;
    uint32_t fNIn = 0, fNOut = 0, fHidden = 0, fK = 0;
    std::vector<float> fDescMean, fDescStd;
    std::vector<float> fW0, fB0, fW2, fB2, fWPi, fBPi, fWMu, fBMu, fWSigma, fBSigma;
};

// Costruisce il vettore di ingresso (ordine = spec.descriptorIndices), applicando log10
// SOLO alle colonne attack_time/decay_time (agents.py _log_transform_descriptors), PRIMA
// della normalizzazione fatta da MdnAgent::predict. descriptors: mappa nome->valore
// (chiavi = kDescriptorKeys); assente -> 0.0 (come Python descriptors.get(k, 0.0)).
inline std::vector<float> buildDescriptorVector(const AgentSpec& spec,
                                                 const std::map<std::string, float>& descriptors) {
    std::vector<float> x;
    x.reserve(spec.descriptorIndices.size());
    for (int idx : spec.descriptorIndices) {
        const char* key = kDescriptorKeys[idx];
        auto it = descriptors.find(key);
        float v = (it != descriptors.end()) ? it->second : 0.0f;
        if (idx == kAttackTimeIdx || idx == kDecayTimeIdx)
            v = std::log10(std::max(v, 1e-8f));
        x.push_back(v);
    }
    return x;
}

} // namespace phimo
