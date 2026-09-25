#pragma once
// PairSelector.hpp -- selettore "Auto pair" eccitatore+risonatore, porting 1:1 di
// pair_selector.py (PairSelectorKnn, k=30, bias_alpha=0.3). Carica
// native/data/selector_corpus.bin (native/tools/build_selector_corpus.py).
//
// Query: z-score (mean/std con NaN ignorati + EPS, come agents._nan_safe_stats, in
// float32 come numpy) -> k vicini (ricerca lineare, distanza euclidea in double come
// cKDTree) -> MEDIANA per coppia degli score dei vicini (float32, pari = media dei due
// centrali come np.median) -> x bias per eccitatore (freq_vittorie/freq_uniforme)^alpha
// con Laplace -> argmin (primo indice; un NaN vince come in np.argmin).
// Come KnnCorpus: descrittori GREZZI (niente log10). Solo worker/thread messaggi: alloca.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ParamRanges.hpp"

namespace phimo {

namespace sel_detail {
// NaN via bit IEEE-754: immune a -ffast-math (vedi bug B, punto 7d)
inline bool isNanBits(float v) {
    uint32_t u; std::memcpy(&u, &v, 4);
    return (u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0;
}
inline bool rd(std::ifstream& f, void* p, size_t n) {
    if (n == 0) return true;
    f.read(reinterpret_cast<char*>(p), (std::streamsize)n);
    return (bool)f;
}
inline bool rdStr(std::ifstream& f, std::string& s) {
    uint16_t n = 0;
    if (!rd(f, &n, 2)) return false;
    s.resize(n);
    return n == 0 || rd(f, &s[0], n);
}
inline bool rdNames(std::ifstream& f, std::vector<std::string>& v) {
    uint16_t n = 0;
    if (!rd(f, &n, 2)) return false;
    v.resize(n);
    for (auto& s : v) if (!rdStr(f, s)) return false;
    return true;
}
} // namespace sel_detail

struct PairChoice {
    std::string exciter, resonator;  // resonator = forma ("bar", ...), senza prefisso
    float score = 0.0f;              // mediana NON corretta dal bias, come query() Python
    bool ok = false;
};

class PairSelector {
public:
    bool load(const std::string& path, std::string* error = nullptr, int k = 30, double biasAlpha = 0.3) {
        using namespace sel_detail;
        auto fail = [&](const std::string& m) { if (error) *error = m; loaded_ = false; return false; };
        std::ifstream f(path, std::ios::binary);
        if (!f) return fail("impossibile aprire " + path);
        char magic[4];
        uint32_t ver = 0;
        if (!rd(f, magic, 4) || std::memcmp(magic, "PHSC", 4) != 0) return fail("magic errato");
        if (!rd(f, &ver, 4) || ver != 2) return fail("versione non supportata (rigenerare con build_selector_corpus.py)");
        std::vector<std::string> desc;
        if (!rdNames(f, desc) || !rdNames(f, exc_) || !rdNames(f, res_)) return fail("intestazione troncata");
        if (desc.size() != kDescriptorKeys.size()) return fail("numero descrittori diverso da kDescriptorKeys");
        for (size_t i = 0; i < desc.size(); ++i)
            if (desc[i] != kDescriptorKeys[i]) return fail("descrittore fuori ordine: " + desc[i]);
        const auto& specs = agentSpecs();
        for (const auto& e : exc_) if (!specs.count(e)) return fail("eccitatore sconosciuto: " + e);
        for (const auto& r : res_) if (!specs.count("resonator_" + r)) return fail("forma sconosciuta: " + r);
        uint32_t rows = 0;
        if (!rd(f, &rows, 4) || rows == 0) return fail("corpus vuoto");
        rows_ = (int)rows;
        nPairs_ = (int)(exc_.size() * res_.size());
        raw_.resize((size_t)rows_ * kD);
        scores_.resize((size_t)rows_ * nPairs_);
        std::vector<int> counts(exc_.size(), 1);  // Laplace
        for (int r = 0; r < rows_; ++r) {
            uint8_t b = 255, br = 255;
            if (!rd(f, &raw_[(size_t)r * kD], sizeof(float) * kD) ||
                !rd(f, &scores_[(size_t)r * nPairs_], sizeof(float) * nPairs_) || !rd(f, &b, 1) || !rd(f, &br, 1))
                return fail("dati troncati");
            if (b < exc_.size()) counts[b]++;
        }
        excBands_.resize(exc_.size() * kD * 2);
        resBands_.resize(res_.size() * kD * 2);
        if (!rd(f, excBands_.data(), sizeof(float) * excBands_.size()) ||
            !rd(f, resBands_.data(), sizeof(float) * resBands_.size()))
            return fail("fasce troncate");
        // mean/std (NaN ignorati) + EPS, poi corpus normalizzato (nan_to_num)
        for (int c = 0; c < kD; ++c) {
            double s = 0.0; int n = 0;
            for (int r = 0; r < rows_; ++r) { float v = raw_[(size_t)r * kD + c]; if (!isNanBits(v)) { s += v; ++n; } }
            const double m = n ? s / n : 0.0;
            double var = 0.0;
            for (int r = 0; r < rows_; ++r) { float v = raw_[(size_t)r * kD + c]; if (!isNanBits(v)) { double d = v - m; var += d * d; } }
            mean_[c] = (float)m;
            std_[c] = (float)(n ? std::sqrt(var / n) : 0.0) + 1e-8f;
        }
        norm_.resize(raw_.size());
        for (size_t i = 0; i < raw_.size(); ++i) {
            const int c = (int)(i % kD);
            const float z = (raw_[i] - mean_[c]) / std_[c];
            norm_[i] = isNanBits(z) ? 0.0f : z;
        }
        // bias per coppia (ordine PAIRS: eccitatore esterno)
        int total = 0;
        for (int c : counts) total += c;
        const double uniform = 1.0 / exc_.size();
        bias_.resize(nPairs_);
        for (size_t e = 0; e < exc_.size(); ++e) {
            const float w = (float)std::pow((double)counts[e] / total / uniform, biasAlpha);
            for (size_t r = 0; r < res_.size(); ++r) bias_[e * res_.size() + r] = w;
        }
        biasAlpha_ = biasAlpha;
        k_ = std::max(1, std::min(k, rows_));
        loaded_ = true;
        return true;
    }

    bool loaded() const { return loaded_; }

    // "Tabella dei margini" (2026-09-25, richiesta utente): bonus sopra il KNN. Per ogni
    // coppia, frazione dei descrittori del target dentro la fascia dell'eccitatore (e
    // della forma); il costo corretto viene ridotto di bonus*frazione (per eccitatore e per
    // forma, moltiplicativo). 0 = selettore identico a pair_selector.py (default, usato
    // dal test di match esatto).
    void setBandBonus(float b) { bandBonus_ = b; }

    PairChoice query(const float target[15]) const {
        using namespace sel_detail;
        PairChoice out;
        if (!loaded_) return out;
        float xn[kD];
        for (int c = 0; c < kD; ++c) {
            const float z = (target[c] - mean_[c]) / std_[c];
            xn[c] = isNanBits(z) ? 0.0f : z;
        }
        std::vector<std::pair<double, int>> d((size_t)rows_);
        for (int r = 0; r < rows_; ++r) {
            double s = 0.0;
            const float* row = &norm_[(size_t)r * kD];
            for (int c = 0; c < kD; ++c) { const double t = (double)row[c] - (double)xn[c]; s += t * t; }
            d[r] = {s, r};
        }
        std::partial_sort(d.begin(), d.begin() + k_, d.end());
        std::vector<float> col((size_t)k_);
        int best = -1;
        float bestAdj = 0.0f, bestAgg = 0.0f;
        for (int p = 0; p < nPairs_; ++p) {
            bool hasNan = false;
            for (int i = 0; i < k_; ++i) {
                col[i] = scores_[(size_t)d[i].second * nPairs_ + p];
                if (isNanBits(col[i])) hasNan = true;
            }
            float agg;
            if (hasNan) agg = std::numeric_limits<float>::quiet_NaN();
            else {
                std::sort(col.begin(), col.end());
                agg = (k_ % 2) ? col[k_ / 2] : (col[k_ / 2 - 1] + col[k_ / 2]) / 2.0f;
            }
            float adj = (biasAlpha_ != 0.0) ? agg * bias_[p] : agg;
            if (bandBonus_ > 0.0f) {
                const size_t e = p / res_.size(), rr = p % res_.size();
                const float m = (1.0f - bandBonus_ * inFrac(&excBands_[e * kD * 2], target, xn)) *
                                (1.0f - bandBonus_ * inFrac(&resBands_[rr * kD * 2], target, xn));
                adj -= std::fabs(adj) * (1.0f - m);  // riduce il costo anche se negativo
            }
            if (isNanBits(adj)) { if (best < 0 || !isNanBits(bestAdj)) { best = p; bestAdj = adj; bestAgg = agg; } break; }
            if (best < 0 || adj < bestAdj) { best = p; bestAdj = adj; bestAgg = agg; }
        }
        out.exciter = exc_[best / res_.size()];
        out.resonator = res_[best % res_.size()];
        out.score = bestAgg;
        out.ok = true;
        return out;
    }

    int rowCount() const { return rows_; }
    const float* rawRow(int r) const { return &raw_[(size_t)r * kD]; }  // per il test "self"

private:
    static constexpr int kD = 15;
    // frazione dei descrittori validi (target non NaN, fascia presente) dentro [lo, hi]
    static float inFrac(const float* band, const float* target, const float*) {
        int n = 0, in = 0;
        for (int c = 0; c < kD; ++c) {
            const float lo = band[2 * c], hi = band[2 * c + 1];
            if (sel_detail::isNanBits(lo) || sel_detail::isNanBits(hi) || sel_detail::isNanBits(target[c])) continue;
            ++n;
            if (target[c] >= lo && target[c] <= hi) ++in;
        }
        return n ? (float)in / n : 0.0f;
    }
    std::vector<std::string> exc_, res_;
    std::vector<float> raw_, norm_, scores_, bias_, excBands_, resBands_;
    float bandBonus_ = 0.0f;
    float mean_[kD] = {}, std_[kD] = {};
    int rows_ = 0, nPairs_ = 0, k_ = 30;
    double biasAlpha_ = 0.3;
    bool loaded_ = false;
};

// ---------------------------------------------------------------------------------
// Thread "Auto pair" (porting di param_candidate._maybe_select_pair): ogni
// kAutoPairPeriodSeconds (= refine_period 0.3 s) interroga il selettore sul target
// corrente e pubblica la coppia scelta come INDICI (ordine dei nomi passati a start(),
// cioe' kExciterEnum/kResonatorEnum del plugin). run() (thread audio) pubblica il target
// con setTarget() (solo store atomici) e raccoglie con takePending(); il tempo minimo di
// hold (AUTO_PAIR_MIN_HOLD = 2 s) e l'applicazione ai parametri host restano in run().
inline constexpr double kAutoPairPeriodSeconds = 0.3;
inline constexpr double kAutoPairMinHoldSeconds = 2.0;
inline constexpr float kAutoPairBandBonus = 0.2f;  // "tabella dei margini": -20% di costo max per eccitatore e per forma

class AutoPairThread {
public:
    ~AutoPairThread() { stop(); }

    // Carica il corpus e avvia il thread. excNames/resNames: nomi nell'ordine degli indici
    // che run() usa. Ritorna false (thread non avviato, Auto pair inattivo) se il corpus
    // manca o un nome scelto dal selettore non e' nella tabella.
    bool start(const std::string& corpusPath, const char* const* excNames, int nExc,
               const char* const* resNames, int nRes, std::string* error = nullptr) {
        if (!sel_.load(corpusPath, error)) return false;
        sel_.setBandBonus(kAutoPairBandBonus);
        for (int i = 0; i < nExc && i < 16; ++i) excNames_[i] = excNames[i];
        for (int i = 0; i < nRes && i < 16; ++i) resNames_[i] = resNames[i];
        nExc_ = nExc; nRes_ = nRes;
        for (auto& t : target_) t.store(0.0f, std::memory_order_relaxed);
        stop_.store(false);
        th_ = std::thread([this] { loop(); });
        return true;
    }
    void stop() {
        stop_.store(true);
        if (th_.joinable()) th_.join();
    }

    // thread audio
    void setEnabled(bool on) { enabled_.store(on, std::memory_order_relaxed); }
    void setTarget(const float t[15]) {
        for (int i = 0; i < 15; ++i) target_[i].store(t[i], std::memory_order_relaxed);
    }
    bool takePending(int& exc, int& res) {
        if (!hasPending_.exchange(false, std::memory_order_acq_rel)) return false;
        exc = pendExc_.load(std::memory_order_relaxed);
        res = pendRes_.load(std::memory_order_relaxed);
        return true;
    }

private:
    void loop() {
        using namespace std::chrono;
        auto next = steady_clock::now();
        while (!stop_.load()) {
            std::this_thread::sleep_for(milliseconds(10));
            if (steady_clock::now() < next) continue;
            next = steady_clock::now() + milliseconds((int)(kAutoPairPeriodSeconds * 1000.0));
            if (!enabled_.load(std::memory_order_relaxed)) continue;
            float t[15];
            for (int i = 0; i < 15; ++i) t[i] = target_[i].load(std::memory_order_relaxed);
            const PairChoice c = sel_.query(t);
            if (!c.ok) continue;
            int e = -1, r = -1;
            for (int i = 0; i < nExc_; ++i) if (c.exciter == excNames_[i]) e = i;
            for (int i = 0; i < nRes_; ++i) if (c.resonator == resNames_[i]) r = i;
            if (e < 0 || r < 0) continue;  // nome sconosciuto: coppia invariata (fallback silenzioso come Python)
            pendExc_.store(e, std::memory_order_relaxed);
            pendRes_.store(r, std::memory_order_relaxed);
            hasPending_.store(true, std::memory_order_release);
        }
    }

    PairSelector sel_;
    std::string excNames_[16], resNames_[16];
    int nExc_ = 0, nRes_ = 0;
    std::atomic<float> target_[15];
    std::atomic<bool> enabled_{false}, stop_{false}, hasPending_{false};
    std::atomic<int> pendExc_{0}, pendRes_{0};
    std::thread th_;
};

} // namespace phimo
