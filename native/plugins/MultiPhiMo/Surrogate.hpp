// Surrogate.hpp -- porting di surrogate.py (2026-09-28): rete diretta parametri->descrittori (ensemble di MLP,
// una per eccitatore, forma del risonatore one-hot in ingresso), solutore sugli ingressi (Adam sui logit, gradiente
// calcolato a mano) e ricerca sul synth vero (RealSearch, analisi-per-sintesi ibrida).
// Pesi: native/data/surrogate/<eccitatore>.bin scritti da native/tools/export_surrogate.py (formato nel suo
// docstring). Tutto per NOME: codifica/decodifica usano i range scritti nel file, non tabelle duplicate.
// Budget del solutore ridotto rispetto a Python (8 partenze x 80 passi invece di 32 x 150): verificato con
// sweep_v5_small.log (slider-ok 69% vs 63%, err 0.68 vs 0.69).
// Niente isnan/isfinite (-ffast-math): validita' sempre con flag espliciti.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace phimo {

inline constexpr int kSgDesc = 15;
inline constexpr int kSgPitch = 14;          // DKEYS.index("pitch")
inline constexpr double kSgPitchTolC = 50.0; // PITCH_TOL_C

struct SgParam { std::string name; float lo = 0, hi = 1; bool logScale = false; };

inline double sgTo01(double v, const SgParam& p) {
    double lo = p.lo, hi = p.hi;
    if (p.logScale && lo > 0) { v = std::log10(std::max(v, 1e-12)); lo = std::log10(lo); hi = std::log10(hi); }
    return std::min(std::max((v - lo) / (hi - lo + 1e-12), 0.0), 1.0);
}
inline double sgFrom01(double u, const SgParam& p) {
    if (p.logScale && p.lo > 0) {
        const double a = std::log10((double)p.lo), b = std::log10((double)p.hi);
        return std::pow(10.0, a + u * (b - a));
    }
    return p.lo + u * (p.hi - p.lo);
}

struct SgLayer { int out = 0, in = 0; std::vector<float> W, WT, b; };  // WT = W trasposta [in][out] (forward vettorizzabile)
struct SgNet { SgLayer L[4]; };

struct SgDecoded {
    std::map<std::string, float> exc, res, coup;
};

class SurrogateModel {
public:
    std::string exc;
    std::vector<SgParam> xp, coup;
    std::vector<std::string> shapes, resNames;
    std::vector<std::vector<SgParam>> shapeParams;
    bool hasOff[kSgDesc] = {};
    double off[kSgDesc] = {}, mu[kSgDesc] = {}, sd[kSgDesc] = {};
    int hidden = 0, nOut = 0, dim = 0, nX = 0, nS = 0, nR = 0, rows = 0, freqIdx = -1;
    std::vector<SgNet> nets;
    std::vector<float> X, Z;
    std::vector<uint8_t> M, S;

    bool load(const std::string& path, std::string& err) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { err = "impossibile aprire " + path; return false; }
        std::vector<uint8_t> buf;
        uint8_t tmp[65536];
        size_t n;
        while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.insert(buf.end(), tmp, tmp + n);
        std::fclose(f);
        size_t pos = 0;
        bool ok = true;
        auto rd = [&](void* dst, size_t sz) {
            if (pos + sz > buf.size()) { ok = false; std::memset(dst, 0, sz); return; }
            std::memcpy(dst, buf.data() + pos, sz); pos += sz;
        };
        auto u32 = [&]() { uint32_t v; rd(&v, 4); return v; };
        auto f32 = [&]() { float v; rd(&v, 4); return v; };
        auto u8 = [&]() { uint8_t v; rd(&v, 1); return v; };
        auto str = [&]() { uint16_t l; rd(&l, 2); std::string s(l, ' '); if (l) rd(&s[0], l); return s; };
        auto prm = [&]() { SgParam p; p.name = str(); p.lo = f32(); p.hi = f32(); p.logScale = u8() != 0; return p; };
        char magic[4];
        rd(magic, 4);
        if (std::memcmp(magic, "PHSG", 4) != 0 || u32() != 1) { err = "formato non valido: " + path; return false; }
        exc = str();
        nX = (int)u32(); xp.clear();
        for (int i = 0; i < nX && ok; ++i) { xp.push_back(prm()); if (xp.back().name == "freq") freqIdx = i; }
        nS = (int)u32(); shapes.clear();
        for (int i = 0; i < nS && ok; ++i) shapes.push_back(str());
        nR = (int)u32(); resNames.clear();
        for (int i = 0; i < nR && ok; ++i) resNames.push_back(str());
        shapeParams.assign(nS, {});
        for (int s = 0; s < nS && ok; ++s) { const int c = (int)u32(); for (int i = 0; i < c && ok; ++i) shapeParams[s].push_back(prm()); }
        const int nC = (int)u32(); coup.clear();
        for (int i = 0; i < nC && ok; ++i) coup.push_back(prm());
        if ((int)u32() != kSgDesc) { err = "numero descrittori inatteso"; return false; }
        for (int j = 0; j < kSgDesc && ok; ++j) { str(); hasOff[j] = u8() != 0; off[j] = f32(); mu[j] = f32(); sd[j] = f32(); }
        hidden = (int)u32(); nOut = (int)u32();
        const int nN = (int)u32();
        dim = nX + nS + nR + (int)coup.size();
        if (nOut != kSgDesc + 1) { err = "uscite inattese"; return false; }
        nets.assign(nN, {});
        for (int k = 0; k < nN && ok; ++k)
            for (int l = 0; l < 4 && ok; ++l) {
                SgLayer& L = nets[k].L[l];
                L.out = (int)u32(); L.in = (int)u32();
                L.W.resize((size_t)L.out * L.in); L.b.resize(L.out);
                rd(L.W.data(), L.W.size() * 4); rd(L.b.data(), L.b.size() * 4);
                L.WT.resize(L.W.size());
                for (int o = 0; o < L.out; ++o)
                    for (int i = 0; i < L.in; ++i) L.WT[(size_t)i * L.out + o] = L.W[(size_t)o * L.in + i];
            }
        rows = (int)u32();
        if ((int)u32() != dim) { err = "dim incoerente"; return false; }
        X.resize((size_t)rows * dim); Z.resize((size_t)rows * kSgDesc);
        M.resize((size_t)rows * kSgDesc); S.resize(rows);
        rd(X.data(), X.size() * 4); rd(Z.data(), Z.size() * 4); rd(M.data(), M.size()); rd(S.data(), S.size());
        if (!ok || nets.empty() || nets[0].L[0].in != dim) { err = "file troncato o incoerente: " + path; return false; }
        return true;
    }

    int shapeIndex(const std::string& s) const {
        for (int i = 0; i < nS; ++i) if (shapes[i] == s) return i;
        return -1;
    }
    int resIndex(const std::string& k) const {
        for (int i = 0; i < nR; ++i) if (resNames[i] == k) return i;
        return -1;
    }

    double tdesc(double v, int j) const { return hasOff[j] ? std::log10(std::max(v, 0.0) + off[j]) : v; }

    // y[o] = b[o] + sum_i W[o][i] x[i], come somma di colonne (axpy su o): nessuna riduzione -> SIMD anche
    // senza -ffast-math (il plugin compila con -fno-fast-math). Stesso risultato a meno dell'ordine delle somme.
    static void layerForward(const SgLayer& L, const float* x, float* y) {
        std::memcpy(y, L.b.data(), sizeof(float) * L.out);
        for (int i = 0; i < L.in; ++i) {
            const float xi = x[i];
            const float* w = &L.WT[(size_t)i * L.out];
            for (int o = 0; o < L.out; ++o) y[o] += w[o] * xi;
        }
    }

    // tutte le reti: attivazioni salvate per il backward (acts[k] = 3 strati nascosti, pre-attivazione)
    void forward(const float* x, float* out) const {
        thread_local std::vector<float> h, h2, o;
        h.resize(hidden); h2.resize(hidden); o.resize(nOut);
        for (int j = 0; j < nOut; ++j) out[j] = 0.0f;
        for (const SgNet& N : nets) {
            layerForward(N.L[0], x, h.data());
            for (float& v : h) v = v / (1.0f + std::exp(-v));
            layerForward(N.L[1], h.data(), h2.data());
            for (float& v : h2) v = v / (1.0f + std::exp(-v));
            layerForward(N.L[2], h2.data(), h.data());
            for (float& v : h) v = v / (1.0f + std::exp(-v));
            layerForward(N.L[3], h.data(), o.data());
            for (int j = 0; j < nOut; ++j) out[j] += o[j];
        }
        const float inv = 1.0f / (float)nets.size();
        for (int j = 0; j < nOut; ++j) out[j] *= inv;
    }

    // uscita media + gradiente d(loss)/dx dato gOut = d(loss)/d(out media)
    void forwardBackward(const float* x, float* out, const std::function<void(const float*, float*)>& lossGrad,
                         float* dx) const {
        const int H = hidden;
        thread_local std::vector<std::vector<float>> pre, act;
        thread_local std::vector<float> o16;
        if (pre.size() < nets.size() * 3) { pre.resize(nets.size() * 3); act.resize(nets.size() * 3); }
        o16.resize(nOut);
        for (int j = 0; j < nOut; ++j) out[j] = 0.0f;
        for (size_t k = 0; k < nets.size(); ++k) {
            const SgNet& N = nets[k];
            const float* in = x;
            for (int l = 0; l < 3; ++l) {
                std::vector<float>& p = pre[k * 3 + l];
                std::vector<float>& q = act[k * 3 + l];
                p.resize(H); q.resize(H);
                layerForward(N.L[l], in, p.data());
                for (int i = 0; i < H; ++i) q[i] = p[i] / (1.0f + std::exp(-p[i]));
                in = q.data();
            }
            layerForward(N.L[3], in, o16.data());
            for (int j = 0; j < nOut; ++j) out[j] += o16[j];
        }
        const float inv = 1.0f / (float)nets.size();
        for (int j = 0; j < nOut; ++j) out[j] *= inv;
        thread_local std::vector<float> g, d, dn;
        g.resize(nOut); d.resize(H); dn.resize(H);
        lossGrad(out, g.data());
        for (int j = 0; j < nOut; ++j) g[j] *= inv;
        for (int i = 0; i < dim; ++i) dx[i] = 0.0f;
        for (size_t k = 0; k < nets.size(); ++k) {
            const SgNet& N = nets[k];
            // strato di uscita -> ultimo nascosto
            {
                const SgLayer& L = N.L[3];
                std::fill(d.begin(), d.end(), 0.0f);
                for (int o = 0; o < L.out; ++o) {
                    const float* w = &L.W[(size_t)o * L.in];
                    const float go = g[o];
                    for (int i = 0; i < L.in; ++i) d[i] += go * w[i];
                }
            }
            for (int l = 2; l >= 0; --l) {
                const float* p = pre[k * 3 + l].data();
                for (int o = 0; o < H; ++o) {
                    const float sg = 1.0f / (1.0f + std::exp(-p[o]));
                    d[o] *= sg * (1.0f + p[o] * (1.0f - sg));  // SiLU'
                }
                const SgLayer& L = N.L[l];
                float* tgt = (l == 0) ? dx : dn.data();
                if (l > 0) std::fill(dn.begin(), dn.end(), 0.0f);
                for (int o = 0; o < L.out; ++o) {
                    const float* w = &L.W[(size_t)o * L.in];
                    const float go = d[o];
                    for (int i = 0; i < L.in; ++i) tgt[i] += go * w[i];
                }
                if (l > 0) d.swap(dn);
            }
        }
    }

    std::vector<float> freeMask(int shape, bool freqFree) const {
        std::vector<float> m(dim, 0.0f);
        for (int i = 0; i < nX; ++i) m[i] = (xp[i].name != "freq" || freqFree) ? 1.0f : 0.0f;
        for (const SgParam& p : shapeParams[shape]) m[nX + nS + resIndex(p.name)] = 1.0f;
        for (int i = nX + nS + nR; i < dim; ++i) m[i] = 1.0f;
        return m;
    }

    SgDecoded decode(const float* v, int shape) const {
        SgDecoded d;
        for (int i = 0; i < nX; ++i) {
            double val = sgFrom01(v[i], xp[i]);
            if (xp[i].name == "n_particles") val = std::nearbyint(val);
            d.exc[xp[i].name] = (float)val;
        }
        for (const SgParam& p : shapeParams[shape]) d.res[p.name] = (float)sgFrom01(v[nX + nS + resIndex(p.name)], p);
        for (size_t i = 0; i < coup.size(); ++i) d.coup[coup[i].name] = (float)sgFrom01(v[nX + nS + nR + i], coup[i]);
        return d;
    }

    // target_z: valori grezzi -> z, w (w=0 dove non valido)
    void targetZ(const double* v, const bool* valid, float* z, float* w) const {
        for (int j = 0; j < kSgDesc; ++j) {
            z[j] = 0.0f; w[j] = 0.0f;
            if (valid[j]) { z[j] = (float)((tdesc(v[j], j) - mu[j]) / sd[j]); w[j] = 1.0f; }
        }
    }
};

// Ultimo problema risolto (Inverter.last in Python): serve a RealSearch
struct SgProblem {
    int shape = 0;
    float z[kSgDesc] = {}, w[kSgDesc] = {};
    std::vector<float> mask, fixed, x;
    bool wantPitched = false, freqFree = true;
    double freq = 0.0;
};

// Inverter.solve (senza il bias del circuito chiuso, scartato)
class SurrogateSolver {
public:
    const SurrogateModel* m = nullptr;
    int nKnn = 7, nRand = 1, steps = 80;
    float lr = 0.05f, prevW = 0.02f, changedGain = 4.0f;
    std::map<int, std::vector<float>> prev;  // forma -> ultima soluzione (continuita')
    bool hasPrevZ = false, hasMoved = false;
    float prevZ[kSgDesc] = {};
    bool moved[kSgDesc] = {};
    SgProblem last;

    void reset() { prev.clear(); hasPrevZ = hasMoved = false; }

    // target: 15 valori grezzi + flag; freq <= 0 = nessuna altezza. Ritorna false se la forma non e' nota.
    bool solve(const double* target, const bool* valid, const std::string& shapeName, double freq,
               std::mt19937& rng, SgDecoded& outDec, float* lossOut = nullptr) {
        const SurrogateModel& M_ = *m;
        const int shape = M_.shapeIndex(shapeName);
        if (shape < 0) return false;
        const bool freqFree = (M_.exc == "noise" || M_.exc == "shaker") || freq <= 0.0 || M_.freqIdx < 0;
        const std::vector<float> mask = M_.freeMask(shape, freqFree);
        float z[kSgDesc], w[kSgDesc];
        M_.targetZ(target, valid, z, w);
        if (hasPrevZ) {
            bool any = false, mv[kSgDesc];
            for (int j = 0; j < kSgDesc; ++j) { mv[j] = std::fabs(z[j] - prevZ[j]) > 0.05f && w[j] > 0; any |= mv[j]; }
            if (any) { std::memcpy(moved, mv, sizeof mv); hasMoved = true; }
        }
        if (hasMoved)
            for (int j = 0; j < kSgDesc; ++j) if (moved[j] && w[j] > 0) w[j] *= changedGain;
        std::memcpy(prevZ, z, sizeof z);
        hasPrevZ = true;
        if (!freqFree) w[kSgPitch] = 0.0f;
        const bool wantPitched = valid[kSgPitch];
        const int D = M_.dim;
        // punti di partenza
        std::vector<std::pair<float, int>> dist;
        for (int r = 0; r < M_.rows; ++r) {
            if (M_.S[r] != shape) continue;
            float d = 0.0f;
            for (int j = 0; j < kSgDesc; ++j) {
                const float e = M_.Z[(size_t)r * kSgDesc + j] - z[j];
                d += e * e * w[j] * (float)M_.M[(size_t)r * kSgDesc + j];
            }
            dist.push_back({d, r});
        }
        const int k = std::min((int)dist.size(), nKnn);
        std::partial_sort(dist.begin(), dist.begin() + k, dist.end());
        std::vector<std::vector<float>> starts;
        for (int i = 0; i < k; ++i) starts.emplace_back(&M_.X[(size_t)dist[i].second * D], &M_.X[(size_t)dist[i].second * D] + D);
        auto pit = prev.find(shape);
        if (pit != prev.end()) starts.push_back(pit->second);
        std::uniform_real_distribution<float> U(0.05f, 0.95f);
        for (int i = 0; i < nRand; ++i) { std::vector<float> s(D); for (float& v : s) v = U(rng); starts.push_back(s); }
        if (starts.empty()) return false;
        std::vector<float> fixed(D, 0.0f);
        fixed[M_.nX + shape] = 1.0f;
        if (!freqFree) {
            const SgParam& fp = M_.xp[M_.freqIdx];
            fixed[M_.freqIdx] = (float)sgTo01(std::min(std::max(freq, (double)fp.lo), (double)fp.hi), fp);
        }
        const std::vector<float> pv = (pit != prev.end()) ? pit->second : starts[0];
        float Wsum = 0.0f;
        for (int j = 0; j < kSgDesc; ++j) Wsum += w[j];
        const float Wd = std::max(Wsum, 1.0f);
        const bool pitchPen = wantPitched && !freqFree;
        std::vector<float> best;
        float bestLoss = 1e30f;
        std::vector<float> u(D), x(D), dx(D), mA(D), vA(D), out(M_.nOut);
        for (const auto& s0 : starts) {
            for (int i = 0; i < D; ++i) {
                const float p = std::min(std::max(s0[i], 0.02f), 0.98f);
                u[i] = std::log(p / (1.0f - p));
            }
            std::fill(mA.begin(), mA.end(), 0.0f);
            std::fill(vA.begin(), vA.end(), 0.0f);
            auto lossGrad = [&](const float* o, float* g) {
                for (int j = 0; j < kSgDesc; ++j) g[j] = 2.0f * w[j] * (o[j] - z[j]) / Wd;
                g[kSgDesc] = pitchPen ? -0.5f / (1.0f + std::exp(o[kSgDesc])) : 0.0f;  // d softplus(-o) = -sigmoid(-o)
            };
            for (int t = 1; t <= steps; ++t) {
                for (int i = 0; i < D; ++i) x[i] = mask[i] / (1.0f + std::exp(-u[i])) + fixed[i];
                M_.forwardBackward(x.data(), out.data(), lossGrad, dx.data());
                const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
                const float bc1 = 1.0f - std::pow(b1, (float)t), bc2 = 1.0f - std::pow(b2, (float)t);
                for (int i = 0; i < D; ++i) {
                    const float sg = 1.0f / (1.0f + std::exp(-u[i]));
                    float gx = dx[i] + prevW * 2.0f * (x[i] - pv[i]) * mask[i] / (float)D;
                    const float gu = gx * mask[i] * sg * (1.0f - sg);
                    mA[i] = b1 * mA[i] + (1.0f - b1) * gu;
                    vA[i] = b2 * vA[i] + (1.0f - b2) * gu * gu;
                    u[i] -= (lr / bc1) * mA[i] / (std::sqrt(vA[i]) / std::sqrt(bc2) + eps);
                }
            }
            for (int i = 0; i < D; ++i) x[i] = mask[i] / (1.0f + std::exp(-u[i])) + fixed[i];
            M_.forward(x.data(), out.data());
            float l = 0.0f;
            for (int j = 0; j < kSgDesc; ++j) l += (out[j] - z[j]) * (out[j] - z[j]) * w[j];
            l /= Wd;
            if (l < bestLoss) { bestLoss = l; best = x; }
        }
        prev[shape] = best;
        last.shape = shape;
        std::memcpy(last.z, z, sizeof z);
        std::memcpy(last.w, w, sizeof w);
        last.mask = mask; last.fixed = fixed; last.x = best;
        last.wantPitched = wantPitched; last.freqFree = freqFree; last.freq = freq;
        outDec = M_.decode(best.data(), shape);
        if (lossOut) *lossOut = bestLoss;
        return true;
    }

    // meas_err: RMS pesato in z sulle misure vere (+ penalita' di altezza con pitch-lock)
    double measErr(const double* meas, const bool* mvalid, const SgProblem& L) const {
        float zm[kSgDesc], wm[kSgDesc];
        m->targetZ(meas, mvalid, zm, wm);
        double W = 0.0, e = 0.0;
        for (int j = 0; j < kSgDesc; ++j) {
            W += L.w[j];
            e += (wm[j] > 0) ? (double)(zm[j] - L.z[j]) * (zm[j] - L.z[j]) * L.w[j] : (double)L.w[j];
        }
        e /= std::max(W, 1e-9);
        if (L.wantPitched && !L.freqFree && L.freq > 0.0) {
            if (!mvalid[kSgPitch] || meas[kSgPitch] <= 0.0) e += 1.0;
            else {
                const double c = std::fabs(1200.0 * std::log2(meas[kSgPitch] / L.freq));
                e += std::min(std::max(c / kSgPitchTolC - 1.0, 0.0), 1.0);
            }
        }
        return std::sqrt(e);
    }
};

// RealSearch: un render reale per step(); render(dec, meas, valid) riempie i 15 descrittori misurati.
class SgRealSearch {
public:
    using RenderFn = std::function<void(const SgDecoded&, double*, bool*)>;
    int group = 5, screen = 8, n = 0, groups = 0;
    float sigma = 0.08f, frac = 0.35f;
    double eBest = 0.0;
    bool started = false, improved = false;
    std::vector<float> best;
    double mBest[kSgDesc] = {};
    bool vBest[kSgDesc] = {};

    void init(const SurrogateSolver& s, unsigned seed) {
        solver_ = &s; L_ = s.last; best = L_.x; rng_.seed(seed);
        n = groups = 0; started = improved = false; queue_.clear(); sigma = 0.08f;
        free_.clear();
        for (int i = 0; i < (int)L_.mask.size(); ++i) if (L_.mask[i] > 0) free_.push_back(i);
    }

    // true se il migliore misurato e' cambiato (escluso il primo = punto della rete)
    bool step(const RenderFn& render) {
        const SurrogateModel& M_ = *solver_->m;
        std::vector<float> x;
        const bool first = !started;
        if (first) x = best;
        else {
            if (queue_.empty()) {
                if (groups) sigma = improved ? std::min(sigma * 1.5f, 0.3f) : std::max(sigma * 0.6f, 0.01f);
                improved = false;
                propose(M_);
                ++groups;
            }
            x = queue_.front();
            queue_.erase(queue_.begin());
        }
        double meas[kSgDesc];
        bool mv[kSgDesc];
        render(M_.decode(x.data(), L_.shape), meas, mv);
        const double e = solver_->measErr(meas, mv, L_);
        ++n;
        if (first || e < eBest) {
            best = x; eBest = e;
            std::memcpy(mBest, meas, sizeof meas);
            std::memcpy(vBest, mv, sizeof mv);
            started = true;
            if (!first) { improved = true; return true; }
        }
        return false;
    }

    SgDecoded result() const { return solver_->m->decode(best.data(), L_.shape); }
    int shape() const { return L_.shape; }

private:
    const SurrogateSolver* solver_ = nullptr;
    SgProblem L_;
    std::mt19937 rng_;
    std::vector<int> free_;
    std::vector<std::vector<float>> queue_;

    void propose(const SurrogateModel& M_) {
        const int D = M_.dim;
        float zm[kSgDesc], wm[kSgDesc], pb[kSgDesc + 1], resid[kSgDesc];
        M_.targetZ(mBest, vBest, zm, wm);
        M_.forward(best.data(), pb);
        for (int j = 0; j < kSgDesc; ++j) resid[j] = wm[j] > 0 ? zm[j] - pb[j] : 0.0f;
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        std::normal_distribution<float> G(0.0f, sigma);
        std::uniform_int_distribution<int> pick(0, (int)free_.size() - 1);
        const int nc = group * screen;
        std::vector<std::pair<float, int>> score;
        std::vector<std::vector<float>> cand(nc, best);
        std::vector<float> out(M_.nOut);
        for (int c = 0; c < nc; ++c) {
            bool any = false;
            for (int i : free_) if (U(rng_) < frac) { cand[c][i] += G(rng_); any = true; }
            if (!any && !free_.empty()) cand[c][free_[pick(rng_)]] += G(rng_);
            for (int i = 0; i < D; ++i) cand[c][i] = std::min(std::max(cand[c][i], 0.0f), 1.0f) * L_.mask[i] + L_.fixed[i];
            M_.forward(cand[c].data(), out.data());
            float s = 0.0f;
            for (int j = 0; j < kSgDesc; ++j) { const float e = out[j] + resid[j] - L_.z[j]; s += e * e * L_.w[j]; }
            score.push_back({s, c});
        }
        std::partial_sort(score.begin(), score.begin() + group, score.end());
        queue_.clear();
        for (int i = 0; i < group; ++i) queue_.push_back(cand[score[i].second]);
    }
};

} // namespace phimo
