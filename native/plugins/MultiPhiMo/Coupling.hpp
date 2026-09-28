// Coupling.hpp -- resonator.py v5 (2026-09-26/27): stadio d'eccitazione, accoppiamento eccitatore/risonatore
// (harmonicity con modi pilotati, pitch_focus, body) e filtro formantico; note_duration().
// Porting 1:1 di resonator.py apply_resonator (ramo v5), _exc_envelope, _harmonize, _driven_modes,
// _formant_filter, _unit_rms, note_duration. Filtri: scipy.signal.butter(2, [lo, hi], btype="band") in forma
// b/a + lfilter (forma diretta II trasposta), come in Python.
// Parametri d'accoppiamento per NOME (stesse chiavi di resonator.COUPLING_RANGES); chiave assente = None in Python.
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "Resonator.hpp"

namespace phimo {

// resonator.COUPLING_RANGES (+ FORMANT_RANGES) e COUPLING_LOG, nell'ordine di Python.
struct CouplingSpec { const char* name; float lo, hi; bool logScale; };
inline constexpr int kCouplingCount = 10;
inline constexpr CouplingSpec kCouplingSpecs[kCouplingCount] = {
    {"harmonicity", 0.0f, 1.0f, false}, {"pitch_focus", 0.0f, 1.0f, false}, {"body", 0.0f, 1.0f, false},
    {"exc_attack", 0.001f, 0.5f, true}, {"exc_hold", 0.03f, 3.0f, true}, {"am_rate", 0.5f, 20.0f, true},
    {"am_depth", 0.0f, 1.0f, false}, {"form_f1", 200.0f, 1000.0f, true}, {"form_f2", 600.0f, 3200.0f, true},
    {"form_amt", 0.0f, 1.0f, false},
};
// Default di Mode=Manual (valori fisici, stesso ordine di kCouplingSpecs). 2026-09-28, scelti con
// diag_manual_defaults.py (dB di variazione spettrale per parametro, mediana eccitatori/risonatori):
// "attuale" h0.5 pf0.5 body0.7 hold0.5 = 7.8/8.8 dB; h1 = 10.2/13.3 ma bird/vocal/plate_rect ~1-2 dB;
// h0 = 6.4/29.4; h0.8 pf0 body1 hold3 = 8.6/10.1 senza crolli (min per eccitatore 5.2, per forma 4.6) -> scelto.
inline constexpr float kCouplingManualDefaults[kCouplingCount] = {0.8f, 0.0f, 1.0f, 0.001f, 3.0f, 5.0f, 0.0f, 500.0f, 1500.0f, 0.0f};
inline constexpr double kExcRelease = 0.02;       // EXC_RELEASE (s)
inline constexpr double kNoteDurationCap = 3.0;   // note_duration(cap=3.0)

// Valore host normalizzato 0-1 <-> fisico (stesso schema log di paramSpecDenormalize), slot Manual/GUI
inline float couplingDenormalize(float t, const CouplingSpec& c) {
    t = std::min(std::max(t, 0.0f), 1.0f);
    if (c.logScale) {
        const float a = std::log10(c.lo), b = std::log10(c.hi);
        return std::pow(10.0f, a + t * (b - a));
    }
    return c.lo + t * (c.hi - c.lo);
}
inline float couplingNormalize(float v, const CouplingSpec& c) {
    float t;
    if (c.logScale) t = (std::log10(std::max(v, 1e-9f)) - std::log10(c.lo)) / (std::log10(c.hi) - std::log10(c.lo));
    else t = (v - c.lo) / (c.hi - c.lo);
    return std::min(std::max(t, 0.0f), 1.0f);
}

namespace coupling_detail {

inline bool get(const std::map<std::string, float>& m, const char* k, double& v) {
    auto it = m.find(k);
    if (it == m.end()) return false;
    v = (double)it->second;
    return true;
}

// numpy.poly per radici complesse (coefficienti reali: le radici sono in coppie coniugate)
inline std::vector<double> polyReal(const std::vector<std::complex<double>>& roots) {
    std::vector<std::complex<double>> c{1.0};
    for (const auto& r : roots) {
        std::vector<std::complex<double>> nc(c.size() + 1, 0.0);
        for (size_t i = 0; i < c.size(); ++i) { nc[i] += c[i]; nc[i + 1] -= r * c[i]; }
        c.swap(nc);
    }
    std::vector<double> out(c.size());
    for (size_t i = 0; i < c.size(); ++i) out[i] = c[i].real();
    return out;
}

// scipy.signal.butter(2, [w1, w2], btype="band") (w normalizzate a Nyquist): buttap -> lp2bp_zpk -> bilinear_zpk
// (fs=2) -> zpk2tf. b, a di 5 coefficienti.
inline void butterBandpass2(double w1, double w2, std::vector<double>& b, std::vector<double>& a) {
    const int N = 2;
    std::vector<std::complex<double>> p;
    for (int m = -N + 1; m < N; m += 2) p.push_back(-std::exp(std::complex<double>(0.0, kPi * m / (2.0 * N))));
    const double fs = 2.0;
    const double wl = 2.0 * fs * std::tan(kPi * w1 / fs), wh = 2.0 * fs * std::tan(kPi * w2 / fs);
    const double bw = wh - wl, wo = std::sqrt(wl * wh);
    std::vector<std::complex<double>> pb, zb;
    std::vector<std::complex<double>> plp;
    for (auto& q : p) plp.push_back(q * bw / 2.0);
    for (auto& q : plp) pb.push_back(q + std::sqrt(q * q - wo * wo));
    for (auto& q : plp) pb.push_back(q - std::sqrt(q * q - wo * wo));
    for (int i = 0; i < N; ++i) zb.push_back(0.0);
    double k = std::pow(bw, N);
    const double fs2 = 2.0 * fs;
    std::complex<double> num = 1.0, den = 1.0;
    std::vector<std::complex<double>> zz, pz;
    for (auto& z : zb) { zz.push_back((fs2 + z) / (fs2 - z)); num *= (fs2 - z); }
    for (auto& q : pb) { pz.push_back((fs2 + q) / (fs2 - q)); den *= (fs2 - q); }
    for (size_t i = zb.size(); i < pb.size(); ++i) zz.push_back(-1.0);
    k *= (num / den).real();
    b = polyReal(zz);
    for (double& v : b) v *= k;
    a = polyReal(pz);
}

// scipy.signal.lfilter(b, a, x), stato iniziale nullo (forma diretta II trasposta)
inline std::vector<double> lfilter(const std::vector<double>& b, const std::vector<double>& a,
                                   const std::vector<double>& x) {
    const size_t K = std::max(a.size(), b.size());
    std::vector<double> bb(K, 0.0), aa(K, 0.0), z(K, 0.0), y(x.size());
    for (size_t i = 0; i < b.size(); ++i) bb[i] = b[i] / a[0];
    for (size_t i = 0; i < a.size(); ++i) aa[i] = a[i] / a[0];
    for (size_t n = 0; n < x.size(); ++n) {
        const double yn = bb[0] * x[n] + z[0];
        for (size_t i = 1; i < K; ++i) z[i - 1] = bb[i] * x[n] + z[i] - aa[i] * yn;
        y[n] = yn;
    }
    return y;
}

inline std::vector<double> unitRms(const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) s += v * v;
    const double d = std::sqrt(x.empty() ? 0.0 : s / (double)x.size()) + kResonatorEps;
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] / d;
    return y;
}

// out = ca * unitRms(a) + cb * unitRms(b)
inline std::vector<double> mixUnit(double ca, const std::vector<double>& a, double cb, const std::vector<double>& b) {
    std::vector<double> ua = unitRms(a), ub = unitRms(b), y(a.size());
    for (size_t i = 0; i < y.size(); ++i) y[i] = ca * ua[i] + cb * ub[i];
    return y;
}

inline std::vector<double> resonantMode(const std::vector<double>& x, double freq, double dampingTime) {
    TwoPoleMode m;
    m.setParams(freq, dampingTime);
    std::vector<double> y;
    m.processBuffer(x, y);
    return y;
}

inline std::vector<double> bandpass(const std::vector<double>& x, double lo, double hi, double sr) {
    std::vector<double> b, a;
    butterBandpass2(lo / (0.5 * sr), hi / (0.5 * sr), b, a);
    return lfilter(b, a, x);
}

// _driven_modes: ogni modo suona alla SUA frequenza, ampiezza = energia dell'eccitazione nella sua banda
inline std::vector<double> drivenModes(const std::vector<double>& x, const ModalBank& bank, double sr) {
    const size_t n = x.size();
    std::vector<double> y(n, 0.0);
    for (size_t i = 0; i < bank.freqs.size(); ++i) {
        const double f = std::min(std::max(bank.freqs[i], 20.0), 0.45 * sr);
        const double bw = f / 6.0;
        std::vector<double> e = bandpass(x, std::max(f - bw / 2.0, 10.0), std::min(f + bw / 2.0, 0.49 * sr), sr);
        const double c = std::exp(-1.0 / (std::max(bank.dampingTimes[i], 2e-3) * sr));
        double env = 0.0;
        const double w = 2.0 * kPi * f / sr, ph = 0.7 * (double)i, g = bank.gains[i];
        for (size_t t = 0; t < n; ++t) {
            env = (1.0 - c) * std::fabs(e[t]) + c * env;
            y[t] += g * env * std::sin(w * (double)t + ph);
        }
    }
    return y;
}

inline std::vector<double> formantFilter(const std::vector<double>& y, double sr, double f1, double f2, double amt) {
    std::vector<double> out(y.size(), 0.0);
    const double fg[2][2] = {{f1, 1.0}, {f2, 0.7}};
    for (const auto& q : fg) {
        const double f = std::min(std::max(q[0], 50.0), 0.45 * sr);
        const double bw = f / 4.0;
        std::vector<double> o = bandpass(y, std::max(f - bw / 2.0, 20.0), std::min(f + bw / 2.0, 0.49 * sr), sr);
        for (size_t i = 0; i < out.size(); ++i) out[i] += q[1] * o[i];
    }
    return mixUnit(1.0 - amt, y, amt, out);
}

inline double medianOf(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

} // namespace coupling_detail

// _exc_envelope applicato in place (solo se almeno uno stadio e' presente, come in apply_resonator)
inline void applyExcitationStage(std::vector<double>& x, const std::map<std::string, float>& cp, double sr) {
    using coupling_detail::get;
    double att, hold, rate, depth;
    const bool hA = get(cp, "exc_attack", att), hH = get(cp, "exc_hold", hold);
    const bool hR = get(cp, "am_rate", rate), hD = get(cp, "am_depth", depth);
    const bool am = hR && hD && depth > 0.0;
    if (!(hA || hH || am)) return;
    const double ta = std::max(att, 1e-4), th = std::max(hold, 1e-3);
    for (size_t n = 0; n < x.size(); ++n) {
        const double t = (double)n / sr;
        double g = 1.0;
        if (hA && t < ta) g = 0.5 - 0.5 * std::cos(kPi * std::min(t / ta, 1.0));
        if (hH) g *= std::min(std::max(1.0 - (t - th) / kExcRelease, 0.0), 1.0);
        if (am) g *= 1.0 - depth * (0.5 - 0.5 * std::cos(2.0 * kPi * rate * t));
        x[n] *= g;
    }
}

// resonator.apply_resonator (v5). f0 <= 0 = None. cp = parametri d'accoppiamento per nome.
inline std::vector<float> applyResonatorV5(const std::vector<float>& excitation, const std::string& shape,
                                           const std::map<std::string, float>& rp,
                                           const std::map<std::string, float>& cp, double f0) {
    using namespace coupling_detail;
    const double sr = (double)kResonatorSR;
    ModalBank bank = computeModalBank(shape, rp);
    std::vector<double> x(excitation.begin(), excitation.end());
    applyExcitationStage(x, cp, sr);
    double h = 0.0, pf = 0.0, bd = 1.0;  // COUPLING_DEFAULTS
    get(cp, "harmonicity", h);
    get(cp, "pitch_focus", pf);
    get(cp, "body", bd);
    const bool hasF0 = f0 > 0.0;
    // _harmonize
    if (hasF0 && h > 0.0) {
        const double hc = std::min(std::max(h, 0.0), 1.0);
        for (double& f : bank.freqs) {
            const double k = std::max(std::nearbyint(f / f0), 1.0);
            f = f * std::pow((k * f0) / f, hc);
        }
    }
    // keep = freqs < 0.45 sr (almeno il primo)
    {
        ModalBank kb;
        for (size_t i = 0; i < bank.freqs.size(); ++i)
            if (bank.freqs[i] < 0.45 * sr) {
                kb.freqs.push_back(bank.freqs[i]);
                kb.dampingTimes.push_back(bank.dampingTimes[i]);
                kb.gains.push_back(bank.gains[i]);
            }
        if (kb.freqs.empty() && !bank.freqs.empty()) {
            kb.freqs.push_back(bank.freqs[0]);
            kb.dampingTimes.push_back(bank.dampingTimes[0]);
            kb.gains.push_back(bank.gains[0]);
        }
        bank.freqs.swap(kb.freqs);
        bank.dampingTimes.swap(kb.dampingTimes);
        bank.gains.swap(kb.gains);
    }
    std::vector<double> y;
    if (shape == "chaotic") {
        auto getp = [&](const char* k, double def) {
            auto it = rp.find(k);
            return it != rp.end() ? (double)it->second : def;
        };
        const size_t M = bank.freqs.size();
        const double beat = getp("beat", 3.0), depth = getp("depth", 0.6);
        std::vector<double> wa(M), dw(M), r(M), ga(M), gb(M);
        for (size_t m = 0; m < M; ++m) {
            wa[m] = 2.0 * kPi * bank.freqs[m] / sr;
            dw[m] = 2.0 * kPi * beat * (1.0 + 0.31 * (double)m) / sr;
            r[m] = std::exp(-1.0 / (std::max(bank.dampingTimes[m], 1e-4) * sr));
            ga[m] = bank.gains[m];
            gb[m] = bank.gains[m] * depth;
        }
        y = chaoticCore(x, wa, dw, r, ga, gb, getp("nonlin", 0.2), getp("chaos", 0.4), getp("speed", 5.0) / sr);
    } else {
        y.assign(x.size(), 0.0);
        for (size_t m = 0; m < bank.freqs.size(); ++m) {
            std::vector<double> mo = resonantMode(x, bank.freqs[m], bank.dampingTimes[m]);
            for (size_t n = 0; n < y.size(); ++n) y[n] += bank.gains[m] * mo[n];
        }
        if (h < 1.0) y = mixUnit(h, y, 1.0 - h, drivenModes(x, bank, sr));
    }
    if (hasF0 && pf > 0.0) {
        const double tau0 = medianOf(bank.dampingTimes);
        y = mixUnit(1.0 - pf, y, pf, resonantMode(x, f0, tau0));
    }
    if (bd < 1.0) y = mixUnit(1.0 - bd, x, bd, y);
    double amt, f1, f2;
    if (get(cp, "form_amt", amt) && amt > 0.0 && get(cp, "form_f1", f1) && get(cp, "form_f2", f2))
        y = formantFilter(y, sr, f1, f2, amt);
    return normalizePeak09(y);
}

// resonator.note_duration(default, shape, rp, exc_hold, cap=3.0)
inline double noteDuration(double defaultDur, const std::string& shape, const std::map<std::string, float>& rp,
                           const std::map<std::string, float>& cp) {
    double tail = 0.0;
    try {
        const ModalBank bank = computeModalBank(shape, rp);
        double mx = 0.0;
        for (double d : bank.dampingTimes) mx = std::max(mx, d);
        tail = 3.0 * mx;
    } catch (const std::exception&) {
        tail = 0.0;
    }
    double hold = 0.0;
    if (!coupling_detail::get(cp, "exc_hold", hold) || hold == 0.0) hold = defaultDur;
    return std::min(std::max(hold + tail, 0.5), kNoteDurationCap);
}

} // namespace phimo
