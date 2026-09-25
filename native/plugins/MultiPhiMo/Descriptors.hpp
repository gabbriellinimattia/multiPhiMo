#pragma once
// Porting di analyzer/descriptors.py. Verificato contro il sorgente reale (device_bash cat,
// 2026-09-22), non da memoria. A differenza di resonator.py/exciters.py, qui l'input e' un
// segnale dato (non generato via RNG in questo file): la validazione usa LO STESSO buffer
// audio esatto in Python e C++ (scritto su disco da questo programma, riletto da Python),
// quindi ci si aspetta match esatto o quasi-esatto, non statistico.
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#ifndef POCKETFFT_NO_MULTITHREADING
#define POCKETFFT_NO_MULTITHREADING
#endif
#include "../../pocketfft/pocketfft_hdronly.h"
#include "Resonator.hpp"  // kPi

namespace phimo {

inline constexpr double kDescEps = 1e-12;

// Test sui bit IEEE-754 (esponente tutto a 1 = NaN/Inf): immune a -ffast-math, che puo'
// riscrivere/eliminare i confronti con NaN.
inline bool descFiniteRange(double v) {
    uint64_t b;
    std::memcpy(&b, &v, sizeof(b));
    return ((b >> 52) & 0x7FFu) != 0x7FFu;
}

// ---------------- FFT helpers condivisi (complex-to-complex, per Hilbert) ----------------

inline std::vector<std::complex<double>> fftComplex(const std::vector<std::complex<double>>& x, bool forward) {
    size_t n = x.size();
    std::vector<std::complex<double>> out(n);
    pocketfft::shape_t shape{n};
    pocketfft::stride_t stride{sizeof(std::complex<double>)};
    double fct = forward ? 1.0 : 1.0 / (double)n;
    pocketfft::c2c(shape, stride, stride, pocketfft::shape_t{0}, forward, x.data(), out.data(), fct);
    return out;
}

// exciters.py rfft/irfft equivalenti (stessa convenzione numpy.fft.rfft/irfft), duplicati qui
// per non dipendere da Exciters.hpp (Descriptors.hpp deve poter essere incluso da solo).
inline std::vector<std::complex<double>> rfftD(const std::vector<double>& x) {
    size_t n = x.size();
    std::vector<std::complex<double>> out(n / 2 + 1);
    pocketfft::shape_t shape{n};
    pocketfft::stride_t strideIn{sizeof(double)};
    pocketfft::stride_t strideOut{sizeof(std::complex<double>)};
    pocketfft::r2c(shape, strideIn, strideOut, size_t(0), true, x.data(), out.data(), 1.0);
    return out;
}

inline std::vector<double> irfftD(const std::vector<std::complex<double>>& X, size_t n) {
    std::vector<double> out(n);
    pocketfft::shape_t shapeOut{n};
    pocketfft::stride_t strideIn{sizeof(std::complex<double>)};
    pocketfft::stride_t strideOut{sizeof(double)};
    pocketfft::c2r(shapeOut, strideIn, strideOut, size_t(0), false, X.data(), out.data(), 1.0 / (double)n);
    return out;
}

inline double median(std::vector<double> v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------------- Windows (2 varianti distinte, come nel sorgente Python) ----------------

// sg.get_window("hann", nperseg) usato da _stft_mag: fftbins=True di default -> Hann PERIODICO
// (denominatore N, non N-1).
inline std::vector<double> hannWindowPeriodic(int n) {
    std::vector<double> w(n);
    for (int i = 0; i < n; ++i) w[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / n);
    return w;
}

// sg.windows.hann(M) usato direttamente in modulation(): sym=True di default -> Hann
// SIMMETRICO (denominatore M-1) -- DIVERSO dal periodico sopra, non riusabile.
inline std::vector<double> hannWindowSymmetric(int n) {
    std::vector<double> w(n);
    if (n <= 1) { if (n == 1) w[0] = 1.0; return w; }
    for (int i = 0; i < n; ++i) w[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / (n - 1));
    return w;
}

// sg.windows.hamming(frame_len) usato in formants(): sym=True di default.
inline std::vector<double> hammingWindowSymmetric(int n) {
    std::vector<double> w(n);
    if (n <= 1) { if (n == 1) w[0] = 1.0; return w; }
    for (int i = 0; i < n; ++i) w[i] = 0.54 - 0.46 * std::cos(2.0 * kPi * i / (n - 1));
    return w;
}

// ---------------- STFT (_stft_mag) ----------------
// scipy.signal.stft(x, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap, boundary=None):
// boundary=None -> nessuna estensione ai bordi; padded=True (default non sovrascritto) -> zero-pad
// SOLO in coda cosi' che (len_padded - nperseg) sia divisibile per hop. scaling di default
// 'spectrum', mode='stft' -> scale = 1/win.sum() (NESSUN raddoppio dei bin, a differenza delle
// stime di potenza psd/welch) -- valore da confermare col probe scipy richiesto all'utente.
struct StftData {
    std::vector<double> freqs;             // n_bins
    std::vector<std::vector<double>> mag;  // mag[frame][bin], n_frames x n_bins
};

inline StftData stftMag(const std::vector<double>& x, int sr, int nperseg = 2048, double hopRatio = 0.5) {
    int n = (int)x.size();
    int seg = std::min(nperseg, n);
    if (seg < 8) seg = n;
    int noverlap = (int)(seg * hopRatio);
    int hop = seg - noverlap;
    if (hop < 1) hop = 1;

    std::vector<double> win = hannWindowPeriodic(seg);
    double winSum = 0.0;
    for (double v : win) winSum += v;
    double scale = 1.0 / winSum;

    int nadd = 0;
    if (n > seg) {
        int rem = (n - seg) % hop;
        if (rem != 0) nadd = hop - rem;
    }
    std::vector<double> xPad(x);
    xPad.resize((size_t)n + nadd, 0.0);
    int nFrames = 1 + ((int)xPad.size() - seg) / hop;

    int nBins = seg / 2 + 1;
    StftData out;
    out.freqs.resize(nBins);
    for (int k = 0; k < nBins; ++k) out.freqs[k] = (double)k * sr / seg;
    out.mag.assign(nFrames, std::vector<double>(nBins, 0.0));

    for (int f = 0; f < nFrames; ++f) {
        std::vector<double> segData(seg);
        int start = f * hop;
        for (int i = 0; i < seg; ++i) segData[i] = xPad[start + i] * win[i];
        std::vector<std::complex<double>> spec = rfftD(segData);
        for (int k = 0; k < nBins; ++k) out.mag[f][k] = std::abs(spec[k]) * scale;
    }
    return out;
}

inline std::vector<int> activeFrames(const std::vector<std::vector<double>>& mag, double relThreshDb = -40.0) {
    int nFrames = (int)mag.size();
    std::vector<double> energy(nFrames, 0.0);
    for (int f = 0; f < nFrames; ++f) {
        double e = 0.0;
        for (double v : mag[f]) e += v * v;
        energy[f] = e;
    }
    double peak = 0.0;
    for (double e : energy) peak = std::max(peak, e);
    std::vector<int> idx;
    if (peak <= kDescEps) {
        idx.resize(nFrames);
        for (int f = 0; f < nFrames; ++f) idx[f] = f;
        return idx;
    }
    double thresh = peak * std::pow(10.0, relThreshDb / 10.0);
    for (int f = 0; f < nFrames; ++f) if (energy[f] >= thresh) idx.push_back(f);
    if (idx.empty()) {
        idx.resize(nFrames);
        for (int f = 0; f < nFrames; ++f) idx[f] = f;
    }
    return idx;
}

inline std::vector<double> avgMagOverActive(const std::vector<std::vector<double>>& mag, const std::vector<int>& active) {
    int K = (int)mag[0].size();
    std::vector<double> avg(K, 0.0);
    for (int f : active) for (int k = 0; k < K; ++k) avg[k] += mag[f][k];
    double cnt = (double)active.size();
    for (double& v : avg) v /= cnt;
    return avg;
}

// ---------------- spectral_centroid_spread / spectral_rolloff / spectral_flatness ----------------

inline void spectralCentroidSpread(const std::vector<double>& freqs, const std::vector<std::vector<double>>& mag,
                                    double& centroidOut, double& spreadOut) {
    std::vector<int> active = activeFrames(mag);
    std::vector<double> cs(active.size()), sps(active.size());
    for (size_t ai = 0; ai < active.size(); ++ai) {
        const auto& row = mag[active[ai]];
        double s = kDescEps;
        for (double v : row) s += v;
        double c = 0.0;
        for (size_t k = 0; k < row.size(); ++k) c += freqs[k] * row[k];
        c /= s;
        double sp = 0.0;
        for (size_t k = 0; k < row.size(); ++k) sp += (freqs[k] - c) * (freqs[k] - c) * row[k];
        sp = std::sqrt(sp / s);
        cs[ai] = c;
        sps[ai] = sp;
    }
    centroidOut = median(cs);
    spreadOut = median(sps);
}

inline double spectralRolloff(const std::vector<double>& freqs, const std::vector<std::vector<double>>& mag, double pct = 0.85) {
    std::vector<int> active = activeFrames(mag);
    std::vector<double> vals(active.size());
    for (size_t ai = 0; ai < active.size(); ++ai) {
        const auto& row = mag[active[ai]];
        int K = (int)row.size();
        double total = 0.0;
        for (double v : row) total += v * v;
        total += kDescEps;
        double target = pct * total;
        int idx = K - 1;
        double running = 0.0;
        for (int k = 0; k < K; ++k) {
            running += row[k] * row[k];
            if (running >= target) { idx = k; break; }
        }
        vals[ai] = freqs[idx];
    }
    return median(vals);
}

inline double spectralFlatness(const std::vector<std::vector<double>>& mag, double relThreshDb = -60.0) {
    std::vector<int> active = activeFrames(mag);
    std::vector<double> ratios;
    for (int f : active) {
        const auto& row = mag[f];
        double peak = 0.0;
        for (double v : row) peak = std::max(peak, v);
        if (peak <= kDescEps) continue;
        double thresh = peak * std::pow(10.0, relThreshDb / 20.0);
        double sumLog = 0.0, sumVal = 0.0;
        int cnt = 0;
        for (double v : row) {
            if (v >= thresh) {
                double g = v + kDescEps;
                sumLog += std::log(g);
                sumVal += g;
                ++cnt;
            }
        }
        if (cnt == 0) continue;
        double gm = std::exp(sumLog / cnt);
        double am = sumVal / cnt;
        ratios.push_back(gm / am);
    }
    if (ratios.empty()) return 0.0;
    return median(ratios);
}

// ---------------- _spectral_peaks / roughness / inharmonicity ----------------

inline void spectralPeaks(const std::vector<double>& freqs, const std::vector<double>& avgMag,
                           std::vector<double>& outFreqs, std::vector<double>& outAmp, int maxPeaks = 40,
                           double relThreshDb = -40.0) {
    outFreqs.clear(); outAmp.clear();
    double peak = 0.0;
    for (double v : avgMag) peak = std::max(peak, v);
    if (peak <= kDescEps) return;
    double thresh = peak * std::pow(10.0, relThreshDb / 20.0);
    int K = (int)avgMag.size();
    std::vector<int> idx;
    for (int i = 1; i < K - 1; ++i) {
        if (avgMag[i] > avgMag[i - 1] && avgMag[i] > avgMag[i + 1] && avgMag[i] > thresh) idx.push_back(i);
    }
    if (idx.empty()) return;
    std::vector<int> order(idx.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return avgMag[idx[a]] < avgMag[idx[b]];
    });
    std::reverse(order.begin(), order.end());
    int take = std::min((int)order.size(), maxPeaks);
    for (int i = 0; i < take; ++i) {
        int bin = idx[order[i]];
        outFreqs.push_back(freqs[bin]);
        outAmp.push_back(avgMag[bin]);
    }
}

inline double roughness(const std::vector<double>& freqs, const std::vector<std::vector<double>>& mag) {
    std::vector<int> active = activeFrames(mag);
    std::vector<double> avgMag = avgMagOverActive(mag, active);
    std::vector<double> pf, pa;
    spectralPeaks(freqs, avgMag, pf, pa);
    int n = (int)pf.size();
    if (n < 2) return 0.0;
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double fi = pf[i], fj = pf[j], ai = pa[i], aj = pa[j];
            double fmin = std::min(fi, fj), fmax = std::max(fi, fj);
            if (!(fmin > 0.0)) continue;
            double df = fmax - fmin;
            double s = 0.24 / (0.021 * fmin + 19.0);
            double x = ai * aj;
            double y = 2.0 * std::min(ai, aj) / (ai + aj + kDescEps);
            double term = std::pow(x, 0.1) * (0.5 * std::pow(y, 3.11)) *
                          (std::exp(-3.5 * s * df) - std::exp(-5.75 * s * df));
            total += term;
        }
    }
    return total;
}

inline double inharmonicity(const std::vector<double>& freqs, const std::vector<std::vector<double>>& mag, double f0) {
    // bit test invece di std::isfinite (-ffast-math, 2026-09-25)
    if (!descFiniteRange(f0) || !(f0 > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    std::vector<int> active = activeFrames(mag);
    std::vector<double> avgMag = avgMagOverActive(mag, active);
    std::vector<double> pf, pa;
    spectralPeaks(freqs, avgMag, pf, pa);
    if (pf.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    std::vector<double> pf2, pa2, h2;
    for (size_t i = 0; i < pf.size(); ++i) {
        double h = std::nearbyint(pf[i] / f0);
        if (h >= 1.0) { pf2.push_back(pf[i]); pa2.push_back(pa[i]); h2.push_back(h); }
    }
    if (pf2.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    double sumDW = 0.0, sumW = 0.0;
    for (size_t i = 0; i < pf2.size(); ++i) {
        double dev = std::fabs(pf2[i] - h2[i] * f0) / f0;
        double w = pa2[i] * pa2[i];
        sumDW += dev * w;
        sumW += w;
    }
    return sumDW / (sumW + kDescEps);
}

// ---------------- _chroma_vector / harmonic_tension ----------------

inline std::array<double, 12> chromaVector(const std::vector<double>& freqs, const std::vector<double>& avgMag, double fmin = 27.5) {
    std::array<double, 12> chroma{};
    for (size_t i = 0; i < freqs.size(); ++i) {
        if (freqs[i] < fmin) continue;
        double pc = std::fmod(69.0 + 12.0 * std::log2(freqs[i] / 440.0), 12.0);
        if (pc < 0.0) pc += 12.0;
        int b = ((int)std::nearbyint(pc)) % 12;
        if (b < 0) b += 12;
        chroma[b] += avgMag[i] * avgMag[i];
    }
    double s = 0.0;
    for (double v : chroma) s += v;
    if (s > kDescEps) for (double& v : chroma) v /= s;
    return chroma;
}

inline double harmonicTension(const std::vector<double>& freqs, const std::vector<std::vector<double>>& mag) {
    std::vector<int> active = activeFrames(mag);
    std::vector<double> avgMag = avgMagOverActive(mag, active);
    std::array<double, 12> c = chromaVector(freqs, avgMag);
    double r1 = 1.0, r2 = 1.0, r3 = 0.5;
    double x1 = 0, y1 = 0, x2 = 0, y2 = 0, x3 = 0, y3 = 0;
    for (int l = 0; l < 12; ++l) {
        x1 += c[l] * std::sin(l * 7.0 * kPi / 6.0);
        y1 += c[l] * std::cos(l * 7.0 * kPi / 6.0);
        x2 += c[l] * std::sin(l * 3.0 * kPi / 2.0);
        y2 += c[l] * std::cos(l * 3.0 * kPi / 2.0);
        x3 += c[l] * std::sin(l * 2.0 * kPi / 3.0);
        y3 += c[l] * std::cos(l * 2.0 * kPi / 3.0);
    }
    x1 *= r1; y1 *= r1; x2 *= r2; y2 *= r2; x3 *= r3; y3 *= r3;
    return std::sqrt(x1 * x1 + y1 * y1 + x2 * x2 + y2 * y2 + x3 * x3 + y3 * y3);
}

// ---------------- envelope_times ----------------

struct EnvelopeResult { double attackTime; double decayTime; bool decayCapped; };

inline EnvelopeResult envelopeTimes(const std::vector<double>& x, int sr, double frameMs = 10.0, double hopMs = 5.0, double floorDb = -60.0) {
    int n = (int)x.size();
    int frameLen = std::max(1, (int)(sr * frameMs / 1000.0));
    int hop = std::max(1, (int)(sr * hopMs / 1000.0));
    int nFrames = std::max(1, (n - frameLen) / hop + 1);
    std::vector<double> cum(n + 1, 0.0);
    for (int i = 0; i < n; ++i) cum[i + 1] = cum[i] + x[i] * x[i];
    std::vector<double> rms(nFrames), times(nFrames);
    for (int f = 0; f < nFrames; ++f) {
        int start = f * hop;
        int end = std::min(start + frameLen, n);
        int segLen = std::max(end - start, 1);
        double sums = cum[end] - cum[start];
        rms[f] = std::sqrt(sums / segLen + kDescEps);
        times[f] = (start + frameLen / 2.0) / sr;
    }
    int peakIdx = 0;
    for (int f = 1; f < nFrames; ++f) if (rms[f] > rms[peakIdx]) peakIdx = f;
    double attackTime = times[peakIdx];
    double thresh = rms[peakIdx] * std::pow(10.0, floorDb / 20.0);
    for (int f = peakIdx; f < nFrames; ++f) {
        if (rms[f] <= thresh) return {attackTime, times[f] - times[peakIdx], false};
    }
    return {attackTime, times[nFrames - 1] - times[peakIdx], true};
}

// ---------------- modulation (Hilbert via FFT) ----------------

inline std::vector<double> hilbertEnvelope(const std::vector<double>& x) {
    size_t n = x.size();
    std::vector<std::complex<double>> xin(n);
    for (size_t i = 0; i < n; ++i) xin[i] = std::complex<double>(x[i], 0.0);
    std::vector<std::complex<double>> X = fftComplex(xin, true);
    std::vector<double> h(n, 0.0);
    if (n % 2 == 0) {
        h[0] = 1.0;
        h[n / 2] = 1.0;
        for (size_t i = 1; i < n / 2; ++i) h[i] = 2.0;
    } else {
        h[0] = 1.0;
        for (size_t i = 1; i < (n + 1) / 2; ++i) h[i] = 2.0;
    }
    for (size_t i = 0; i < n; ++i) X[i] *= h[i];
    std::vector<std::complex<double>> analytic = fftComplex(X, false);
    std::vector<double> env(n);
    for (size_t i = 0; i < n; ++i) env[i] = std::abs(analytic[i]);
    return env;
}

struct ModulationResult { double modRate; double modDepth; };

inline ModulationResult modulation(const std::vector<double>& x, int sr, double bandLo = 0.5, double bandHi = 20.0) {
    EnvelopeResult et = envelopeTimes(x, sr);
    int n = (int)x.size();
    double clipped = std::min(std::max(et.attackTime, 0.0), (n - 1) / (double)sr);
    int start = (int)(clipped * sr);
    std::vector<double> xTrim;
    if (n - start >= 8) xTrim.assign(x.begin() + start, x.end());
    else xTrim = x;

    std::vector<double> env = hilbertEnvelope(xTrim);
    double envMean = 0.0;
    for (double v : env) envMean += v;
    envMean /= (double)env.size();
    std::vector<double> envAc(env.size());
    for (size_t i = 0; i < env.size(); ++i) envAc[i] = env[i] - envMean;

    if (envAc.size() < 8) return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};

    std::vector<double> win = hannWindowSymmetric((int)envAc.size());
    std::vector<double> windowed(envAc.size());
    for (size_t i = 0; i < envAc.size(); ++i) windowed[i] = envAc[i] * win[i];
    std::vector<std::complex<double>> specC = rfftD(windowed);
    int nBins = (int)specC.size();
    std::vector<double> spec(nBins);
    for (int i = 0; i < nBins; ++i) spec[i] = std::abs(specC[i]);
    std::vector<double> freqs(nBins);
    int nEnv = (int)envAc.size();
    for (int i = 0; i < nBins; ++i) freqs[i] = (double)i * sr / nEnv;

    std::vector<int> idxs;
    for (int i = 0; i < nBins; ++i) if (freqs[i] >= bandLo && freqs[i] <= bandHi) idxs.push_back(i);
    if (idxs.empty()) return {0.0, 0.0};
    int best = idxs[0];
    for (int i : idxs) if (spec[i] > spec[best]) best = i;

    double delta = 0.0;
    if (best > 0 && best < nBins - 1) {
        double a = spec[best - 1], b = spec[best], c = spec[best + 1];
        double denom = a - 2.0 * b + c;
        if (std::fabs(denom) > kDescEps) delta = 0.5 * (a - c) / denom;
        delta = std::min(std::max(delta, -0.5), 0.5);
    }
    double binWidth = freqs[1] - freqs[0];
    double modRate = freqs[best] + delta * binWidth;
    double modDepth = spec[best] / (envMean * nEnv / 2.0 + kDescEps);
    return {modRate, modDepth};
}

// ---------------- pitch_autocorr (YIN via FFT) ----------------

inline double pitchAutocorr(const std::vector<double>& xIn, int sr, double fmin = 40.0, double fmax = 2000.0, double voicedThresh = 0.15) {
    int n = (int)xIn.size();
    double mean = 0.0;
    for (double v : xIn) mean += v;
    mean /= n;
    std::vector<double> x(n);
    double sumSq = 0.0;
    for (int i = 0; i < n; ++i) { x[i] = xIn[i] - mean; sumSq += x[i] * x[i]; }
    if (sumSq < kDescEps || n < 2) return std::numeric_limits<double>::quiet_NaN();

    int tauMin = std::max((int)(sr / fmax), 1);
    int tauMax = std::min((int)(sr / fmin), n - 1);
    if (tauMax <= tauMin) return std::numeric_limits<double>::quiet_NaN();

    size_t target = (size_t)(2 * n - 1);
    size_t nfft = 1;
    while (nfft <= target) nfft <<= 1;

    std::vector<double> xPad(nfft, 0.0);
    for (int i = 0; i < n; ++i) xPad[i] = x[i];
    std::vector<std::complex<double>> spec = rfftD(xPad);
    std::vector<std::complex<double>> specSq(spec.size());
    for (size_t i = 0; i < spec.size(); ++i) specSq[i] = spec[i] * std::conj(spec[i]);
    std::vector<double> acFull = irfftD(specSq, nfft);

    std::vector<double> cum(n + 1, 0.0);
    for (int i = 0; i < n; ++i) cum[i + 1] = cum[i] + x[i] * x[i];

    std::vector<double> d(tauMax + 1, 0.0);
    for (int tau = 0; tau <= tauMax; ++tau) {
        double val = cum[n - tau] + cum[n] - cum[tau] - 2.0 * acFull[tau];
        d[tau] = std::max(val, 0.0);
    }

    std::vector<double> cmndf(tauMax + 1, 1.0);
    double running = 0.0;
    for (int tau = 1; tau <= tauMax; ++tau) {
        running += d[tau];
        if (running > kDescEps) cmndf[tau] = d[tau] * tau / running;
    }

    int start = std::max(tauMin, 2);
    int tau = -1;
    double threshList[3] = {voicedThresh, 0.25, 0.35};
    for (double thresh : threshList) {
        int found = -1;
        for (int t = start; t <= tauMax; ++t) {
            if (cmndf[t] < thresh) { found = t; break; }
        }
        if (found < 0) continue;
        tau = found;
        while (tau + 1 <= tauMax && cmndf[tau + 1] < cmndf[tau]) ++tau;
        break;
    }
    if (tau < 0) return std::numeric_limits<double>::quiet_NaN();

    double tauF = (double)tau;
    if (tauMin < tau && tau < tauMax) {
        double y0 = cmndf[tau - 1], y1 = cmndf[tau], y2 = cmndf[tau + 1];
        double denom = y0 - 2.0 * y1 + y2;
        if (std::fabs(denom) > kDescEps) {
            double delta = 0.5 * (y0 - y2) / denom;
            delta = std::min(std::max(delta, -1.0), 1.0);
            tauF = tau + delta;
        }
    }
    if (tauF <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    return sr / tauF;
}

// ---------------- pitch_mpm (McLeod Pitch Method) -- 2026-09-25 ----------------
// Porting 1:1 di analyzer/descriptors.py pitch_mpm (sostituisce pitch_autocorr/YIN come
// stimatore di analyze_signal dal 24/9, range PITCH_FMIN/FMAX 25-4500 Hz). NSDF = 2*acf/m',
// primo picco >= k*max, interpolazione parabolica del valore del picco LIMITATA a [y1, 1] e
// del lag clip [-1, 1]. NaN se segnale quasi nullo / nessun picco / chiarezza < clarity_min.
// pitchAutocorr (YIN) resta nel file, non piu' usato da analyzeSignal.
inline constexpr double kPitchFmin = 25.0;
inline constexpr double kPitchFmax = 4500.0;

inline double pitchMpm(const std::vector<double>& xIn, int sr, double fmin = kPitchFmin, double fmax = kPitchFmax,
                       double k = 0.93, double clarityMin = 0.5) {
    const double nanV = std::numeric_limits<double>::quiet_NaN();
    const int n = (int)xIn.size();
    if (n < 64) return nanV;
    double mean = 0.0;
    for (double v : xIn) mean += v;
    mean /= n;
    std::vector<double> x(n);
    double energy = 0.0;
    for (int i = 0; i < n; ++i) { x[i] = xIn[i] - mean; energy += x[i] * x[i]; }
    if (energy < 1e-12) return nanV;

    // nfft = 1 << (2n-1).bit_length()
    size_t v = (size_t)(2 * n - 1), bits = 0;
    while (v) { ++bits; v >>= 1; }
    const size_t nfft = (size_t)1 << bits;
    std::vector<double> xp(nfft, 0.0);
    std::copy(x.begin(), x.end(), xp.begin());
    std::vector<std::complex<double>> X = rfftD(xp);
    for (auto& c : X) c = c * std::conj(c);
    std::vector<double> acf = irfftD(X, nfft);

    std::vector<double> cs(n + 1, 0.0);
    for (int i = 0; i < n; ++i) cs[i + 1] = cs[i] + x[i] * x[i];
    std::vector<double> nsdf(n, 0.0);
    for (int tau = 0; tau < n; ++tau) {
        const double m = cs[n - tau] + (cs[n] - cs[tau]);
        nsdf[tau] = (m > 1e-12) ? 2.0 * acf[tau] / std::max(m, 1e-12) : 0.0;
    }
    const int tmin = std::max((int)(sr / fmax), 2);
    const int tmax = std::min((int)(sr / fmin), n - 2);
    int neg0 = -1;
    for (int i = 0; i < n; ++i) if (nsdf[i] < 0.0) { neg0 = i; break; }
    if (neg0 < 0) return nanV;

    std::vector<int> peaks;
    int t = std::max(neg0, tmin);
    while (t <= tmax) {
        if (nsdf[t] > 0.0) {
            int e = t;
            while (e + 1 <= tmax && nsdf[e + 1] > 0.0) ++e;
            int j = t;
            for (int q = t + 1; q <= e; ++q) if (nsdf[q] > nsdf[j]) j = q;  // np.argmax: primo massimo
            peaks.push_back(j);
            t = e + 1;
        } else {
            ++t;
        }
    }
    if (peaks.empty()) return nanV;

    auto pv = [&](int j) -> double {
        if (j >= 1 && j < n - 1) {
            const double y0 = nsdf[j - 1], y1 = nsdf[j], y2 = nsdf[j + 1];
            const double d = y0 - 2.0 * y1 + y2;
            if (d < -1e-12)
                return std::min(std::max(y1 - (y0 - y2) * (y0 - y2) / (8.0 * d), y1), 1.0);
        }
        return nsdf[j];
    };
    std::vector<double> vals(peaks.size());
    double hi = -std::numeric_limits<double>::max();
    for (size_t i = 0; i < peaks.size(); ++i) { vals[i] = pv(peaks[i]); hi = std::max(hi, vals[i]); }
    if (hi < clarityMin) return nanV;
    size_t sel = 0;
    for (size_t i = 0; i < vals.size(); ++i) if (vals[i] >= k * hi) { sel = i; break; }
    const int j = peaks[sel];
    double tt = (double)j;
    if (j >= 1 && j < n - 1) {
        const double y0 = nsdf[j - 1], y1 = nsdf[j], y2 = nsdf[j + 1];
        const double d = y0 - 2.0 * y1 + y2;
        if (std::fabs(d) > 1e-12)
            tt += std::min(std::max(0.5 * (y0 - y2) / d, -1.0), 1.0);
    }
    return (tt > 0.0) ? (double)sr / tt : nanV;
}

// ---------------- Formanti: LPC (Levinson-Durbin) + root-finding (Durand-Kerner) ----------------
// np.roots (autovalori della companion matrix, LAPACK) sostituito con Durand-Kerner: metodo
// iterativo auto-contenuto per tutte le radici simultaneamente, nessuna nuova dipendenza (deciso
// con l'utente 2026-09-22, alternativa scartata: vendorizzare una libreria di algebra lineare
// solo per questo). Adatto a polinomi di grado ~12 (ordine LPC) con radici ragionevolmente
// separate, caso tipico per formanti vocali/fisiche.

inline bool lpcLevinson(const std::vector<double>& frame, int order, std::vector<double>& aOut) {
    int N = (int)frame.size();
    std::vector<double> ac(order + 1, 0.0);
    for (int lag = 0; lag <= order; ++lag) {
        double s = 0.0;
        for (int j = 0; j + lag < N; ++j) s += frame[j] * frame[j + lag];
        ac[lag] = s;
    }
    if (ac[0] <= kDescEps) return false;
    std::vector<double> a(order + 1, 0.0);
    a[0] = 1.0;
    double e = ac[0];
    for (int i = 1; i <= order; ++i) {
        double acc = ac[i];
        for (int j = 1; j < i; ++j) acc += a[j] * ac[i - j];
        double k = -acc / e;
        std::vector<double> aPrev = a;
        for (int j = 1; j < i; ++j) a[j] = aPrev[j] + k * aPrev[i - j];
        a[i] = k;
        e *= (1.0 - k * k);
        if (e <= 0.0) break;
    }
    aOut = a;
    return true;
}

inline std::vector<std::complex<double>> durandKerner(const std::vector<double>& coeffsDesc, int maxIter = 200, double tol = 1e-12) {
    int n = (int)coeffsDesc.size() - 1;
    if (n <= 0) return {};
    std::vector<std::complex<double>> roots(n);
    std::complex<double> seed(0.4, 0.9);
    std::complex<double> p = 1.0;
    for (int k = 0; k < n; ++k) { roots[k] = p; p *= seed; }
    auto evalPoly = [&](std::complex<double> z) {
        std::complex<double> result = 0.0;
        for (double c : coeffsDesc) result = result * z + c;
        return result;
    };
    for (int iter = 0; iter < maxIter; ++iter) {
        double maxDelta = 0.0;
        for (int k = 0; k < n; ++k) {
            std::complex<double> num = evalPoly(roots[k]);
            std::complex<double> denom = 1.0;
            for (int j = 0; j < n; ++j) if (j != k) denom *= (roots[k] - roots[j]);
            if (std::abs(denom) < 1e-300) continue;
            std::complex<double> delta = num / denom;
            roots[k] -= delta;
            maxDelta = std::max(maxDelta, std::abs(delta));
        }
        if (maxDelta < tol) break;
    }
    return roots;
}

inline std::vector<double> formantsFromLpc(const std::vector<double>& a, int sr) {
    std::vector<std::complex<double>> roots = durandKerner(a);
    std::vector<double> freqs;
    for (auto& r : roots) {
        if (r.imag() <= 0.0) continue;
        double f = std::arg(r) * sr / (2.0 * kPi);
        double bw = -0.5 * (sr / kPi) * std::log(std::abs(r) + kDescEps);
        if (f > 90.0 && f < sr / 2.0 - 100.0 && bw < 400.0) freqs.push_back(f);
    }
    std::sort(freqs.begin(), freqs.end());
    return freqs;
}

// ---------------- formants(): resample + pre-enfasi + framing + LPC + root-finding ----------------
// Resampling: NON un porting fedele di scipy.signal.resample_poly (deciso con l'utente
// 2026-09-22) -- kernel sinc troncato con finestra di Hann, buona qualita' anti-aliasing ma
// non identico a scipy. E' l'unico punto approssimato di descriptors.py: le formanti restano
// gia' segnalate come "TARATURA APPROSSIMATIVA" anche nella docstring Python originale.
inline std::vector<double> resampleSinc(const std::vector<double>& x, int srIn, int srOut, int halfWidth = 32) {
    if (srOut == srIn) return x;
    double ratio = (double)srOut / (double)srIn;
    int nIn = (int)x.size();
    int nOut = (int)std::llround((double)nIn * ratio);
    std::vector<double> y(nOut, 0.0);
    double fc = 0.5 * std::min(srIn, srOut) / (double)srIn;  // taglio anti-aliasing, normalizzato su srIn
    for (int n = 0; n < nOut; ++n) {
        double posInSrc = (double)n * srIn / (double)srOut;
        int center = (int)std::floor(posInSrc);
        double acc = 0.0;
        for (int k = center - halfWidth; k <= center + halfWidth; ++k) {
            if (k < 0 || k >= nIn) continue;
            double t = posInSrc - k;
            double arg = 2.0 * fc * t;
            double h = (std::fabs(arg) < 1e-9) ? 1.0 : std::sin(kPi * arg) / (kPi * arg);
            double win = (std::fabs(t) > halfWidth) ? 0.0 : (0.5 + 0.5 * std::cos(kPi * t / (halfWidth + 1)));
            acc += x[k] * (2.0 * fc) * h * win;
        }
        y[n] = acc;
    }
    return y;
}

struct FormantsResult { double f1; double f2; double f3; };

inline FormantsResult formants(const std::vector<double>& x, int sr, int targetSr = 10000,
                                double frameMs = 25.0, double hopMs = 10.0, int order = 12) {
    const double nanV = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> xA;
    int srA;
    if (sr > targetSr) {
        xA = resampleSinc(x, sr, targetSr);
        srA = targetSr;
    } else {
        xA = x;
        srA = sr;
    }
    int nA = (int)xA.size();
    std::vector<double> pre(nA);
    if (nA > 0) pre[0] = xA[0];
    for (int i = 1; i < nA; ++i) pre[i] = xA[i] - 0.97 * xA[i - 1];

    int frameLen = (int)(srA * frameMs / 1000.0);
    int hop = std::max(1, (int)(srA * hopMs / 1000.0));
    if (frameLen < 8 || (int)pre.size() < frameLen) return {nanV, nanV, nanV};

    std::vector<double> win = hammingWindowSymmetric(frameLen);
    std::vector<double> f1s, f2s, f3s;
    for (int start = 0; start + frameLen <= (int)pre.size(); start += hop) {
        std::vector<double> frame(frameLen);
        double energy = 0.0;
        for (int i = 0; i < frameLen; ++i) {
            frame[i] = pre[start + i] * win[i];
            energy += frame[i] * frame[i];
        }
        if (energy < kDescEps) continue;
        std::vector<double> a;
        if (!lpcLevinson(frame, order, a)) continue;
        std::vector<double> fm = formantsFromLpc(a, srA);
        if (fm.size() >= 3) {
            f1s.push_back(fm[0]);
            f2s.push_back(fm[1]);
            f3s.push_back(fm[2]);
        }
    }
    if (f1s.empty()) return {nanV, nanV, nanV};
    return {median(f1s), median(f2s), median(f3s)};
}

// ---------------- analyze_signal (aggregatore) -- 2026-09-25 ----------------
// Porting di analyzer/descriptors.py analyze_signal(x, sr, extract_pitch=True, normalize=False)
// (usato dall'audio-in). Valori in ordine agents.DESCRIPTOR_KEYS (= kDescriptorNames:
// centroid, spread, rolloff, flatness, roughness, tension, inharmonicity, f1, f2, f3,
// mod_rate, mod_depth, attack, decay, pitch). NaN dove Python da' NaN; il chiamante nel
// plugin NON deve usare isnan (-ffast-math, vedi bug B punto 7d): usare valid[] calcolato qui
// con un confronto di range (v == v e' inaffidabile sotto fast-math, un range check no).
struct DescriptorSet {
    double v[15];
    bool valid[15];
    bool decayCapped;
};


inline DescriptorSet analyzeSignal(const std::vector<double>& x, int sr, bool extractPitch = true) {
    DescriptorSet out{};
    StftData stft = stftMag(x, sr);
    double centroid = 0.0, spread = 0.0;
    spectralCentroidSpread(stft.freqs, stft.mag, centroid, spread);
    const double rolloff = spectralRolloff(stft.freqs, stft.mag);
    const double flat = spectralFlatness(stft.mag);
    const double rough = roughness(stft.freqs, stft.mag);
    const double tension = harmonicTension(stft.freqs, stft.mag);
    const double f0 = extractPitch ? pitchMpm(x, sr) : std::numeric_limits<double>::quiet_NaN();
    const double inharm = inharmonicity(stft.freqs, stft.mag, f0);
    const FormantsResult fm = formants(x, sr);
    const ModulationResult mod = modulation(x, sr);
    const EnvelopeResult env = envelopeTimes(x, sr);
    const double vals[15] = {centroid, spread, rolloff, flat, rough, tension, inharm,
                             fm.f1, fm.f2, fm.f3, mod.modRate, mod.modDepth,
                             env.attackTime, env.decayTime, f0};
    for (int i = 0; i < 15; ++i) { out.v[i] = vals[i]; out.valid[i] = descFiniteRange(vals[i]); }
    out.decayCapped = env.decayCapped;
    return out;
}

} // namespace phimo
