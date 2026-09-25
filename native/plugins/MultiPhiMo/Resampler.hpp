#pragma once
// Resampler.hpp -- 2026-09-25: conversione di sample rate in streaming.
// Il motore rende SEMPRE a 44.1 kHz (kResonatorSR: corpus, pesi, fisica, morph, limiter).
// Con host a un altro rate: UN resampler sull'uscita (pull: il mixer genera esattamente i
// campioni interni che servono al blocco host) e uno sull'audio-in (host -> 44.1 kHz, nel
// thread d'analisi). Stesso schema dei plugin a rate interno fisso / emulatori.
// Sinc con finestra di Kaiser (beta 8, ~-80 dB), 2H tap con H = 40 / min(1, out/in) (il
// filtro si allarga quando si scende di rate), taglio a 0.45 * min(in, out); tabella
// polifase di kPhases+1 righe con interpolazione lineare tra fasi (rapporto arbitrario).
// init() alloca (mai dal thread audio); inputNeeded/push/process/drain non allocano.
// Latenza ~H campioni d'ingresso (< 1 ms). Nessun isnan/isfinite (-ffast-math).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace phimo {

class StreamResampler {
public:
    static constexpr int kPhases = 256;
    static constexpr int kBaseHalf = 40;
    static constexpr double kBeta = 8.0;
    static constexpr double kCutoff = 0.45;  // frazione del rate piu' basso tra in e out

    void init(double inRate, double outRate, uint32_t maxOut) {
        step_ = inRate / outRate;
        const double r = std::min(1.0, outRate / inRate);
        half_ = (int)std::ceil(kBaseHalf / r);
        taps_ = 2 * half_;
        const double fc = kCutoff * r;  // cicli per campione d'ingresso
        const double kPiR = 3.14159265358979323846;
        const double i0b = besselI0(kBeta);
        table_.assign((size_t)(kPhases + 1) * taps_, 0.0f);
        for (int p = 0; p <= kPhases; ++p) {
            const double frac = (double)p / kPhases;
            for (int j = 0; j < taps_; ++j) {
                const double x = (double)(j - half_ + 1) - frac;  // distanza dal centro
                const double u = x / half_;
                const double w = (std::fabs(u) >= 1.0) ? 0.0 : besselI0(kBeta * std::sqrt(1.0 - u * u)) / i0b;
                const double s = (std::fabs(x) < 1e-12) ? 2.0 * fc : std::sin(2.0 * kPiR * fc * x) / (kPiR * x);
                table_[(size_t)p * taps_ + j] = (float)(s * w);
            }
        }
        maxIn_ = (uint32_t)std::ceil(maxOut * step_) + 4;
        hist_.assign((size_t)2 * taps_ + maxIn_ + 8, 0.0f);
        reset();
    }

    void reset() {
        std::fill(hist_.begin(), hist_.end(), 0.0f);
        n_ = (uint32_t)half_;       // storia iniziale di zeri
        pos_ = (double)(half_ - 1);  // centro della prossima uscita (in campioni d'ingresso)
    }

    uint32_t maxInput() const { return maxIn_; }

    // Campioni d'ingresso da passare a push() prima di process(out, outFrames).
    uint32_t inputNeeded(uint32_t outFrames) const {
        if (outFrames == 0) return 0;
        const double tLast = pos_ + (double)(outFrames - 1) * step_;
        const int64_t need = (int64_t)std::floor(tLast) + half_ + 1 - (int64_t)n_;
        return need > 0 ? (uint32_t)need : 0;
    }

    // n <= maxInput(), e la storia va consumata (process/drain) prima del push successivo.
    void push(const float* in, uint32_t n) {
        std::memcpy(hist_.data() + n_, in, sizeof(float) * n);
        n_ += n;
    }

    // Produce outFrames campioni (servono inputNeeded(outFrames) campioni gia' in push).
    void process(float* out, uint32_t outFrames) {
        for (uint32_t k = 0; k < outFrames; ++k) out[k] = next();
        trim();
    }

    // Modo push (audio-in): produce fino a maxOut campioni con la storia disponibile.
    uint32_t drain(float* out, uint32_t maxOut) {
        uint32_t k = 0;
        while (k < maxOut && std::floor(pos_) + half_ + 1 <= (double)n_) out[k++] = next();
        trim();
        return k;
    }

private:
    float next() {
            const int64_t i = (int64_t)std::floor(pos_);
            const double ph = (pos_ - (double)i) * kPhases;
            const int p0 = std::min((int)ph, kPhases - 1);
            const float fr = (float)(ph - p0);
            const float* c0 = table_.data() + (size_t)p0 * taps_;
            const float* c1 = c0 + taps_;
            const float* h = hist_.data() + (i - half_ + 1);
            float acc = 0.0f;
            for (int j = 0; j < taps_; ++j) acc += h[j] * (c0[j] + fr * (c1[j] - c0[j]));
            pos_ += step_;
            return acc;
    }

    void trim() {
        const int64_t keepFrom = (int64_t)std::floor(pos_) - half_ + 1;
        if (keepFrom <= 0) return;
        const uint32_t d = (uint32_t)std::min<int64_t>(keepFrom, (int64_t)n_);
        std::memmove(hist_.data(), hist_.data() + d, sizeof(float) * (n_ - d));
        n_ -= d;
        pos_ -= (double)d;
    }

    static double besselI0(double x) {
        double sum = 1.0, term = 1.0;
        const double q = 0.25 * x * x;
        for (int k = 1; k < 50; ++k) {
            term *= q / ((double)k * k);
            sum += term;
            if (term < 1e-12 * sum) break;
        }
        return sum;
    }

    std::vector<float> table_;
    std::vector<float> hist_;
    double step_ = 1.0;
    double pos_ = 0.0;
    int half_ = 0;
    int taps_ = 0;
    uint32_t n_ = 0;
    uint32_t maxIn_ = 0;
};

} // namespace phimo
