#pragma once
// SpectralMorph.hpp -- morph spettrale continuo, modello SINUSOIDI + RUMORE (2026-09-25,
// v2). Sostituisce la v1 a phase vocoder per bin, scartata dopo il test dal vivo
// (wabble = battimenti tra i bin vicini di uno stesso parziale; il rumore congelato
// diventava migliaia di sinusoidi fisse -> risonanze e bande fischianti).
//
// Metodo (scelto con l'utente dopo una ricerca, fonti in ~/Desktop/multiPhiMo/fonti):
// Spectral Modeling Synthesis / Harmonic+Noise (Serra; Haken "Additive Sound Morphing";
// Caetano-Rodet "Sound morphing by feature interpolation"):
//  - ANALISI (worker, computeMorphSnapshot): spettro medio (potenza) sulla parte stabile
//    del render; fino a kMaxPartials picchi spettrali (anche inarmonici) con frequenza e
//    ampiezza per interpolazione parabolica in dB; inviluppo del RUMORE = mediana della
//    magnitudine per bande di 1/3 d'ottava (robusta ai picchi), interpolata per bin.
//  - MORPH (thread audio, setTarget): i parziali nuovi vengono ABBINATI a quelli attuali
//    in scala log-frequenza dopo aver stimato un rapporto globale di trasposizione r
//    (cosi' un cambio di altezza diventa un GLISSANDO dei parziali abbinati, non una
//    dissolvenza tra due spettri); non abbinati: nascono/muoiono in dissolvenza.
//    Frequenze (in log) e ampiezze scivolano con tau = Smoothing.
//  - SINTESI (thread audio): banco di oscillatori a rotazione complessa (fase continua,
//    niente sin() per campione) + rumore bianco a fasi casuali filtrato dall'inviluppo
//    (overlap-add FFT, fasi NUOVE a ogni frame -> rumore vero, niente ringing).
// Real-time: nessuna allocazione dopo il costruttore, FFT radix-2 propria (pocketfft
// alloca internamente -> usata solo nel worker).

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <vector>

#include "Descriptors.hpp"

namespace phimo {

inline constexpr int kMorphN = 4096;
inline constexpr int kMorphHop = kMorphN / 4;
inline constexpr int kMorphBins = kMorphN / 2 + 1;
inline constexpr int kMaxPartials = 96;
inline constexpr int kMaxOsc = 3 * kMaxPartials;      // parziali che entrano + che escono
inline constexpr int kOscBlock = 32;                   // aggiornamento freq/amp ogni 32 campioni
// v2.1 (rifinitura dopo il test: livello, artefatti, saturazione progressiva in bande di rumore)
inline constexpr double kPeakSalienceDb = 12.0;        // picco = parziale solo se sopra il fondo di rumore di tanto
inline constexpr int kNoiseMinBandBins = 12;           // bande del rumore larghe almeno 12 bin (mediana = fondo, non picco)
inline constexpr double kMorphTargetRms = 0.2;         // livello costante di ogni istantanea (prima di gain/limiter)
inline constexpr double kMorphRenderSeconds = 1.5;     // durata massima dei render per il morph
inline constexpr double kMorphRequestSeconds = 0.3;    // un nuovo render ogni hop di audio_input

struct MorphSnapshot {
    int nPart = 0;
    float freq[kMaxPartials];
    float amp[kMaxPartials];      // ampiezza della sinusoide (non magnitudine FFT)
    float noise[kMorphBins];      // magnitudine per bin del rumore (scala FFT finestrata)
};

// ------------------------------------------------------------------ analisi (worker)
inline bool computeMorphSnapshot(const std::vector<float>& audio, int sr, MorphSnapshot& out) {
    const int N = kMorphN, hop = kMorphHop;
    std::vector<double> win(N);
    for (int i = 0; i < N; ++i) win[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / N);

    std::vector<double> x(audio.begin(), audio.end());
    if ((int)x.size() < N) x.resize(N, 0.0);
    const int nFrames = 1 + ((int)x.size() - N) / hop;

    std::vector<std::vector<double>> pw(nFrames, std::vector<double>(kMorphBins));
    std::vector<double> energy(nFrames, 0.0), frame(N);
    double maxE = 0.0;
    for (int t = 0; t < nFrames; ++t) {
        for (int i = 0; i < N; ++i) frame[i] = x[t * hop + i] * win[i];
        const auto X = rfftD(frame);
        for (int k = 0; k < kMorphBins; ++k) { pw[t][k] = std::norm(X[k]); energy[t] += pw[t][k]; }
        maxE = std::max(maxE, energy[t]);
    }
    if (!(maxE > 1e-12)) return false;

    // parte stabile: energia >= -10 dB del frame piu' forte, dopo i primi 50 ms (attacco)
    const int skip = (int)(0.05 * sr / hop);
    std::vector<int> steady;
    for (int t = 0; t < nFrames; ++t) if (energy[t] >= 0.1 * maxE && t >= skip) steady.push_back(t);
    if (steady.empty()) for (int t = 0; t < nFrames; ++t) if (energy[t] >= 0.1 * maxE) steady.push_back(t);

    std::vector<double> mag(kMorphBins, 0.0), db(kMorphBins);
    for (int k = 0; k < kMorphBins; ++k) {
        double p = 0.0;
        for (int t : steady) p += pw[t][k];
        mag[k] = std::sqrt(p / steady.size());
        db[k] = 20.0 * std::log10(mag[k] + 1e-12);
    }
    const double binHz = (double)sr / N;
    const double maxDb = *std::max_element(db.begin(), db.end());

    // --- rumore (fondo): mediana della magnitudine per bande di 1/3 d'ottava larghe almeno
    //     kNoiseMinBandBins bin (sotto ~400 Hz una banda di 1/3 d'ottava ha 1-3 bin e la
    //     mediana cadeva SUL parziale -> il parziale veniva risintetizzato anche come rumore)
    std::vector<double> bandF, bandM;
    for (double lo = 20.0; lo < 0.5 * sr;) {
        const double hi = std::max(lo * std::pow(2.0, 1.0 / 3.0), lo + kNoiseMinBandBins * binHz);
        std::vector<double> v;
        for (int k = (int)std::ceil(lo / binHz); k < kMorphBins && k * binHz < hi; ++k) v.push_back(mag[k]);
        if (!v.empty()) {
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            bandF.push_back(std::sqrt(lo * hi));
            bandM.push_back(v[v.size() / 2]);
        }
        lo = hi;
    }
    std::vector<double> floorMag(kMorphBins, 0.0);
    for (int k = 0; k < kMorphBins; ++k) {
        const double f = std::max(k * binHz, 1.0);
        double m = 0.0;
        if (!bandF.empty()) {
            if (f <= bandF.front()) m = bandM.front();
            else if (f >= bandF.back()) m = bandM.back();
            else {
                size_t j = 1;
                while (bandF[j] < f) ++j;
                const double u = std::log(f / bandF[j - 1]) / std::log(bandF[j] / bandF[j - 1]);
                m = bandM[j - 1] + (bandM[j] - bandM[j - 1]) * u;
            }
        }
        floorMag[k] = m;
    }

    // --- picchi: massimi locali sopra (max - 70 dB) E sopra il fondo di rumore di
    //     kPeakSalienceDb, 20 Hz .. 0.45*sr. Senza il test di salienza nelle regioni
    //     rumorose i massimi casuali del rumore diventavano sinusoidi (fino a 96), riabbinate
    //     a caso ogni 0.3 s -> grappoli densi = "bande di rumore filtrato".
    struct Peak { double f, a; };
    std::vector<Peak> peaks;
    const int kLo = std::max(2, (int)(20.0 / binHz)), kHi = std::min(kMorphBins - 2, (int)(0.45 * sr / binHz));
    for (int k = kLo; k <= kHi; ++k) {
        const double floorDb = 20.0 * std::log10(floorMag[k] + 1e-12);
        if (db[k] > db[k - 1] && db[k] >= db[k + 1] && db[k] > maxDb - 70.0 && db[k] > floorDb + kPeakSalienceDb) {
            const double a = db[k - 1], b = db[k], c = db[k + 1];
            const double den = a - 2.0 * b + c;
            const double p = (std::fabs(den) > 1e-12) ? std::min(std::max(0.5 * (a - c) / den, -0.5), 0.5) : 0.0;
            const double peakDb = b - 0.25 * (a - c) * p;
            // Hann: sinusoide di ampiezza A -> picco di magnitudine A*N/4
            peaks.push_back({(k + p) * binHz, std::pow(10.0, peakDb / 20.0) * 4.0 / N});
        }
    }
    std::sort(peaks.begin(), peaks.end(), [](const Peak& u, const Peak& v) { return u.a > v.a; });
    out.nPart = std::min((int)peaks.size(), kMaxPartials);
    for (int i = 0; i < out.nPart; ++i) { out.freq[i] = (float)peaks[i].f; out.amp[i] = (float)peaks[i].a; }

    // --- livello: ogni istantanea normalizzata alla stessa potenza prevista in sintesi
    //     (parziali: a^2/2; rumore: 16/(3N^2)*sum(m^2), vedi scala in synthNoiseHop).
    //     Prima il livello era quello assoluto della parte stabile di ogni render ->
    //     sostenuto troppo forte rispetto alle note normali e salti tra un render e l'altro.
    double pow2 = 0.0;
    for (int i = 0; i < out.nPart; ++i) pow2 += 0.5 * (double)out.amp[i] * out.amp[i];
    double pn = 0.0;
    for (int k = 0; k < kMorphBins; ++k) pn += floorMag[k] * floorMag[k];
    pow2 += pn * 16.0 / (3.0 * (double)N * N);
    const double g = (pow2 > 1e-20) ? kMorphTargetRms / std::sqrt(pow2) : 0.0;
    for (int i = 0; i < out.nPart; ++i) out.amp[i] = (float)(out.amp[i] * g);
    for (int k = 0; k < kMorphBins; ++k) out.noise[k] = (float)(floorMag[k] * g);
    return true;
}

// ------------------------------------------------------------------ voce (thread audio)
class MorphVoice {
public:
    MorphVoice() : rev_(kMorphN), twRe_(kMorphN / 2), twIm_(kMorphN / 2), win_(kMorphN),
                   re_(kMorphN), im_(kMorphN), ola_(kMorphN, 0.0f), noiseHop_(kMorphHop, 0.0f) {
        int bits = 0;
        while ((1 << bits) < kMorphN) ++bits;
        for (int i = 0; i < kMorphN; ++i) {
            int r = 0;
            for (int b = 0; b < bits; ++b) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
            rev_[i] = r;
        }
        for (int i = 0; i < kMorphN / 2; ++i) {
            twRe_[i] = std::cos(2.0 * kPi * i / kMorphN);
            twIm_[i] = std::sin(2.0 * kPi * i / kMorphN);
        }
        for (int i = 0; i < kMorphN; ++i) win_[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / kMorphN);
        reset();
    }

    bool active() const { return held_ || releasing_; }
    bool held() const { return held_; }
    int32_t noteId() const { return noteId_; }

    void noteOn(int32_t noteId, float gain) {
        if (!active()) reset();
        held_ = true; releasing_ = false; env_ = 1.0f;
        noteId_ = noteId; gain_ = gain;
    }
    void noteOff(int32_t noteId, int sr, double fadeMs) {
        if (!held_ || noteId != noteId_) return;
        held_ = false; releasing_ = true;
        envStep_ = 1.0f / std::max(1.0f, (float)(sr * fadeMs / 1000.0));
    }
    void forceRelease(int sr, double fadeMs) { if (held_) noteOff(noteId_, sr, fadeMs); }

    // Nuova istantanea: abbinamento dei parziali + nuovo inviluppo del rumore.
    void setTarget(const MorphSnapshot& s) {
        std::memcpy(tgtNoise_, s.noise, sizeof(tgtNoise_));
        const bool first = !hasTarget_;
        hasTarget_ = true;
        if (first) {
            std::memcpy(curNoise_, s.noise, sizeof(curNoise_));
            for (int i = 0; i < s.nPart; ++i) spawn(s.freq[i], s.amp[i], s.amp[i]);
            attack_ = 0.0f;  // rampa d'attacco breve (20 ms) invece del fade-in lungo
            return;
        }

        // 1) rapporto di trasposizione globale r: tra i candidati (rapporti fra i 5 parziali
        //    piu' forti nuovi/vecchi, + 1.0) quello che allinea piu' ampiezza nuova.
        int oldIdx[kMaxOsc]; int nOld = 0;
        for (int o = 0; o < kMaxOsc; ++o) if (osc_[o].alive && osc_[o].at > 0.0f) oldIdx[nOld++] = o;
        double bestR = 1.0, bestScore = -1.0;
        // log2 pre-calcolati: il punteggio e' O(nuovi x vecchi) per ~26 candidati, solo
        // sottrazioni nel thread audio (una volta ogni ~0.3 s).
        double lnNew[kMaxPartials], lnOld[kMaxOsc];
        for (int i = 0; i < s.nPart; ++i) lnNew[i] = std::log2(std::max(s.freq[i], 1.0f));
        for (int q = 0; q < nOld; ++q) lnOld[q] = std::log2(std::max(osc_[oldIdx[q]].ft, 1.0));
        auto score = [&](double r) {
            const double lr = std::log2(r);
            double sc = 0.0;
            for (int i = 0; i < s.nPart; ++i) {
                for (int q = 0; q < nOld; ++q) {
                    if (std::fabs(lnNew[i] - lr - lnOld[q]) < 0.025) { sc += s.amp[i]; break; }
                }
            }
            return sc;
        };
        {
            const double sc1 = score(1.0);
            bestScore = sc1;
            const int nTopNew = std::min(s.nPart, 5);
            int topOld[5]; int nTopOld = 0;
            for (int q = 0; q < nOld && nTopOld < 5; ++q) {  // i 5 vecchi piu' forti
                int best = -1;
                for (int z = 0; z < nOld; ++z) {
                    bool used = false;
                    for (int u = 0; u < nTopOld; ++u) if (topOld[u] == oldIdx[z]) used = true;
                    if (!used && (best < 0 || osc_[oldIdx[z]].at > osc_[best].at)) best = oldIdx[z];
                }
                if (best >= 0) topOld[nTopOld++] = best;
            }
            for (int i = 0; i < nTopNew; ++i)
                for (int u = 0; u < nTopOld; ++u) {
                    const double r = s.freq[i] / osc_[topOld[u]].ft;
                    if (r < 0.25 || r > 4.0) continue;
                    const double sc = score(r);
                    if (sc > bestScore * 1.05) { bestScore = sc; bestR = r; }
                }
        }

        // 2) abbinamento greedy (nuovi in ordine di ampiezza) entro mezzo semitono dopo /r
        bool matched[kMaxOsc] = {};
        for (int i = 0; i < s.nPart; ++i) {
            int bestO = -1; double bestD = 1.0 / 24.0;
            for (int q = 0; q < nOld; ++q) {
                const int o = oldIdx[q];
                if (matched[o]) continue;
                const double d = std::fabs(lnNew[i] - std::log2(bestR) - lnOld[q]);
                if (d < bestD) { bestD = d; bestO = o; }
            }
            if (bestO >= 0) { matched[bestO] = true; osc_[bestO].ft = s.freq[i]; osc_[bestO].at = s.amp[i]; }
            else spawn(s.freq[i], 0.0f, s.amp[i]);
        }
        for (int q = 0; q < nOld; ++q) if (!matched[oldIdx[q]]) osc_[oldIdx[q]].at = 0.0f;  // esce in dissolvenza
    }

    // Somma in out[] (non sovrascrive). smoothingMs = tau del morph.
    void render(float* out, uint32_t frames, int sr, float smoothingMs) {
        if (!active()) return;
        for (uint32_t i = 0; i < frames; ++i) {
            if (noisePos_ >= kMorphHop) { synthNoiseHop(sr, smoothingMs); noisePos_ = 0; }
            if (blockPos_ >= kOscBlock) { updateOscBlock(sr, smoothingMs); blockPos_ = 0; }
            float y = noiseHop_[noisePos_++];
            for (int o = 0; o < kMaxOsc; ++o) {
                Osc& v = osc_[o];
                if (!v.alive) continue;
                y += v.a * v.s;
                const double c = v.c * v.cr - v.s * v.sr;
                v.s = v.c * v.sr + v.s * v.cr;
                v.c = c;
                v.a += v.ainc;
            }
            ++blockPos_;
            float e = 1.0f;
            if (hasTarget_ && attack_ < 1.0f) { attack_ = std::min(1.0f, attack_ + 1.0f / (0.02f * sr)); e = attack_; }
            if (!hasTarget_) e = 0.0f;
            if (releasing_) {
                env_ -= envStep_;
                if (env_ <= 0.0f) { releasing_ = false; reset(); return; }
                e *= env_;
            }
            out[i] += y * gain_ * e;
        }
    }

private:
    struct Osc {
        bool alive = false;
        double f = 0, ft = 0;   // freq corrente / target (Hz)
        double a = 0, at = 0;   // ampiezza corrente / target
        double ainc = 0;        // rampa d'ampiezza per campione nel blocco
        double die = 0;         // passo della rampa lineare d'uscita (per blocco), 0 = vivo
        double c = 1, s = 0;    // rotatore (cos, sin) -- fase continua
        double cr = 1, sr = 0;  // rotazione per campione
    };

    void spawn(double f, double a, double at) {
        for (int o = 0; o < kMaxOsc; ++o) {
            if (osc_[o].alive) continue;
            Osc& v = osc_[o];
            v = Osc();
            v.alive = true; v.f = v.ft = f; v.a = a; v.at = at;
            const double ph = 2.0 * kPi * nextRand();  // fase iniziale casuale
            v.c = std::cos(ph); v.s = std::sin(ph);
            return;
        }
    }

    void updateOscBlock(int sr, float smoothingMs) {
        const double tau = std::max((double)smoothingMs, 1.0) / 1000.0;
        const double coef = 1.0 - std::exp(-(double)kOscBlock / sr / tau);
        for (int o = 0; o < kMaxOsc; ++o) {
            Osc& v = osc_[o];
            if (!v.alive) continue;
            if (v.ft > 0.0 && v.f > 0.0) v.f *= std::pow(v.ft / v.f, coef);  // glide in log-frequenza
            double aNew;
            if (v.at <= 0.0) {
                // parziale uscito: rampa LINEARE a zero in tau (prima esponenziale fino a 1e-6
                // = ~11*tau -> con Smoothing alto decine di parziali "zombie" si accumulavano,
                // saturavano gli slot e addensavano lo spettro in modo progressivo)
                if (v.die <= 0.0) v.die = std::max(v.a, 1e-9) * kOscBlock / (tau * sr);
                aNew = v.a - v.die;
                if (aNew <= 0.0) { v.alive = false; continue; }
            } else {
                const double at = (v.f > 0.45 * sr) ? 0.0 : v.at;             // mai oltre Nyquist
                aNew = v.a + (at - v.a) * coef;
            }
            v.ainc = (aNew - v.a) / kOscBlock;
            const double w = 2.0 * kPi * v.f / sr;
            v.cr = std::cos(w); v.sr = std::sin(w);
            const double r = std::sqrt(v.c * v.c + v.s * v.s);                // rinormalizza il rotatore
            if (r > 0.0) { v.c /= r; v.s /= r; }
        }
    }

    // Rumore: magnitudine interpolata verso il target, fasi casuali NUOVE a ogni frame.
    void synthNoiseHop(int sr, float smoothingMs) {
        const double tau = std::max((double)smoothingMs, 1.0) / 1000.0;
        const float coef = (float)(1.0 - std::exp(-(double)kMorphHop / sr / tau));
        for (int k = 0; k < kMorphBins; ++k) {
            curNoise_[k] += (tgtNoise_[k] - curNoise_[k]) * coef;
            const double ph = 2.0 * kPi * nextRand();
            re_[k] = curNoise_[k] * std::cos(ph);
            im_[k] = curNoise_[k] * std::sin(ph);
        }
        re_[0] = 0.0; im_[0] = 0.0;
        im_[kMorphN / 2] = 0.0;
        for (int k = kMorphBins; k < kMorphN; ++k) { re_[k] = re_[kMorphN - k]; im_[k] = -im_[kMorphN - k]; }
        ifft();
        // varianza: irfft(1/N) -> 0.375*var originale; finestra di sintesi Hann + OLA N/4 ->
        // x1.5 -> 0.5625; compensazione 1/sqrt(0.5625) = 4/3.
        const float scale = (4.0f / 3.0f) / kMorphN;
        for (int i = 0; i < kMorphN; ++i) ola_[i] += (float)(re_[i] * win_[i]) * scale;
        for (int i = 0; i < kMorphHop; ++i) noiseHop_[i] = ola_[i];
        std::memmove(ola_.data(), ola_.data() + kMorphHop, sizeof(float) * (kMorphN - kMorphHop));
        std::fill(ola_.begin() + (kMorphN - kMorphHop), ola_.end(), 0.0f);
    }

    void reset() {
        for (auto& v : osc_) v = Osc();
        std::fill(std::begin(curNoise_), std::end(curNoise_), 0.0f);
        std::fill(std::begin(tgtNoise_), std::end(tgtNoise_), 0.0f);
        std::fill(ola_.begin(), ola_.end(), 0.0f);
        std::fill(noiseHop_.begin(), noiseHop_.end(), 0.0f);
        hasTarget_ = false; noisePos_ = kMorphHop; blockPos_ = kOscBlock; env_ = 1.0f; attack_ = 1.0f;
    }

    double nextRand() {  // xorshift32 -> [0,1)
        rng_ ^= rng_ << 13; rng_ ^= rng_ >> 17; rng_ ^= rng_ << 5;
        return (rng_ >> 8) / 16777216.0;
    }

    // FFT complessa radix-2 in-place, segno + (inversa, non normalizzata).
    void ifft() {
        for (int i = 0; i < kMorphN; ++i) {
            const int j = rev_[i];
            if (j > i) { std::swap(re_[i], re_[j]); std::swap(im_[i], im_[j]); }
        }
        for (int len = 2; len <= kMorphN; len <<= 1) {
            const int half = len >> 1, step = kMorphN / len;
            for (int i = 0; i < kMorphN; i += len) {
                for (int j = 0; j < half; ++j) {
                    const double wr = twRe_[j * step], wi = twIm_[j * step];
                    const double xr = re_[i + j + half], xi = im_[i + j + half];
                    const double tr = xr * wr - xi * wi, ti = xr * wi + xi * wr;
                    re_[i + j + half] = re_[i + j] - tr; im_[i + j + half] = im_[i + j] - ti;
                    re_[i + j] += tr; im_[i + j] += ti;
                }
            }
        }
    }

    std::vector<int> rev_;
    std::vector<double> twRe_, twIm_, win_, re_, im_;
    std::vector<float> ola_, noiseHop_;
    Osc osc_[kMaxOsc];
    float curNoise_[kMorphBins], tgtNoise_[kMorphBins];
    bool held_ = false, releasing_ = false, hasTarget_ = false;
    int32_t noteId_ = -1;
    float gain_ = 1.0f, env_ = 1.0f, envStep_ = 0.0f, attack_ = 1.0f;
    int noisePos_ = kMorphHop, blockPos_ = kOscBlock;
    uint32_t rng_ = 0x9E3779B9u;
};

} // namespace phimo
