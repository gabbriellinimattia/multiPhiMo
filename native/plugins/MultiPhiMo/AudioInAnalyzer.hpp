#pragma once
// AudioInAnalyzer.hpp -- porting di audio_input.py (2026-09-25): descrittori target
// ANALIZZATI dall'audio in ingresso della traccia (in Python: microfono) invece che da
// slider/CC/OSC.
//
// Stessa pipeline di AudioDescriptorSource: finestra scorrevole win_s=0.5 s, analisi
// (phimo::analyzeSignal = analyze_signal, extract_pitch=True, normalize=False) ogni
// hop_s=0.3 s, poi FADE ESPONENZIALE continuo verso l'ultimo set analizzato ogni
// interp_period=0.03 s con tau = smoothing_ms (20-1000, parametro host). Valori non
// validi (NaN) saltati come in Python; decay_capped ignorato.
//
// Thread: il thread audio (run()) SOLO pushAudio() in un ring SPSC pre-allocato (nessuna
// allocazione/lock). Analisi + interpolazione girano in UN thread proprio (in Python sono
// due: qui l'analisi C++ costa pochi ms, l'interpolazione si ferma solo per quel tempo).
// L'uscita e' lo stesso schema del drain OSC: pending[15] + hasPending[15] atomici,
// drenati da run() e scritti all'host (requestParameterValueChange).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#include "Descriptors.hpp"
#include "Resampler.hpp"

namespace phimo {

class AudioInAnalyzer {
public:
    static constexpr uint32_t kRingSize = 1u << 19;  // ~11.9 s a 44.1 kHz, potenza di 2
    static constexpr double kWinS = 0.5;
    static constexpr double kHopS = 0.3;
    static constexpr double kInGateLin = 0.00316;  // -50 dBFS: sotto, finestra ignorata (valori tenuti)
    static constexpr double kInterpPeriodS = 0.03;
    // 2026-09-25: analisi SEMPRE a 44.1 kHz (range descrittori, corpus KNN e selettore sono
    // a 44.1 kHz): l'ingresso al rate host passa per StreamResampler in questo thread.
    static constexpr double kAnalysisSr = 44100.0;

    AudioInAnalyzer() : ring_(kRingSize, 0.0f) {
        for (int k = 0; k < 15; ++k) { pending_[k].store(0.0f); hasPending_[k].store(false); }
    }
    ~AudioInAnalyzer() { stop(); }

    void start() {
        stop_.store(false);
        thread_ = std::thread([this]() { loop(); });
    }
    void stop() {
        if (thread_.joinable()) { stop_.store(true); thread_.join(); }
    }

    // --- chiamate dal thread audio / host (solo store atomici) ---
    void setEnabled(bool on) { enabled_.store(on, std::memory_order_relaxed); }
    void setSmoothingMs(float ms) { smoothingMs_.store(ms, std::memory_order_relaxed); }
    void setSampleRate(double sr) { if (sr > 1000.0) sampleRate_.store(sr, std::memory_order_relaxed); }

    // Thread audio: best-effort, se il ring e' pieno i campioni in eccesso sono scartati.
    void pushAudio(const float* in, uint32_t n) {
        const uint32_t w = write_.load(std::memory_order_relaxed);
        const uint32_t r = read_.load(std::memory_order_acquire);
        const uint32_t freeSpace = kRingSize - 1u - (w - r);
        const uint32_t m = std::min(n, freeSpace);
        for (uint32_t i = 0; i < m; ++i) ring_[(w + i) & (kRingSize - 1u)] = in[i];
        write_.store(w + m, std::memory_order_release);
    }

    // Come pushAudio, ma da un ingresso stereo: analizza (L+R)/2.
    void pushAudioStereo(const float* l, const float* r, uint32_t n) {
        const uint32_t w = write_.load(std::memory_order_relaxed);
        const uint32_t rd = read_.load(std::memory_order_acquire);
        const uint32_t freeSpace = kRingSize - 1u - (w - rd);
        const uint32_t m = std::min(n, freeSpace);
        for (uint32_t i = 0; i < m; ++i) ring_[(w + i) & (kRingSize - 1u)] = 0.5f * (l[i] + r[i]);
        write_.store(w + m, std::memory_order_release);
    }

    // Thread audio: preleva un valore smussato pronto per il descrittore k (ordine
    // kDescriptorNames). true = c'e' un valore nuovo.
    bool takePending(int k, float& out) {
        if (!hasPending_[k].exchange(false, std::memory_order_acq_rel)) return false;
        out = pending_[k].load(std::memory_order_relaxed);
        return true;
    }

private:
    void loop() {
        std::vector<double> buf;
        std::vector<double> norm;  // copia normalizzata a picco 0.9 per l'analisi
        double bufSr = 0.0;
        uint32_t sinceHop = 0;
        bool wasEnabled = false;
        bool rawReady = false;
        double raw[15] = {};
        bool rawValid[15] = {};
        double smooth[15] = {};
        bool smoothValid[15] = {};
        bool smoothInit = false;
        std::vector<float> chunk;
        chunk.reserve(kRingSize);
        std::vector<float> conv;       // chunk ricampionato a kAnalysisSr
        conv.reserve((size_t)kRingSize * 3);
        std::vector<float> tmp(4096);
        StreamResampler rs;
        bool useRs = false;

        while (!stop_.load(std::memory_order_relaxed)) {
            const bool on = enabled_.load(std::memory_order_relaxed);
            const double sr = sampleRate_.load(std::memory_order_relaxed);

            // drena sempre il ring (anche da spento, per non accumulare audio vecchio)
            chunk.clear();
            {
                const uint32_t r = read_.load(std::memory_order_relaxed);
                const uint32_t w = write_.load(std::memory_order_acquire);
                for (uint32_t i = r; i != w; ++i) chunk.push_back(ring_[i & (kRingSize - 1u)]);
                read_.store(w, std::memory_order_release);
            }

            // start() di Python: all'accensione (o cambio sample rate) si riparte da zero
            if (!on || !wasEnabled || sr != bufSr) {
                const size_t win = (size_t)(kWinS * kAnalysisSr);
                if (sr != bufSr) {
                    useRs = std::fabs(sr - kAnalysisSr) > 0.5;
                    if (useRs) rs.init(sr, kAnalysisSr, (uint32_t)tmp.size());
                }
                if (useRs) rs.reset();
                buf.assign(win, 0.0);  // come np.zeros(win_samples) in _capture_loop
                norm.assign(win, 0.0);
                bufSr = sr;
                sinceHop = 0;
                rawReady = false;
                smoothInit = false;
                for (int k = 0; k < 15; ++k) smoothValid[k] = false;
            }
            wasEnabled = on;
            if (!on) {
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                continue;
            }

            if (useRs && !chunk.empty()) {
                conv.clear();
                size_t off = 0;
                while (off < chunk.size()) {
                    const uint32_t n = (uint32_t)std::min<size_t>(chunk.size() - off, rs.maxInput());
                    rs.push(chunk.data() + off, n);
                    off += n;
                    uint32_t m;
                    while ((m = rs.drain(tmp.data(), (uint32_t)tmp.size())) > 0)
                        conv.insert(conv.end(), tmp.begin(), tmp.begin() + m);
                }
                chunk.swap(conv);
            }

            // finestra scorrevole (stessa logica di buf = concatenate([buf[len(chunk):], chunk]))
            if (!chunk.empty()) {
                const size_t win = buf.size();
                if (chunk.size() >= win) {
                    for (size_t i = 0; i < win; ++i) buf[i] = chunk[chunk.size() - win + i];
                } else {
                    std::move(buf.begin() + chunk.size(), buf.end(), buf.begin());
                    for (size_t i = 0; i < chunk.size(); ++i) buf[win - chunk.size() + i] = chunk[i];
                }
                sinceHop += (uint32_t)chunk.size();
            }

            const uint32_t hop = std::max<uint32_t>(1, (uint32_t)(kHopS * kAnalysisSr));
            if (sinceHop >= hop) {
                sinceHop = 0;
                // 2026-09-25 (deviazione da audio_input.py, normalize=False): GATE + normalizzazione.
                // Dal microfono gli slider restavano quasi fermi mentre da file funzionava:
                // sotto il gate (silenzio/rumore di fondo) i valori restano quelli dell'ultimo
                // suono invece di descrivere il rumore della stanza; sopra, la finestra viene
                // portata a picco 0.9 come i render del corpus (livello del mic ininfluente).
                double pk = 0.0;
                for (double v : buf) pk = std::max(pk, std::fabs(v));
                if (pk >= kInGateLin) {
                    const double g = 0.9 / pk;
                    for (size_t i = 0; i < buf.size(); ++i) norm[i] = buf[i] * g;
                    const DescriptorSet ds = analyzeSignal(norm, (int)kAnalysisSr, true);
                    for (int k = 0; k < 15; ++k) { raw[k] = ds.v[k]; rawValid[k] = ds.valid[k]; }
                    rawReady = true;
                }
            }

            // _interp_loop
            if (rawReady) {
                if (!smoothInit) {
                    for (int k = 0; k < 15; ++k) { smooth[k] = raw[k]; smoothValid[k] = rawValid[k]; }
                    smoothInit = true;
                } else {
                    const double tau = std::max((double)smoothingMs_.load(std::memory_order_relaxed), 1.0) / 1000.0;
                    const double coef = 1.0 - std::exp(-kInterpPeriodS / tau);
                    for (int k = 0; k < 15; ++k) {
                        if (!rawValid[k]) continue;
                        if (smoothValid[k]) smooth[k] += (raw[k] - smooth[k]) * coef;
                        else { smooth[k] = raw[k]; smoothValid[k] = true; }
                    }
                }
                for (int k = 0; k < 15; ++k) {
                    if (!smoothValid[k]) continue;
                    pending_[k].store((float)smooth[k], std::memory_order_relaxed);
                    hasPending_[k].store(true, std::memory_order_release);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    }

    std::vector<float> ring_;
    std::atomic<uint32_t> write_{0};
    std::atomic<uint32_t> read_{0};

    std::atomic<bool> enabled_{false};
    std::atomic<float> smoothingMs_{200.0f};
    std::atomic<double> sampleRate_{44100.0};
    std::atomic<bool> stop_{false};
    std::thread thread_;

    std::atomic<float> pending_[15];
    std::atomic<bool> hasPending_[15];
};

} // namespace phimo
