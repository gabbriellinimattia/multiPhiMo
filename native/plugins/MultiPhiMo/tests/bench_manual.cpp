// tests/bench_manual.cpp -- prestazioni del motore C++ v5 per il manuale d'uso (2026-09-28).
// Per ogni eccitatore: N target RAGGIUNGIBILI (descrittori di righe del dataset contenute nel .bin, forma della
// riga, pitch della riga come freq), rete (Surrogate.hpp) -> render + analisi (stesso percorso del plugin), poi
// ricerca reale fino a 30 prove. Errore per descrittore = |misurato - target| / (p95 - p5 del descrittore
// sull'eccitatore), in unita' trasformate (log dove surrogate.OFFSET), in % (stessa metrica di "ERRORE" del
// training). Tempi = ritardo reale sul thread di calcolo (un core): rete = risoluzione + primo render.
// Uso (da native/plugins/MultiPhiMo/tests):
//   g++ -std=c++17 -O3 bench_manual.cpp -o bench_manual && ./bench_manual ../../../data/surrogate prestazioni_v5.txt 20
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../Render.hpp"
#include "../Surrogate.hpp"

using namespace phimo;
using Clock = std::chrono::steady_clock;

static const char* kExc[10] = {"bird", "blow", "bow", "chaos", "mechanical", "noise", "pluck", "shaker", "strike", "vocal"};
static const int kBudgets[3] = {1, 10, 30};

static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[(size_t)std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5))];
}

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("uso: bench_manual <cartella surrogate> <file.txt> [N target per eccitatore]\n"); return 1; }
    const std::string dir = argv[1];
    const int N = argc > 3 ? std::atoi(argv[3]) : 20;
    FILE* f = std::fopen(argv[2], "w");
    if (!f) { std::printf("impossibile scrivere %s\n", argv[2]); return 1; }
    std::fprintf(f, "multiPhiMo 0.2.1 -- prestazioni del motore C++ (rete v5 + ricerca reale), %d target raggiungibili per eccitatore\n", N);
    std::fprintf(f, "Errore per descrittore: |misurato - richiesto| / (p95 - p5 del descrittore per quell'eccitatore), in %%.\n");
    std::fprintf(f, "Colonne: rete = prima nota (stima della rete); +10 / +30 = dopo 10 / 30 render di ricerca reale.\n");
    std::fprintf(f, "Tempi (mediana, un core): ritardo della prima nota (risoluzione + render) e tempo per arrivare a 10 / 30 prove.\n\n");
    std::mt19937 rng(1234);
    for (const char* exc : kExc) {
        SurrogateModel m;
        std::string err;
        if (!m.load(dir + "/" + exc + ".bin", err)) { std::fprintf(f, "%s: %s\n", exc, err.c_str()); continue; }
        // scala per descrittore: p95 - p5 dei valori trasformati delle righe
        double scale[kSgDesc];
        for (int j = 0; j < kSgDesc; ++j) {
            std::vector<double> t;
            for (int r = 0; r < m.rows; ++r)
                if (m.M[(size_t)r * kSgDesc + j]) t.push_back(m.Z[(size_t)r * kSgDesc + j] * m.sd[j] + m.mu[j]);
            scale[j] = std::max(pct(t, 0.95) - pct(t, 0.05), 1e-9);
        }
        std::vector<double> errSum[3], errCnt[3];
        for (auto& v : errSum) v.assign(kSgDesc, 0.0);
        for (auto& v : errCnt) v.assign(kSgDesc, 0.0);
        std::vector<double> tFirst, t10, t30, tSolve;
        std::uniform_int_distribution<int> pick(0, m.rows - 1);
        int done = 0, fails = 0;
        while (done < N && fails < 5 * N) {
            const int r = pick(rng);
            double tv[kSgDesc], ttr[kSgDesc];
            bool valid[kSgDesc];
            for (int j = 0; j < kSgDesc; ++j) {
                valid[j] = m.M[(size_t)r * kSgDesc + j] != 0;
                ttr[j] = m.Z[(size_t)r * kSgDesc + j] * m.sd[j] + m.mu[j];
                tv[j] = m.hasOff[j] ? std::pow(10.0, ttr[j]) - m.off[j] : ttr[j];
            }
            const std::string shape = m.shapes[m.S[r]];
            const double freq = valid[kSgPitch] ? tv[kSgPitch] : 0.0;
            try {
                SurrogateSolver s;
                s.m = &m;
                SgDecoded dec;
                const auto t0 = Clock::now();
                if (!s.solve(tv, valid, shape, freq, rng, dec)) { ++fails; continue; }
                tSolve.push_back(std::chrono::duration<double>(Clock::now() - t0).count());
                SgRealSearch rs;
                rs.init(s, rng());
                auto render = [&](const SgDecoded& d, double* meas, bool* mv) {
                    const std::vector<float> y = renderRaw(exc, shape, d.exc, d.res, d.coup, rng());
                    const std::vector<double> x(y.begin(), y.end());
                    const DescriptorSet ds = analyzeSignal(x, kResonatorSR, true);
                    for (int j = 0; j < kSgDesc; ++j) { meas[j] = ds.v[j]; mv[j] = ds.valid[j]; }
                };
                int bi = 0;
                while (rs.n < kBudgets[2]) {
                    rs.step(render);
                    if (rs.n == kBudgets[bi]) {
                        const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
                        (bi == 0 ? tFirst : bi == 1 ? t10 : t30).push_back(dt);
                        for (int j = 0; j < kSgDesc; ++j) {
                            if (!valid[j] || !rs.vBest[j]) continue;
                            const double tm = m.tdesc(rs.mBest[j], j);
                            errSum[bi][j] += std::fabs(tm - ttr[j]) / scale[j];
                            errCnt[bi][j] += 1.0;
                        }
                        ++bi;
                    }
                }
                ++done;
            } catch (const std::exception& e) {
                ++fails;
            }
        }
        std::fprintf(f, "== %s  (%d target)\n", exc, done);
        std::fprintf(f, "  %-18s %8s %8s %8s\n", "descrittore", "rete", "+10", "+30");
        double mean[3] = {0, 0, 0};
        int under10[3] = {0, 0, 0};
        for (int j = 0; j < kSgDesc; ++j) {
            std::fprintf(f, "  %-18s", j == 5 ? "tonal_focus" : kDescriptorKeys[j]);
            for (int b = 0; b < 3; ++b) {
                const double e = errCnt[b][j] > 0 ? 100.0 * errSum[b][j] / errCnt[b][j] : -1.0;
                if (e >= 0) std::fprintf(f, " %7.1f%%", e); else std::fprintf(f, " %8s", "n/d");
                if (e >= 0) { mean[b] += e / kSgDesc; under10[b] += e < 10.0; }
            }
            std::fprintf(f, "\n");
        }
        std::fprintf(f, "  %-18s %7.1f%% %7.1f%% %7.1f%%\n", "media", mean[0], mean[1], mean[2]);
        std::fprintf(f, "  %-18s %5d/15 %5d/15 %5d/15\n", "sotto il 10%", under10[0], under10[1], under10[2]);
        std::fprintf(f, "  ritardo prima nota %.2f s (di cui rete %.2f s) | 10 prove %.1f s | 30 prove %.1f s\n\n",
                     pct(tFirst, 0.5), pct(tSolve, 0.5), pct(t10, 0.5), pct(t30, 0.5));
        std::fflush(f);
        std::printf("%s fatto (%d target)\n", exc, done);
    }
    std::fclose(f);
    return 0;
}
