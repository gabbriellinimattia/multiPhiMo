// tests/bench_errors.cpp (2026-09-30) -- errore MIN / MEDIO / MAX di ogni descrittore per ogni eccitatore x forma del
// risonatore, per documentare il grado di errore del motore C++ (stesso percorso di bench_manual.cpp e del plugin).
// Per ogni eccitatore e ogni forma: N target RAGGIUNGIBILI (righe del .bin con quella forma: descrittori della riga,
// pitch della riga come freq) -> rete -> ricerca reale fino a 30 render. Errore per target e descrittore:
//   %   = |misurato - richiesto| / (p95 - p5 del descrittore per l'eccitatore), in unita' trasformate (come bench_manual);
//   ass = |misurato - richiesto| nelle unita' del descrittore (Hz, s, ...).
// Uscite: <prefisso>.csv (eccitatore, forma, budget, descrittore, n, min/p10/media/p90/max in %, min/media/max assoluti)
// e <prefisso>.txt (riepiloghi per eccitatore e per forma: min, p10, media, p90, max in %, rete e +30).
// Uso (da native/plugins/MultiPhiMo/tests):
//   g++ -std=c++17 -O3 bench_errors.cpp -o bench_errors && taskpolicy -b ./bench_errors ../../../data/surrogate errori_descrittori 10
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "../Render.hpp"
#include "../Surrogate.hpp"

using namespace phimo;

static const char* kExc[10] = {"bird", "blow", "bow", "chaos", "mechanical", "noise", "pluck", "shaker", "strike", "vocal"};
static const int kBudgets[3] = {1, 10, 30};
static const char* kBudgetName[3] = {"rete", "+10", "+30"};

static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[(size_t)std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5))];
}
static const char* dname(int j) { return j == 5 ? "tonal_focus" : kDescriptorKeys[j]; }

struct Acc { std::vector<double> p, a; };            // errori in % e assoluti
using Key = std::string;                               // "exc|shape"
static std::map<Key, Acc> E[3][kSgDesc];

static void stats(const std::vector<double>& v, double& mn, double& me, double& mx) {
    mn = *std::min_element(v.begin(), v.end());
    mx = *std::max_element(v.begin(), v.end());
    me = 0.0;
    for (double x : v) me += x;
    me /= v.size();
}
static double q(std::vector<double> v, double p) {   // percentile con interpolazione lineare
    std::sort(v.begin(), v.end());
    const double x = p * (v.size() - 1);
    const size_t i = (size_t)x;
    return i + 1 < v.size() ? v[i] + (x - i) * (v[i + 1] - v[i]) : v[i];
}

// riepilogo: raggruppa per eccitatore (byExc) o per forma, righe = descrittori, colonne = rete e +30: min, p10, media, p90, max
static void summary(FILE* f, bool byExc, const std::vector<std::string>& groups) {
    for (const auto& g : groups) {
        std::fprintf(f, "== %s %s\n  %-18s | %-34s | %-34s\n  %-18s | %6s %6s %6s %6s %6s | %6s %6s %6s %6s %6s\n",
                     byExc ? "eccitatore" : "forma", g.c_str(), "", "rete (prima nota), %", "+30 render, %", "descrittore",
                     "min", "p10", "media", "p90", "max", "min", "p10", "media", "p90", "max");
        for (int j = 0; j < kSgDesc; ++j) {
            std::fprintf(f, "  %-18s", dname(j));
            for (int b : {0, 2}) {
                std::vector<double> v;
                for (const auto& kv : E[b][j]) {
                    const std::string k = kv.first;
                    const std::string part = byExc ? k.substr(0, k.find('|')) : k.substr(k.find('|') + 1);
                    if (part == g) v.insert(v.end(), kv.second.p.begin(), kv.second.p.end());
                }
                if (v.empty()) { std::fprintf(f, " | %34s", "n/d"); continue; }
                double mn, me, mx;
                stats(v, mn, me, mx);
                std::fprintf(f, " | %6.1f %6.1f %6.1f %6.1f %6.1f", mn, q(v, 0.1), me, q(v, 0.9), mx);
            }
            std::fprintf(f, "\n");
        }
        // % media: media delle medie dei 15 descrittori
        std::fprintf(f, "  %-18s", "MEDIA (15 descr.)");
        for (int b : {0, 2}) {
            double tot = 0.0;
            int nd = 0;
            for (int j = 0; j < kSgDesc; ++j) {
                std::vector<double> v;
                for (const auto& kv : E[b][j]) {
                    const std::string k = kv.first;
                    const std::string part = byExc ? k.substr(0, k.find('|')) : k.substr(k.find('|') + 1);
                    if (part == g) v.insert(v.end(), kv.second.p.begin(), kv.second.p.end());
                }
                if (v.empty()) continue;
                double mn, me, mx;
                stats(v, mn, me, mx);
                tot += me;
                ++nd;
            }
            std::fprintf(f, " | %13s %6.1f %13s", "", nd ? tot / nd : 0.0, "");
        }
        std::fprintf(f, "\n\n");
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("uso: bench_errors <cartella surrogate> <prefisso uscite> [N target per eccitatore x forma]\n"); return 1; }
    const std::string dir = argv[1], out = argv[2];
    const int N = argc > 3 ? std::atoi(argv[3]) : 10;
    std::mt19937 rng(1234);
    std::vector<std::string> allShapes;
    for (const char* exc : kExc) {
        SurrogateModel m;
        std::string err;
        if (!m.load(dir + "/" + exc + ".bin", err)) { std::printf("%s: %s\n", exc, err.c_str()); continue; }
        double scale[kSgDesc];
        for (int j = 0; j < kSgDesc; ++j) {
            std::vector<double> t;
            for (int r = 0; r < m.rows; ++r)
                if (m.M[(size_t)r * kSgDesc + j]) t.push_back(m.Z[(size_t)r * kSgDesc + j] * m.sd[j] + m.mu[j]);
            scale[j] = std::max(pct(t, 0.95) - pct(t, 0.05), 1e-9);
        }
        for (int si = 0; si < (int)m.shapes.size(); ++si) {
            const std::string shape = m.shapes[si];
            if (std::find(allShapes.begin(), allShapes.end(), shape) == allShapes.end()) allShapes.push_back(shape);
            std::vector<int> rowsS;
            for (int r = 0; r < m.rows; ++r) if (m.S[r] == si) rowsS.push_back(r);
            if (rowsS.empty()) continue;
            std::uniform_int_distribution<int> pick(0, (int)rowsS.size() - 1);
            const Key key = std::string(exc) + "|" + shape;
            int done = 0, fails = 0;
            while (done < N && fails < 5 * N) {
                const int r = rowsS[pick(rng)];
                double tv[kSgDesc], ttr[kSgDesc];
                bool valid[kSgDesc];
                for (int j = 0; j < kSgDesc; ++j) {
                    valid[j] = m.M[(size_t)r * kSgDesc + j] != 0;
                    ttr[j] = m.Z[(size_t)r * kSgDesc + j] * m.sd[j] + m.mu[j];
                    tv[j] = m.hasOff[j] ? std::pow(10.0, ttr[j]) - m.off[j] : ttr[j];
                }
                const double freq = valid[kSgPitch] ? tv[kSgPitch] : 0.0;
                try {
                    SurrogateSolver s;
                    s.m = &m;
                    SgDecoded dec;
                    if (!s.solve(tv, valid, shape, freq, rng, dec)) { ++fails; continue; }
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
                            for (int j = 0; j < kSgDesc; ++j) {
                                if (!valid[j] || !rs.vBest[j]) continue;
                                E[bi][j][key].p.push_back(100.0 * std::fabs(m.tdesc(rs.mBest[j], j) - ttr[j]) / scale[j]);
                                E[bi][j][key].a.push_back(std::fabs(rs.mBest[j] - tv[j]));
                            }
                            ++bi;
                        }
                    }
                    ++done;
                } catch (const std::exception&) {
                    ++fails;
                }
            }
            std::printf("%s / %s: %d target\n", exc, shape.c_str(), done);
            std::fflush(stdout);
        }
    }
    FILE* c = std::fopen((out + ".csv").c_str(), "w");
    if (!c) { std::printf("impossibile scrivere %s.csv\n", out.c_str()); return 1; }
    std::fprintf(c, "eccitatore,forma,budget,descrittore,n,min_pct,p10_pct,media_pct,p90_pct,max_pct,min_ass,media_ass,max_ass\n");
    for (int b = 0; b < 3; ++b)
        for (int j = 0; j < kSgDesc; ++j)
            for (const auto& kv : E[b][j]) {
                const std::string& k = kv.first;
                double mn, me, mx, amn, ame, amx;
                stats(kv.second.p, mn, me, mx);
                stats(kv.second.a, amn, ame, amx);
                std::fprintf(c, "%s,%s,%s,%s,%zu,%.2f,%.2f,%.2f,%.2f,%.2f,%.6g,%.6g,%.6g\n", k.substr(0, k.find('|')).c_str(),
                             k.substr(k.find('|') + 1).c_str(), kBudgetName[b], dname(j), kv.second.p.size(), mn,
                             q(kv.second.p, 0.1), me, q(kv.second.p, 0.9), mx, amn, ame, amx);
            }
    std::fclose(c);
    FILE* f = std::fopen((out + ".txt").c_str(), "w");
    if (!f) { std::printf("impossibile scrivere %s.txt\n", out.c_str()); return 1; }
    std::fprintf(f, "multiPhiMo -- errore min-max per descrittore (motore C++: rete v5 + ricerca reale), %d target per "
                    "eccitatore x forma.\nErrore = |misurato - richiesto| / (p95 - p5 del descrittore per l'eccitatore), in %%. "
                    "rete = prima nota; +30 = dopo 30 render di ricerca reale.\nmin/max = target migliore/peggiore; p10/p90 = "
                    "10%% dei target sotto/sopra; media = errore medio.\nDettaglio per coppia eccitatore/forma e "
                    "errori assoluti (unita' del descrittore) in %s.csv.\n\n", N, out.c_str());
    std::fprintf(f, "######## PER ECCITATORE (tutte le forme)\n\n");
    std::vector<std::string> excs(kExc, kExc + 10);
    summary(f, true, excs);
    std::fprintf(f, "######## PER FORMA DEL RISONATORE (tutti gli eccitatori)\n\n");
    summary(f, false, allShapes);
    std::fclose(f);
    std::printf("scritti %s.csv e %s.txt\n", out.c_str(), out.c_str());
    return 0;
}
