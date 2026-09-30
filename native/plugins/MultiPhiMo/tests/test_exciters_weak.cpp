// tests/test_exciters_weak.cpp (2026-09-28) -- parametri deboli: bow/blow t60, strike material (inharmScale 0.5),
// shaker energy (ampiezze k^(-3(1-energy))). Scrive i render grezzi (float32) in <dir>/<exc>_<i>.f32 per
// native/tools/ref_exciters_weak.py (confronto con exciters.py). Stessi punti base del riferimento Python.
// Compilazione ed esecuzione (da native/plugins/MultiPhiMo/tests):
//   g++ -std=c++17 -O2 test_exciters_weak.cpp -o test_exciters_weak && ./test_exciters_weak out_weak
#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "../Exciters.hpp"

using namespace phimo;

static void save(const std::string& dir, const char* exc, int i, const std::vector<float>& y) {
    const std::string p = dir + "/" + exc + "_" + std::to_string(i) + ".f32";
    FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) { std::printf("ERRORE scrittura %s\n", p.c_str()); return; }
    std::fwrite(y.data(), sizeof(float), y.size(), f);
    std::fclose(f);
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "out_weak";
    mkdir(dir.c_str(), 0755);
    const double D = 1.0, F = 220.0;
    const double t60[5] = {0.05, 0.158113883008419, 0.5, 1.58113883008419, 5.0};
    const double mat[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
    const double en[5] = {0.1, 0.325, 0.55, 0.775, 1.0};
    for (int i = 0; i < 5; ++i) {
        save(dir, "bow", i, bow(D, F, 0.525, 0.525, 0.235, 0.525, t60[i]));
        save(dir, "blow", i, blow(D, F, 0.55, 0.5, 0.3, 0.525, t60[i], kResonatorSR, 0));
        save(dir, "blow0", i, blow(D, F, 0.55, 0.5, 0.0, 0.525, t60[i], kResonatorSR, 0));  // senza rumore: deterministico
        save(dir, "strike", i, strike(D, F, 0.55, 0.01, 1e7, 1.75, mat[i], 0.5));
        save(dir, "shaker", i, shaker(D, F, 50.0, en[i], 1.05, 0.5, kResonatorSR, 0));
    }
    std::printf("scritti 25 file in %s\n", dir.c_str());
    return 0;
}
