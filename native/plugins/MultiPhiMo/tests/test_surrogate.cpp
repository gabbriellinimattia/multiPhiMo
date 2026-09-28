// tests/test_surrogate.cpp -- confronto con native/tools/ref_surrogate.py (stesso formato di stampa).
//   g++ -std=c++17 -O2 test_surrogate.cpp -o test_surrogate && ./test_surrogate ../../../data/surrogate/bow.bin
#include <cstdio>
#include <random>
#include "../Surrogate.hpp"

using namespace phimo;

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("uso: test_surrogate <eccitatore>.bin\n"); return 1; }
    SurrogateModel m;
    std::string err;
    if (!m.load(argv[1], err)) { std::printf("ERRORE %s\n", err.c_str()); return 1; }
    std::printf("exc=%s dim=%d nets=%d rows=%d hidden=%d\n", m.exc.c_str(), m.dim, (int)m.nets.size(), m.rows, m.hidden);
    for (int r = 0; r < 3; ++r) {
        float out[16];
        m.forward(&m.X[(size_t)r * m.dim], out);
        std::printf("fwd %d %.6g %.6g %.6g %.6g %.6g\n", r, out[0], out[5], out[6], out[13], out[15]);
    }
    const double tgt[15] = {1500, 1200, 3000, 0.05, 0.1, 0.5, 0.05, 500, 1500, 2500, 5, 0.1, 0.05, 0.8, 220};
    bool valid[15];
    for (bool& v : valid) v = true;
    SurrogateSolver s;
    s.m = &m;
    s.nRand = 0;
    std::mt19937 rng(0);
    SgDecoded d;
    float loss = 0;
    s.solve(tgt, valid, "bar", 220.0, rng, d, &loss);
    std::printf("solve loss=%.5g x0-7=", loss);
    for (int i = 0; i < 8; ++i) std::printf(" %.4f", s.last.x[i]);
    std::printf("\n");
    for (auto& kv : d.exc) std::printf("exc %s=%.5g\n", kv.first.c_str(), kv.second);
    for (auto& kv : d.coup) std::printf("coup %s=%.5g\n", kv.first.c_str(), kv.second);
    return 0;
}
