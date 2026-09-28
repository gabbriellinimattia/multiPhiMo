// tests/test_coupling.cpp -- confronto con native/tools/ref_coupling.py (stesso formato di stampa).
// Compilazione (da native/plugins/MultiPhiMo/tests):
//   g++ -std=c++17 -O2 test_coupling.cpp -o test_coupling && ./test_coupling
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "../Exciters.hpp"
#include "../Coupling.hpp"

using namespace phimo;
using M = std::map<std::string, float>;

struct Cfg { const char* shape; M rp; M cp; double f0; };

int main() {
    {
        std::vector<double> b, a;
        coupling_detail::butterBandpass2(0.02, 0.03, b, a);
        std::printf("butter b=%.12g %.12g %.12g %.12g %.12g a=%.12g %.12g %.12g %.12g %.12g\n",
                    b[0], b[1], b[2], b[3], b[4], a[0], a[1], a[2], a[3], a[4]);
    }
    const M full{{"harmonicity", 0.3f}, {"pitch_focus", 0.5f}, {"body", 0.8f}, {"exc_attack", 0.02f},
                 {"exc_hold", 0.4f}, {"am_rate", 5.0f}, {"am_depth", 0.3f}, {"form_f1", 500.0f},
                 {"form_f2", 1500.0f}, {"form_amt", 0.5f}};
    const M bar{{"size", 0.3f}, {"thickness", 0.01f}, {"density", 2700.0f}, {"stiffness", 7e10f},
                {"loss", 0.005f}, {"mode_falloff", 0.5f}};
    std::vector<Cfg> cfgs = {
        {"bar", bar, full, 220.0},
        {"bar", bar, M{{"harmonicity", 1.0f}, {"pitch_focus", 0.0f}, {"body", 1.0f}}, 220.0},
        {"plate_circ", M{{"size", 0.2f}, {"thickness", 0.002f}, {"density", 7800.0f}, {"stiffness", 2e11f},
                         {"loss", 0.002f}, {"mode_falloff", 0.3f}},
         M{{"harmonicity", 0.0f}, {"pitch_focus", 0.8f}, {"body", 0.5f}, {"exc_hold", 1.0f}, {"exc_attack", 0.005f}}, 330.0},
        {"tube", M{{"size", 0.6f}, {"radius", 0.012f}, {"closure", 0.5f}, {"flare", 0.2f}, {"loss", 0.005f},
                   {"mode_falloff", 0.5f}},
         M{{"harmonicity", 0.7f}, {"pitch_focus", 0.2f}, {"body", 0.9f}, {"form_f1", 700.0f}, {"form_f2", 2200.0f},
           {"form_amt", 1.0f}}, 147.0},
        {"soundboard", M{{"size", 0.4f}, {"aspect", 0.75f}, {"thickness", 0.0045f}, {"stiffness", 1.2e10f},
                         {"ortho", 15.0f}, {"cavity", 0.005f}, {"hole", 0.04f}, {"loss", 0.005f}, {"mode_falloff", 0.5f}},
         M{{"harmonicity", 0.5f}, {"exc_hold", 0.2f}, {"am_rate", 12.0f}, {"am_depth", 0.8f}}, 440.0},
        {"chaotic", M{{"size", 0.1f}, {"thickness", 0.006f}, {"stiffness", 7e10f}, {"loss", 0.005f},
                      {"mode_falloff", 0.5f}, {"nonlin", 0.2f}, {"beat", 3.0f}, {"depth", 0.6f}, {"chaos", 0.4f},
                      {"speed", 5.0f}},
         M{{"harmonicity", 0.4f}, {"pitch_focus", 0.3f}, {"body", 0.7f}, {"exc_attack", 0.05f}, {"exc_hold", 0.6f}}, 110.0},
        {"membrane", M{{"size", 0.3f}, {"thickness", 0.0005f}, {"density", 1000.0f}, {"stiffness", 3000.0f},
                       {"loss", 0.01f}, {"mode_falloff", 0.6f}},
         M{{"harmonicity", 0.1f}, {"pitch_focus", 0.9f}}, 0.0},
    };
    for (size_t i = 0; i < cfgs.size(); ++i) {
        const Cfg& c = cfgs[i];
        const double dur = noteDuration(1.0, c.shape, c.rp, c.cp);
        std::vector<float> raw = strike(dur, c.f0 > 0 ? c.f0 : 220.0, 0.8, 0.02, 5e7, 1.5, 0.3, 0.4);
        std::vector<float> y = applyResonatorV5(raw, c.shape, c.rp, c.cp, c.f0);
        double s = 0, ss = 0;
        for (float v : y) { s += v; ss += (double)v * v; }
        std::printf("cfg %zu %s dur=%.10g n=%zu sum=%.8g sumsq=%.8g y1000=%.8g y20000=%.8g\n", i, c.shape, dur,
                    y.size(), s, ss, (double)y[1000], (double)y[20000]);
    }
    return 0;
}
