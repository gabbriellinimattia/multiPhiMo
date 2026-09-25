#pragma once
// Porting di resonator.py (risonatore condiviso, agente 8): calcolo analitico dei parametri
// modali (resonator_params) + banco di filtri modali a 2 poli (apply_resonator) per le 4
// forme originali (bar, plate_rect, plate_circ, membrane). tube/soundboard/chaotic sono il
// prossimo passo (formule piu' elaborate, chaotic e' non lineare). Verificato contro il
// sorgente reale (device_bash cat, 2026-09-22), non da memoria.
// Richiede C++17.
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace phimo {

inline constexpr int kResonatorSR = 44100;
inline constexpr double kResonatorEps = 1e-9;  // resonator.py EPS (diverso da agents.py EPS=1e-8)
inline constexpr double kPi = 3.14159265358979323846;

// rapporti f_n/f_1 per modo (Fletcher & Rossing), letteratura -- resonator.py MODE_RATIOS.
// plate_rect non e' qui: calcolato da plateRectRatios() (dipende dall'aspect ratio).
inline const std::map<std::string, std::vector<double>>& modeRatios() {
    static const std::map<std::string, std::vector<double>> m = {
        {"bar",        {1.0, 2.756, 5.404, 8.933, 13.34, 18.64}},
        {"plate_circ", {1.0, 2.08, 3.41, 5.00, 6.82, 8.95}},
        {"membrane",   {1.0, 1.594, 2.136, 2.296, 2.653, 2.918}},
    };
    return m;
}

// resonator.py _plate_rect_ratios: f_mn ~ m^2 + (n/aspect)^2, aspect = lato_b/lato_a.
inline std::vector<double> plateRectRatios(double aspect, int nModes) {
    aspect = std::max(aspect, 0.1);
    static const int pairs[8][2] = {{1,1},{2,1},{1,2},{2,2},{3,1},{1,3},{3,2},{2,3}};
    std::vector<double> vals;
    for (auto& p : pairs) vals.push_back(p[0] * p[0] + (p[1] / aspect) * (p[1] / aspect));
    std::sort(vals.begin(), vals.end());
    vals.resize(std::min((size_t)nModes, vals.size()));
    double v0 = vals[0];
    for (auto& v : vals) v /= v0;
    return vals;
}

// resonator.py _fundamental_freq: legge di scala standard (Euler-Bernoulli/Kirchhoff-Love per
// barra/piastra, onda su membrana tesa per membrane). Costanti illustrative (non calibrate).
inline double fundamentalFreq(const std::string& shape, double size, double thickness,
                               double density, double stiffness) {
    size = std::max(size, 1e-3);
    thickness = std::max(thickness, 1e-5);
    double f1;
    if (shape == "membrane") {
        double rhoAreal = std::max(density * thickness, kResonatorEps);
        double c = std::sqrt(std::max(stiffness, kResonatorEps) / rhoAreal);
        f1 = 0.383 * c / size;
    } else {
        f1 = 0.161 * (thickness / (size * size)) *
             std::sqrt(std::max(stiffness, kResonatorEps) / std::max(density, kResonatorEps));
    }
    return std::max(f1, 20.0);
}

// resonator.py _mode_dampings: T60 (-60dB) da 'loss' (stesso per tutti i modi), poi convertito
// in tau (1/e); i modi piu' acuti decadono piu' in fretta (tau / sqrt(f/f0)).
inline std::vector<double> modeDampings(const std::vector<double>& freqs, double loss) {
    constexpr double kT60LossK = 0.005, kT60Min = 0.05, kT60Max = 10.0;
    const double lnThousand = std::log(1000.0);
    double t60 = std::min(std::max(kT60LossK / std::max(loss, 1e-4), kT60Min), kT60Max);
    double tau = t60 / lnThousand;
    std::vector<double> out(freqs.size());
    double f0 = freqs[0];
    for (size_t i = 0; i < freqs.size(); ++i)
        out[i] = tau / std::sqrt(std::max(freqs[i] / f0, 1.0));
    return out;
}

struct ModalBank {
    std::vector<double> freqs, dampingTimes, gains;
};

// resonator.py resonator_params() per le 4 forme originali (bar/plate_rect/plate_circ/
// membrane). n_modes fisso a 6 (default Python quando non specificato -- in produzione i
// parametri arrivano sempre da MdnAgent::predict, mai n_modes esplicito). params: size,
// aspect (solo plate_rect), thickness, density, stiffness, loss, mode_falloff -- stessi nomi
// di resonator.PARAM_RANGES. I default di fallback qui sotto (usati solo se una chiave manca)
// NON sono i default per-forma di resonator.py (_DEFAULTS): in produzione i parametri sono
// sempre completi (output di predict()), quindi non dovrebbero mai scattare.
inline ModalBank resonatorParamsOriginal(const std::string& shape, const std::map<std::string, float>& params) {
    auto get = [&](const char* k, double def) {
        auto it = params.find(k);
        return it != params.end() ? (double)it->second : def;
    };
    const int nModes = 6;
    std::vector<double> ratios = (shape == "plate_rect")
        ? plateRectRatios(get("aspect", 1.0), nModes)
        : modeRatios().at(shape);

    double f1 = fundamentalFreq(shape, get("size", 0.3), get("thickness", 0.006),
                                 get("density", 2700.0), get("stiffness", 7e10));
    ModalBank bank;
    bank.freqs.resize(ratios.size());
    for (size_t i = 0; i < ratios.size(); ++i) bank.freqs[i] = f1 * ratios[i];
    bank.dampingTimes = modeDampings(bank.freqs, get("loss", 0.01));

    double modeFalloff = std::min(std::max(get("mode_falloff", 0.5), 0.0), 1.0);
    bank.gains.resize(ratios.size());
    for (size_t i = 0; i < ratios.size(); ++i)
        bank.gains[i] = 1.0 / std::pow((double)(i + 1), 0.5 + 1.5 * modeFalloff);
    return bank;
}

inline constexpr double kSpeedOfSound = 343.0;  // C_AIR, m/s

// resonator.py _t60_to_tau: gamma = tasso di decadimento in ampiezza (1/s) -> tau (1/e),
// con T60 in [0.02, 10.0] (floor piu' basso delle 4 forme originali: i modi alti di
// tube/soundboard decadono in fretta). Opera elemento per elemento (gamma e' un vettore).
inline std::vector<double> t60ToTau(const std::vector<double>& gamma) {
    constexpr double kT60MinNew = 0.02, kT60Max = 10.0;
    const double lnThousand = std::log(1000.0);
    std::vector<double> tau(gamma.size());
    for (size_t i = 0; i < gamma.size(); ++i) {
        double t60 = std::min(std::max(lnThousand / std::max(gamma[i], 1e-9), kT60MinNew), kT60Max);
        tau[i] = t60 / lnThousand;
    }
    return tau;
}

// resonator.py _tube_params: colonna d'aria, fino a 48 modi. size=lunghezza L (m),
// radius=raggio interno (m), closure 0=aperto-aperto..1=chiuso-aperto, flare=svasatura.
// Correzione di estremita' non flangiata (Levine-Schwinger), perdite viscotermiche +
// radiazione per il tasso di decadimento.
inline ModalBank resonatorParamsTube(const std::map<std::string, float>& params) {
    auto get = [&](const char* k, double def) {
        auto it = params.find(k);
        return it != params.end() ? (double)it->second : def;
    };
    double a = std::max(get("radius", 0.012), 1e-3);
    double u = std::min(std::max(get("closure", 0.0), 0.0), 1.0);
    double fl = std::min(std::max(get("flare", 0.0), 0.0), 1.0);
    double Leff = std::max(get("size", 0.6), 1e-2) + 2.0 * 0.6133 * a;
    double fo = kSpeedOfSound / (2.0 * Leff);
    double fc = 3.0 * fl * fo;

    std::vector<double> fAll;
    fAll.reserve(399);
    for (int k = 1; k < 400; ++k) {
        double val = (k - u / 2.0) * fo;
        fAll.push_back(std::sqrt(val * val + fc * fc));
    }
    std::vector<double> fSel;
    for (double v : fAll) if (v < 12000.0) fSel.push_back(v);
    if (fSel.size() < 3) fSel.assign(fAll.begin(), fAll.begin() + std::min((size_t)3, fAll.size()));
    if (fSel.size() > 48) fSel.resize(48);

    ModalBank bank;
    bank.freqs = fSel;
    double loss = std::max(get("loss", 0.005), 1e-5);
    double modeFalloff = std::min(std::max(get("mode_falloff", 0.5), 0.0), 1.0);
    double pw = 1.5 * modeFalloff - 0.25;
    std::vector<double> gamma(bank.freqs.size());
    bank.gains.resize(bank.freqs.size());
    for (size_t i = 0; i < bank.freqs.size(); ++i) {
        double f = bank.freqs[i];
        double gVisc = kSpeedOfSound * 3e-5 * std::sqrt(f) / a;
        double ka = 2.0 * kPi * f / kSpeedOfSound * a;
        double gRad = (2.0 - u) * std::min(ka * ka, 1.0) * kSpeedOfSound / (4.0 * Leff);
        gamma[i] = (loss / 0.005) * (gVisc + gRad);
        bank.gains[i] = std::pow((double)(i + 1), -pw);
    }
    bank.dampingTimes = t60ToTau(gamma);
    return bank;
}

// resonator.py _soundboard_params: piastra ortotropa (rho fisso 450 kg/m3) + cavita' di
// Helmholtz accoppiata al modo (1,1) (2 DOF piastra-aria, il fondamentale diventa un
// doppietto di 2 frequenze accoppiate). size=lato lungo, aspect=b/a, stiffness=E_L,
// ortho=E_L/E_R, cavity=volume, hole=raggio buca.
inline ModalBank resonatorParamsSoundboard(const std::map<std::string, float>& params) {
    auto get = [&](const char* k, double def) {
        auto it = params.find(k);
        return it != params.end() ? (double)it->second : def;
    };
    const double rho = 450.0;
    double a = std::max(get("size", 0.4), 1e-2);
    double asp = std::max(get("aspect", 0.75), 0.1);
    double h = std::max(get("thickness", 0.0045), 1e-4);
    double E = std::max(get("stiffness", 1.2e10), 1e6);
    double ortho = std::max(get("ortho", 15.0), 1.0);
    double loss = std::max(get("loss", 0.005), 1e-5);
    double mf = std::min(std::max(get("mode_falloff", 0.5), 0.0), 1.0);
    double scale = 0.478 * h / (a * a) * std::sqrt(E / rho);

    // meshgrid(1..12, 1..12, indexing="ij").ravel(): mm esterno, nn interno -- stesso
    // ordine di Python, necessario per riprodurre l'ordinamento stabile a parita' di M.
    struct Mode { int mm, nn; double M; };
    std::vector<Mode> modes;
    modes.reserve(144);
    for (int mm = 1; mm <= 12; ++mm)
        for (int nn = 1; nn <= 12; ++nn)
            modes.push_back({mm, nn, mm * (double)mm + (nn / asp) * (nn / asp) / std::sqrt(ortho)});
    std::stable_sort(modes.begin(), modes.end(), [](const Mode& x, const Mode& y) { return x.M < y.M; });

    auto gShapeOf = [](const Mode& md) {
        double s1 = std::sin(md.mm * kPi * 0.31), s2 = std::sin(md.nn * kPi * 0.43);
        return s1 * s1 * s2 * s2;
    };
    std::vector<double> f, gShape;
    for (auto& md : modes) {
        double fv = scale * md.M;
        if (fv < 14000.0) { f.push_back(fv); gShape.push_back(gShapeOf(md)); }
    }
    if (f.size() < 3) {
        f.clear(); gShape.clear();
        for (size_t i = 0; i < std::min((size_t)3, modes.size()); ++i) {
            f.push_back(scale * modes[i].M);
            gShape.push_back(gShapeOf(modes[i]));
        }
    }
    const size_t nmax = 40;
    if (f.size() > nmax) { f.resize(nmax); gShape.resize(nmax); }

    std::vector<double> gains(f.size());
    double falloffExp = -(1.5 * mf - 0.25);
    for (size_t i = 0; i < f.size(); ++i) gains[i] = gShape[i] * std::pow(f[i] / f[0], falloffExp);

    double V = std::max(get("cavity", 0.012), 1e-4);
    double rh = std::max(get("hole", 0.04), 1e-3);
    double fH = kSpeedOfSound / (2.0 * kPi) * std::sqrt(kPi * rh * rh / (V * (0.003 + 1.7 * rh)));
    double wb2 = std::pow(2.0 * kPi * f[0], 2.0);
    double wH2 = std::pow(2.0 * kPi * fH, 2.0);
    double kk = 0.4 * 1.4e5 * a * (a * asp) / (V * rho * h);
    double S2 = wb2 + kk + wH2;
    double disc = std::sqrt(std::max(S2 * S2 - 4.0 * wb2 * wH2, 0.0));
    double xs[2] = {(S2 - disc) / 2.0, (S2 + disc) / 2.0};
    double k12 = kk * wH2;
    double fC2[2], pp[2], gC2[2];
    for (int i = 0; i < 2; ++i) {
        fC2[i] = std::sqrt(xs[i]) / (2.0 * kPi);
        pp[i] = k12 / (k12 + std::pow(wb2 + kk - xs[i], 2.0) + 1e-30);
        gC2[i] = gains[0] * (pp[i] + 0.6 * (1.0 - pp[i]));
    }

    ModalBank bank;
    bank.freqs = {fC2[0], fC2[1]};
    bank.gains = {gC2[0], gC2[1]};
    for (size_t i = 1; i < f.size(); ++i) { bank.freqs.push_back(f[i]); bank.gains.push_back(gains[i]); }

    std::vector<double> gam(bank.freqs.size());
    for (size_t i = 0; i < gam.size(); ++i)
        gam[i] = kPi * 300.0 * loss * std::sqrt(std::max(bank.freqs[i], 1.0) / 300.0);
    gam[0] *= (1.0 + 2.0 * (1.0 - pp[0]));
    gam[1] *= (1.0 + 2.0 * (1.0 - pp[1]));
    bank.dampingTimes = t60ToTau(gam);
    return bank;
}

// resonator.py _lorenz_rk4: un passo RK4 del sistema di Lorenz (sigma=10, rho=28, beta=8/3),
// sorgente di "casualita'" deterministica per chaotic (stato iniziale fisso -> riproducibile).
inline void lorenzRk4(double& x, double& y, double& z, double dt) {
    auto deriv = [](double px, double py, double pz, double& dx, double& dy, double& dz) {
        dx = 10.0 * (py - px);
        dy = px * (28.0 - pz) - py;
        dz = px * py - (8.0 / 3.0) * pz;  // 2.6666666666666665 == 8/3
    };
    double k1x, k1y, k1z; deriv(x, y, z, k1x, k1y, k1z);
    double x2 = x + 0.5 * dt * k1x, y2 = y + 0.5 * dt * k1y, z2 = z + 0.5 * dt * k1z;
    double k2x, k2y, k2z; deriv(x2, y2, z2, k2x, k2y, k2z);
    double x3 = x + 0.5 * dt * k2x, y3 = y + 0.5 * dt * k2y, z3 = z + 0.5 * dt * k2z;
    double k3x, k3y, k3z; deriv(x3, y3, z3, k3x, k3y, k3z);
    double x4 = x + dt * k3x, y4 = y + dt * k3y, z4 = z + dt * k3z;
    double k4x, k4y, k4z; deriv(x4, y4, z4, k4x, k4y, k4z);
    x = x + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
    y = y + dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
    z = z + dt / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
}

// resonator.py _chaotic_core: M doppietti di rotatori (fasori z=zr+i*zi), A a w_m, B a
// w_m+dw_m*(1+chaos*C_m(t)) (battimento). z <- r*e^{i*theta}*z + g*x, theta scalato dalla
// tension modulation (1+nonlin*E/E_picco). C_m(t) = proiezione decorrelata di Lorenz
// (riscaldato 20s prima di iniziare -> deterministico). Scambio di energia nel doppietto:
// rotazione di Givens (norma conservata) di angolo chaos*C2_m*0.001 per campione.
// dt = speed/sr (passo del sistema di Lorenz, NON 1/sr).
inline std::vector<double> chaoticCore(const std::vector<double>& x, const std::vector<double>& wa,
                                        const std::vector<double>& dw, const std::vector<double>& r,
                                        const std::vector<double>& ga, const std::vector<double>& gb,
                                        double nonlin, double chaos, double dt) {
    const size_t n = x.size();
    const size_t M = wa.size();
    std::vector<double> zra(M, 0.0), zia(M, 0.0), zrb(M, 0.0), zib(M, 0.0);
    std::vector<double> cp1(M), sp1(M), cp2(M), sp2(M);
    for (size_t m = 0; m < M; ++m) {
        cp1[m] = std::cos(0.7 * (double)m); sp1[m] = std::sin(0.7 * (double)m);
        cp2[m] = std::cos(0.7 * (double)m + 1.3); sp2[m] = std::sin(0.7 * (double)m + 1.3);
    }
    std::vector<double> y(n, 0.0);
    double lx = 1.0, ly = 1.0, lz = 1.0;
    for (int i = 0; i < 2000; ++i) lorenzRk4(lx, ly, lz, 0.01);  // riscaldamento (stato iniziale deterministico)
    double epk = 1e-30;
    for (size_t i = 0; i < n; ++i) {
        lorenzRk4(lx, ly, lz, dt);
        double cx = lx / 20.0, cy = ly / 28.0, cz = (lz - 25.0) / 15.0;
        double E = 0.0;
        for (size_t m = 0; m < M; ++m) E += zra[m]*zra[m] + zia[m]*zia[m] + zrb[m]*zrb[m] + zib[m]*zib[m];
        if (E > epk) epk = E;
        double gl = 1.0 + nonlin * (E / epk);
        double xi = x[i];
        double acc = 0.0;
        for (size_t m = 0; m < M; ++m) {
            double c = std::min(std::max(cp1[m]*cx + sp1[m]*cy, -1.0), 1.0);
            double c2 = std::min(std::max(cp2[m]*cy + sp2[m]*cz, -1.5), 1.5);
            double tha = wa[m] * gl;
            double thb = (wa[m] + dw[m] * (1.0 + chaos * c)) * gl;
            double csa = r[m] * std::cos(tha), sna = r[m] * std::sin(tha);
            double csb = r[m] * std::cos(thb), snb = r[m] * std::sin(thb);
            double a1 = csa*zra[m] - sna*zia[m] + ga[m]*xi;
            double b1 = sna*zra[m] + csa*zia[m];
            double a2 = csb*zrb[m] - snb*zib[m] + gb[m]*xi;
            double b2 = snb*zrb[m] + csb*zib[m];
            double ph = chaos * c2 * 0.001;
            double cg = 1.0 - 0.5 * ph * ph;
            zra[m] = cg*a1 + ph*a2;
            zrb[m] = -ph*a1 + cg*a2;
            zia[m] = cg*b1 + ph*b2;
            zib[m] = -ph*b1 + cg*b2;
            acc += zra[m] + zrb[m];
        }
        y[i] = acc;
    }
    return y;
}

// resonator.py _chaotic_params: banco "base" (fino a 6 doppietti) usato per derivare w/dw/r/g
// di chaoticCore -- NON un banco di filtri lineari (chaotic non passa mai per applyLinearBank).
// f1 = fondamentale di una piastra circolare in acciaio (density fissa 7800) con
// size/thickness/stiffness dell'agente chaotic.
inline ModalBank resonatorParamsChaoticBase(const std::map<std::string, float>& params) {
    auto get = [&](const char* k, double def) {
        auto it = params.find(k);
        return it != params.end() ? (double)it->second : def;
    };
    static const double kChaoticRatios[6] = {1.0, 2.08, 3.41, 5.00, 6.82, 8.95};
    double f1 = std::min(fundamentalFreq("plate_circ", get("size", 0.1), get("thickness", 0.006),
                                          7800.0, get("stiffness", 7e10)), 13000.0);
    std::vector<double> f;
    for (double ratio : kChaoticRatios) f.push_back(f1 * ratio);

    std::vector<double> fKept;
    for (double v : f) if (v * 1.6 < 0.5 * kResonatorSR) fKept.push_back(v);
    if (fKept.empty()) fKept.push_back(f[0]);

    ModalBank bank;
    bank.freqs = fKept;
    double modeFalloff = std::min(std::max(get("mode_falloff", 0.5), 0.0), 1.0);
    bank.gains.resize(fKept.size());
    for (size_t i = 0; i < fKept.size(); ++i)
        bank.gains[i] = 1.0 / std::pow((double)(i + 1), 0.5 + 1.5 * modeFalloff);
    bank.dampingTimes = modeDampings(bank.freqs, std::max(get("loss", 0.005), 1e-4));
    return bank;
}

// Dispatcher generale (porting di resonator.py resonator_params(): stessa istruzione
// if/elif per forma, 'chaotic' incluso -- COMMENTO PRECEDENTE ERA OBSOLETO: chaotic e'
// stato portato al punto 3 (vedi applyResonatorChaotic sotto e vst3_native_stato.md),
// qui manca solo perche' applyResonatorOriginal gestiva chaotic con un ramo separato
// PRIMA di chiamare questa funzione. Aggiunto al punto 7 per la stima durata
// (estimateDuration in Render.hpp: serve damping_times anche per chaotic).
inline ModalBank computeModalBank(const std::string& shape, const std::map<std::string, float>& params) {
    if (shape == "tube") return resonatorParamsTube(params);
    if (shape == "soundboard") return resonatorParamsSoundboard(params);
    if (shape == "chaotic") return resonatorParamsChaoticBase(params);
    return resonatorParamsOriginal(shape, params);  // bar/plate_rect/plate_circ/membrane
}

// resonator.py apply_resonator, cap sui decadimenti dei modi quando f0 (= freq
// dell'eccitatore, se pitch-locked) e' noto (audit pitch 2026-09-24): tau <= capK/f0
// (al piu' capK periodi di f0) e tau <= capQ/(pi*f_modo) (Q massimo per modo) -- il
// risonatore colora senza poter "suonare" una nota propria e spostare il pitch percepito.
// f0<=0 = nessun cap (comportamento precedente, invariato -- porting di f0=None).
inline void capDampingTimesForPitch(ModalBank& bank, double f0, double capK = 2.0, double capQ = 4.0) {
    if (f0 <= 0.0) return;
    for (size_t i = 0; i < bank.dampingTimes.size(); ++i) {
        bank.dampingTimes[i] = std::min(bank.dampingTimes[i], capK / f0);
        bank.dampingTimes[i] = std::min(bank.dampingTimes[i], capQ / (kPi * std::max(bank.freqs[i], 1.0)));
    }
}

// resonator.py _resonant_mode: filtro a 2 poli, y[n] = x[n] + 2*r*cos(theta)*y[n-1] - r^2*y[n-2]
// (equivalente a scipy lfilter([1.0], [1, -2r cos(theta), r^2], x)). fY1/fY2 riservati per
// un uso realtime a blocchi (stato persistente tra chiamate, non ancora agganciato); qui
// processBuffer() e' offline (stato locale, azzerato ad ogni chiamata) per il confronto 1:1
// con Python.
class TwoPoleMode {
public:
    void setParams(double freq, double dampingTime, int sr = kResonatorSR) {
        fR = std::exp(-1.0 / (std::max(dampingTime, 1e-4) * sr));
        double f = std::min(std::max(freq, 1.0), sr / 2.0 - 1.0);
        fTheta = 2.0 * kPi * f / sr;
    }
    void reset() { fY1 = fY2 = 0.0; }

    void processBuffer(const std::vector<double>& x, std::vector<double>& yOut) const {
        double y1 = 0.0, y2 = 0.0;
        double a1 = 2.0 * fR * std::cos(fTheta), a2 = fR * fR;
        yOut.resize(x.size());
        for (size_t n = 0; n < x.size(); ++n) {
            double y = x[n] + a1 * y1 - a2 * y2;
            yOut[n] = y;
            y2 = y1; y1 = y;
        }
    }

private:
    double fR = 0.0, fTheta = 0.0;
    double fY1 = 0.0, fY2 = 0.0;  // riservato, non ancora usato (vedi commento sopra)
};

// resonator.py apply_resonator() per le 4 forme originali: somma pesata dei modi, poi
// normalizza il picco a 0.9 (comportamento offline/non causale, replicato per il confronto
// diretto con Python -- l'integrazione realtime, senza normalizzazione globale sull'intero
// buffer, e' un passo successivo: vedi vst3_native_stato.md punto 7/8).
// resonator.py apply_resonator(): normalizzazione finale del picco a 0.9, condivisa da
// TUTTE le forme (lineari e chaotic).
inline std::vector<float> normalizePeak09(const std::vector<double>& y) {
    double peak = 0.0;
    for (double v : y) peak = std::max(peak, std::fabs(v));
    std::vector<float> out(y.size());
    if (peak > kResonatorEps) {
        double scale = 0.9 / peak;
        for (size_t n = 0; n < y.size(); ++n) out[n] = (float)(y[n] * scale);
    } else {
        for (size_t n = 0; n < y.size(); ++n) out[n] = (float)y[n];
    }
    return out;
}

inline std::vector<float> applyLinearBank(const std::vector<float>& excitation, const ModalBank& bank) {
    std::vector<double> x(excitation.begin(), excitation.end());
    std::vector<double> y(x.size(), 0.0), modeOut;
    for (size_t m = 0; m < bank.freqs.size(); ++m) {
        TwoPoleMode mode;
        mode.setParams(bank.freqs[m], bank.dampingTimes[m]);
        mode.processBuffer(x, modeOut);
        for (size_t n = 0; n < y.size(); ++n) y[n] += bank.gains[m] * modeOut[n];
    }
    return normalizePeak09(y);
}

// resonator.py apply_resonator() ramo "chaotic": deriva w/dw/r/g dal banco base e chiama
// chaoticCore(), poi normalizza il picco a 0.9 (stessa normalizzazione di tutte le altre
// forme). params deve contenere anche nonlin/beat/depth/chaos/speed oltre a
// size/thickness/stiffness/loss/mode_falloff (i 10 parametri di resonator_chaotic).
inline std::vector<float> applyResonatorChaoticFromBank(const std::vector<float>& excitation,
                                                          const std::map<std::string, float>& params,
                                                          const ModalBank& base) {
    auto get = [&](const char* k, double def) {
        auto it = params.find(k);
        return it != params.end() ? (double)it->second : def;
    };
    const size_t M = base.freqs.size();
    const double sr = (double)kResonatorSR;
    double beat = get("beat", 3.0), depth = get("depth", 0.6);
    std::vector<double> wa(M), dw(M), r(M), ga(M), gb(M);
    for (size_t m = 0; m < M; ++m) {
        wa[m] = 2.0 * kPi * base.freqs[m] / sr;
        dw[m] = 2.0 * kPi * beat * (1.0 + 0.31 * (double)m) / sr;
        r[m] = std::exp(-1.0 / (std::max(base.dampingTimes[m], 1e-4) * sr));
        ga[m] = base.gains[m];
        gb[m] = base.gains[m] * depth;
    }
    double nonlin = get("nonlin", 0.2), chaos = get("chaos", 0.4), speed = get("speed", 5.0);
    double dt = speed / sr;

    std::vector<double> x(excitation.begin(), excitation.end());
    std::vector<double> y = chaoticCore(x, wa, dw, r, ga, gb, nonlin, chaos, dt);
    return normalizePeak09(y);
}

// Firma originale (nessun cap f0) -- usata solo dai test/chiamanti pre-pitch-lock,
// mantenuta per compatibilita'.
inline std::vector<float> applyResonatorChaotic(const std::vector<float>& excitation,
                                                  const std::map<std::string, float>& params) {
    return applyResonatorChaoticFromBank(excitation, params, resonatorParamsChaoticBase(params));
}

// Convenienza: calcola il banco (con eventuale cap f0, vedi capDampingTimesForPitch sopra)
// e applica la sintesi in un solo passo -- "chaotic" ha un ramo dedicato (nonlineare,
// applyResonatorChaoticFromBank), le altre 6 forme LINEARI passano da applyLinearBank.
// f0<=0 (default) = nessun cap, comportamento invariato rispetto a prima del pitch-lock.
inline std::vector<float> applyResonatorOriginal(const std::vector<float>& excitation,
                                                   const std::string& shape,
                                                   const std::map<std::string, float>& params,
                                                   double f0 = 0.0) {
    ModalBank bank = computeModalBank(shape, params);
    if (f0 > 0.0) capDampingTimesForPitch(bank, f0);
    if (shape == "chaotic") return applyResonatorChaoticFromBank(excitation, params, bank);
    return applyLinearBank(excitation, bank);
}

} // namespace phimo
