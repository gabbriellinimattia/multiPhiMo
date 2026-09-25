#pragma once
// Porting di exciters.py (motore di sintesi, 10 eccitatori): bow/strike/shaker in questo
// primo blocco (nessuna FFT richiesta). blow/pluck/chaos/mechanical/bird/vocal e noise
// (richiede FFT, vedi vst3_native_stato.md) sono i prossimi blocchi. Verificato contro il
// sorgente reale (device_bash cat, 2026-09-22), non da memoria.
// Richiede C++17.
#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

#define POCKETFFT_NO_MULTITHREADING  // uso sempre nthreads=1, evita la dipendenza da <thread>/pthread
#include "../../pocketfft/pocketfft_hdronly.h"
#include "Resonator.hpp"  // TwoPoleMode, normalizePeak09, kResonatorEps, kResonatorSR
#include "Descriptors.hpp"  // pitchMpm, descFiniteRange (calibrazione di bird)

namespace phimo {

// exciters.py _modal_bank: banco di N modi (freq = baseFreq*ratios[i]) eccitati dallo stesso
// ingresso x, sommati con ampiezza amps[i] (default 1/(i+1), stesso schema del risonatore
// legacy). Riusa TwoPoleMode (stesso identico filtro IIR a 2 poli di resonator.py).
inline std::vector<double> applyModalBank(const std::vector<double>& x, double baseFreq,
                                           const std::vector<double>& ratios, double dampingTime,
                                           const std::vector<double>* amps = nullptr) {
    std::vector<double> y(x.size(), 0.0), modeOut;
    for (size_t i = 0; i < ratios.size(); ++i) {
        double amp = amps ? (*amps)[i] : 1.0 / (double)(i + 1);
        TwoPoleMode mode;
        mode.setParams(baseFreq * ratios[i], dampingTime);
        mode.processBuffer(x, modeOut);
        for (size_t n = 0; n < y.size(); ++n) y[n] += amp * modeOut[n];
    }
    return y;
}

// =====================================================================================
// PITCH PER COSTRUZIONE (exciters.py 24-25/9, sezione dopo "PITCH PER COSTRUZIONE"):
// helper comuni per le nuove fisiche bow/blow/strike/pluck/chaos/mechanical/bird/vocal.
// =====================================================================================

// _split_delay: total (campioni) = linea intera N + allpass del 1o ordine con ritardo di
// FASE d in [0.5, 1.5) a w0 (eta esatto a w0; w0 <= 1e-6 -> formula DC).
struct SplitDelay { int n; double eta; };
inline SplitDelay splitDelay(double total, int floorInt, double w0) {
    const int N = std::max(floorInt, (int)std::floor(total - 0.5));
    const double d = std::min(std::max(total - N, 0.5), 1.5);
    if (w0 <= 1e-6) return {N, (1.0 - d) / (1.0 + d)};
    return {N, std::sin(w0 * (1.0 - d) / 2.0) / std::sin(w0 * (1.0 + d) / 2.0)};
}

// _lp_pd: ritardo di fase a w0 del passa-basso y = (1-p)x + p*y_prev.
inline double lpPhaseDelay(double p, double w0) {
    if (w0 <= 1e-6) return p / (1.0 - p);
    return std::atan2(p * std::sin(w0), 1.0 - p * std::cos(w0)) / w0;
}

// _ap_pd: ritardo di fase a w0 dell'allpass (c + z^-1)/(1 + c z^-1).
inline double apPhaseDelay(double c, double w0) {
    if (w0 <= 1e-6) return (1.0 - c) / (1.0 + c);
    const std::complex<double> z = std::exp(std::complex<double>(0.0, -w0));
    return -std::arg((c + z) / (1.0 + c * z)) / w0;
}

// scipy.signal.butter(order, Wn, btype="high", output="sos") -- porting del percorso
// buttap -> lp2hp_zpk -> bilinear_zpk(fs=2) -> zpk2sos(pairing="nearest") per ordine PARI
// (qui sempre 4). Sezioni ordinate come scipy: la coppia di poli piu' vicina al cerchio
// unitario va nell'ULTIMA sezione, il guadagno nella prima.
struct Biquad { double b0, b1, b2, a1, a2; };
inline std::vector<Biquad> butterHighpassSos(int order, double Wn) {
    using C = std::complex<double>;
    const double warped = 4.0 * std::tan(kPi * Wn / 2.0);   // 2*fs*tan(pi*Wn/fs), fs=2
    std::vector<C> pz;
    C prodNegP(1.0, 0.0), prod4mP(1.0, 0.0);
    for (int m = -order + 1; m < order; m += 2) {
        const C pa = -std::exp(C(0.0, kPi * m / (2.0 * order)));   // buttap
        const C ph = warped / pa;                                    // lp2hp_zpk
        prodNegP *= -pa;   // lp2hp: k_hp = k*real(prod(-z)/prod(-p)) sui poli PASSA-BASSO
        prod4mP *= (4.0 - ph);
        pz.push_back((4.0 + ph) / (4.0 - ph));                       // bilinear_zpk
    }
    // guadagno: lp2hp k = real(1/prod(-p)); bilinear k *= real(prod(4 - 0)/prod(4 - p))
    const double kHp = std::real(C(1.0, 0.0) / prodNegP);
    const double k = kHp * std::real(C(std::pow(4.0, order), 0.0) / prod4mP);
    // solo i poli a parte immaginaria positiva (uno per coppia coniugata)
    std::vector<C> pos;
    for (const C& q : pz) if (std::imag(q) > 0.0) pos.push_back(q);
    // ordine delle sezioni: distanza |1-|p|| decrescente (la piu' vicina al cerchio in fondo)
    std::stable_sort(pos.begin(), pos.end(), [](const C& a, const C& b) {
        return std::fabs(1.0 - std::abs(a)) > std::fabs(1.0 - std::abs(b));
    });
    std::vector<Biquad> sos;
    for (size_t i = 0; i < pos.size(); ++i) {
        const double g = (i == 0) ? k : 1.0;
        sos.push_back({g, -2.0 * g, g, -2.0 * std::real(pos[i]), std::norm(pos[i])});
    }
    return sos;
}

// scipy.signal.sosfilt (zi = 0): forma diretta II trasposta, sezione per sezione.
inline std::vector<double> sosFilt(const std::vector<Biquad>& sos, const std::vector<double>& x) {
    std::vector<double> y(x);
    for (const Biquad& s : sos) {
        double z0 = 0.0, z1 = 0.0;
        for (double& v : y) {
            const double xin = v;
            const double out = s.b0 * xin + z0;
            z0 = s.b1 * xin - s.a1 * out + z1;
            z1 = s.b2 * xin - s.a2 * out;
            v = out;
        }
    }
    return y;
}

// HP di uscita di blow/chaos: butter(4, min(hp*freq/(0.5*sr), 0.99), "high") + sosfilt.
inline std::vector<double> outputHighpass(const std::vector<double>& x, double hp, double freq, int sr) {
    if (!(hp > 0.0)) return x;
    return sosFilt(butterHighpassSos(4, std::min(hp * freq / (0.5 * sr), 0.99)), x);
}

// _unit: x / (rms + EPS), EPS = 1e-9 (exciters.py).
inline std::vector<double> unitRms(const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) s += v * v;
    const double r = std::sqrt(s / std::max<size_t>(x.size(), 1)) + 1e-9;
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] / r;
    return y;
}

// _harm_tone: somma di armoniche k*freq (solo sotto 0.45*sr), ampiezze amps[k-1], inviluppo opzionale.
inline std::vector<double> harmTone(double freq, int n, int sr, const std::vector<double>& amps,
                                    const std::vector<double>* env = nullptr) {
    std::vector<double> y(n, 0.0);
    for (size_t k = 1; k <= amps.size(); ++k) {
        if (k * freq >= 0.45 * sr) continue;
        const double c = 2.0 * kPi * (double)k * freq;   // stesso ordine di valutazione di numpy
        for (int i = 0; i < n; ++i) y[i] += amps[k - 1] * std::sin(c * ((double)i / sr));
    }
    if (env) for (int i = 0; i < n; ++i) y[i] *= (*env)[i];
    return y;
}

// exciters.py bow() (v2 dal 25/9, vedi sotto). bow_force/bow_velocity/bow_position/brightness/damping qui sono
// SEMPRE scalari (in produzione i parametri sono costanti per nota, MDN predict() non
// produce traiettorie -- la "morphing B" di agents.py/_as_traj non e' portata).
inline std::vector<float> bow(double duration, double freq, double bowForce, double bowVelocity,
                               double bowPosition, double brightness, double damping,
                               int sr = kResonatorSR) {
    // exciters.py bow() v2 (pitch per costruzione, 24/9): guida d'onda a DUE linee (nut/ponte,
    // riflessioni invertite = loop non invertente, periodo = sr/freq) + tabella d'attrito
    // (Smith 1986, struttura STK Bowed). Ritardo di loop = Nn + Nb + allpass di accordatura +
    // ritardo di fase del passa-basso al ponte = sr/freq. Deterministico (nessun RNG).
    const int n = (int)(duration * sr);
    const double p = 0.7 * (1.0 - std::min(std::max(brightness, 0.0), 1.0));
    const double w0 = 2.0 * kPi * freq / sr;
    const SplitDelay sd = splitDelay(sr / freq - lpPhaseDelay(p, w0), 2, w0);
    const int Ntot = sd.n;
    const double eta = sd.eta;
    const double t = bowPosition * Ntot;
    const int Nb = (int)std::min(std::max(std::nearbyint(t), 1.0), (double)(Ntot - 1));  // round() Python = pari
    const int Nn = Ntot - Nb;
    std::vector<double> bufB(Nb, 0.0), bufN(Nn, 0.0), out(n, 0.0);
    const double slope = 5.0 - 4.0 * std::min(std::max(bowForce, 0.0), 1.0);
    const double vb = 0.02 + 0.2 * bowVelocity;
    const int ramp = (int)(0.02 * sr);
    double lp = 0.0, ax = 0.0, ay = 0.0;
    int pB = 0, pN = 0;
    for (int i = 0; i < n; ++i) {
        const double bOut = bufB[pB], nOut = bufN[pN];
        lp = (1.0 - p) * bOut + p * lp;
        const double bridge = -damping * lp;
        const double ap = eta * nOut + ax - eta * ay;
        ax = nOut; ay = ap;
        const double nut = -ap;
        const double vd = vb * (ramp > 0 ? std::min(1.0, (double)i / ramp) : 1.0) - (bridge + nut);
        const double mu = std::pow(std::fabs(vd * slope) + 0.75, -4.0);
        const double newv = vd * std::min(std::max(mu, 0.01), 0.98);
        bufN[pN] = bridge + newv;
        bufB[pB] = nut + newv;
        out[i] = bridge;
        if (++pB == Nb) pB = 0;
        if (++pN == Nn) pN = 0;
    }
    return normalizePeak09(out);
}

// exciters.py strike(): contatto non lineare hertziano/Hunt-Crossley integrato esplicitamente
// (pochi ms) -> eccita un banco modale a 5 modi. Deterministico (nessun RNG).
inline std::vector<float> strike(double duration, double freq, double impactVelocity,
                                  double hammerMass, double hammerStiffness, double nonlinearity,
                                  double material, double sizeDamping, int sr = kResonatorSR,
                                  double inharmScale = 0.1) {
    double v = impactVelocity * 2.0;
    double x = 0.0;
    double dt = 1.0 / sr;
    int maxSteps = (int)(0.02 * sr);
    std::vector<double> forces;
    forces.reserve(maxSteps);
    for (int i = 0; i < maxSteps; ++i) {
        double f = hammerStiffness * std::pow(std::max(x, 0.0), nonlinearity);
        double a = -f / hammerMass;
        v += a * dt;
        x += v * dt;
        forces.push_back(std::max(f, 0.0));
        if (x < 0) break;
    }
    if (forces.empty()) forces.push_back(1.0);
    double peak = 0.0;
    for (double f : forces) peak = std::max(peak, std::fabs(f));
    if (peak > kResonatorEps) for (auto& f : forces) f /= peak;

    int n = (int)(duration * sr);
    std::vector<double> xIn(n, 0.0);
    int copyLen = std::min((int)forces.size(), n);
    for (int i = 0; i < copyLen; ++i) xIn[i] = forces[i];

    // pitch per costruzione (exciters.py 24/9): rapporti quasi armonici scalati da
    // inharm_scale (0.1), modi oltre 0.45*sr scartati, ampiezze 1/k sui modi tenuti.
    const double s = inharmScale;
    const double all[5] = {1.0, 2.0 + material * 0.6 * s, 3.0 + material * 1.3 * s,
                           4.0 + material * 2.1 * s, 5.0 + material * 3.0 * s};
    std::vector<double> ratios;
    for (double r : all) if (freq * r < 0.45 * sr) ratios.push_back(r);
    if (ratios.empty()) ratios.push_back(1.0);
    double dampingTime = 0.05 + (1.0 - sizeDamping) * 1.5;
    return normalizePeak09(applyModalBank(xIn, freq, ratios, dampingTime));
}

// exciters.py shaker(): PhISM/PhISEM, collisioni stocastiche di particelle -> banco modale a
// 4 modi. USA un RNG (numpy default_rng, non seedato): qui std::mt19937 con seed esplicito
// (non bit-per-bit uguale a Python, per design del progetto non serve -- stessa tolleranza
// gia' accettata per _sample_mixture della MDN e per il caso "descrittori uguali possono dare
// suoni diversi"). Validazione statistica (rms/peak), non campione-per-campione.
inline std::vector<float> shaker(double duration, double freq, double nParticles, double energy,
                                  double decayTime, double material, int sr = kResonatorSR,
                                  unsigned rngSeed = 0) {
    int n = (int)(duration * sr);
    std::vector<double> impulses(n, 0.0);
    std::mt19937 rng(rngSeed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double decayClamped = std::max(decayTime, 0.01);
    for (int i = 0; i < n; ++i) {
        double t = (double)i / sr;
        double sysEnergy = energy * std::exp(-t / decayClamped);
        double prob = std::min(std::max(nParticles / 800.0 * sysEnergy, 0.0), 0.9);
        if (uni(rng) < prob) {
            double sign = (uni(rng) < 0.5) ? -1.0 : 1.0;
            impulses[i] = sign * std::sqrt(sysEnergy + kResonatorEps);
        }
    }
    std::vector<double> ratios = {1.0, 1.8 + material, 2.6 + 2.0 * material, 3.4 + 3.0 * material};
    return normalizePeak09(applyModalBank(impulses, freq, ratios, 0.05));
}

// exciters.py blow(): ancia singola non lineare (curva di apertura, stile clarinetto
// Smith/McIntyre-Woodhouse-Schumacher) + rumore di turbolenza proporzionale al flusso, su
// canna a delay singolo. USA un RNG per-campione (rumore di turbolenza sempre presente, non
// solo se breath_noise>0) -- stessa tolleranza statistica di shaker/noise, oltre alla
// sensibilita' da retroazione gia' osservata su bow (stesso schema: delay buffer + non
// linearita' che si autoalimenta).
inline std::vector<float> blow(double duration, double freq, double mouthPressure, double reedStiffness,
                                double breathNoise, double brightness, double damping,
                                int sr = kResonatorSR, unsigned rngSeed = 0, double hp = 0.4) {
    // exciters.py blow() v2 (pitch per costruzione, 24/9): ancia singola su canna cilindrica,
    // tabella d'ancia Smith/STK Clarinet (rho = clip(0.7 + slope*dp, -1, 1)), riflessione
    // invertita (loop invertente -> periodo = 2*ritardo = sr/freq, allpass di accordatura e
    // passa-basso compensati), rumore di respiro gaussiano, HP Butterworth 4o a hp*freq.
    // RNG diverso da numpy: validazione statistica + pitch.
    const int n = (int)(duration * sr);
    const double mp = 0.45 + 0.55 * std::min(std::max(mouthPressure, 0.0), 1.0);
    const double slope = -0.44 + 0.26 * std::min(std::max(reedStiffness, 0.0), 1.0);
    const double bn = 0.25 * breathNoise;
    const double p = 0.7 * (1.0 - std::min(std::max(brightness, 0.0), 1.0));
    const double w0 = 2.0 * kPi * freq / sr;
    const SplitDelay sd = splitDelay(sr / (2.0 * freq) - lpPhaseDelay(p, w0), 2, w0);
    const int N = sd.n;
    const double eta = sd.eta;
    std::mt19937 rng(rngSeed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const int ramp = (int)(0.02 * sr);
    std::vector<double> buf(N, 0.0), out(n, 0.0);
    double lp = 0.0, ax = 0.0, ay = 0.0;
    int ptr = 0;
    for (int i = 0; i < n; ++i) {
        const double dOut = buf[ptr];
        lp = (1.0 - p) * dOut + p * lp;
        double breath = mp * (ramp > 0 ? std::min(1.0, (double)i / ramp) : 1.0);
        breath += breath * bn * gauss(rng);
        const double pd = -damping * lp - breath;
        const double rho = std::min(std::max(0.7 + slope * pd, -1.0), 1.0);
        const double nw = breath + pd * rho;
        const double ap = eta * nw + ax - eta * ay;
        ax = nw; ay = ap;
        buf[ptr] = ap;
        out[i] = nw;
        if (++ptr == N) ptr = 0;
    }
    return normalizePeak09(outputHighpass(out, hp, freq, sr));
}

// exciters.py pluck(): Karplus-Strong esteso / commuted waveguide synthesis -- burst di
// rumore filtrato (posizione+durezza) iniettato in un loop a delay con filtro media mobile
// di decadimento (esponenziale liscio, nessuna soglia rigida) e allpass opzionale di
// dispersione. USA un RNG solo per il burst iniziale (N campioni, non per-sample come blow) --
// il resto e' lineare/deterministico, quindi ci si aspetta un match migliore di bow/blow ma
// non esatto (dipende comunque dal burst casuale iniziale).
inline std::vector<double> onePoleFilter(const std::vector<double>& x, double b0, double a1) {
    std::vector<double> y(x.size());
    double yPrev = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double yi = b0 * x[i] - a1 * yPrev;
        y[i] = yi;
        yPrev = yi;
    }
    return y;
}

// irfft e' definita piu' sotto (insieme a rfft, per noise()): dichiarazione anticipata.
inline std::vector<double> irfft(const std::vector<std::complex<double>>& X, size_t n);

inline std::vector<float> pluck(double duration, double freq, double pluckPosition,
                                 double pluckHardness, double decayTime, double dispersion,
                                 int sr = kResonatorSR, unsigned rngSeed = 0) {
    // exciters.py pluck() v2 (pitch per costruzione, 24/9): Karplus-Strong con allpass di
    // accordatura (ring = N - 0.5 + ritardo di fase dell'allpass + dispersione = sr/freq) e
    // burst a spettro piatto (irfft di fasi casuali a modulo 1). RNG diverso da numpy:
    // validazione statistica + pitch, non campione per campione.
    const int n = (int)(duration * sr);
    const double disp = dispersion;
    const double w0 = 2.0 * kPi * freq / sr;
    const double dDisp = disp > 0.0 ? apPhaseDelay(-0.5 * disp, w0) : 0.0;
    const SplitDelay sd = splitDelay(sr / freq + 0.5 - dDisp, 4, w0);
    const int N = sd.n;
    const double eta = sd.eta;

    std::mt19937 rng(rngSeed);
    std::uniform_real_distribution<double> uni(0.0, 2.0 * kPi);
    std::vector<std::complex<double>> spec(N / 2 + 1);
    for (auto& c : spec) c = std::polar(1.0, uni(rng));
    spec[0] = 0.0;
    if (N % 2 == 0) spec.back() = 1.0;
    std::vector<double> burst = irfft(spec, (size_t)N);
    if (pluckHardness < 1.0) {
        double alpha = 0.05 + 0.9 * (1.0 - pluckHardness);
        burst = onePoleFilter(burst, alpha, -(1.0 - alpha));
    }
    const int tap = std::max(1, std::min((int)std::nearbyint(pluckPosition * N), N - 1));
    {
        std::vector<double> orig = burst;
        for (int i = tap; i < N; ++i) burst[i] += -orig[i - tap];
    }
    std::vector<double> buf = burst;

    const double decayPerSample = std::exp(-6.91 / (std::max(decayTime, 0.01) * sr));
    const double apCoef = -disp * 0.5;
    double ax = 0.0, ay = 0.0;   // allpass di dispersione
    double tx = 0.0, ty = 0.0;   // allpass di accordatura
    std::vector<double> out(n, 0.0);
    int ptr = 0;
    for (int i = 0; i < n; ++i) {
        const double prev = buf[ptr];
        const double nxt = buf[(ptr + 1) % N];
        double filtered = 0.5 * (prev + nxt) * decayPerSample;
        const double tOut = eta * filtered + tx - eta * ty;
        tx = filtered; ty = tOut;
        filtered = tOut;
        if (disp > 0.0) {
            const double aOut = apCoef * filtered + ax - apCoef * ay;
            ax = filtered; ay = aOut;
            filtered = aOut;
        }
        buf[ptr] = filtered;
        out[i] = filtered;
        ptr = (ptr + 1) % N;
    }
    return normalizePeak09(out);
}

// exciters.py chaos(): mappa logistica x_{n+1}=r*x*(1-x) (regime caotico, r in [3.57,3.99])
// come sorgente di eccitazione, campionata a coupling_rate Hz e agganciata a un loop
// waveguide. Deterministico (nessun RNG) MA caotico per costruzione (sensibilita' esponenziale
// alle condizioni iniziali, il caso peggiore della categoria gia' vista con bow: li' la
// sensibilita' veniva da una soglia rigida in un ciclo limite, qui e' proprio una mappa
// caotica da manuale) -- aspettarsi una divergenza anche marcata da Python, non un bug.
inline std::vector<float> chaos(double duration, double freq, double bifurcation,
                                 double couplingRate, double x0, double brightness,
                                 double damping, int sr = kResonatorSR, unsigned rngSeed = 0,
                                 double hp = 0.7) {
    // exciters.py chaos() v3 (pitch per costruzione, 24/9): la mappa logistica genera IMPULSI
    // (variazione della mappa x segno casuale) dentro un loop di ritardo totale sr/freq (linea
    // intera + allpass di accordatura, passa-basso compensato); HP Butterworth 4o a hp*freq.
    // Il segno casuale usa un RNG diverso da numpy: validazione statistica + pitch.
    const int n = (int)(duration * sr);
    const double br = brightness;
    const double bMean = std::max(br, 1e-3);
    const double w0 = 2.0 * kPi * freq / sr;
    const SplitDelay sd = splitDelay(sr / freq - lpPhaseDelay(1.0 - bMean, w0), 2, w0);
    const int N = sd.n;
    const double eta = sd.eta;
    const int step = std::max(1, (int)std::nearbyint(sr / couplingRate));
    std::mt19937 rng(rngSeed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double x = std::min(std::max(x0, 1e-4), 1.0 - 1e-4);
    const double r = 3.57 + 0.42 * std::min(std::max(bifurcation, 0.0), 1.0);
    std::vector<double> buf(N, 0.0), out(n, 0.0);
    double lp = 0.0, ax = 0.0, ay = 0.0, drivePrev = 0.0;
    int ptr = 0;
    for (int i = 0; i < n; ++i) {
        double imp = 0.0;
        if (i % step == 0) {
            x = r * x * (1.0 - x);
            const double drive = 2.0 * x - 1.0;
            imp = 0.3 * (drive - drivePrev) * (uni(rng) < 0.5 ? 1.0 : -1.0);
            drivePrev = drive;
        }
        const double raw = damping * buf[ptr] + imp;
        lp = br * raw + (1.0 - br) * lp;
        const double ap = eta * lp + ax - eta * ay;
        ax = lp; ay = ap;
        buf[ptr] = ap;
        out[i] = ap;
        if (++ptr == N) ptr = 0;
    }
    return normalizePeak09(outputHighpass(out, hp, freq, sr));
}

// wrapper minimale su PocketFFT (la stessa libreria usata internamente da numpy: stesso
// algoritmo del riferimento Python, non solo una FFT qualsiasi) -- convenzioni identiche a
// numpy.fft.rfft/irfft: rfft NON normalizzata, irfft normalizzata per 1/n.
inline std::vector<std::complex<double>> rfft(const std::vector<double>& x) {
    size_t n = x.size();
    std::vector<std::complex<double>> out(n / 2 + 1);
    pocketfft::shape_t shape{n};
    pocketfft::stride_t strideIn{sizeof(double)};
    pocketfft::stride_t strideOut{sizeof(std::complex<double>)};
    pocketfft::r2c(shape, strideIn, strideOut, size_t(0), true, x.data(), out.data(), 1.0);
    return out;
}
inline std::vector<double> irfft(const std::vector<std::complex<double>>& X, size_t n) {
    std::vector<double> out(n);
    pocketfft::shape_t shapeOut{n};
    pocketfft::stride_t strideIn{sizeof(std::complex<double>)};
    pocketfft::stride_t strideOut{sizeof(double)};
    pocketfft::c2r(shapeOut, strideIn, strideOut, size_t(0), false, X.data(), out.data(), 1.0 / (double)n);
    return out;
}

// scipy.signal.lfilter([b0],[1,a1],x) per un filtro a 1 polo: y[i] = b0*x[i] - a1*y[i-1].
// exciters.py noise(): rumore bianco con tilt spettrale via FFT (X *= freqs^(color*1.5)),
// gating granulare opzionale (density<1), smoothing temporale opzionale (correlation>0),
// risonanza tonale opzionale miscelata (tone_amount>0, riusa TwoPoleMode = _resonant_filter).
// USA un RNG (numpy default_rng, non seedato): stessa tolleranza di shaker() (confronto
// statistico, non campione-per-campione -- vedi Exciters.hpp).
inline std::vector<float> noise(double duration, double color, double density, double correlation,
                                 double freq, double toneAmount, double toneQ,
                                 int sr = kResonatorSR, unsigned rngSeed = 0) {
    int n = (int)(duration * sr);
    std::mt19937 rng(rngSeed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) x[i] = gauss(rng);

    auto X = rfft(x);
    size_t nf = X.size();
    std::vector<double> freqs(nf);
    for (size_t k = 0; k < nf; ++k) freqs[k] = (double)k * sr / (double)n;
    if (nf > 1) freqs[0] = freqs[1];
    double p = color * 1.5;
    for (size_t k = 0; k < nf; ++k) X[k] *= std::pow(freqs[k], p);
    x = irfft(X, (size_t)n);

    if (density < 1.0) {
        int gateN = std::max(1, (int)(sr * 0.02));
        int nGates = n / gateN + 1;
        std::vector<double> gate(n, 0.0);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for (int g = 0; g < nGates; ++g) {
            double val = (uni(rng) < density) ? 1.0 : 0.0;
            int hi = std::min((g + 1) * gateN, n);
            for (int i = g * gateN; i < hi; ++i) gate[i] = val;
        }
        gate = onePoleFilter(gate, 0.1, -0.9);
        for (int i = 0; i < n; ++i) x[i] *= gate[i];
    }

    if (correlation > 0.0) {
        double alpha = 0.01 + 0.98 * correlation;
        x = onePoleFilter(x, alpha, -(1.0 - alpha));
    }

    if (toneAmount > 0.0) {
        double dampingTime = 0.01 + toneQ * 0.3;
        TwoPoleMode mode;
        mode.setParams(freq, dampingTime, sr);
        std::vector<double> tone;
        mode.processBuffer(x, tone);
        double peakX = 0.0, peakTone = 0.0;
        for (double v : x) peakX = std::max(peakX, std::fabs(v));
        for (double v : tone) peakTone = std::max(peakTone, std::fabs(v));
        if (peakTone > kResonatorEps) {
            double scale = (peakX > kResonatorEps ? peakX : 1.0) / peakTone;
            for (auto& v : tone) v *= scale;
        }
        for (int i = 0; i < n; ++i) x[i] = (1.0 - toneAmount) * x[i] + toneAmount * tone[i];
    }

    return normalizePeak09(x);
}

// exciters.py mechanical(): rumori meccanici caotico-stocastici (motori, ingranaggi, aria
// compressa) -- treno di impulsi con jitter da mappa logistica (eccita 3 modi inarmonici a
// Q=15 fisso, riusa TwoPoleMode) + rumore turbolento colorato (1 polo) con inviluppo pulsato
// dopo ogni impulso. Il seed di default (0, vedi docstring Python "render deterministico")
// e' voluto per ripetibilita', ma numpy default_rng (PCG64) e std::mt19937 restano algoritmi
// RNG diversi -> nessun match bit-esatto possibile nemmeno a parita' di seed, stessa
// categoria statistica di blow/noise/pluck (confronto su rms/peak, non campione-per-campione).
inline std::vector<float> mechanical(double duration, double freq, double rate, double jitter,
                                      double airMix, double airColor, double loadMod, double toneMix,
                                      int sr = kResonatorSR, unsigned rngSeed = 0) {
    // exciters.py mechanical() (pitch per costruzione, 24/9): come l'originale (treno d'impulsi
    // jittered + modi + aria) piu' un nucleo tonale ARMONICO a freq (fino a 60 armoniche 1/k^0.7,
    // modulato dall'inviluppo pulsato); modi a impulsi armonici (1,2,3) se tone_mix >= 0.9,
    // inarmonici (1,2.76,5.40) se tone_mix <= 0.4. Uscita = (1-air_mix)*(tone_mix*nucleo +
    // (1-tone_mix)*modi) + air_mix*aria (ognuno a rms unitario).
    int n = (int)(duration * sr);
    std::mt19937 rng(rngSeed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> unit01(0.0, 1.0);
    std::uniform_real_distribution<double> posDist(0.05, 0.5);

    std::vector<double> noise1(n);
    for (int i = 0; i < n; ++i) noise1[i] = gauss(rng);

    double aLp = 1.0 - std::exp(-2.0 * kPi * 2.0 / sr);
    std::vector<double> lam = onePoleFilter(noise1, aLp, -(1.0 - aLp));
    double mean = 0.0;
    for (double v : lam) mean += v;
    mean /= n;
    double var = 0.0;
    for (double v : lam) var += (v - mean) * (v - mean);
    var /= n;
    double sd = std::sqrt(var);
    for (double& v : lam) v = std::min(std::max(v / (sd + kResonatorEps), -2.0), 2.0);

    std::vector<double> rEff(n);
    for (int i = 0; i < n; ++i) rEff[i] = std::max(rate * (1.0 + loadMod * lam[i]), 0.3 * rate);

    double mu = 3.4 + 0.59 * jitter;
    double z = 0.3 + 0.4 * unit01(rng);
    double pos = posDist(rng) * sr / rate;
    std::vector<int> idx;
    std::vector<double> amp;
    while (pos < n) {
        int i = (int)pos;
        z = std::min(std::max(mu * z * (1.0 - z), 1e-6), 1.0 - 1e-6);
        double dev = std::min(std::max(2.0 * (z - 0.6), -1.0), 1.0);
        idx.push_back(i);
        amp.push_back(1.0 + 0.3 * jitter * dev);
        double T = sr / rEff[i];
        pos += std::max(0.2 * T, T * (1.0 + jitter * dev));
    }
    std::vector<double> imp(n, 0.0);
    for (size_t k = 0; k < idx.size(); ++k) imp[(size_t)idx[k]] += amp[k];

    std::vector<double> pulse(n, 0.0);
    const double qInh = std::min(std::max((0.9 - toneMix) / 0.5, 0.0), 1.0);
    const double kRatios[3] = {1.0, 2.0 + 0.76 * qInh, 3.0 + 2.40 * qInh};
    const double kAmps[3] = {1.0, 0.6, 0.35};
    for (int k = 0; k < 3; ++k) {
        double f = freq * kRatios[k];
        if (f < 0.45 * sr) {
            TwoPoleMode mode;
            mode.setParams(f, 15.0 / (kPi * f), sr);
            std::vector<double> modeOut;
            mode.processBuffer(imp, modeOut);
            for (int i = 0; i < n; ++i) pulse[i] += kAmps[k] * modeOut[i];
        }
    }

    std::vector<double> noise2(n);
    for (int i = 0; i < n; ++i) noise2[i] = gauss(rng);
    std::vector<double> air = onePoleFilter(noise2, 1.0, -airColor);

    double d = std::exp(-1.0 / (0.4 / rate * sr));
    std::vector<double> indicator(n, 0.0);
    for (int i = 0; i < n; ++i) indicator[i] = (imp[i] != 0.0) ? 1.0 : 0.0;
    std::vector<double> E = onePoleFilter(indicator, 1.0, -d);
    for (double& v : E) v = std::min(v, 1.0);
    for (int i = 0; i < n; ++i) air[i] *= (0.4 + 0.6 * E[i]);

    // nucleo ricco (buzz 1/k^0.7 fino a 60 armoniche), inviluppo 0.55 + 0.45*env
    const int nh = std::max(1, std::min(60, (int)(0.45 * sr / freq)));
    std::vector<double> amps(nh), coreEnv(n);
    for (int k = 1; k <= nh; ++k) amps[k - 1] = std::pow((double)k, -0.7);
    for (int i = 0; i < n; ++i) coreEnv[i] = 0.55 + 0.45 * E[i];
    const std::vector<double> core = unitRms(harmTone(freq, n, sr, amps, &coreEnv));
    const std::vector<double> pulseU = unitRms(pulse), airU = unitRms(air);

    std::vector<double> out(n);
    for (int i = 0; i < n; ++i)
        out[i] = (1.0 - airMix) * (toneMix * core[i] + (1.0 - toneMix) * pulseU[i]) + airMix * airU[i];
    return normalizePeak09(out);
}

// exciters.py bird(): canto d'uccello, forma normale di Mindlin-Laje a 2 sorgenti di siringa
// (RK4, doppio oscillatore accoppiato) su una trachea a ritardo con retroazione (linea di
// ritardo interpolata linearmente + passa-basso 1 polo). Loop per-campione fedele
// all'originale (stesso ordine: leggi il ritardo, integra RK4 nsub volte, poi passa al
// campione successivo). y0a/y0b iniziali sono gli UNICI valori stocastici (RNG non seedato
// in Python -> confronto statistico come blow/noise/pluck/mechanical).
inline double birdF(double x, double Y, double al, double be, double F) {
    return -al - be * x - x * x * x - x * x * Y + x * x - x * Y + F;
}

inline void birdStep(double& x, double& Y, double al, double be, double F, double h) {
    double k1x = Y;
    double k1y = birdF(x, Y, al, be, F);
    double k2x = Y + 0.5 * h * k1y;
    double k2y = birdF(x + 0.5 * h * k1x, Y + 0.5 * h * k1y, al, be, F);
    double k3x = Y + 0.5 * h * k2y;
    double k3y = birdF(x + 0.5 * h * k2x, Y + 0.5 * h * k2y, al, be, F);
    double k4x = Y + h * k3y;
    double k4y = birdF(x + h * k3x, Y + h * k3y, al, be, F);
    double xNew = x + h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
    double yNew = Y + h / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
    x = xNew;
    Y = yNew;
}

inline std::vector<double> birdCore(const std::vector<double>& alphaArr, double beta, double eps,
                                     double D, double rhoF, double h, int nsub,
                                     double y0a, double y0b, double g, double aLp, double mixB) {
    int n = (int)alphaArr.size();
    std::vector<double> W(n + 2, 0.0);
    double xa = 0.0, ya = y0a, xb = 0.0, yb = y0b, lp = 0.0;
    double hb = h * rhoF;
    for (int k = 0; k < n; ++k) {
        double wd = 0.0;
        double idx = k - D;
        if (idx >= 1.0) {
            int i0 = (int)idx;
            double fr = idx - i0;
            wd = W[i0] * (1.0 - fr) + W[i0 + 1] * fr;
        }
        lp = (1.0 - aLp) * wd + aLp * lp;
        double F = eps * g * lp;
        W[k] = ya + mixB * yb - g * lp;
        double al = alphaArr[k];
        for (int s = 0; s < nsub; ++s) {
            birdStep(xa, ya, al, beta, F, h);
            birdStep(xb, yb, al, beta, F, hb);
        }
        if (!(std::fabs(xa) < 1e3 && std::fabs(xb) < 1e3)) {
            xa = 0.0; ya = 0.05; xb = 0.0; yb = 0.05;
        }
    }
    W.resize(n);
    return W;
}

// exciters.py _pitch_med: mediana a frame (0.3 s, passo 0.15 s, solo frame con rms > 5% del
// massimo) di pitchMpm; NaN se nessun valore valido. Usata dalla calibrazione di bird.
inline double pitchMed(const std::vector<double>& x, int sr, double L = 0.3, double H = 0.15) {
    const int Ln = (int)(L * sr), Hn = (int)(H * sr);
    if ((int)x.size() < Ln) return pitchMpm(x, sr);
    std::vector<double> vals, rms;
    for (int s0 = 0; s0 + Ln <= (int)x.size(); s0 += Hn) {
        const std::vector<double> f(x.begin() + s0, x.begin() + s0 + Ln);
        double q = 0.0;
        for (double v : f) q += v * v;
        rms.push_back(std::sqrt(q / Ln));
        vals.push_back(pitchMpm(f, sr));
    }
    const double rmax = *std::max_element(rms.begin(), rms.end());
    std::vector<double> good;
    for (size_t k = 0; k < vals.size(); ++k)
        if (rms[k] > 0.05 * rmax && descFiniteRange(vals[k]) && vals[k] > 0.0) good.push_back(vals[k]);
    if (good.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(good.begin(), good.end());
    const size_t m = good.size();
    return (m % 2) ? good[m / 2] : (good[m / 2 - 1] + good[m / 2]) / 2.0;
}

// exciters.py bird() (pitch per costruzione, 24/9): siringa originale (2 oscillatori
// Mindlin-Laje + ritardo tracheale) CALIBRATA in frequenza: un primo render completo a
// gamma0 = 2*pi*freq/2.4 misura la frequenza dominante f_m (pitchMed, stesso stimatore
// dell'analyzer), il render finale usa gamma = gamma0*freq/f_m (2 render: CPU x2). Piu' un
// nucleo armonico (3 armoniche col gate del modello, peso core = 2.5).
inline std::vector<float> bird(double duration, double freq, double alpha0, double beta, double eps,
                                double tauD, double duty, double rhoF, double gateRate,
                                int sr = kResonatorSR, unsigned rngSeed = 0, double core = 2.5) {
    const int n = (int)(duration * sr);

    std::vector<double> gEnv(n);
    if (duty >= 0.999) {
        std::fill(gEnv.begin(), gEnv.end(), 1.0);
    } else {
        double r = std::min({0.1, 0.5 * duty, 0.5 * (1.0 - duty)});
        for (int i = 0; i < n; ++i) {
            double phi = std::fmod((double)i / sr * gateRate, 1.0);
            double g;
            if (phi <= duty - r) {
                g = 1.0;
            } else if (phi < duty) {
                g = 0.5 * (1.0 + std::cos(kPi * (phi - (duty - r)) / r));
            } else if (phi >= 1.0 - r) {
                g = 0.5 * (1.0 - std::cos(kPi * (phi - (1.0 - r)) / r));
            } else {
                g = 0.0;
            }
            gEnv[i] = g;
        }
    }
    std::vector<double> alphaArr(n);
    for (int i = 0; i < n; ++i) alphaArr[i] = alpha0 * gEnv[i] - 0.2 * (1.0 - gEnv[i]);

    std::mt19937 rng(rngSeed);
    std::uniform_real_distribution<double> unit01(0.0, 1.0);
    const double y0a = 0.05 + 0.02 * unit01(rng);
    const double y0b = 0.03 + 0.02 * unit01(rng);

    auto run = [&](double fg) {
        const double gamma = 2.0 * kPi * fg;
        const int nsub = std::max(1, (int)std::ceil(gamma * std::max(1.0, rhoF) / (sr * 0.2)));
        const double h = gamma / (sr * nsub);
        const double D = tauD * sr / gamma;
        const std::vector<double> w = birdCore(alphaArr, beta, eps, D, rhoF, h, nsub, y0a, y0b, 0.98, 0.3, 0.7);
        std::vector<double> d(n);
        d[0] = w[0];
        for (int i = 1; i < n; ++i) d[i] = w[i] - w[i - 1];
        return d;
    };

    double fg = freq / 2.4;
    std::vector<double> col = run(fg);
    const int skip = (int)(0.15 * sr);
    const std::vector<double> tail(col.begin() + std::min(skip, n), col.end());
    const double fm = pitchMed(tail, sr);
    if (descFiniteRange(fm) && fm > 0.0) {
        fg = fg * freq / fm;
        col = run(fg);
    }
    std::vector<double> out = unitRms(col);
    if (core > 0.0) {
        const std::vector<double> tone = unitRms(harmTone(freq, n, sr, {1.0, 0.4, 0.15}, &gEnv));
        for (int i = 0; i < n; ++i) out[i] = out[i] + core * tone[i];
    }
    return normalizePeak09(out);
}

// exciters.py vocal(): piega vocale a 2 masse (Ishizaka-Flanagan) + flusso di Bernoulli in
// forma chiusa (radice di una quadratica, nessuna iterazione per-campione) + tratto vocale a
// 4 risonatori passa-banda RBJ (retroazione sulla pressione sottoglottica via inertanza) +
// rumore turbolento nu*|U|*w. Loop per-campione fedele all'originale, incluso il ramo
// asimmetrico p1/p2 quando A1<A2 (copiato cosi' com'e' dal sorgente, non "raddrizzato").
// Come mechanical/bird, RNG (rumore turbolento + x1i/x2i infinitesimi) -> confronto
// statistico anche col seed di default (numpy PCG64 vs std::mt19937 restano diversi).
inline std::vector<double> vocalCore(const std::vector<double>& psArr, const std::vector<double>& w,
                                      double x0, double T, double m1, double m2, double k1, double k2,
                                      double kc, double r1, double r2, const double c1[4],
                                      const double c2[4], const double G[4], double Z, double nu,
                                      double e, double x1i, double x2i, double aL, double Li) {
    int n = (int)psArr.size();
    const double Lg = 1.4e-2, d1 = 2.5e-3, d2 = 5.0e-4, rho = 1.2;
    std::vector<double> out(n, 0.0);
    double y[4] = {0.0, 0.0, 0.0, 0.0};
    double yp[4] = {0.0, 0.0, 0.0, 0.0};
    double x1 = x1i, x2 = x2i, x1p = x1i, x2p = x2i, ph = 0.0;
    double den1 = m1 + 0.5 * r1 * T, den2 = m2 + 0.5 * r2 * T;
    const double epsA = 1e-9;
    double ut1 = 0.0, ut2 = 0.0, sl1 = 0.0, sl2 = 0.0;
    for (int k = 0; k < n; ++k) {
        double Ps = psArr[k];
        double A1 = 2.0 * Lg * std::max(x0 + x1, 0.0);
        double A2 = 2.0 * Lg * std::max(x0 + x2, 0.0);
        double Amin = std::min(A1, A2);
        double D = Ps - ph;
        double U = 0.0;
        if (Amin > epsA) {
            double a = rho / (2.0 * Amin * Amin);
            double aD = std::fabs(D);
            U = 2.0 * aD / (Z + std::sqrt(Z * Z + 4.0 * a * aD));
            if (D < 0.0) U = -U;
        }
        double pin = ph + Z * U;
        double p1, p2;
        if (Amin > epsA) {
            if (A1 >= A2) {
                p1 = Ps - rho * U * U / (2.0 * A1 * A1);
                p2 = pin;
            } else {
                p1 = pin;
                p2 = pin;
            }
        } else if (A1 <= epsA) {
            p1 = Ps;
            p2 = ph;
        } else {
            p1 = Ps;
            p2 = Ps;
        }
        double F1 = Lg * d1 * p1;
        double F2 = Lg * d2 * p2;
        double x1n = (T * T * (F1 - k1 * x1 - kc * (x1 - x2)) + 2.0 * m1 * x1 - (m1 - 0.5 * r1 * T) * x1p) / den1;
        double x2n = (T * T * (F2 - k2 * x2 - kc * (x2 - x1)) + 2.0 * m2 * x2 - (m2 - 0.5 * r2 * T) * x2p) / den2;
        if (x0 + x1n < 0.0) x1n = -x0 + e * (-(x0 + x1n));
        if (x0 + x2n < 0.0) x2n = -x0 + e * (-(x0 + x2n));
        x1p = x1; x2p = x2; x1 = x1n; x2 = x2n;
        double Ut = U + nu * std::fabs(U) * w[k];
        double dl = Ut - ut1;
        sl1 = aL * sl1 + (1.0 - aL) * dl;
        sl2 = aL * sl2 + (1.0 - aL) * sl1;
        double dU = Ut - ut2;
        ut2 = ut1;
        ut1 = Ut;
        double sOld = 0.0, sNew = 0.0;
        for (int i = 0; i < 4; ++i) {
            sOld += y[i];
            double yn = c1[i] * y[i] + c2[i] * yp[i] + G[i] * dU;
            yp[i] = y[i];
            y[i] = yn;
            sNew += yn;
        }
        ph = sNew + Li * sl2;
        if (ph > 5000.0) ph = 5000.0;
        else if (ph < -5000.0) ph = -5000.0;
        out[k] = sNew - sOld;
    }
    return out;
}

inline std::vector<double> vocalRender(double duration, int sr, double ps, double freq, double gap,
                                        double foldQ, double kcScale, double jaw, double tongue,
                                        std::mt19937& rng, double tract, double f3) {
    int n = (int)(duration * sr);
    double T = 1.0 / sr;
    double q = freq / 140.0;
    double m1 = 1.25e-4 / (q * q), m2 = 2.5e-5 / (q * q);
    double k1 = 80.0, k2 = 8.0, kc = 25.0 * kcScale;
    double r1 = std::sqrt(k1 * m1) / foldQ, r2 = std::sqrt(k2 * m2) / foldQ;
    double f1 = 250.0 + 700.0 * jaw;
    double f2 = std::max(700.0 + 1900.0 * tongue, 1.1 * f1);
    double f3v = std::max(f3, 1.15 * f2);
    double srF = (double)sr;
    double Fs[4] = {std::min(f1, 0.45 * srF), std::min(f2, 0.45 * srF), std::min(f3v, 0.45 * srF),
                    std::min(std::max(3500.0, 1.1 * f3v), 0.45 * srF)};
    double Bw[4] = {60.0, 90.0, 150.0, 200.0};
    double c1[4], c2[4], G[4];
    double Z = 1.2 * 343.0 / 3e-4;
    double gBase[4] = {1.0, 0.7, 0.4, 0.25};
    for (int i = 0; i < 4; ++i) {
        double w0 = 2.0 * kPi * Fs[i] / srF;
        double al = std::sin(w0) / (2.0 * (Fs[i] / Bw[i]));
        c1[i] = 2.0 * std::cos(w0) / (1.0 + al);
        c2[i] = -(1.0 - al) / (1.0 + al);
        G[i] = Z * gBase[i] * al / (1.0 + al) * tract;
    }
    double LIn = 1.2 * 0.17 / 3e-4;
    double inert = 1.0 + 0.5 * std::min(std::max((100.0 - freq) / 30.0, 0.0), 1.0);
    double aL = std::exp(-2.0 * kPi * std::min(std::max(freq, 300.0), 700.0) / srF);

    std::vector<double> psArr(n);
    for (int i = 0; i < n; ++i) psArr[i] = ps * std::min(1.0, (double)i / (0.03 * srF));

    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<double> raw(n);
    for (int i = 0; i < n; ++i) raw[i] = gauss(rng);
    std::vector<double> w(n);
    w[0] = raw[0];
    for (int i = 1; i < n; ++i) w[i] = raw[i] - raw[i - 1];
    for (double& v : w) v /= std::sqrt(2.0);

    std::uniform_real_distribution<double> unit01(0.0, 1.0);
    double x1i = 1e-6 * unit01(rng);
    double x2i = 1e-6 * unit01(rng);

    return vocalCore(psArr, w, gap * 1e-3, T, m1, m2, k1, k2, kc, r1, r2, c1, c2, G,
                      0.15 * Z, 0.4, 0.5, x1i, x2i, aL, inert * LIn * srF);
}

inline std::vector<float> vocalOrig(double duration, double freq, double ps, double gap, double foldQ,
                                 double kcScale, double jaw, double tongue, double f3, double tilt,
                                 int sr = kResonatorSR, unsigned rngSeed = 0) {
    std::mt19937 rng(rngSeed);
    std::vector<double> out = vocalRender(duration, sr, ps, freq, gap, foldQ, kcScale, jaw, tongue,
                                           rng, 1.0, f3);
    double a = std::exp(-2.0 * kPi * std::min(400.0 * std::pow(30.0, tilt), 0.45 * sr) / sr);
    for (int pass = 0; pass < 2; ++pass) {
        out = onePoleFilter(out, 1.0 - a, -a);
    }
    return normalizePeak09(out);
}

// exciters.py vocal() (pitch per costruzione, 24/9): nucleo tonale = serie armonica a freq (1/k,
// formanti F1-F3 come nel modello, tilt, attacco 30 ms) + vocal originale (vocalOrig, pieghe a
// 2 masse) come colore. Peso del nucleo core_w*clip(1-(gap-0.4)/0.9): bisbiglio (gap > ~1.3) =
// solo originale -> non intonato.
inline std::vector<float> vocal(double duration, double freq, double ps, double gap, double foldQ,
                                 double kcScale, double jaw, double tongue, double f3, double tilt,
                                 int sr = kResonatorSR, unsigned rngSeed = 0, double coreW = 0.9) {
    const int n = (int)(duration * sr);
    const std::vector<float> colF = vocalOrig(duration, freq, ps, gap, foldQ, kcScale, jaw, tongue, f3, tilt,
                                              sr, rngSeed);
    const std::vector<double> col(colF.begin(), colF.end());
    const double f1 = 250.0 + 700.0 * jaw;
    const double f2 = std::max(700.0 + 1900.0 * tongue, 1.1 * f1);
    const double f3v = std::max(f3, 1.15 * f2);
    const int nh = std::max(1, (int)(0.45 * sr / freq));
    const double tiltFc = 400.0 * std::pow(30.0, tilt);
    std::vector<double> core(n, 0.0);
    for (int k = 1; k <= nh; ++k) {
        const double fk = k * freq;
        double H = 0.1;
        const double F[3] = {f1, f2, f3v}, Bw[3] = {80.0, 120.0, 200.0}, g[3] = {1.0, 0.7, 0.4};
        for (int m = 0; m < 3; ++m) {
            const double u = (fk - F[m]) / (1.5 * Bw[m]);
            H = H + g[m] / (1.0 + u * u);
        }
        const double v = fk / tiltFc;
        const double a = H / k / (1.0 + v * v);
        const double c = 2.0 * kPi * fk;
        for (int i = 0; i < n; ++i) core[i] += a * std::sin(c * ((double)i / sr));
    }
    for (int i = 0; i < n; ++i) core[i] *= std::min(1.0, ((double)i / sr) / 0.03);
    const double wc = coreW * std::min(std::max(1.0 - (gap - 0.4) / 0.9, 0.0), 1.0);
    const std::vector<double> cu = unitRms(core), vu = unitRms(col);
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i) out[i] = wc * cu[i] + (1.0 - wc) * vu[i];
    return normalizePeak09(out);
}

} // namespace phimo
