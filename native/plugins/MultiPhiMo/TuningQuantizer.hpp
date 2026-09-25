#pragma once
// TuningQuantizer.hpp -- porting di tuning.py (punto 8A, round 3.3a, 2026-09-23).
// Solo scale built-in (edo12/24/31, perfect, harmonic) -- import .scl custom rimandato
// al round 3.3b, vedi claude/vst3_native_stato.md. Nessuna allocazione a runtime dopo
// il primo uso (tabelle "static const" locali per grado di scala, calcolate una sola
// volta); usato SOLO nel worker thread (VoiceEngine.hpp), mai nel thread audio.
// scaleIndex e' un int (mai una stringa) apposta per restare fuori dal percorso
// realtime/coda SPSC (TriggerRequest), coerente col resto del progetto.

#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace phimo {

enum ScaleIndex : int {
    kScaleNone = -1,  // Scale disattivo, o nessun override -- nearestGridFreq salta
    kScaleEdo12 = 0,
    kScaleEdo24,
    kScaleEdo31,
    kScalePerfect,
    kScaleHarmonic,
    kScaleCustom,  // round 3.3b: scala importata da .scl (vedi CustomScale sotto)
    kScaleCount
};

inline const char* scaleIndexToName(int idx)
{
    switch (idx)
    {
        case kScaleEdo12:    return "edo12";
        case kScaleEdo24:    return "edo24";
        case kScaleEdo31:    return "edo31";
        case kScalePerfect:  return "perfect";
        case kScaleHarmonic: return "harmonic";
        case kScaleCustom:   return "custom";
        default:             return "edo12";
    }
}

inline int scaleNameToIndex(const char* name)
{
    if (std::strcmp(name, "edo12") == 0)    return kScaleEdo12;
    if (std::strcmp(name, "edo24") == 0)    return kScaleEdo24;
    if (std::strcmp(name, "edo31") == 0)    return kScaleEdo31;
    if (std::strcmp(name, "perfect") == 0)  return kScalePerfect;
    if (std::strcmp(name, "harmonic") == 0) return kScaleHarmonic;
    if (std::strcmp(name, "custom") == 0)   return kScaleCustom;
    return kScaleEdo12;  // default -- mai kScaleNone qui (quello significa "nessun override")
}

// edo_ratios(n) -- n gradi equal-tempered per ottava, rapporti sul grado 0.
template <int N>
inline const std::array<double, N>& edoRatios()
{
    static const std::array<double, N> r = []() {
        std::array<double, N> a{};
        for (int i = 0; i < N; ++i) a[i] = std::pow(2.0, (double)i / (double)N);
        return a;
    }();
    return r;
}

// JUST_MAJOR_RATIOS / HARMONIC_RATIOS -- porting 1:1 di tuning.py.
inline const std::array<double, 7>& justMajorRatios()
{
    static const std::array<double, 7> r = {1.0 / 1, 9.0 / 8, 5.0 / 4, 4.0 / 3, 3.0 / 2, 5.0 / 3, 15.0 / 8};
    return r;
}
inline const std::array<double, 8>& harmonicRatios()
{
    static const std::array<double, 8> r = {8.0 / 8, 9.0 / 8, 10.0 / 8, 11.0 / 8, 12.0 / 8, 13.0 / 8, 14.0 / 8, 15.0 / 8};
    return r;
}

// nearest_grid_freq() -- porting 1:1 (periodo sempre 2.0/ottava per le scale built-in,
// come in tuning.py; il periodo custom da .scl arrivera' col round 3.3b).
inline double nearestGridFreq(double freq, const double* degrees, int nDegrees,
                               double period, double anchorFreq)
{
    if (!(freq > 0.0) || !(anchorFreq > 0.0) || !std::isfinite(freq))
        return freq;  // difesa in profondita', stesso guard di tuning.py
    const int m = (int)std::lround(std::log(freq / anchorFreq) / std::log(period));
    double best = freq, bestDist = 1e300;
    for (int mm = m - 1; mm <= m + 1; ++mm)
    {
        const double base = anchorFreq * std::pow(period, (double)mm);
        for (int i = 0; i < nDegrees; ++i)
        {
            const double cand = base * degrees[i];
            const double dist = std::fabs(std::log2(cand / freq));
            if (dist < bestDist) { bestDist = dist; best = cand; }
        }
    }
    return best;
}

// ---------------------------------------------------------------------------------
// Round 3.3b (2026-09-24): scala custom da file .scl. POD a dimensione fissa (nessuna
// allocazione, copiabile per puntatore in TriggerRequest). `degrees` = gradi INCLUSO
// il grado 0 (1.0) ed ESCLUSA la fine periodo, `period` = rapporto di ripetizione --
// stessa forma di tuning.py::load_scl(), che ritorna (degrees[:-1], degrees[-1]).
constexpr int kMaxCustomDegrees = 128;

struct CustomScale {
    int n = 0;
    double period = 2.0;
    double degrees[kMaxCustomDegrees] = {};
};

namespace detail {
inline std::string trimStr(const std::string& s)
{
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}
inline bool parseIntStrict(const std::string& s0, int& out)
{
    const std::string s = trimStr(s0);
    if (s.empty() || s.size() > 6) return false;
    for (char c : s) if (c < '0' || c > '9') return false;
    out = std::atoi(s.c_str());
    return true;
}
// Un valore .scl: "n/d" rapporto, con '.' = cents, altrimenti intero n = n/1 (stessa
// regola di load_scl() in tuning.py e dello standard Scala). Solo il primo token della
// riga conta (lo standard ammette testo libero dopo il valore).
inline bool parseSclValue(const std::string& line0, double& out)
{
    std::string line = line0;
    const size_t bang = line.find('!');
    if (bang != std::string::npos) line = line.substr(0, bang);
    line = trimStr(line);
    const size_t sp = line.find_first_of(" \t");
    if (sp != std::string::npos) line = line.substr(0, sp);
    if (line.empty()) return false;
    // Solo cifra/'.'/segno in testa: scarta "nan"/"inf" prima di atof (il progetto gira con
    // -ffast-math nei plugin DPF, dove isnan/isfinite sono inaffidabili, vedi punto 7d bug B).
    const char c0 = line[0];
    if (!((c0 >= '0' && c0 <= '9') || c0 == '.' || c0 == '+' || c0 == '-')) return false;
    const size_t slash = line.find('/');
    double v;
    if (slash != std::string::npos) {
        const double num = std::atof(line.substr(0, slash).c_str());
        const double den = std::atof(line.substr(slash + 1).c_str());
        if (!(den != 0.0)) return false;
        v = num / den;
    } else if (line.find('.') != std::string::npos) {
        v = std::pow(2.0, std::atof(line.c_str()) / 1200.0);
    } else {
        v = std::atof(line.c_str());
    }
    if (!(v > 1e-6 && v < 1e6)) return false;
    out = v;
    return true;
}
}  // namespace detail

// Parsing di un file .scl (testo intero). DIFFERENZA voluta da tuning.py::load_scl():
// li' l'intero N e' letto da lines[0] dopo aver scartato TUTTE le righe vuote, quindi un
// file standard con descrizione non vuota (prima riga utile = testo) farebbe fallire
// int() -- funziona solo con descrizione vuota. Qui lo standard: righe '!' = commento,
// prima riga (anche vuota) = descrizione, seconda = N; fallback "senza descrizione"
// (prima riga = N) se la seconda non e' un intero.
inline bool parseScl(const std::string& text, CustomScale& out)
{
    std::vector<std::string> L;
    {
        std::string cur;
        for (size_t i = 0; i <= text.size(); ++i) {
            if (i == text.size() || text[i] == '\n') {
                if (!cur.empty() && cur.back() == '\r') cur.pop_back();
                if (detail::trimStr(cur).rfind('!', 0) != 0) L.push_back(cur);
                cur.clear();
            } else cur.push_back(text[i]);
        }
    }
    int n = 0;
    size_t start;
    if (L.size() >= 2 && detail::parseIntStrict(L[1], n)) start = 2;
    else if (!L.empty() && detail::parseIntStrict(L[0], n)) start = 1;
    else return false;
    if (n < 1 || n > kMaxCustomDegrees) return false;

    std::vector<double> vals;
    vals.push_back(1.0);
    for (size_t i = start; i < L.size() && (int)vals.size() < n + 1; ++i) {
        if (detail::trimStr(L[i]).empty()) continue;
        double v;
        if (!detail::parseSclValue(L[i], v)) return false;
        vals.push_back(v);
    }
    if ((int)vals.size() != n + 1) return false;
    const double period = vals.back();
    if (!(period > 1.0000001 && period < 1e6)) return false;
    out.n = n;
    out.period = period;
    for (int i = 0; i < n; ++i) out.degrees[i] = vals[i];
    return true;
}

// Formato canonico su una riga (per lo state 'scl_data'): "n period d0 d1 ... d(n-1)".
// Solo ASCII, nessun byte 0xFF (separatore dello state chunk VST3 di DPF).
inline std::string serializeCustomScale(const CustomScale& s)
{
    std::string r = std::to_string(s.n);
    char buf[40];
    std::snprintf(buf, sizeof(buf), " %.12g", s.period);
    r += buf;
    for (int i = 0; i < s.n; ++i) {
        std::snprintf(buf, sizeof(buf), " %.12g", s.degrees[i]);
        r += buf;
    }
    return r;
}

inline bool deserializeCustomScale(const char* str, CustomScale& out)
{
    if (str == nullptr) return false;
    char* p = const_cast<char*>(str);
    char* e = nullptr;
    const long n = std::strtol(p, &e, 10);
    if (e == p || n < 1 || n > kMaxCustomDegrees) return false;
    p = e;
    const double period = std::strtod(p, &e);
    if (e == p || !(period > 1.0000001 && period < 1e6)) return false;
    p = e;
    CustomScale tmp;
    tmp.n = (int)n;
    tmp.period = period;
    for (long i = 0; i < n; ++i) {
        const double v = std::strtod(p, &e);
        if (e == p || !(v > 1e-6 && v < 1e6)) return false;
        tmp.degrees[i] = v;
        p = e;
    }
    out = tmp;
    return true;
}

// Pubblicazione senza lock verso il worker thread: kSlots scale IMMUTABILI una volta
// pubblicate; setState() (thread messaggi, unico scrittore) riempie il prossimo slot e
// poi scambia il puntatore atomico; run() copia solo il puntatore in TriggerRequest
// (POD, nessun refcount) e il worker legge lo slot puntato. Uno slot viene riscritto
// solo dopo kSlots importazioni successive: con import manuali (secondi tra l'uno e
// l'altro) e coda trigger di 32 elementi non puo' coincidere con una request in volo.
class CustomScaleStore {
public:
    static constexpr int kSlots = 8;
    void publish(const CustomScale& s)
    {
        const int idx = (int)(next_++ % kSlots);
        slots_[idx] = s;
        current_.store(&slots_[idx], std::memory_order_release);
    }
    void clear() { current_.store(nullptr, std::memory_order_release); }
    const CustomScale* current() const { return current_.load(std::memory_order_acquire); }
private:
    CustomScale slots_[kSlots];
    std::atomic<const CustomScale*> current_{nullptr};
    uint32_t next_ = 0;
};

// Equivalente di make_quantizer(scaleName, a4)(freq), indicizzato per int invece che
// per stringa. scaleIndex==kScaleNone -> nessuna quantizzazione (ritorna freq com'e').
// kScaleCustom con custom==nullptr (nessun .scl caricato) -> nessuna quantizzazione.
inline float quantizeFreqToScale(float freq, int scaleIndex, float a4,
                                  const CustomScale* custom = nullptr)
{
    switch (scaleIndex)
    {
        case kScaleCustom:
            if (custom == nullptr || custom->n < 1) return freq;
            return (float)nearestGridFreq(freq, custom->degrees, custom->n, custom->period, a4);
        case kScaleEdo12:    { const auto& d = edoRatios<12>(); return (float)nearestGridFreq(freq, d.data(), 12, 2.0, a4); }
        case kScaleEdo24:    { const auto& d = edoRatios<24>(); return (float)nearestGridFreq(freq, d.data(), 24, 2.0, a4); }
        case kScaleEdo31:    { const auto& d = edoRatios<31>(); return (float)nearestGridFreq(freq, d.data(), 31, 2.0, a4); }
        case kScalePerfect:  { const auto& d = justMajorRatios(); return (float)nearestGridFreq(freq, d.data(), 7, 2.0, a4); }
        case kScaleHarmonic: { const auto& d = harmonicRatios(); return (float)nearestGridFreq(freq, d.data(), 8, 2.0, a4); }
        default:             return freq;
    }
}

// Nota MIDI -> frequenza cromatica 12-TET standard (A4=nota 69). E' il punto di
// partenza che poi viene "agganciato" (snap) alla scala attiva via
// quantizeFreqToScale() -- round 3.3a, estensione NUOVA rispetto al prototipo Python
// (che non ha mai un input MIDI che sovrascrive il pitch), decisa con l'utente
// 2026-09-23: canale MIDI riservato al pulsante Play (vedi kPlayMidiChannel in
// ParamLayout.hpp) esclude l'override, qualunque altro canale lo applica se Scale e'
// attivo -- vedi PluginMultiPhiMo.cpp::run()/VoiceEngine.hpp::workerLoop().
inline float midiNoteToFreq(int note, float a4)
{
    return a4 * std::pow(2.0f, (float)(note - 69) / 12.0f);
}

}  // namespace phimo
