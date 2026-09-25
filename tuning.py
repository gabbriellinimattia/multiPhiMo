"""
tuning.py -- quantizzazione del descrittore "pitch" alla nota temperata piu' vicina
(toggle "Scale" in gui.py, punto 3 del prompt 2026-09-16).

Scelta di design: nessuna mappatura a nomi di nota/tonalita' -- la frequenza scelta nel
menu "La" (A4, es. 440/442 Hz) e' semplicemente l'ANCORA del grado 0 della scala
(period-based grid: ancora * grado * periodo**m, m intero, esteso su piu' periodi/ottave
in entrambe le direzioni). Per le scale EDO/giusta/armonica built-in il periodo e'
sempre l'ottava (2/1); i file .scl importati possono avere un periodo diverso (es.
Bohlen-Pierce, 3/1) e viene rispettato. Semplice, generalizza a qualunque scala senza
bisogno di teoria delle tonalita', coerente con "matching sui descrittori, non sul
timbro" del progetto: qui e' solo un vincolo sul valore target di pitch.
"""
import math


def edo_ratios(n):
    """n gradi equal-tempered per periodo (ottava), rapporti di frequenza sul grado 0."""
    return [2 ** (i / n) for i in range(n)]


# "Giusta" -- scala diatonica maggiore a intervalli razionali semplici (just intonation)
JUST_MAJOR_RATIOS = [1 / 1, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8]

# Serie armonica: armoniche 8-15 normalizzate sull'ottava base (8:8 -> 16:8 = un'ottava
# esatta, 8 gradi per ottava, spaziatura via via piu' stretta salendo -- caratteristica
# della serie armonica, non equal-tempered)
HARMONIC_RATIOS = [h / 8 for h in range(8, 16)]

# name -> (gradi esclusa la fine periodo, periodo)
BUILTIN_SCALES = {
    "edo12": (edo_ratios(12), 2.0),
    "edo24": (edo_ratios(24), 2.0),
    "edo31": (edo_ratios(31), 2.0),
    "perfect": (JUST_MAJOR_RATIOS, 2.0),
    "harmonic": (HARMONIC_RATIOS, 2.0),
}


def load_scl(path):
    """Legge un file .scl (formato Scala, http://www.huygens-fokker.org/scala/scl_format.html):
    righe '!' = commento, prima riga utile = descrizione (ignorata), seconda = numero di
    gradi N, poi N righe con un valore per grado (cents es. '700.00', o rapporto 'n/d',
    o intero 'n' = n/1) -- l'ultimo dei N e' per convenzione il periodo di ripetizione
    (di norma l'ottava, 2/1, ma non sempre). Ritorna (gradi_esclusa_fine_periodo, periodo),
    stessa forma di BUILTIN_SCALES."""
    # Standard Scala: le righe '!' sono commenti, la PRIMA riga rimasta (anche vuota) e' la
    # descrizione, la seconda e' N. (Prima le righe vuote venivano scartate e N letto da
    # lines[0]: falliva con qualunque descrizione non vuota.) Fallback per file senza
    # descrizione: se la seconda riga non e' un intero ma la prima si', la prima e' N.
    with open(path) as f:
        raw = [l.rstrip("\r\n") for l in f if not l.strip().startswith("!")]
    if len(raw) >= 2 and raw[1].strip().isdigit():
        n, start = int(raw[1]), 2
    elif raw and raw[0].strip().isdigit():
        n, start = int(raw[0]), 1
    else:
        raise ValueError(f"file .scl non valido (numero di gradi non trovato): {path}")
    # solo il primo token di ogni riga di valore conta (testo libero ammesso dopo)
    vals = [l.split("!", 1)[0].split()[0] for l in raw[start:] if l.split("!", 1)[0].strip()]
    degrees = [1.0]
    for line in vals[:n]:
        if "/" in line:
            num, den = line.split("/")
            degrees.append(float(num) / float(den))
        elif "." in line:
            degrees.append(2 ** (float(line) / 1200.0))
        else:
            degrees.append(float(line))
    return degrees[:-1], degrees[-1]


def nearest_grid_freq(freq, degrees, period, anchor_freq):
    """Frequenza piu' vicina (in scala log2, cioe' a orecchio) sulla griglia
    anchor_freq * grado * periodo**m. Controlla m-1/m/m+1 per coprire i bordi del
    periodo in cui il grado piu' vicino puo' cadere nel periodo adiacente."""
    if freq <= 0 or anchor_freq <= 0 or not math.isfinite(freq):
        # difesa in profondita' (2026-09-16): un valore non finito non deve MAI arrivare
        # qui nella pipeline normale (audio_input.py ora lo filtra a monte), ma un
        # secondo guard qui costa nulla e protegge da qualunque altra sorgente futura
        # (es. OSC malformato con "nan") -- ritorna la frequenza cosi' com'e' invece di
        # far esplodere math.log su NaN/inf.
        return freq
    m = round(math.log(freq / anchor_freq, period))
    best, best_dist = freq, float("inf")
    for mm in (m - 1, m, m + 1):
        base = anchor_freq * (period ** mm)
        for r in degrees:
            cand = base * r
            dist = abs(math.log2(cand / freq))
            if dist < best_dist:
                best, best_dist = cand, dist
    return best


def make_quantizer(scale_name, a4=440.0, custom=None):
    """Ritorna una funzione freq->freq. scale_name in BUILTIN_SCALES, oppure
    scale_name=="custom" con custom=(degrees, period) da load_scl."""
    degrees, period = custom if scale_name == "custom" else BUILTIN_SCALES[scale_name]

    def _q(freq):
        return nearest_grid_freq(freq, degrees, period, a4)

    return _q
