"""
Analizzatore descrittori — sez.1 di pipeline_multiPhiMo.txt.

Ogni funzione produce uno scalare costante per l'intero render (mediana sui
frame attivi), coerente con la convenzione "descrittori costanti in ingresso"
del progetto. Nessun timbral matching: i valori bastano a se stessi.

VERSIONE FINALE (2026-09-13): harness_timing.py aveva misurato una scalata
SUPERLINEARE del tempo di analisi con la durata del render (~1.3-1.6s per
analizzare 1.5s di audio, quasi quanto il tempo reale). Causa isolata:
pitch_autocorr usava np.correlate(x, x, mode="full") sull'INTERO segnale —
autocorrelazione diretta, O(n^2). Riscritta via FFT (Wiener-Khinchin),
O(n log n): era il collo di bottiglia dominante, non il render né le altre
funzioni di analisi. spectral_centroid_spread, spectral_flatness e roughness
vettorializzate (numpy broadcasting invece di loop Python per-frame/per-coppia)
per ridurre l'overhead dell'interprete, minore ma non trascurabile su molti
frame. Risultati numerici identici alla versione precedente (stesse formule).

Riferimenti:
- centroid/spread/flatness: Peeters (2004), standard STFT-based features.
- roughness: Vassilakis (2001), modello psicoacustico su coppie di parziali.
- harmonic_tension: proxy via 6D Tonal Centroid (Harte, Sandler, Gasser 2006)
  sul vettore di chroma — stessa trasformata usata da librosa.feature.tonnetz.
  Non è il modello spiral-array di Chew (richiede contesto tonale/nota per
  nota non disponibile su texture sintetiche arbitrarie): qui la norma del
  vettore 6D è usata come proxy deterministico di "focalizzazione tonale".
- formanti: LPC (Levinson-Durbin) su segnale ricampionato per stabilità
  numerica, root-finding del polinomio A(z) — metodo classico da speech
  processing.
- inviluppo (attack/decay): RMS a finestre scorrevoli nel dominio del tempo.
- modulazione: inviluppo di ampiezza (Hilbert) -> FFT dell'inviluppo.
- pitch: autocorrelazione normalizzata via FFT (fallback semplice, non YIN
  completo).
"""
import numpy as np
from scipy import signal as sg

EPS = 1e-12


# ---------------- I/O helpers ----------------

def to_mono(x):
    return x.mean(axis=1) if x.ndim > 1 else x


def peak_normalize(x):
    peak = np.max(np.abs(x))
    return x / peak if peak > EPS else x


# ---------------- STFT spectral descriptors ----------------

def _stft_mag(x, sr, nperseg=2048, hop_ratio=0.5):
    nperseg = min(nperseg, len(x))
    if nperseg < 8:
        nperseg = len(x)
    noverlap = int(nperseg * hop_ratio)
    freqs, _, Zxx = sg.stft(x, fs=sr, window="hann", nperseg=nperseg,
                             noverlap=noverlap, boundary=None)
    return freqs, np.abs(Zxx)  # (n_freqs, n_frames)


def _active_frames(mag, rel_thresh_db=-40):
    energy = np.sum(mag ** 2, axis=0)
    peak = energy.max()
    if peak <= EPS:
        return np.arange(mag.shape[1])
    thresh = peak * 10 ** (rel_thresh_db / 10)
    idx = np.where(energy >= thresh)[0]
    return idx if len(idx) else np.arange(mag.shape[1])


def spectral_centroid_spread(freqs, mag):
    active = _active_frames(mag)
    sub = mag[:, active]                            # (n_freqs, n_active)
    s = sub.sum(axis=0) + EPS                        # (n_active,)
    c = (freqs[:, None] * sub).sum(axis=0) / s        # (n_active,)
    sp = np.sqrt(((freqs[:, None] - c[None, :]) ** 2 * sub).sum(axis=0) / s)
    return float(np.median(c)), float(np.median(sp))


def spectral_rolloff(freqs, mag, pct=0.85):
    """Frequenza sotto la quale cade il pct (default 85%, MPEG-7/Peeters 2004)
    dell'energia spettrale totale -- descrittore MIR standard, complementare al
    centroide: misura il BORDO della banda occupata (due spettri con lo stesso
    centroide possono avere rolloff molto diverso, es. banda stretta vs. coda lunga
    di armoniche deboli). Vettorizzata, nessun loop Python sui frame."""
    active = _active_frames(mag)
    sub = mag[:, active]
    energy = sub ** 2
    cum = np.cumsum(energy, axis=0)
    total = cum[-1] + EPS
    reached = cum >= (pct * total)[None, :]
    idx = np.argmax(reached, axis=0)  # primo bin che soddisfa la soglia, per frame
    return float(np.median(freqs[idx]))


def spectral_flatness(mag, rel_thresh_db=-60):
    # 2026-09-14: la flatness (Wiener entropy, gm/am) va calcolata solo sui bin con energia
    # reale, non su tutto lo spettro fino a Nyquist -- la media geometrica e' fortemente
    # sensibile ai valori piccoli, quindi i tanti bin ad alta frequenza vicini al floor
    # numerico (fuori dal contenuto armonico di uno strumento fisico) dominano il risultato
    # anziche' il vero timbro. Gate in frequenza per-frame (stesso schema soglia relativa
    # di _active_frames, ma sull'asse frequenza invece che tempo), non un taglio fisso in Hz
    # -- funziona sia per bow (armonico, energia concentrata) sia per noise (banda larga).
    # 2026-09-16f (revisione descrittori): vettorizzata via mascheramento+nanmean --
    # stesso identico calcolo per-frame di prima (gm/am sui SOLI bin sopra soglia,
    # shift di +EPS prima del log come originale), solo senza il loop Python esplicito
    # sui frame -- 'peak' e' sempre incluso nel gate (thresh<=peak per costruzione),
    # quindi nessuna colonna valida puo' risultare tutta NaN.
    active = _active_frames(mag)
    sub = mag[:, active]
    peak = sub.max(axis=0)
    valid = peak > EPS
    if not np.any(valid):
        return 0.0
    sub, peak = sub[:, valid], peak[valid]
    thresh = peak * 10 ** (rel_thresh_db / 20)
    gated = sub + EPS
    masked = np.where(sub >= thresh[None, :], gated, np.nan)
    gm = np.exp(np.nanmean(np.log(masked), axis=0))
    am = np.nanmean(masked, axis=0)
    return float(np.median(gm / am))


# ---------------- Roughness (Vassilakis 2001) ----------------

def _spectral_peaks(freqs, avg_mag, max_peaks=40, rel_thresh_db=-40):
    peak = avg_mag.max()
    if peak <= EPS:
        return np.array([]), np.array([])
    thresh = peak * 10 ** (rel_thresh_db / 20)
    idx = np.where(
        (avg_mag[1:-1] > avg_mag[:-2]) &
        (avg_mag[1:-1] > avg_mag[2:]) &
        (avg_mag[1:-1] > thresh)
    )[0] + 1
    if len(idx) == 0:
        return np.array([]), np.array([])
    order = np.argsort(avg_mag[idx])[::-1][:max_peaks]
    idx = idx[order]
    return freqs[idx], avg_mag[idx]


def roughness(freqs, mag):
    # 2026-09-14: prima mag.mean(axis=1) su TUTTI i frame (silenzio ai bordi incluso),
    # incoerente con centroid/spread/flatness che filtrano sui frame attivi (_active_frames)
    # -- i picchi usati per il calcolo venivano diluiti da porzioni non rilevanti del buffer.
    active = _active_frames(mag)
    avg_mag = mag[:, active].mean(axis=1)
    pf, pa = _spectral_peaks(freqs, avg_mag)
    n = len(pf)
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    fi, fj = pf[iu[0]], pf[iu[1]]
    ai, aj = pa[iu[0]], pa[iu[1]]
    fmin = np.minimum(fi, fj)
    fmax = np.maximum(fi, fj)
    valid = fmin > 0
    fmin, fmax, ai, aj = fmin[valid], fmax[valid], ai[valid], aj[valid]
    df = fmax - fmin
    s = 0.24 / (0.021 * fmin + 19.0)
    x = ai * aj
    y = 2 * np.minimum(ai, aj) / (ai + aj + EPS)
    total = (x ** 0.1) * (0.5 * y ** 3.11) * (np.exp(-3.5 * s * df) - np.exp(-5.75 * s * df))
    return float(np.sum(total))


# ---------------- Inharmonicity (deviazione parziali dalla serie armonica) --------

def inharmonicity(freqs, mag, f0):
    """Deviazione media (pesata per ampiezza^2, normalizzata su f0) dei parziali
    rilevati dalla serie armonica ideale n*f0 -- misura MODEL-FREE (nessun fit fisico
    alla Fletcher/corda tesa: valida anche per risonatori che non seguono affatto
    quella legge, es. barre/piastre/membrane in resonator.py hanno rapporti tra i
    modi completamente diversi) usata in letteratura MIR (Krimphoff et al. 1994,
    stesso principio della famiglia di descrittori armonici di Peeters 2004). 0 =
    armonico perfetto, valori piu' alti = parziali via via piu' scollegati da un
    fondamentale comune. Aggiunta 2026-09-16f, motivata direttamente dal lavoro sul
    pitch dei risonatori inarmonici (vedi resonator.fundamental_freq): quantifica
    esattamente la proprieta' fisica che ha reso ambigua la stima acustica del pitch
    su bar/piastre/membrana. Complementare a harmonic_tension (quello misura la
    messa a fuoco tonale/chroma, non la relazione coi multipli del fondamentale).
    NaN se f0 non e' disponibile (stesso caso di pitch=NaN, es. extract_pitch=False
    nelle probe SPSA -- per questo NON e' incluso in param_candidate.SPSA_DESCRIPTORS,
    stesso motivo per cui pitch stesso non lo e'):
    """
    if f0 is None or not np.isfinite(f0) or f0 <= 0:
        return float("nan")
    active = _active_frames(mag)
    avg_mag = mag[:, active].mean(axis=1)
    pf, pa = _spectral_peaks(freqs, avg_mag)
    if len(pf) < 2:
        return float("nan")
    h = np.round(pf / f0)
    keep = h >= 1
    pf, pa, h = pf[keep], pa[keep], h[keep]
    if len(pf) < 2:
        return float("nan")
    dev = np.abs(pf - h * f0) / f0
    w = pa ** 2
    return float(np.sum(dev * w) / (np.sum(w) + EPS))


# ---------------- Harmonic tension (tonal-centroid proxy) ----------------

# 2026-09-16f (revisione descrittori): vettorizzata via np.bincount -- stesso identico
# binning per-bin di prima (arrotondamento al semitono piu' vicino, accumulo di m^2),
# solo senza il loop Python esplicito su ~1000+ bin per chiamata.
def _chroma_vector(freqs, avg_mag, fmin=27.5):
    valid = freqs >= fmin
    f, m = freqs[valid], avg_mag[valid]
    if len(f) == 0:
        return np.zeros(12)
    pc = (69.0 + 12.0 * np.log2(f / 440.0)) % 12.0
    b = np.round(pc).astype(int) % 12
    chroma = np.bincount(b, weights=m ** 2, minlength=12)
    s = chroma.sum()
    return chroma / s if s > EPS else chroma


# 2026-09-16f: gating sui frame attivi (STESSO fix gia' applicato a roughness il
# 2026-09-14, mai riportato qui -- la media su TUTTI i frame, silenzio ai bordi
# incluso, diluiva il contenuto tonale reale esattamente come succedeva a roughness).
def harmonic_tension(freqs, mag):
    active = _active_frames(mag)
    avg_mag = mag[:, active].mean(axis=1)
    c = _chroma_vector(freqs, avg_mag)
    l = np.arange(12)
    r1, r2, r3 = 1.0, 1.0, 0.5
    x1 = r1 * np.sum(c * np.sin(l * 7 * np.pi / 6))
    y1 = r1 * np.sum(c * np.cos(l * 7 * np.pi / 6))
    x2 = r2 * np.sum(c * np.sin(l * 3 * np.pi / 2))
    y2 = r2 * np.sum(c * np.cos(l * 3 * np.pi / 2))
    x3 = r3 * np.sum(c * np.sin(l * 2 * np.pi / 3))
    y3 = r3 * np.sum(c * np.cos(l * 2 * np.pi / 3))
    vec = np.array([x1, y1, x2, y2, x3, y3])
    return float(np.linalg.norm(vec))


# ---------------- Formanti (LPC / Levinson-Durbin) ----------------

def _lpc_levinson(frame, order):
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1: len(frame) + order]
    if ac[0] <= EPS:
        return None
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = ac[0]
    for i in range(1, order + 1):
        acc = ac[i] + np.dot(a[1:i], ac[i - 1:0:-1])
        k = -acc / e
        a_prev = a.copy()
        for j in range(1, i):
            a[j] = a_prev[j] + k * a_prev[i - j]
        a[i] = k
        e *= (1 - k ** 2)
        if e <= 0:
            break
    return a


def _formants_from_lpc(a, sr):
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]
    if len(roots) == 0:
        return np.array([])
    freqs = np.angle(roots) * sr / (2 * np.pi)
    bw = -0.5 * (sr / np.pi) * np.log(np.abs(roots) + EPS)
    keep = (freqs > 90) & (freqs < sr / 2 - 100) & (bw < 400)
    freqs = freqs[keep]
    return np.sort(freqs)


def formants(x, sr, target_sr=10000, frame_ms=25, hop_ms=10, order=12):
    if sr > target_sr:
        x_a = sg.resample_poly(x, target_sr, sr)
        sr_a = target_sr
    else:
        x_a, sr_a = x, sr
    pre = np.append(x_a[0], x_a[1:] - 0.97 * x_a[:-1])
    frame_len = int(sr_a * frame_ms / 1000)
    hop = max(1, int(sr_a * hop_ms / 1000))
    if frame_len < 8 or len(pre) < frame_len:
        return float("nan"), float("nan"), float("nan")
    win = sg.windows.hamming(frame_len)
    f1s, f2s, f3s = [], [], []
    for start in range(0, len(pre) - frame_len + 1, hop):  # +1: include l'ultimo frame pieno
        frame = pre[start:start + frame_len] * win
        if np.sum(frame ** 2) < EPS:
            continue
        a = _lpc_levinson(frame, order)
        if a is None:
            continue
        fm = _formants_from_lpc(a, sr_a)
        if len(fm) >= 3:
            f1s.append(fm[0]); f2s.append(fm[1]); f3s.append(fm[2])
    if not f1s:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(f1s)), float(np.median(f2s)), float(np.median(f3s))


# ---------------- Inviluppo temporale (attack/decay) ----------------

def envelope_times(x, sr, frame_ms=10, hop_ms=5, floor_db=-60):
    # 2026-09-16f: vettorizzata via cumsum dei quadrati (stesso schema di d(tau) in
    # pitch_autocorr) -- stesso RMS per-frame di prima, "end" tronca l'ultimo segmento
    # se piu' corto di frame_len esattamente come faceva lo slicing numpy permissivo
    # dell'originale (mai un IndexError anche su buffer piu' corti di un frame).
    x = np.asarray(x, dtype=np.float64)
    frame_len = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    n_frames = max(1, (len(x) - frame_len) // hop + 1)
    starts = np.arange(n_frames) * hop
    ends = np.minimum(starts + frame_len, len(x))
    seg_len = np.maximum(ends - starts, 1)
    cum = np.concatenate(([0.0], np.cumsum(x ** 2)))
    sums = cum[ends] - cum[starts]
    rms = np.sqrt(sums / seg_len + EPS)
    times = (starts + frame_len / 2) / sr
    peak_idx = int(np.argmax(rms))
    attack_time = float(times[peak_idx])
    thresh = rms[peak_idx] * 10 ** (floor_db / 20)
    after = np.where(rms[peak_idx:] <= thresh)[0]
    if len(after):
        decay_idx = peak_idx + after[0]
        return attack_time, float(times[decay_idx] - times[peak_idx]), False
    return attack_time, float(times[-1] - times[peak_idx]), True  # decay_capped


# ---------------- Modulazione (rate + depth) ----------------

def modulation(x, sr, band=(0.5, 20.0)):
    # 2026-09-14 (rev.2): il trim del transiente d'attacco va fatto sul picco
    # dell'inviluppo RMS frame-based di envelope_times() (10ms/5ms, la stessa identica
    # metrica di attack_time), NON sul np.argmax dell'inviluppo di Hilbert campione per
    # campione: per un tono ricco di armoniche/beating come bow, l'inviluppo istantaneo ha
    # fluttuazioni spurie in tutta la porzione sostenuta e il suo massimo GLOBALE cade
    # spesso vicino alla fine del buffer per puro beating, non per un vero attacco lento --
    # prima causava NaN su quasi meta' delle righe (env post-trim troppo corto).
    attack_t, _, _ = envelope_times(x, sr)
    start = int(np.clip(attack_t, 0.0, (len(x) - 1) / sr) * sr)
    x_trim = x[start:] if len(x) - start >= 8 else x
    env = np.abs(sg.hilbert(x_trim))
    env_ac = env - env.mean()
    if len(env_ac) < 8:
        return float("nan"), float("nan")
    win = sg.windows.hann(len(env_ac))
    spec = np.abs(np.fft.rfft(env_ac * win))
    freqs = np.fft.rfftfreq(len(env_ac), d=1 / sr)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0, 0.0
    idxs = np.flatnonzero(mask)
    i = int(idxs[np.argmax(spec[idxs])])
    # interpolazione parabolica tra i 3 bin attorno al picco (2026-09-14): senza, mod_rate
    # e' quantizzato al passo del bin FFT (1/durata, es. 0.667Hz a 1.5s) -- stima sub-bin.
    if 0 < i < len(spec) - 1:
        a_, b_, c_ = spec[i - 1], spec[i], spec[i + 1]
        denom = a_ - 2 * b_ + c_
        delta = 0.5 * (a_ - c_) / denom if abs(denom) > EPS else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    bin_width = freqs[1] - freqs[0]
    mod_rate = float(freqs[i] + delta * bin_width)
    mod_depth = float(spec[i] / (env.mean() * len(env_ac) / 2 + EPS))
    return mod_rate, mod_depth


# ---------------- Pitch (autocorrelazione via FFT) ----------------

def pitch_autocorr(x, sr, fmin=40, fmax=2000, voiced_thresh=0.15):
    """Stima f0 via YIN (de Cheveigne' & Kawahara 2002, JASA 111:1917-1930) -- sostituisce
    l'argmax su autocorrelazione grezza normalizzata su ac[0] (bug diagnosticato
    2026-09-16c/d: test_pitch_match.py + ispezione dei dataset di training, 38-69% dei
    render con pitch "incollato" allo stesso lag_min = sr/int(sr/fmax); l'autocorrelazione
    grezza e' polarizzata verso i lag corti perche' l'overlap disponibile si riduce al
    crescere del lag anche su un segnale perfettamente periodico -- un lag corto puo'
    sembrare "piu' periodico" per puro artefatto di normalizzazione su un termine fisso).

    YIN usa la funzione di differenza d(tau) = Somma_j (x[j]-x[j+tau])^2 sull'overlap
    disponibile (d(0)=0 per costruzione, mai il lag piu' corto per bias) invece
    dell'autocorrelazione grezza, poi la normalizza sulla propria MEDIA CUMULATIVA
    (CMNDF, par.3 del paper): d'(tau) ~ 1 per un segnale non periodico, scende sotto
    voiced_thresh solo al vero periodo (e ai suoi multipli -- da cui si cerca il PRIMO
    minimo sotto soglia, non il minimo globale, per evitare errori di ottava, par.4).
    Interpolazione parabolica sub-campione alla fine (par.6). Resta O(n log n): d(tau) si
    ricava dalla STESSA autocorrelazione FFT di prima (Wiener-Khinchin, invariata) piu'
    le energie cumulative dei due segmenti che si accorciano con tau -- nessuna nuova
    trasformata, nessuna dipendenza aggiuntiva.

    Sui risonatori fisici (bar/piastre/membrana, inarmonici per costruzione) questa resta
    comunque una stima acustica intrinsecamente ambigua (vedi Terhardt 1974 sul pitch
    virtuale) -- per le loro etichette di training dataset_gen.py NON usa piu' questa
    funzione, usa il fondamentale analitico di resonator.fundamental_freq (esatto, noto
    per costruzione). Questa funzione resta l'unica stima disponibile per gli eccitatori
    (via analyze_signal, sia offline sia dal vivo in audio_input.py/SPSA)."""
    x = x - x.mean()
    n = len(x)
    if np.sum(x ** 2) < EPS or n < 2:
        return float("nan")
    tau_min = max(int(sr / fmax), 1)
    tau_max = min(int(sr / fmin), n - 1)
    if tau_max <= tau_min:
        return float("nan")

    # autocorrelazione lineare via Wiener-Khinchin (FFT): O(n log n) invece di O(n^2) di
    # np.correlate(x, x, "full") sull'intero segnale (vedi harness_timing.py). nfft =
    # prima potenza di 2 >= 2n-1 evita contaminazione da wraparound circolare.
    nfft = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(x, n=nfft)
    ac_full = np.fft.irfft(spec * np.conj(spec), n=nfft)[:n]  # lag 0..n-1

    cum = np.concatenate(([0.0], np.cumsum(x ** 2)))  # cum[k] = energia x[0:k]
    taus = np.arange(tau_max + 1)
    # d(tau) = Somma_{j=0}^{n-1-tau} (x[j]-x[j+tau])^2, sviluppato in energie (dai
    # cumulativi sopra) + il prodotto incrociato (l'autocorrelazione gia' calcolata) --
    # l'overlap si accorcia con tau esattamente come prima, ma qui NON si normalizza piu'
    # su un termine fisso (ac[0]): la normalizzazione sotto e' sulla media cumulativa di
    # d stesso, dove sta il fix vero e proprio.
    d = cum[n - taus] + cum[n] - cum[taus] - 2.0 * ac_full[taus]
    d = np.maximum(d, 0.0)  # solo errori di arrotondamento in teoria

    running = np.cumsum(d[1:])
    counts = np.arange(1, tau_max + 1)
    cmndf = np.ones(tau_max + 1)
    ok = running > EPS
    cmndf[1:][ok] = d[1:][ok] * counts[ok] / running[ok]

    # decisione di voicing (par.4 YIN) A SOGLIE CRESCENTI (idea pYIN, Mauch & Dixon
    # 2014): preferisce sempre il candidato trovato con la soglia PIU' bassa (piu'
    # sicuro), allarga solo se non trova nulla. voiced_thresh=0.15 da solo si e' rivelato
    # troppo severo su segnali reali non puliti (filtrati/smorzati, non toni da
    # laboratorio -- freq220_q0_full/freq220_q1_half tornavano NaN pur essendo
    # genuinamente intonati, 2026-09-16g). Tetto a 0.35, non 1.0 come nel fallback
    # rimosso il turno precedente: quello accettava quasi qualunque minimo casuale su
    # materiale non periodico (baseline_atonal). 0.35 resta un ordine di grandezza piu'
    # severo, e la ricerca del minimo locale sotto (while) scarta comunque i punti
    # isolati non davvero al fondo di una valle.
    start = max(tau_min, 2)
    tau = None
    for thresh in (voiced_thresh, 0.25, 0.35):
        below = np.where(cmndf[start:] < thresh)[0]
        if len(below) == 0:
            continue
        tau = start + int(below[0])
        while tau + 1 <= tau_max and cmndf[tau + 1] < cmndf[tau]:
            tau += 1  # segue il minimo locale, non il primo campione sotto soglia
        break
    if tau is None:
        return float("nan")

    if tau_min < tau < tau_max:  # interpolazione parabolica sub-campione (par.6 YIN)
        y0, y1, y2 = cmndf[tau - 1], cmndf[tau], cmndf[tau + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > EPS:
            tau = tau + float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))

    if tau <= 0:
        return float("nan")
    return float(sr / tau)


# ---------------- Pitch MPM (McLeod & Wyvill 2005), dal 2026-09-24 (audit pitch r13-r16) ----------------

PITCH_FMIN = 25.0    # copre la tastiera del pianoforte (27.5 - 4186 Hz)
PITCH_FMAX = 4500.0


def pitch_mpm(x, sr, fmin=PITCH_FMIN, fmax=PITCH_FMAX, k=0.93, clarity_min=0.5):
    """Stima f0 con MPM (NSDF = 2*acf / (m'), primo picco >= k*max, interpolazione parabolica del lag).
    Sostituisce pitch_autocorr (YIN 40-2000 Hz) come stimatore di analyze_signal: YIN dava NaN sul 25-42% di strike/blow/bow
    e valori incollati a fmax=2000; sopra ~1.8 kHz e' stato necessario interpolare il valore dei picchi (T ~ 15 campioni),
    LIMITATO a [valore intero, 1] (la parabola puo' sovrastimare su picchi stretti: blow, -6000 c). NaN se il segnale e'
    quasi nullo, non ha picchi o il picco massimo dell'NSDF e' < clarity_min. Nota: Descriptors.hpp va rivalidato."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 64 or np.sum(x ** 2) < 1e-12:
        return float("nan")
    nfft = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(X * np.conj(X), nfft)[:n]
    cs = np.concatenate(([0.0], np.cumsum(x ** 2)))
    tau = np.arange(n)
    m = cs[n - tau] + (cs[n] - cs[tau])
    nsdf = np.where(m > 1e-12, 2.0 * acf / np.maximum(m, 1e-12), 0.0)
    tmin, tmax = max(int(sr / fmax), 2), min(int(sr / fmin), n - 2)
    neg = np.where(nsdf < 0)[0]
    if len(neg) == 0:
        return float("nan")
    s = max(int(neg[0]), tmin)
    peaks = []
    t = s
    while t <= tmax:
        if nsdf[t] > 0:
            e = t
            while e + 1 <= tmax and nsdf[e + 1] > 0:
                e += 1
            j = t + int(np.argmax(nsdf[t:e + 1]))
            peaks.append(j)
            t = e + 1
        else:
            t += 1
    if not peaks:
        return float("nan")

    def _pv(j):
        if 1 <= j < n - 1:
            y0, y1, y2 = nsdf[j - 1], nsdf[j], nsdf[j + 1]
            d = y0 - 2 * y1 + y2
            if d < -1e-12:
                return float(min(max(y1 - (y0 - y2) ** 2 / (8.0 * d), y1), 1.0))
        return float(nsdf[j])
    vals = np.array([_pv(j) for j in peaks])
    hi = vals.max()
    if hi < clarity_min:
        return float("nan")
    j = peaks[int(np.argmax(vals >= k * hi))]
    tt = float(j)
    if 1 <= j < n - 1:
        y0, y1, y2 = nsdf[j - 1], nsdf[j], nsdf[j + 1]
        d = y0 - 2 * y1 + y2
        if abs(d) > 1e-12:
            tt += float(np.clip(0.5 * (y0 - y2) / d, -1, 1))
    return float(sr / tt) if tt > 0 else float("nan")


# ---------------- Entry point ----------------

def analyze_signal(x, sr, extract_pitch=True, normalize=False):
    x = to_mono(np.asarray(x, dtype=np.float64))
    if normalize:
        x = peak_normalize(x)
    freqs, mag = _stft_mag(x, sr)
    centroid, spread = spectral_centroid_spread(freqs, mag)
    rolloff = spectral_rolloff(freqs, mag)
    flat = spectral_flatness(mag)
    rough = roughness(freqs, mag)
    tension = harmonic_tension(freqs, mag)
    f0 = pitch_mpm(x, sr) if extract_pitch else float("nan")
    inharm = inharmonicity(freqs, mag, f0)
    f1, f2, f3 = formants(x, sr)
    mod_rate, mod_depth = modulation(x, sr)
    attack, decay, decay_capped = envelope_times(x, sr)
    result = {
        "spectral_centroid": centroid,
        "spectral_spread": spread,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flat,
        "roughness": rough,
        "harmonic_tension": tension,
        "inharmonicity": inharm,
        "formant_f1": f1,
        "formant_f2": f2,
        "formant_f3": f3,
        "mod_rate": mod_rate,
        "mod_depth": mod_depth,
        "attack_time": attack,
        "decay_time": decay,
        "decay_capped": decay_capped,
    }
    if extract_pitch:
        result["pitch"] = f0
    return result


def analyze_file(path, **kwargs):
    import soundfile as sf
    x, sr = sf.read(path, always_2d=False)
    return analyze_signal(x, sr, **kwargs)


# ---------------- Estrazione a finestre (curva nel tempo) ----------------

def analyze_signal_windowed(x, sr, win_s=0.5, hop_s=0.3, extract_pitch=True,
                             normalize=False, min_last_frac=0.5):
    """Curva di descrittori: applica analyze_signal() (invariata) a finestre scorrevoli
    invece che sul segnale intero -- nessuna funzione di analisi duplicata, ogni finestra
    e' trattata come un mini-segnale a se stante.

    win_s: durata finestra (default 0.5s: abbastanza lunga per formanti stabili, ~20
    sotto-frame LPC da 25ms dentro formants(), e per pitch_autocorr fino a fmin=40Hz).
    hop_s: passo tra finestre (default 0.3s, dentro il budget realtime 0.2-0.5s del
    progetto, vedi runtime_architettura_realtime.md).
    normalize: applicato UNA VOLTA sul segnale intero prima di affettare (non per
    finestra), altrimenti si perderebbero le differenze di livello relativo tra finestre.
    min_last_frac: l'ultima finestra, se piu' corta di min_last_frac*win_s, e' scartata
    (evita stime rumorose su un frammento finale troppo breve).

    Ritorna una lista di dict, stessi campi di analyze_signal() piu' "t_start"/"t_end"
    (secondi, riferiti all'inizio del segnale).
    """
    x = to_mono(np.asarray(x, dtype=np.float64))
    if normalize:
        x = peak_normalize(x)
    win_len = max(1, int(round(win_s * sr)))
    hop_len = max(1, int(round(hop_s * sr)))
    min_len = int(round(min_last_frac * win_len))
    n = len(x)
    curve = []
    start = 0
    while start < n:
        end = min(start + win_len, n)
        seg = x[start:end]
        if len(seg) < min_len:
            break
        d = analyze_signal(seg, sr, extract_pitch=extract_pitch, normalize=False)
        d["t_start"] = start / sr
        d["t_end"] = end / sr
        curve.append(d)
        if end >= n:
            break
        start += hop_len
    return curve


def analyze_file_windowed(path, **kwargs):
    import soundfile as sf
    x, sr = sf.read(path, always_2d=False)
    return analyze_signal_windowed(x, sr, **kwargs)
