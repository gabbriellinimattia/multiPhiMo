"""
exciters.py — Motore di sintesi, sez.2 di pipeline_multiPhiMo.txt.

10 eccitatori (7 + mechanical + bird + vocal, 2026-09-20), ciascuno una funzione pura: parametri di controllo -> np.ndarray audio
(float32, SR fisso). Implementazione offline (loop Python per-sample dove necessario):
NON vincolata ai 25ms (quel vincolo vale solo per la reimplementazione runtime nativa,
sez.1 del doc). Qui l'obiettivo e' generare render per il dataset (sez.4) e per la
validazione dei descrittori — vedi claude/exciters_modelli_scelti.md per la scelta di
ciascun modello e i riferimenti bibliografici.

Ogni eccitatore include un risonatore minimo "di comodo" (loop waveguide o piccolo banco
modale) solo per rendere il suono udibile stand-alone: verra' sostituito dal risonatore
condiviso (agente 8, sez.3) quando pronto. Non e' timbral matching: bastano i descrittori,
quindi le costanti fisiche qui sotto sono punti di partenza plausibili, non calibrati —
da aggiustare con i log/descrittori reali una volta testato.
"""
import numpy as np
from scipy import signal as sg

SR = 44100
EPS = 1e-9


def _as_traj(val, n):
    """Broadcasta uno scalare a un array di lunghezza n, o lo passa invariato se gia'
    un array (2026-09-18, morphing B): permette a bow/blow/chaos/pluck di ricevere
    parametri che scivolano nel tempo invece di costanti fisse per l'intera nota,
    riusando lo stesso loop per-campione -- comportamento IDENTICO a prima quando il
    chiamante passa ancora uno scalare (retrocompatibile)."""
    arr = np.asarray(val, dtype=np.float64)
    return np.full(n, val, dtype=np.float64) if arr.ndim == 0 else arr


# ---------------- helper condivisi ----------------

def _resonant_filter(x, freq, damping_time, sr=SR):
    """Filtro risonante a 2 poli (un modo): eccitato da x, decade in damping_time secondi."""
    r = np.exp(-1.0 / (max(damping_time, 1e-4) * sr))
    theta = 2 * np.pi * min(max(freq, 1.0), sr / 2 - 1) / sr
    a = [1.0, -2 * r * np.cos(theta), r * r]
    return sg.lfilter([1.0], a, x)


def _modal_bank(x, base_freq, ratios, damping_time, amps=None, sr=SR):
    """Banco di N modi (frequenze = base_freq*ratios) eccitati dallo stesso ingresso x."""
    if amps is None:
        amps = 1.0 / np.arange(1, len(ratios) + 1)
    y = np.zeros_like(x, dtype=np.float64)
    for ratio, amp in zip(ratios, amps):
        y += amp * _resonant_filter(x, base_freq * ratio, damping_time, sr)
    return y


def _peak_normalize(x, target=0.9):
    peak = np.max(np.abs(x))
    return (x / peak * target).astype(np.float32) if peak > EPS else x.astype(np.float32)


# ---------------- 1. BOW ----------------

def bow(duration=1.5, sr=SR, freq=220.0, bow_force=0.5, bow_velocity=0.35,
        bow_position=0.12, brightness=0.5, damping=0.9997):
    """
    Digital waveguide a loop singolo + curva di attrito differenziale (semplificazione a
    singola polarizzazione di McIntyre-Woodhouse-Schumacher, stile Smith/CCRMA MUS420).
    Semplificazione voluta rispetto al modello a due segmenti/polarizzazioni: qui la
    posizione d'arco agisce solo come comb filter in uscita, non come giunzione fisica.

    freq: frequenza fondamentale (Hz) -> lunghezza del loop.
    bow_force: 0-1, pressione arco (ampiezza curva di attrito statico/dinamico).
    bow_velocity: 0-1, velocita' di trascinamento dell'arco.
    bow_position: 0-1, posizione lungo la corda (comb filter in uscita).
    brightness: 0-1, quanto il filtro di loop lascia passare le alte frequenze.
    damping: <1, perdite per giro (insieme alla forza d'arco fissa l'ampiezza di regime).
    """
    n = int(duration * sr)
    N = max(4, int(round(sr / freq)))
    # 2026-09-18, morphing B: bow_force/bow_velocity/brightness/damping possono essere
    # array di lunghezza n (traiettoria nel tempo) oltre che scalari -- vedi _as_traj.
    # freq/bow_position restano scalari (freq fissa la lunghezza del loop N).
    bow_force = _as_traj(bow_force, n)
    bow_velocity = _as_traj(bow_velocity, n)
    brightness = _as_traj(brightness, n)
    damping = _as_traj(damping, n)
    buf = np.zeros(N)
    out = np.zeros(n)
    lp_state = 0.0
    v0 = 0.1
    ptr = 0
    for i in range(n):
        y_prev = buf[ptr]
        v_rel = bow_velocity[i] - y_prev
        mu_s, mu_d = 0.8 * bow_force[i], 0.2 * bow_force[i]
        mu = mu_d + (mu_s - mu_d) * np.exp(-abs(v_rel) / v0)
        force = mu * np.sign(v_rel) if abs(v_rel) > 1e-6 else 0.0
        raw = damping[i] * y_prev + force
        lp_state = brightness[i] * raw + (1 - brightness[i]) * lp_state
        buf[ptr] = lp_state
        out[i] = lp_state
        ptr = (ptr + 1) % N

    # comb dipendente dalla posizione d'arco (enfasi/attenuazione armonica)
    tap = max(1, min(int(round(bow_position * N)), n - 1))
    comb = out.copy()
    comb[tap:] -= 0.5 * out[:-tap]
    return _peak_normalize(comb)


# ---------------- 2. BLOW ----------------

def blow(duration=1.5, sr=SR, freq=220.0, mouth_pressure=0.6, reed_stiffness=0.5,
         breath_noise=0.15, brightness=0.6, damping=0.999):
    """
    Ancia singola non lineare (curva di apertura, stile clarinetto Smith/McIntyre-Woodhouse-
    Schumacher) + rumore di turbolenza proporzionale al flusso, su canna a delay singolo
    (aperta: riflessione invertita in fondo). reed_stiffness basso avvicina il comportamento
    a un edge-tone (piu' rumore, meno regime a riflessione).

    freq: frequenza fondamentale (Hz) -> lunghezza canna.
    mouth_pressure: 0-1, pressione in bocca (energia immessa).
    reed_stiffness: 0-1, rigidita' ancia.
    breath_noise: 0-1, quota di rumore di turbolenza mescolato al flusso.
    brightness: 0-1, perdite/filtro passa-basso in canna.
    damping: <1, perdite per giro in canna.
    """
    n = int(duration * sr)
    N = max(4, int(round(sr / (2 * freq))))
    # 2026-09-18, morphing B: vedi bow() sopra, stesso meccanismo.
    mouth_pressure = _as_traj(mouth_pressure, n)
    reed_stiffness = _as_traj(reed_stiffness, n)
    breath_noise = _as_traj(breath_noise, n)
    brightness = _as_traj(brightness, n)
    damping = _as_traj(damping, n)
    buf = np.zeros(N)
    out = np.zeros(n)
    lp_state = 0.0
    ptr = 0
    noise = np.random.default_rng().standard_normal(n)
    for i in range(n):
        p_bore = buf[ptr]
        p_diff = mouth_pressure[i] - p_bore
        opening = np.clip(1.0 - reed_stiffness[i] * p_diff, 0.0, 1.2)
        flow = mouth_pressure[i] * opening + breath_noise[i] * mouth_pressure[i] * noise[i]
        raw = damping[i] * (-p_bore) + flow
        lp_state = brightness[i] * raw + (1 - brightness[i]) * lp_state
        buf[ptr] = lp_state
        out[i] = lp_state
        ptr = (ptr + 1) % N
    return _peak_normalize(out)


# ---------------- 3. STRIKE ----------------

def strike(duration=1.0, sr=SR, freq=220.0, impact_velocity=0.8, hammer_mass=0.02,
           hammer_stiffness=5e7, nonlinearity=1.5, material=0.3, size_damping=0.4):
    """
    Contatto non lineare hertziano/Hunt-Crossley (martelletto -> superficie), integrato
    esplicitamente per pochi ms, che eccita un piccolo banco modale (Avanzini/Rocchesso;
    van den Doel/Pai).

    freq: frequenza del modo fondamentale del risonatore.
    impact_velocity: 0-1, velocita' d'impatto.
    hammer_mass, hammer_stiffness, nonlinearity: parametri del contatto (F = k*x^nonlinearity).
    material: 0-1, inarmonicita' del banco modale (0=quasi armonico, 1=molto inarmonico).
    size_damping: 0-1, velocita' di decadimento dei modi (piu' alto = decadimento piu' rapido).
    """
    v = impact_velocity * 2.0
    x = 0.0
    dt = 1.0 / sr
    max_steps = int(0.02 * sr)
    forces = []
    for _ in range(max_steps):
        f = hammer_stiffness * max(x, 0.0) ** nonlinearity
        a = -f / hammer_mass
        v += a * dt
        x += v * dt
        forces.append(max(f, 0.0))
        if x < 0:
            break
    force = np.array(forces) if forces else np.array([1.0])
    peak = np.max(np.abs(force))
    force = force / peak if peak > EPS else force

    n = int(duration * sr)
    x_in = np.zeros(n)
    x_in[:min(len(force), n)] = force[:n]
    ratios = [1.0 + material * 0.05, 2.0 + material * 0.6, 3.0 + material * 1.3,
              4.2 + material * 2.1, 5.4 + material * 3.0]
    damping_time = 0.05 + (1 - size_damping) * 1.5
    return _peak_normalize(_modal_bank(x_in, freq, ratios, damping_time, sr=sr))


# ---------------- 4. PLUCK ----------------

def pluck(duration=2.0, sr=SR, freq=220.0, pluck_position=0.2, pluck_hardness=0.5,
          decay_time=1.5, dispersion=0.0):
    """
    Karplus-Strong esteso / commuted waveguide synthesis (Jaffe & Smith 1983): burst di
    rumore filtrato (posizione + durezza) iniettato in un loop a delay con filtro medio
    mobile di decadimento e allpass opzionale di dispersione (corde rigide/metalliche).

    freq: frequenza fondamentale -> lunghezza loop.
    pluck_position: 0-1, punto di pizzico (comb filter sul burst iniziale).
    pluck_hardness: 0-1, durezza plettro/dito (0=morbido/scuro, 1=duro/brillante).
    decay_time: secondi a -60dB.
    dispersion: 0-1, inarmonicita' introdotta dall'allpass.
    """
    n = int(duration * sr)
    N = max(4, int(round(sr / freq)))

    burst = np.random.default_rng().standard_normal(N)
    if pluck_hardness < 1.0:
        alpha = 0.05 + 0.9 * (1 - pluck_hardness)
        burst = sg.lfilter([alpha], [1, -(1 - alpha)], burst)
    tap = max(1, min(int(round(pluck_position * N)), N - 1))
    if tap < N:
        burst[tap:] += -burst[:-tap]
    buf = burst.copy()

    # 2026-09-18, morphing B: decay_time/dispersion possono variare nel tempo (letti ad
    # ogni iterazione del loop Karplus-Strong, stessa famiglia di damping/brightness in
    # bow/blow/chaos) -- pluck_position/pluck_hardness/freq restano scalari (modellano
    # solo il burst iniziale, one-shot per costruzione).
    decay_time = _as_traj(decay_time, n)
    dispersion = _as_traj(dispersion, n)
    decay_per_sample = np.exp(-6.91 / (np.maximum(decay_time, 0.01) * sr))
    ap_coef = -dispersion * 0.5
    ap_x_prev, ap_y_prev = 0.0, 0.0
    out = np.zeros(n)
    ptr = 0
    for i in range(n):
        prev = buf[ptr]
        nxt = buf[(ptr + 1) % N]
        filtered = 0.5 * (prev + nxt) * decay_per_sample[i]
        if dispersion[i] > 0:
            ap_out = ap_coef[i] * filtered + ap_x_prev - ap_coef[i] * ap_y_prev
            ap_x_prev, ap_y_prev = filtered, ap_out
            filtered = ap_out
        buf[ptr] = filtered
        out[i] = filtered
        ptr = (ptr + 1) % N
    return _peak_normalize(out)


# ---------------- 5. SHAKER ----------------

def shaker(duration=1.5, sr=SR, freq=800.0, n_particles=50, energy=0.7,
           decay_time=0.8, material=0.3):
    """
    PhISM/PhISEM (Cook 1996-97): collisioni stocastiche di particelle (tasso legato a
    n_particles ed energia residua, approssimazione tipo Poisson) in una cavita' risonante
    approssimata da un piccolo banco modale.

    freq: frequenza base della cavita' (Hz).
    n_particles: numero di particelle (piu' alto = texture piu' densa/continua).
    energy: 0-1, energia iniziale della "shakata".
    decay_time: secondi, costante di decadimento dell'energia del sistema.
    material: 0-1, inarmonicita' dei modi della cavita'/guscio.
    """
    n = int(duration * sr)
    t = np.arange(n) / sr
    sys_energy = energy * np.exp(-t / max(decay_time, 0.01))
    rng = np.random.default_rng()
    prob = np.clip(n_particles / 800.0 * sys_energy, 0.0, 0.9)
    hits = rng.random(n) < prob
    impulses = np.where(hits, rng.choice([-1.0, 1.0], size=n) * np.sqrt(sys_energy + EPS), 0.0)
    ratios = [1.0, 1.8 + material, 2.6 + 2 * material, 3.4 + 3 * material]
    return _peak_normalize(_modal_bank(impulses, freq, ratios, 0.05, sr=sr))


# ---------------- 6. NOISE ----------------

def noise(duration=1.0, sr=SR, color=0.0, density=1.0, correlation=0.0,
          freq=440.0, tone_amount=0.0, tone_q=0.5):
    """
    Rumore bianco sagomato spettralmente (approccio alla Zhu & Wyse, "Sound Texture
    Modelling with Linear Prediction", qui semplificato con un tilt in frequenza via FFT
    invece di un vero modello a predizione lineare), con risonanza tonale opzionale.

    color: -1..1, tilt spettrale (negativo=scuro/pink-brown, positivo=chiaro/blue-violet).
    density: 0-1, densita' temporale (1=continuo, <1 introduce gating granulare casuale).
    correlation: 0-1, smoothing temporale aggiuntivo (piu' alto = piu' "colloso"/correlato).
    freq: Hz, centro della risonanza tonale opzionale (aggiunta fase 2, 2026-09-15: prima
      'noise' non aveva alcun parametro di frequenza fondamentale, quindi non poteva mai
      inseguire un pitch target reale -- vedi criticita_modelli.md). Ha effetto solo se
      tone_amount > 0: default 0.0 mantiene INVARIATO il comportamento precedente.
    tone_amount: 0-1, quanto della risonanza a 'freq' e' miscelata nel segnale (0 =
      rumore puramente atonale come prima; valori piu' alti = componente tonale via via
      piu' presente/dominante).
    tone_q: 0-1, "risonanza" del picco tonale -- riusa _resonant_filter (stesso helper
      degli altri eccitatori): piu' alto = picco piu' stretto/sostenuto, piu' basso =
      accenno di intonazione piu' largo/sfumato.
    """
    n = int(duration * sr)
    rng = np.random.default_rng()
    x = rng.standard_normal(n)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    if len(freqs) > 1:
        freqs[0] = freqs[1]
    X *= freqs ** (color * 1.5)
    x = np.fft.irfft(X, n)

    if density < 1.0:
        gate_n = max(1, int(sr * 0.02))
        n_gates = n // gate_n + 1
        gate_vals = (rng.random(n_gates) < density).astype(float)
        gate = np.repeat(gate_vals, gate_n)[:n]
        gate = sg.lfilter([0.1], [1, -0.9], gate)
        x *= gate

    if correlation > 0:
        alpha = 0.01 + 0.98 * correlation
        x = sg.lfilter([alpha], [1, -(1 - alpha)], x)

    if tone_amount > 0:
        damping_time = 0.01 + tone_q * 0.3
        tone = _resonant_filter(x, freq, damping_time, sr)
        peak_x, peak_tone = np.max(np.abs(x)), np.max(np.abs(tone))
        if peak_tone > EPS:
            tone = tone / peak_tone * (peak_x if peak_x > EPS else 1.0)
        x = (1 - tone_amount) * x + tone_amount * tone

    return _peak_normalize(x)


def noise_morph(params_from, params_to, duration, sr=SR, morph_ms=150.0,
                 block_ms=60.0, hop_ms=30.0):
    """Morphing 'A' per noise (2026-09-18, vedi
    claude/morphing_spettrale_ricerca_proposta.md): noise() e' block-based (FFT
    sull'intero buffer per il tilt spettrale), non ha stato per-campione da far
    scivolare come bow/blow/chaos/pluck (morphing B). Qui si procede a BLOCCHI
    SOVRAPPOSTI (overlap-add, finestra di Hann): ogni blocco chiama noise() SENZA
    modificarla, con i parametri interpolati linearmente tra params_from e params_to
    (0->1 durante i primi morph_ms, poi fermi su params_to) -- rumore fresco per
    blocco e' corretto qui: materiale stocastico non ha una fase da preservare come
    il materiale armonico (bow/blow/pluck), a differenza del vocoder di fase che
    servirebbe per quelli. Usata solo intra-coppia (stesso eccitatore 'noise' prima e
    dopo), mai per cambi di eccitatore/risonatore."""
    # 2026-09-18e: gli OLA a blocchi (ognuno con una FFT dentro noise()) servono SOLO
    # durante la transizione (morph_n campioni) -- prima facevano OLA per l'INTERA nota,
    # sprecando calcolo (e potenzialmente aggiungendo latenza di render) sulla parte
    # gia' stabile al target. Dopo morph_n si passa a UNA chiamata sola a noise() coi
    # parametri finali -- oltretutto piu' pulita (nessuna cucitura OLA sulla parte
    # sostenuta).
    n = int(duration * sr)
    morph_n = max(1, min(int(sr * morph_ms / 1000.0), n))
    block_n = max(1, int(sr * block_ms / 1000.0))
    keys = ("color", "density", "correlation", "freq", "tone_amount", "tone_q")
    tail_params = {k: params_to.get(k, params_from.get(k, 0.0))
                   for k in keys if k in params_from or k in params_to}
    if morph_n <= block_n:
        # transizione troppo corta per blocchi utili: equivalente a noise(**params_to)
        return _peak_normalize(noise(duration=duration, sr=sr, **tail_params))
    hop_n = max(1, int(sr * hop_ms / 1000.0))
    window = np.hanning(block_n) if block_n > 1 else np.ones(block_n)
    out = np.zeros(morph_n + block_n)
    wsum = np.zeros(morph_n + block_n)
    pos = 0
    while pos < morph_n:
        t = min(pos / morph_n, 1.0)
        blk_params = {}
        for k in keys:
            a = params_from.get(k, params_to.get(k, 0.0))
            b = params_to.get(k, a)
            blk_params[k] = a + t * (b - a)
        blk = noise(duration=block_n / sr, sr=sr, **blk_params)
        L = min(block_n, len(blk))
        out[pos:pos + L] += blk[:L] * window[:L]
        wsum[pos:pos + L] += window[:L]
        pos += hop_n
    wsum[wsum < 1e-6] = 1.0
    head = out[:morph_n] / wsum[:morph_n]
    if morph_n < n:
        tail = noise(duration=(n - morph_n) / sr, sr=sr, **tail_params)
        full = np.concatenate([head, tail])[:n]
    else:
        full = head[:n]
    return _peak_normalize(full)


# ---------------- 7. CHAOS ----------------

def chaos(duration=1.5, sr=SR, freq=220.0, bifurcation=0.9, coupling_rate=200.0,
          x0=0.6, brightness=0.5, damping=0.998):
    """
    Mappa logistica x_{n+1} = r*x_n*(1-x_n) come sorgente di eccitazione, campionata a
    coupling_rate Hz e agganciata a un loop waveguide (stile "Connecting Chaotic Maps to
    Digital Waveguides", NIME 2018).

    freq: frequenza del loop waveguide (Hz).
    bifurcation: 0-1, mappato su r in [3.57, 3.99] (regime caotico).
    coupling_rate: Hz, frequenza di aggiornamento del valore iniettato nel loop.
    x0: 0-1 (evitare 0, 0.5, 1 esatti), condizione iniziale della mappa.
    brightness: 0-1, filtro di loop.
    damping: <1, perdite per giro.
    """
    n = int(duration * sr)
    N = max(4, int(round(sr / freq)))
    # 2026-09-18, morphing B: bifurcation/damping/brightness possono variare nel tempo;
    # coupling_rate/x0/freq restano scalari (strutturali/condizione iniziale).
    bifurcation = _as_traj(bifurcation, n)
    damping = _as_traj(damping, n)
    brightness = _as_traj(brightness, n)
    step = max(1, int(round(sr / coupling_rate)))
    x = np.clip(x0, 1e-4, 1 - 1e-4)
    buf = np.zeros(N)
    out = np.zeros(n)
    lp_state = 0.0
    ptr = 0
    drive = 0.0
    for i in range(n):
        if i % step == 0:
            r = 3.57 + 0.42 * np.clip(bifurcation[i], 0.0, 1.0)
            x = r * x * (1 - x)
            drive = 2 * x - 1
        raw = damping[i] * buf[ptr] + 0.3 * drive
        lp_state = brightness[i] * raw + (1 - brightness[i]) * lp_state
        buf[ptr] = lp_state
        out[i] = lp_state
        ptr = (ptr + 1) % N
    return _peak_normalize(out)


# ---------------- 8. MECHANICAL ----------------

def mechanical(duration=1.5, sr=SR, freq=800.0, rate=30.0, jitter=0.3, air_mix=0.3,
               air_color=0.0, load_mod=0.1, seed=0):
    """
    Rumori meccanici caotico-stocastici (motori, ingranaggi, aria compressa), 2026-09-20 --
    vedi claude/nuovi_eccitatori_dettaglio_equazioni.md sez.1. Composizione di due rami
    interamente vettorizzati (lfilter, nessun loop per-campione, il piu' economico degli
    eccitatori): treno di impulsi con jitter deterministico-caotico (mappa logistica, come
    chaos) che eccita un piccolo banco modale ("dente"/cassa) + rumore turbolento sagomato
    con inviluppo pulsato (aria compressa).

    freq: Hz, modo fondamentale della risonanza colpita (dente/cassa); nome 'freq' per
      compatibilita' con gli altri 7 eccitatori (non e' un pitch percepito preciso).
    rate: impulsi/s (velocita' di rotazione), 2-200.
    jitter: 0-1, irregolarita' (0 = treno perfettamente regolare e ampiezze costanti;
      1 = mappa logistica ~caotica sia sugli intervalli sia sulle ampiezze).
    air_mix: 0-1, quota rumore d'aria (0 = solo impulsi, 1 = solo aria compressa).
    air_color: -0.9..0.9, tilt del rumore (>0 scuro, <0 chiaro), IIR a 1 polo.
    seed: int, seme del rumore (default 0 = render deterministico: stessi parametri -> stesso
      audio; il pavimento di ripetibilita' di mod_rate/mod_depth/attack_time era NMAE 0.6-0.95).
    load_mod: 0-0.5, profondita' della modulazione lenta del tasso (~2 Hz, "carico
      variabile": motore che arranca).
    """
    n = int(duration * sr)
    rng = np.random.default_rng(seed)  # seed fisso: render deterministico (2026-09-21, vedi nuovi_eccitatori_stato.md)

    # carico lento: rumore passa-basso ~2 Hz a varianza 1 -> modula il tasso istantaneo
    a_lp = 1.0 - np.exp(-2 * np.pi * 2.0 / sr)
    lam = sg.lfilter([a_lp], [1, -(1 - a_lp)], rng.standard_normal(n))
    lam = np.clip(lam / (np.std(lam) + EPS), -2.0, 2.0)
    r_eff = np.maximum(rate * (1.0 + load_mod * lam), 0.3 * rate)

    # treno di impulsi: intervalli/ampiezze da mappa logistica (mu<=3.99: a mu=4 la
    # mappa puo' collassare a 0 per errore numerico)
    mu = 3.4 + 0.59 * jitter
    z = 0.3 + 0.4 * rng.random()
    pos = rng.uniform(0.05, 0.5) * sr / rate
    idx, amp = [], []
    while pos < n:
        i = int(pos)
        z = float(np.clip(mu * z * (1.0 - z), 1e-6, 1.0 - 1e-6))
        dev = float(np.clip(2.0 * (z - 0.6), -1.0, 1.0))
        idx.append(i)
        amp.append(1.0 + 0.3 * jitter * dev)
        T = sr / r_eff[i]
        pos += max(0.2 * T, T * (1.0 + jitter * dev))
    imp = np.zeros(n)
    np.add.at(imp, np.array(idx, dtype=int), np.array(amp))

    def _unit_rms(x):
        return x / (np.sqrt(np.mean(x ** 2)) + EPS)

    # ramo impulsi: 3 modi inarmonici (piastra), Q fisso 15 -> tau = Q/(pi*f)
    pulse = np.zeros(n)
    for ratio, a in zip((1.0, 2.76, 5.40), (1.0, 0.6, 0.35)):
        f = freq * ratio
        if f < 0.45 * sr:
            pulse += a * _resonant_filter(imp, f, 15.0 / (np.pi * f), sr)

    # ramo aria compressa: bianco -> IIR 1 polo (colore), inviluppo pulsato dopo ogni impulso
    air = sg.lfilter([1.0], [1.0, -air_color], rng.standard_normal(n))
    d = np.exp(-1.0 / (0.4 / rate * sr))
    E = np.minimum(sg.lfilter([1.0], [1.0, -d], (imp != 0).astype(np.float64)), 1.0)
    air = air * (0.4 + 0.6 * E)

    out = (1.0 - air_mix) * _unit_rms(pulse) + air_mix * _unit_rms(air)
    return _peak_normalize(out)


# ---------------- 9. BIRD ----------------
# Loop per-campione (RK4 su 2 oscillatori + linea di ritardo): con numba (opzionale,
# `pip install numba`) ~10-50 ms/render, in Python puro ~1-5 s -- stesso codice in entrambi i casi.
try:
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

    def _njit(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda f: f


@_njit(cache=True)
def _bird_f(x, Y, al, be, F):
    return -al - be * x - x * x * x - x * x * Y + x * x - x * Y + F


@_njit(cache=True)
def _bird_step(x, Y, al, be, F, h):
    k1x = Y
    k1y = _bird_f(x, Y, al, be, F)
    k2x = Y + 0.5 * h * k1y
    k2y = _bird_f(x + 0.5 * h * k1x, Y + 0.5 * h * k1y, al, be, F)
    k3x = Y + 0.5 * h * k2y
    k3y = _bird_f(x + 0.5 * h * k2x, Y + 0.5 * h * k2y, al, be, F)
    k4x = Y + h * k3y
    k4y = _bird_f(x + h * k3x, Y + h * k3y, al, be, F)
    return (x + h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
            Y + h / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y))


@_njit(cache=True)
def _bird_core(alpha_arr, beta, eps, D, rho_f, h, nsub, y0a, y0b, g, a_lp, mix_b):
    n = alpha_arr.shape[0]
    W = np.zeros(n + 2)
    xa = 0.0
    ya = y0a
    xb = 0.0
    yb = y0b
    lp = 0.0
    hb = h * rho_f
    for k in range(n):
        wd = 0.0
        idx = k - D
        if idx >= 1.0:
            i0 = int(idx)
            fr = idx - i0
            wd = W[i0] * (1.0 - fr) + W[i0 + 1] * fr
        lp = (1.0 - a_lp) * wd + a_lp * lp
        F = eps * g * lp
        W[k] = ya + mix_b * yb - g * lp
        al = alpha_arr[k]
        for _ in range(nsub):
            xa, ya = _bird_step(xa, ya, al, beta, F, h)
            xb, yb = _bird_step(xb, yb, al, beta, F, hb)
        if not (abs(xa) < 1e3 and abs(xb) < 1e3):
            xa = 0.0
            ya = 0.05
            xb = 0.0
            yb = 0.05
    return W[:n]


def bird(duration=1.5, sr=SR, freq=1500.0, alpha0=0.5, beta=1.0, eps=0.2, tau_d=5.0,
         duty=1.0, rho_f=1.5, gate_rate=12.0):
    """
    Canto d'uccello: 2 sorgenti di siringa (forma normale di Mindlin-Laje, sx/dx) su una
    trachea condivisa a ritardo (chiusa-aperta, alto Q) con retroazione -- 2026-09-20, vedi
    claude/nuovi_eccitatori_dettaglio_equazioni.md sez.2 e scan_bird_regimes.py (regimi).
    Forma adimensionale (tau = gamma*t, gamma = 2*pi*freq):  x' = Y;
    Y' = -alpha - beta*x - x^3 - x^2*Y + x^2 - x*Y + F,  F = eps*g*LP(w[t - tau_d/gamma]),
    w = Y_A + 0.7*Y_B - g*LP(w ritardato); oscillatore B con gamma_B = rho_f*gamma.

    freq: Hz, gamma/(2*pi) -- frequenza di riferimento (f reale ~ omega_nd*freq, omega_nd 0.3-2.3,
      mediana ~1.6, dipende da alpha0/beta).
    alpha0: 0.05-1.5, "pressione"; alpha<=0 = silenzio (soglia di Hopf a alpha=0).
    beta: 0.2-1.8, "tensione".
    eps: 0-0.6, accoppiamento della retroazione tracheale (sub-armoniche/caos da ~0.15-0.35
      in su, a seconda di alpha0 e tau_d; sotto: solo tono).
    tau_d: 1-12, ritardo adimensionale della trachea (gamma*2L/c): le regioni complesse sono
      strette in tau_d (finestre ~1 di larghezza).
    duty: 0.1-1.0, frazione attiva del gate (1.0 = sostenuto; <1 = impulsivo). Fase
      spenta: alpha = -0.2 (silenzio).
    rho_f: 1-2.5, rapporto di frequenza tra le due sorgenti (1 = battimenti, 2 = armonico,
      non intero = inarmonico).
    gate_rate: 2-25 Hz, frequenza del gate (solo se duty<1): fissa mod_rate (nel sanity con
      12 Hz fisso mod_rate valeva ~12 per il 99% dei campioni -> descrittore non pilotabile).
    """
    n = int(duration * sr)
    rng = np.random.default_rng()
    gamma = 2.0 * np.pi * freq
    if duty >= 0.999:
        g_env = np.ones(n)
    else:
        r = min(0.1, 0.5 * duty, 0.5 * (1.0 - duty))
        phi = (np.arange(n) / sr * gate_rate) % 1.0  # fase deterministica: attack_time riproducibile
        g_env = np.zeros(n)
        g_env[phi <= duty - r] = 1.0
        m = (phi > duty - r) & (phi < duty)
        g_env[m] = 0.5 * (1.0 + np.cos(np.pi * (phi[m] - (duty - r)) / r))
        m = phi >= 1.0 - r
        g_env[m] = 0.5 * (1.0 - np.cos(np.pi * (phi[m] - (1.0 - r)) / r))
    alpha_arr = alpha0 * g_env - 0.2 * (1.0 - g_env)
    nsub = max(1, int(np.ceil(gamma * max(1.0, rho_f) / (sr * 0.2))))
    h = gamma / (sr * nsub)
    D = tau_d * sr / gamma
    w = _bird_core(alpha_arr, float(beta), float(eps), float(D), float(rho_f), float(h), nsub,
                   0.05 + 0.02 * rng.random(), 0.03 + 0.02 * rng.random(), 0.98, 0.3, 0.7)
    return _peak_normalize(np.diff(w, prepend=0.0))


# ---------------- 10. VOCAL ----------------
import math


@_njit(cache=True)
def _vocal_core(Ps_arr, w, x0, T, m1, m2, k1, k2, kc, r1, r2, c1, c2, G, Z, nu, e, x1i, x2i, a_l, Li):
    """Piega vocale a 2 masse (Ishizaka-Flanagan) + flusso di Bernoulli in forma chiusa (radice
    di una quadratica, nessuna iterazione) + tratto a 4 risonatori. Update esplicito."""
    n = Ps_arr.shape[0]
    Lg = 1.4e-2
    d1 = 2.5e-3
    d2 = 5.0e-4
    rho = 1.2
    out = np.zeros(n)
    Ua = np.zeros(n)
    X1 = np.zeros(n)
    X2 = np.zeros(n)
    y = np.zeros(4)
    yp = np.zeros(4)
    x1 = x1i
    x2 = x2i
    x1p = x1i
    x2p = x2i
    ph = 0.0
    den1 = m1 + 0.5 * r1 * T
    den2 = m2 + 0.5 * r2 * T
    epsA = 1e-9
    ut1 = 0.0
    ut2 = 0.0
    sl1 = 0.0
    sl2 = 0.0
    for k in range(n):
        Ps = Ps_arr[k]
        A1 = 2.0 * Lg * max(x0 + x1, 0.0)
        A2 = 2.0 * Lg * max(x0 + x2, 0.0)
        Amin = min(A1, A2)
        D = Ps - ph
        U = 0.0
        if Amin > epsA:
            a = rho / (2.0 * Amin * Amin)
            aD = abs(D)
            U = 2.0 * aD / (Z + math.sqrt(Z * Z + 4.0 * a * aD))
            if D < 0.0:
                U = -U
        pin = ph + Z * U
        if Amin > epsA:
            if A1 >= A2:
                p1 = Ps - rho * U * U / (2.0 * A1 * A1)
                p2 = pin
            else:
                p1 = pin
                p2 = pin
        elif A1 <= epsA:
            p1 = Ps
            p2 = ph
        else:
            p1 = Ps
            p2 = Ps
        F1 = Lg * d1 * p1
        F2 = Lg * d2 * p2
        x1n = (T * T * (F1 - k1 * x1 - kc * (x1 - x2)) + 2.0 * m1 * x1 - (m1 - 0.5 * r1 * T) * x1p) / den1
        x2n = (T * T * (F2 - k2 * x2 - kc * (x2 - x1)) + 2.0 * m2 * x2 - (m2 - 0.5 * r2 * T) * x2p) / den2
        if x0 + x1n < 0.0:
            x1n = -x0 + e * (-(x0 + x1n))
        if x0 + x2n < 0.0:
            x2n = -x0 + e * (-(x0 + x2n))
        x1p = x1
        x2p = x2
        x1 = x1n
        x2 = x2n
        Ut = U + nu * abs(U) * w[k]
        dl = Ut - ut1          # derivata del flusso per l'inertanza del tratto
        sl1 = a_l * sl1 + (1.0 - a_l) * dl
        sl2 = a_l * sl2 + (1.0 - a_l) * sl1
        dU = Ut - ut2          # numeratore [1,0,-1] del passa-banda: guadagno DC nullo
        ut2 = ut1
        ut1 = Ut
        s_old = 0.0
        s_new = 0.0
        for i in range(4):
            s_old += y[i]
            yn = c1[i] * y[i] + c2[i] * yp[i] + G[i] * dU
            yp[i] = y[i]
            y[i] = yn
            s_new += yn
        ph = s_new + Li * sl2
        if ph > 5000.0:
            ph = 5000.0
        elif ph < -5000.0:
            ph = -5000.0
        out[k] = s_new - s_old
        Ua[k] = U
        X1[k] = x1
        X2[k] = x2
    return out, Ua, X1, X2


def _vocal_render(duration, sr, ps, freq, gap, fold_q, kc_scale, jaw, tongue, rng=None, tract=1.0, inert=None, f3=None):
    """Ritorna (audio grezzo, flusso U, x1, x2) -- usata anche da scan_vocal_regimes.py."""
    if rng is None:
        rng = np.random.default_rng()
    n = int(duration * sr)
    T = 1.0 / sr
    q = freq / 140.0                      # frequenza base della coppia di masse ~140 Hz
    m1, m2 = 1.25e-4 / q ** 2, 2.5e-5 / q ** 2   # f ∝ q con k fisse (compliance statica costante)
    k1, k2, kc = 80.0, 8.0, 25.0 * kc_scale
    r1, r2 = np.sqrt(k1 * m1) / fold_q, np.sqrt(k2 * m2) / fold_q
    f1 = 250.0 + 700.0 * jaw
    f2 = max(700.0 + 1900.0 * tongue, 1.1 * f1)
    f3v = 2500.0 + 500.0 * tongue if f3 is None else max(f3, 1.15 * f2)   # F3 indipendente (con F3>F2)
    Fs = np.minimum(np.array([f1, f2, f3v, max(3500.0, 1.1 * f3v)]), 0.45 * sr)
    Bw = np.array([60.0, 90.0, 150.0, 200.0])
    # passa-banda RBJ (picco costante, zero in DC): il carico DC sul flusso e' solo 0.15*Z
    w0 = 2.0 * np.pi * Fs / sr
    al = np.sin(w0) / (2.0 * (Fs / Bw))
    c1, c2 = 2.0 * np.cos(w0) / (1.0 + al), -(1.0 - al) / (1.0 + al)
    # Z = impedenza caratteristica (solo per i PICCHI dei formanti); il carico istantaneo (DC) sul
    # flusso e' 0.15*Z: con Z pieno il flusso diventava quasi indipendente dall'area glottica
    # (U ~ Ps/Z) e la piega non oscillava mai (scan_vocal_regimes.py, 2026-09-21).
    Z = 1.2 * 343.0 / 3e-4
    G = Z * np.array([1.0, 0.7, 0.4, 0.25]) * al / (1.0 + al) * tract   # tract=0: solo ppieghe (diagnostica)
    # inertanza del tratto sopraglottico L = rho*l/A (~680 kg/m^4), passa-basso doppio: ph += L*dU/dt.
    # E' cio' che fa oscillare le pieghe a f0 ~ freq (senza, oscillavano solo per risonanza di F1,
    # diag_vocal.py 2026-09-21); corner limitato per stabilita' del loop esplicito (guadagno <1).
    L_in = 1.2 * 0.17 / 3e-4
    if inert is None:   # 1.0 da 100 Hz in su; sotto, fino a 1.5 a 70 Hz (a 70 Hz con 1.0 non oscilla)
        inert = 1.0 + 0.5 * min(max((100.0 - freq) / 30.0, 0.0), 1.0)
    a_l = float(np.exp(-2.0 * np.pi * min(max(freq, 300.0), 700.0) / sr))
    Ps_arr = ps * np.minimum(1.0, np.arange(n) / (0.03 * sr))
    w = np.diff(rng.standard_normal(n), prepend=0.0) / np.sqrt(2.0)
    return _vocal_core(Ps_arr, w, gap * 1e-3, T, m1, m2, k1, k2, kc, float(r1), float(r2),
                       c1, c2, G, 0.15 * Z, 0.4, 0.5, 1e-6 * rng.random(), 1e-6 * rng.random(),
                       a_l, inert * L_in * sr)


def vocal(duration=1.5, sr=SR, freq=150.0, ps=800.0, gap=0.2, fold_q=8.0, kc_scale=1.0,
          jaw=0.5, tongue=0.5, f3=2750.0, tilt=1.0, seed=0):
    """
    Voce dal bisbiglio al grido, 2026-09-20 -- vedi claude/nuovi_eccitatori_dettaglio_equazioni.md
    sez.3. Piega vocale a 2 masse (Ishizaka-Flanagan, valori base m1=0.125 g, m2=0.025 g, k1=80,
    k2=8, kc=25 N/m, d1=2.5, d2=0.5 mm, Lg=14 mm) con flusso di Bernoulli in forma chiusa,
    contatto per riflessione con restituzione 0.5, tratto vocale a 4 formanti (banco di
    risonatori pilotati dal flusso, con retroazione sulla pressione sottoglottica) e rumore
    turbolento nu*|U|*w sagomato dalle formanti. TARATURA APPROSSIMATIVA: da verificare con
    scan_vocal_regimes.py (zone di oscillazione in (ps, gap, freq)).

    freq: Hz, f0 (masse scalate ∝ 1/q^2 con q=freq/140: frequenza ∝ q, rigidezze fisse).
    ps: Pa, pressione sottoglottica (100-4000).
    gap: mm, semi-apertura a riposo/adduzione (-0.1..1.5; grande = bisbiglio, ~0 = pieno/pressato).
    fold_q: 2-20, fattore di qualita' delle pieghe (basso = smorzato/soffiato).
    kc_scale: 0.3-3, scala dell'accoppiamento tra le masse (fase verticale).
    jaw: 0-1 -> F1 = 250 + 700*jaw.   tongue: 0-1 -> F2 = max(700 + 1900*tongue, 1.1*F1).
    f3: Hz, 1500-3500, terza formante (>= 1.15*F2). tilt: 0-1, brillantezza: doppio passa-basso a
    fc = 400*30**tilt Hz (400..12000; 1 = nessun taglio). Aggiunti 2026-09-21 dopo la valutazione su
    suoni reali (centroide mediano del vocal 3.6 kHz vs 2.0 kHz reale; F3 confinata a 2.5-3 kHz).
    """
    out = _vocal_render(duration, sr, ps, freq, gap, fold_q, kc_scale, jaw, tongue,
                        np.random.default_rng(seed), 1.0, None, f3)[0]   # deterministico
    a = float(np.exp(-2.0 * np.pi * min(400.0 * 30.0 ** tilt, 0.45 * sr) / sr))
    for _ in range(2):
        out = sg.lfilter([1.0 - a], [1.0, -a], out)
    return _peak_normalize(out)


# ======================================================================================================
# PITCH PER COSTRUZIONE (audit 2026-09-24, vedi claude/pitch_stabilita_audit.md): le funzioni qui sotto SOSTITUISCONO
# (stesso nome, definite dopo) bow/blow/strike/pluck/chaos/mechanical/bird/vocal di sopra, che restano solo come
# riferimento storico (dead code, tranne vocal originale = colore del nuovo vocal e i kernel _bird_*). f0 = `freq` esatta
# per costruzione (ritardi di loop con allpass a fase esatta, modi armonici, ecc.); il risonatore ricevera' f0 (cap
# sui decadimenti, resonator.apply_resonator(f0=...)). shaker/noise invariati (pitch sempre NaN per scelta).
# ======================================================================================================
_vocal_orig = vocal


def _pitch_med(x, sr, L=0.3, H=0.15):
    """Mediana a frame (0.3 s, passo 0.15 s, solo frame attivi) di analyzer.descriptors.pitch_mpm; NaN se nessuno valido.
    Usata dalla calibrazione di bird (stesso stimatore dell'analyzer)."""
    from analyzer.descriptors import pitch_mpm
    Ln, Hn = int(L * sr), int(H * sr)
    if len(x) < Ln:
        return pitch_mpm(x, sr)
    vals, rms = [], []
    for s in range(0, len(x) - Ln + 1, Hn):
        f = x[s:s + Ln]
        rms.append(float(np.sqrt(np.mean(f ** 2))))
        vals.append(pitch_mpm(f, sr))
    rms, vals = np.array(rms), np.array(vals)
    v = vals[rms > 0.05 * rms.max()]
    good = v[np.isfinite(v) & (v > 0)]
    return float(np.median(good)) if len(good) else float("nan")


def _split_delay(total, floor_int=2, w0=None):
    """total (campioni) = linea intera N + allpass 1o ordine con ritardo di FASE d in [0.5, 1.5) alla pulsazione w0
    (eta = sin(w0(1-d)/2)/sin(w0(1+d)/2), esatto a w0; senza w0: formula DC). Ritorna (N, eta)."""
    N = max(floor_int, int(np.floor(total - 0.5)))
    d = min(max(total - N, 0.5), 1.5)
    if w0 is None or w0 <= 1e-6:
        return N, (1.0 - d) / (1.0 + d)
    return N, float(np.sin(w0 * (1.0 - d) / 2.0) / np.sin(w0 * (1.0 + d) / 2.0))


def _lp_pd(p, w0):
    """Ritardo di FASE a w0 del passa-basso y = (1-p)x + p*y_prev (a w0 piccolo = ritardo DC p/(1-p); a f alta e' minore:
    causa del bias di +5..45 cents di chaos/bow a freq alta/brightness bassa con la compensazione DC)."""
    if w0 <= 1e-6:
        return p / (1.0 - p)
    return float(np.arctan2(p * np.sin(w0), 1.0 - p * np.cos(w0)) / w0)


def _ap_pd(c, w0):
    """Ritardo di fase a w0 dell'allpass (c + z^-1)/(1 + c z^-1)."""
    if w0 <= 1e-6:
        return (1.0 - c) / (1.0 + c)
    z = np.exp(-1j * w0)
    return float(-np.angle((c + z) / (1.0 + c * z)) / w0)


def bow(duration=1.5, sr=SR, freq=220.0, bow_force=0.5, bow_velocity=0.35,
        bow_position=0.12, brightness=0.5, damping=0.9997):
    """v2 (dopo il collaudo r1: la correzione del solo ritardo NON bastava, 83-92% NaN): guida d'onda di corda sfregata a
    DUE linee (nut/ponte, riflessioni invertite = loop non invertente, periodo = sr/freq) + tabella di attrito
    (Smith 1986; struttura STK Bowed): moto di Helmholtz stabile invece del "bang-bang" a periodo doppio.
    Ritardo di loop totale = Nn + Nb + allpass + ritardo DC del passa-basso al ponte = sr/freq.
    bow_position = punto d'arco (frazione della lunghezza: Nb = pos*Ntot); brightness = 1 - polo del passa-basso al ponte;
    damping = guadagno per giro; bow_force -> pendenza tabella (5-4*f); bow_velocity -> velocita' arco (0.02+0.2*v)."""
    n = int(duration * sr)
    bf, bv = _as_traj(bow_force, n), _as_traj(bow_velocity, n)
    br, dm = _as_traj(brightness, n), _as_traj(damping, n)
    p = 0.7 * (1.0 - np.clip(br, 0.0, 1.0))
    p_mean = float(np.mean(p))
    w0 = 2.0 * np.pi * freq / sr
    Ntot, eta = _split_delay(sr / freq - _lp_pd(p_mean, w0), floor_int=2, w0=w0)
    t = bow_position * Ntot
    Nb = int(min(max(round(t), 1), Ntot - 1))
    Nn = Ntot - Nb
    bufB, bufN = np.zeros(Nb), np.zeros(Nn)
    slope = 5.0 - 4.0 * np.clip(bf, 0.0, 1.0)
    vb = 0.02 + 0.2 * bv
    ramp = int(0.02 * sr)
    out = np.zeros(n)
    lp = 0.0
    ax = ay = 0.0
    pB = pN = 0
    for i in range(n):
        b_out, n_out = bufB[pB], bufN[pN]
        lp = (1.0 - p[i]) * b_out + p[i] * lp
        bridge = -dm[i] * lp
        ap = eta * n_out + ax - eta * ay
        ax, ay = n_out, ap
        nut = -ap
        vd = vb[i] * (min(1.0, i / ramp) if ramp > 0 else 1.0) - (bridge + nut)
        mu = (abs(vd * slope[i]) + 0.75) ** -4.0
        newv = vd * min(max(mu, 0.01), 0.98)
        bufN[pN] = bridge + newv
        bufB[pB] = nut + newv
        out[i] = bridge
        pB += 1
        if pB == Nb:
            pB = 0
        pN += 1
        if pN == Nn:
            pN = 0
    return _peak_normalize(out)


def pluck(duration=2.0, sr=SR, freq=220.0, pluck_position=0.2, pluck_hardness=0.5,
          decay_time=1.5, dispersion=0.0, seed=0):
    n = int(duration * sr)
    disp = float(np.mean(dispersion))
    w0 = 2.0 * np.pi * freq / sr
    d_disp = _ap_pd(-0.5 * disp, w0) if disp > 0 else 0.0  # ritardo di fase a f0 dell'allpass di dispersione
    total = sr / freq + 0.5 - d_disp  # ring: N - 0.5 (media) + d_tune + d_disp = sr/freq
    N, eta = _split_delay(total, floor_int=4, w0=w0)
    rng = np.random.default_rng(seed)
    spec = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, N // 2 + 1))
    spec[0] = 0.0
    if N % 2 == 0:
        spec[-1] = 1.0
    burst = np.fft.irfft(spec, n=N)
    if pluck_hardness < 1.0:
        alpha = 0.05 + 0.9 * (1 - pluck_hardness)
        burst = sg.lfilter([alpha], [1, -(1 - alpha)], burst)
    tap = max(1, min(int(round(pluck_position * N)), N - 1))
    burst[tap:] += -burst[:-tap]
    buf = burst.copy()
    decay_time = _as_traj(decay_time, n)
    decay_per_sample = np.exp(-6.91 / (np.maximum(decay_time, 0.01) * sr))
    ap_coef = -disp * 0.5
    ax = ay = 0.0     # allpass di dispersione
    tx = ty = 0.0     # allpass di accordatura
    out = np.zeros(n)
    ptr = 0
    for i in range(n):
        prev = buf[ptr]
        nxt = buf[(ptr + 1) % N]
        filtered = 0.5 * (prev + nxt) * decay_per_sample[i]
        t_out = eta * filtered + tx - eta * ty
        tx, ty = filtered, t_out
        filtered = t_out
        if disp > 0:
            a_out = ap_coef * filtered + ax - ap_coef * ay
            ax, ay = filtered, a_out
            filtered = a_out
        buf[ptr] = filtered
        out[i] = filtered
        ptr = (ptr + 1) % N
    return _peak_normalize(out)


def strike(duration=1.0, sr=SR, freq=220.0, impact_velocity=0.8, hammer_mass=0.02,
           hammer_stiffness=5e7, nonlinearity=1.5, material=0.3, size_damping=0.4, inharm_scale=0.1):
    v = impact_velocity * 2.0
    x = 0.0
    dt = 1.0 / sr
    forces = []
    for _ in range(int(0.02 * sr)):
        f = hammer_stiffness * max(x, 0.0) ** nonlinearity
        v += (-f / hammer_mass) * dt
        x += v * dt
        forces.append(max(f, 0.0))
        if x < 0:
            break
    force = np.array(forces) if forces else np.array([1.0])
    pk = np.max(np.abs(force))
    force = force / pk if pk > EPS else force
    n = int(duration * sr)
    x_in = np.zeros(n)
    x_in[:min(len(force), n)] = force[:n]
    s = inharm_scale
    ratios = [1.0, 2.0 + material * 0.6 * s, 3.0 + material * 1.3 * s, 4.0 + material * 2.1 * s, 5.0 + material * 3.0 * s]
    keep = [r for r in ratios if freq * r < 0.45 * sr] or [1.0]
    damping_time = 0.05 + (1 - size_damping) * 1.5
    amps = 1.0 / np.arange(1, len(keep) + 1)
    return _peak_normalize(_modal_bank(x_in, freq, keep, damping_time, amps=amps, sr=sr))


def blow(duration=1.5, sr=SR, freq=220.0, mouth_pressure=0.6, reed_stiffness=0.5,
         breath_noise=0.15, brightness=0.6, damping=0.999, seed=0, hp=0.4):
    """v2 (baseline: 50-75% NaN a secco: l'unica non linearita' era il clip dell'apertura, oscillava raramente): ancia
    singola su canna cilindrica, tabella d'ancia di Smith/STK Clarinet (rho = clip(0.7 + slope*dp, -1, 1), slope = -0.44 +
    0.26*rigidita'), riflessione invertita in fondo (loop invertente -> periodo = 2*ritardo = sr/freq).
    mouth_pressure 0-1 -> pressione in bocca 0.45-1.0 (sempre sopra la soglia di oscillazione); breath_noise -> rumore di
    turbolenza 0.25*b; brightness = 1 - polo del passa-basso (ritardo compensato); damping = guadagno per giro; seed = RNG."""
    n = int(duration * sr)
    mp = 0.45 + 0.55 * np.clip(_as_traj(mouth_pressure, n), 0.0, 1.0)
    slope = -0.44 + 0.26 * np.clip(_as_traj(reed_stiffness, n), 0.0, 1.0)
    bn = 0.25 * _as_traj(breath_noise, n)
    br, dm = _as_traj(brightness, n), _as_traj(damping, n)
    p = 0.7 * (1.0 - np.clip(br, 0.0, 1.0))
    p_mean = float(np.mean(p))
    w0 = 2.0 * np.pi * freq / sr
    N, eta = _split_delay(sr / (2.0 * freq) - _lp_pd(p_mean, w0), floor_int=2, w0=w0)
    buf = np.zeros(N)
    noise = np.random.default_rng(seed).standard_normal(n)
    ramp = int(0.02 * sr)
    out = np.zeros(n)
    lp = 0.0
    ax = ay = 0.0
    ptr = 0
    for i in range(n):
        d_out = buf[ptr]
        lp = (1.0 - p[i]) * d_out + p[i] * lp
        breath = mp[i] * (min(1.0, i / ramp) if ramp > 0 else 1.0)
        breath += breath * bn[i] * noise[i]
        pd = -dm[i] * lp - breath
        rho = min(max(0.7 + slope[i] * pd, -1.0), 1.0)
        new = breath + pd * rho
        ap = eta * new + ax - eta * ay
        ax, ay = new, ap
        buf[ptr] = ap
        out[i] = new
        ptr += 1
        if ptr == N:
            ptr = 0
    if hp > 0:  # r15: l'uscita ha una componente DC/lenta (pressione della bocca): MPM a volte la legge come 25 Hz (-6000 c)
        sos = sg.butter(4, min(hp * freq / (0.5 * sr), 0.99), btype="high", output="sos")
        out = sg.sosfilt(sos, out)
    return _peak_normalize(out)


def chaos(duration=1.5, sr=SR, freq=220.0, bifurcation=0.9, coupling_rate=200.0,
          x0=0.6, brightness=0.5, damping=0.998, seed=0, hp=0.7):
    """v3 (r4: bias +5..45 c da ritardo DC del passa-basso; cfg con bifurcation bassa a -1160/-2306 c: pitch del ciclo
    periodico della mappa) -- v2 (baseline: 42-75% NaN): la mappa logistica (invariata) non e' piu' iniettata come valore TENUTO (a bassi
    coupling_rate i gradini lenti dominavano l'energia e il loop non suonava come nota) ma come IMPULSI di ampiezza pari
    alla variazione della mappa a ogni aggiornamento (treno d'impulsi caotico, a coupling_rate alto resta rumore bianco
    filtrato dal loop); loop con ritardo totale sr/freq (linea intera + allpass di accordatura, passa-basso compensato)."""
    n = int(duration * sr)
    bif, dm, br = _as_traj(bifurcation, n), _as_traj(damping, n), _as_traj(brightness, n)
    b_mean = max(float(np.mean(br)), 1e-3)
    w0 = 2.0 * np.pi * freq / sr
    N, eta = _split_delay(sr / freq - _lp_pd(1.0 - b_mean, w0), floor_int=2, w0=w0)
    step = max(1, int(round(sr / coupling_rate)))
    rng = np.random.default_rng(seed)  # segno casuale degli impulsi: i cicli periodici della mappa (bifurcation bassa,
    # finestre periodiche) non generano righe spettrali proprie -> il pitch lo decide solo il loop (audit: -1160/-2306 c)
    x = float(np.clip(x0, 1e-4, 1 - 1e-4))
    buf = np.zeros(N)
    out = np.zeros(n)
    lp = 0.0
    ax = ay = 0.0
    drive_prev = 0.0
    ptr = 0
    for i in range(n):
        imp = 0.0
        if i % step == 0:
            r = 3.57 + 0.42 * min(max(bif[i], 0.0), 1.0)
            x = r * x * (1 - x)
            drive = 2 * x - 1
            imp = 0.3 * (drive - drive_prev) * (1.0 if rng.random() < 0.5 else -1.0)
            drive_prev = drive
        raw = dm[i] * buf[ptr] + imp
        lp = br[i] * raw + (1 - br[i]) * lp
        ap = eta * lp + ax - eta * ay
        ax, ay = lp, ap
        buf[ptr] = ap
        out[i] = ap
        ptr += 1
        if ptr == N:
            ptr = 0
    if hp > 0:  # r6: il loop ha guadagno ~1/(1-damping) a DC/basse freq (random walk degli impulsi): riga sotto f0 che
        # eccita i modi bassi del risonatore (-1200/-2400/-3600 c). HP 4o ordine a hp*f0 sull'uscita (non tocca il periodo).
        sos = sg.butter(4, min(hp * freq / (0.5 * sr), 0.99), btype="high", output="sos")
        out = sg.sosfilt(sos, out)
    return _peak_normalize(out)


# ---------------- autonomi: nucleo tonale a f0=freq + fisica originale come "colore" (r9) ----------------
def _unit(x):
    return x / (np.sqrt(np.mean(x * x)) + EPS)


def _harm_tone(freq, n, sr, amps, env=None):
    t = np.arange(n) / sr
    y = np.zeros(n)
    for k, a in enumerate(amps, 1):
        if k * freq < 0.45 * sr:
            y += a * np.sin(2.0 * np.pi * k * freq * t)
    return y if env is None else y * env


def mechanical(duration=1.5, sr=SR, freq=800.0, rate=30.0, jitter=0.3, air_mix=0.3,
               air_color=0.0, load_mod=0.1, seed=0, tone_mix=0.85):
    """tone_mix (0-1): quota del nucleo tonale (motore/ronzio) rispetto ai modi metallici a impulsi (ingranaggi, stridori).
    Pitch definito solo se tone_mix*(1-air_mix) >~ 0.3 (da confermare col test); sotto: non intonato (sbuffi, clic, stridori).
    Come exciters.mechanical (treno d'impulsi jittered + modi + aria) piu' un nucleo tonale ARMONICO a freq (3 armoniche,
    ampiezza modulata dall'inviluppo pulsato: 0.55+0.45*env). Uscita = (1-air_mix)*(0.65*nucleo + 0.35*modi) + air_mix*aria.
    Pitch definito solo per air_mix bassa (regione non intonata da dichiarare in dataset)."""
    n = int(duration * sr)
    rng = np.random.default_rng(seed)
    a_lp = 1.0 - np.exp(-2 * np.pi * 2.0 / sr)
    lam = sg.lfilter([a_lp], [1, -(1 - a_lp)], rng.standard_normal(n))
    lam = np.clip(lam / (np.std(lam) + EPS), -2.0, 2.0)
    r_eff = np.maximum(rate * (1.0 + load_mod * lam), 0.3 * rate)
    mu = 3.4 + 0.59 * jitter
    z = 0.3 + 0.4 * rng.random()
    pos = rng.uniform(0.05, 0.5) * sr / rate
    idx, amp = [], []
    while pos < n:
        i = int(pos)
        z = float(np.clip(mu * z * (1.0 - z), 1e-6, 1.0 - 1e-6))
        dev = float(np.clip(2.0 * (z - 0.6), -1.0, 1.0))
        idx.append(i)
        amp.append(1.0 + 0.3 * jitter * dev)
        T = sr / r_eff[i]
        pos += max(0.2 * T, T * (1.0 + jitter * dev))
    imp = np.zeros(n)
    np.add.at(imp, np.array(idx, dtype=int), np.array(amp))
    pulse = np.zeros(n)
    q_inh = min(max((0.9 - tone_mix) / 0.5, 0.0), 1.0)  # r14: modi a impulsi ARMONICI (1,2,3) se tone_mix>=0.9, inarmonici
    # metallici (1,2.76,5.40) se tone_mix<=0.4: i modi inarmonici ai risonatori rovinavano il pitch (bar 32%, membrane 32-42%)
    for ratio, a in zip((1.0, 2.0 + 0.76 * q_inh, 3.0 + 2.40 * q_inh), (1.0, 0.6, 0.35)):
        f = freq * ratio
        if f < 0.45 * sr:
            pulse += a * _resonant_filter(imp, f, 15.0 / (np.pi * f), sr)
    air = sg.lfilter([1.0], [1.0, -air_color], rng.standard_normal(n))
    d = np.exp(-1.0 / (0.4 / rate * sr))
    env = np.minimum(sg.lfilter([1.0], [1.0, -d], (imp != 0).astype(np.float64)), 1.0)
    air = air * (0.4 + 0.6 * env)
    nh = max(1, min(60, int(0.45 * sr / freq)))  # r11: nucleo RICCO (buzz a 1/k^0.7, fino a 60 armoniche): con 3 armoniche i
    # risonatori modali (bar/membrane) lo lasciavano quasi passare -> NaN 46-64%; il ronzio di motore/ingranaggi e' ricco
    core = _harm_tone(freq, n, sr, [k ** -0.7 for k in range(1, nh + 1)], 0.55 + 0.45 * env)
    tonal = tone_mix * _unit(core) + (1.0 - tone_mix) * _unit(pulse)
    out = (1.0 - air_mix) * tonal + air_mix * _unit(air)
    return _peak_normalize(out)


def bird(duration=1.5, sr=SR, freq=1500.0, alpha0=0.5, beta=1.0, eps=0.2, tau_d=5.0,
         duty=1.0, rho_f=1.5, gate_rate=12.0, seed=0, core=2.5):
    """Canto d'uccello: siringa/trachea originale (2 oscillatori Mindlin-Laje + ritardo tracheale) CALIBRATA in frequenza
    + nucleo tonale armonico a `freq`. Il sistema e' a tempo normalizzato (frequenza ∝ gamma, D in unita' di gamma): un
    primo render completo (gate incluso) a gamma0 = 2*pi*freq/2.4 misura la frequenza dominante f_m (mediana MPM a
    frame, stesso stimatore dell'analyzer) e il render finale usa gamma = gamma0*freq/f_m. Il nucleo tonale (3 armoniche
    col gate del modello, peso `core`=2.5 rispetto alla fisica) rende il pitch univoco anche con 2 sorgenti a rapporto
    non intero (`rho_f`) o regimi caotici; `core`=0 = solo fisica (copertura pitch molto piu' bassa). seed = condizioni
    iniziali (l'originale usava default_rng() non seminato). Parametri come exciters.bird originale."""
    n = int(duration * sr)
    rng = np.random.default_rng(seed)
    if duty >= 0.999:
        g_env = np.ones(n)
    else:
        r = min(0.1, 0.5 * duty, 0.5 * (1.0 - duty))
        phi = (np.arange(n) / sr * gate_rate) % 1.0
        g_env = np.zeros(n)
        g_env[phi <= duty - r] = 1.0
        m = (phi > duty - r) & (phi < duty)
        g_env[m] = 0.5 * (1.0 + np.cos(np.pi * (phi[m] - (duty - r)) / r))
        m = phi >= 1.0 - r
        g_env[m] = 0.5 * (1.0 - np.cos(np.pi * (phi[m] - (1.0 - r)) / r))
    alpha_arr = alpha0 * g_env - 0.2 * (1.0 - g_env)
    y0a, y0b = 0.05 + 0.02 * rng.random(), 0.03 + 0.02 * rng.random()

    def run(fg, arr):
        gamma = 2.0 * np.pi * fg
        nsub = max(1, int(np.ceil(gamma * max(1.0, rho_f) / (sr * 0.2))))
        h = gamma / (sr * nsub)
        D = tau_d * sr / gamma
        w = _bird_core(arr, float(beta), float(eps), float(D), float(rho_f), float(h), nsub,
                         y0a, y0b, 0.98, 0.3, 0.7)
        return np.diff(w, prepend=0.0)

    f_g = freq / 2.4  # r12: il pre-render senza gate (r11) NON rappresentava il render finale (pitch scattato di +-1200..3000 c):
    # ora si calibra sul render COMPLETO (gate incluso), misurando con lo stesso stimatore del collaudo (mediana MPM a frame)
    col = run(f_g, alpha_arr)
    f_m = _pitch_med(col[int(0.15 * sr):], sr)
    if np.isfinite(f_m) and f_m > 0:
        f_g = f_g * freq / f_m
        col = run(f_g, alpha_arr)
    out = _unit(col)
    if core > 0:
        out = out + core * _unit(_harm_tone(freq, n, sr, (1.0, 0.4, 0.15), g_env))
    return _peak_normalize(out)


def vocal(duration=1.5, sr=SR, freq=150.0, ps=800.0, gap=0.2, fold_q=8.0, kc_scale=1.0,
          jaw=0.5, tongue=0.5, f3=2750.0, tilt=1.0, seed=0, core_w=0.9):
    """Nucleo tonale = serie armonica a freq (1/k, formanti F1-F3 come nel modello, tilt) + vocal originale (pieghe a 2 masse)
    come colore. Peso del nucleo core_w*clip(1-(gap-0.4)/0.9): bisbiglio (gap > ~1.3) = solo originale -> non intonato."""
    n = int(duration * sr)
    col = np.asarray(_vocal_orig(duration=duration, sr=sr, freq=freq, ps=ps, gap=gap, fold_q=fold_q, kc_scale=kc_scale,
                             jaw=jaw, tongue=tongue, f3=f3, tilt=tilt, seed=seed), dtype=np.float64)
    f1 = 250.0 + 700.0 * jaw
    f2 = max(700.0 + 1900.0 * tongue, 1.1 * f1)
    f3v = max(f3, 1.15 * f2)
    nh = max(1, int(0.45 * sr / freq))
    k = np.arange(1, nh + 1)
    fk = k * freq
    H = 0.1
    for F, Bw, g in ((f1, 80.0, 1.0), (f2, 120.0, 0.7), (f3v, 200.0, 0.4)):
        H = H + g / (1.0 + ((fk - F) / (1.5 * Bw)) ** 2)
    a = H / k / (1.0 + (fk / (400.0 * 30.0 ** tilt)) ** 2)
    t = np.arange(n) / sr
    core = np.zeros(n)
    for ki in range(nh):
        core += a[ki] * np.sin(2.0 * np.pi * fk[ki] * t)
    core *= np.minimum(1.0, t / 0.03)
    wc = core_w * min(max(1.0 - (gap - 0.4) / 0.9, 0.0), 1.0)
    out = wc * _unit(core) + (1.0 - wc) * _unit(col)
    return _peak_normalize(out)


# ---------------- dispatcher + range parametri (per LHS/Sobol, sez.4) ----------------

EXCITERS = {
    "bow": bow,
    "blow": blow,
    "strike": strike,
    "pluck": pluck,
    "shaker": shaker,
    "noise": noise,
    "chaos": chaos,
    "mechanical": mechanical,
    "bird": bird,
    "vocal": vocal,
}

# Range freq allargati 2026-09-15 (fase 2, punto 1) sulla base dei percentili reali
# per categoria in real_descriptors_full.csv (tutto il corpus ~/Desktop/sample, non un
# campione casuale globale -- vedi analyze_real_ranges.py). p5/p95 usati come guida,
# min/max scartati come guida diretta (il pitch tracker si appiattisce su ~40Hz/~2005Hz
# anche su materiale non pitchato come noise -- bound di ricerca del tracker, non il vero
# range dei suoni reali). shaker gia' copriva la distribuzione osservata, invariato.
PARAM_RANGES = {
    "bow": dict(freq=(65, 2000), bow_force=(0.05, 1.0), bow_velocity=(0.05, 1.0),
                bow_position=(0.02, 0.45), brightness=(0.1, 0.95), damping=(0.995, 0.9999)),  # max 0.45: a 0.5 sub-armonica -1200 c
    "blow": dict(freq=(55, 1200), mouth_pressure=(0.1, 1.0), reed_stiffness=(0.0, 1.0),
                 breath_noise=(0.0, 0.6), brightness=(0.1, 0.95), damping=(0.99, 0.9999)),
    "strike": dict(freq=(65, 2400), impact_velocity=(0.1, 1.0), hammer_mass=(0.001, 0.1),
                   hammer_stiffness=(1e5, 1e9), nonlinearity=(1.0, 2.5), material=(0.0, 1.0),
                   size_damping=(0.0, 1.0)),
    "pluck": dict(freq=(55, 1800), pluck_position=(0.01, 0.5), pluck_hardness=(0.0, 1.0),
                  decay_time=(0.2, 4.0), dispersion=(0.0, 1.0)),
    "shaker": dict(freq=(200, 4000), n_particles=(5, 500), energy=(0.1, 1.0),
                   decay_time=(0.1, 2.0), material=(0.0, 1.0)),
    "noise": dict(color=(-1.0, 1.0), density=(0.1, 1.0), correlation=(0.0, 0.95),
                  freq=(50, 2000), tone_amount=(0.0, 1.0), tone_q=(0.0, 1.0)),
    "chaos": dict(freq=(60, 1200), bifurcation=(0.0, 1.0), coupling_rate=(20, 2000),
                  x0=(0.05, 0.95), brightness=(0.1, 0.95), damping=(0.99, 0.9999)),
    "mechanical": dict(freq=(80, 4500), rate=(2, 200), jitter=(0.0, 1.0), air_mix=(0.0, 1.0),
                       air_color=(-0.9, 0.9), load_mod=(0.0, 0.5), tone_mix=(0.0, 1.0)),  # freq <= FMAX pitch (4500)
    "bird": dict(freq=(400, 3500), alpha0=(0.05, 1.5), beta=(0.2, 1.8), eps=(0.0, 0.6),
                 tau_d=(1.0, 12.0), duty=(0.1, 1.0), rho_f=(1.0, 2.5), gate_rate=(2.0, 25.0)),
    "vocal": dict(freq=(70, 900), ps=(100, 4000), gap=(-0.1, 1.5), fold_q=(2.0, 20.0),
                  kc_scale=(0.3, 3.0), jaw=(0.0, 1.0), tongue=(0.0, 1.0),
                  f3=(1500.0, 3500.0), tilt=(0.0, 1.0)),
}


def generate(name, **params):
    """Dispatcher: generate(nome_eccitatore, **parametri) -> (audio float32, sr)."""
    if name not in EXCITERS:
        raise ValueError(f"eccitatore sconosciuto: {name} (validi: {list(EXCITERS)})")
    return EXCITERS[name](**params), SR


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    import soundfile as sf

    p = argparse.ArgumentParser(
        description="Demo: genera un render con parametri di default per ciascuno dei 7 eccitatori.")
    p.add_argument("--out-dir", default="renders_demo", help="cartella di output")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, fn in EXCITERS.items():
        audio = fn()
        path = out_dir / f"{name}.wav"
        sf.write(path, audio, SR)
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        print(f"[{name}] {len(audio) / SR:.2f}s  peak={peak:.3f}  rms={rms:.3f}  -> {path}")

    print(f"\nFatto: {len(EXCITERS)} render in {out_dir}/", file=sys.stderr)
