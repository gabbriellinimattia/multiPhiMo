"""
resonator.py — Risonatore condiviso (agente 8), sez.3 pipeline_multiPhiMo.txt +
claude/risonatore_modello_scelto.md.

Oggi: generatore ANALITICO di parametri modali (formule standard di barra/piastra/membrana,
Fletcher & Rossing) per 4 topologie, da dimensione+forma+materiale. E' la stessa funzione che
servira' a costruire il dataset sintetico di sez.4 (forma/materiale -> parametri modali): il
regressore leggero (MLP) descritto in claude/risonatore_modello_scelto.md sostituira' in futuro
resonator_params() SENZA cambiare l'interfaccia a valle (apply_resonator(excitation, ...)).

I rapporti tra i modi (MODE_RATIOS) sono valori di letteratura, approssimati: l'accuratezza
esatta non e' critica (il progetto fa matching solo sui descrittori, non timbrico). La costante
in _fundamental_freq e' illustrativa (pensata per restare in banda audio con i default), non
calibrata su un caso reale.

Uso tipico: audio = apply_resonator(eccitazione_da_exciters_py, shape="plate_circ", size=0.2, ...)
"""
import numpy as np
from scipy import signal as sg

# numba OPZIONALE (stesso pattern di exciters.py): serve al risonatore `chaotic` (loop per-campione);
# senza numba il fallback Python puro e' identico ma ~100x piu' lento (inutilizzabile per i dataset).
try:
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

    def _njit(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda f: f

SR = 44100
EPS = 1e-9

# rapporti f_n/f_1 per modo, da letteratura (Fletcher & Rossing, "The Physics of Musical
# Instruments" / "Science of Percussion Instruments"). plate_rect e' calcolato, non tabulato
# (dipende dall'aspect ratio, vedi _plate_rect_ratios).
MODE_RATIOS = {
    "bar":      [1.0, 2.756, 5.404, 8.933, 13.34, 18.64],   # barra libera-libera (xilofono/marimba)
    "plate_circ": [1.0, 2.08, 3.41, 5.00, 6.82, 8.95],      # piastra circolare incastrata (approx.)
    "membrane": [1.0, 1.594, 2.136, 2.296, 2.653, 2.918],   # membrana circolare tesa (timpano)
}

# default fisicamente plausibili per ciascuna topologia (size in metri, thickness in metri,
# density in kg/m3, stiffness = modulo elastico E in Pa per bar/piastre, tensione superficiale
# T in N/m per membrane)
_DEFAULTS = {
    "bar":        dict(size=0.30, aspect=1.0, thickness=0.006, density=2700.0, stiffness=7.0e10, loss=0.010),
    "plate_rect": dict(size=0.30, aspect=1.4, thickness=0.002, density=2700.0, stiffness=7.0e10, loss=0.010),
    "plate_circ": dict(size=0.15, aspect=1.0, thickness=0.002, density=2700.0, stiffness=7.0e10, loss=0.010),
    "membrane":   dict(size=0.20, aspect=1.0, thickness=0.0001, density=1400.0, stiffness=3000.0, loss=0.020),
}

PARAM_RANGES = {
    "bar":        dict(size=(0.05, 1.5), thickness=(0.005, 0.05), density=(400, 8000),
                        stiffness=(5e9, 2.1e11), loss=(0.0005, 0.05), mode_falloff=(0.0, 1.0)),
    "plate_rect": dict(size=(0.05, 1.0), aspect=(0.3, 3.0), thickness=(0.0005, 0.02),
                        density=(400, 8000), stiffness=(5e9, 2.1e11), loss=(0.0005, 0.05),
                        mode_falloff=(0.0, 1.0)),
    "plate_circ": dict(size=(0.03, 0.6), thickness=(0.0005, 0.02), density=(400, 8000),
                        stiffness=(5e9, 2.1e11), loss=(0.0005, 0.05), mode_falloff=(0.0, 1.0)),
    "membrane":   dict(size=(0.05, 0.6), thickness=(0.00005, 0.002), density=(200, 2000),
                        stiffness=(500, 8000), loss=(0.0005, 0.05), mode_falloff=(0.0, 1.0)),
}


# ---------------- nuove forme (2026-09-21): tube, soundboard, chaotic ----------------
# Spec: claude/nuovi_risonatori_proposta.md. Tutte le formule sono approssimazioni: il progetto
# fa matching solo sui descrittori, non timbrico.
_NEW_SHAPES = ("tube", "soundboard", "chaotic")
C_AIR = 343.0           # m/s
_T60_MIN_NEW = 0.02     # s, floor T60 per le nuove forme (i modi alti di tube/soundboard decadono in fretta)

_DEFAULTS.update({
    "tube":       dict(size=0.6, radius=0.012, closure=0.0, flare=0.0, loss=0.005, mode_falloff=0.5),
    "soundboard": dict(size=0.4, aspect=0.75, thickness=0.0045, stiffness=1.2e10, ortho=15.0,
                        cavity=0.012, hole=0.04, loss=0.005, mode_falloff=0.5),
    "chaotic":    dict(size=0.1, thickness=0.006, stiffness=7.0e10, loss=0.005, mode_falloff=0.5,
                        nonlin=0.2, beat=3.0, depth=0.6, chaos=0.4, speed=5.0),
})

PARAM_RANGES.update({
    # colonna d'aria: size = lunghezza L (m), radius = raggio interno (m), closure 0=aperto-aperto 1=chiuso-aperto,
    # flare 0-1 = svasatura (frequenza di taglio f_c = 3*flare*f_o)
    "tube":       dict(size=(0.05, 3.0), radius=(0.004, 0.06), closure=(0.0, 1.0), flare=(0.0, 1.0),
                        loss=(0.0005, 0.05), mode_falloff=(0.0, 1.0)),
    # piastra ortotropa + cavita' di Helmholtz: size = lato lungo (m), aspect = b/a, stiffness = E_L (Pa),
    # ortho = E_L/E_R, cavity = volume (m3), hole = raggio della buca (m); densita' fissa 450 kg/m3
    "soundboard": dict(size=(0.08, 0.6), aspect=(0.3, 1.0), thickness=(0.001, 0.02),
                        stiffness=(5e9, 8e10), ortho=(1.0, 30.0), cavity=(0.0005, 0.01),
                        hole=(0.01, 0.06), loss=(0.0005, 0.05), mode_falloff=(0.0, 1.0)),
    # piastra circolare (densita' fissa 7800) con modi a DOPPIETTO (coppie quasi degeneri): nonlin = glide
    # relativo al picco (tension modulation), beat = battimento del doppietto (Hz, log), depth = ampiezza del
    # partner (0 = niente doppietti), chaos = modulazione caotica (Lorenz) del battimento + scambio di energia
    # nel doppietto, speed = scala dei tempi di Lorenz. Serve a coprire mod_rate/mod_depth/roughness.
    "chaotic":    dict(size=(0.04, 0.25), thickness=(0.002, 0.03), stiffness=(5e9, 2.1e11),
                        loss=(0.0005, 0.05), mode_falloff=(0.0, 1.0), nonlin=(0.0, 0.5),
                        beat=(0.5, 25.0), depth=(0.0, 1.0), chaos=(0.0, 1.0), speed=(0.5, 50.0)),
})

_CHAOTIC_RATIOS = np.array([1.0, 2.08, 3.41, 5.00, 6.82, 8.95, 11.3, 14.0])


def _t60_to_tau(gamma):
    """gamma: tasso di decadimento in ampiezza (1/s) -> tau (1/e) con T60 in [_T60_MIN_NEW, _T60_MAX]."""
    t60 = np.clip(_LN1000 / np.maximum(gamma, 1e-9), _T60_MIN_NEW, _T60_MAX)
    return t60 / _LN1000


def _tube_params(p, n_modes):
    a = max(float(p["radius"]), 1e-3)
    u = float(np.clip(p["closure"], 0.0, 1.0))
    fl = float(np.clip(p["flare"], 0.0, 1.0))
    L_eff = max(float(p["size"]), 1e-2) + 2 * 0.6133 * a           # end correction non flangiata (Levine-Schwinger)
    f_o = C_AIR / (2 * L_eff)
    f_c = 3.0 * fl * f_o                                            # taglio della svasatura (Webster esponenziale)
    k = np.arange(1, 400)
    f = np.sqrt(((k - u / 2) * f_o) ** 2 + f_c ** 2)
    sel = f < 12000.0
    if sel.sum() < 3:
        sel = np.arange(len(f)) < 3
    f = f[sel][:(48 if n_modes is None else max(int(n_modes), 1))]
    n = np.arange(1, len(f) + 1)
    g_visc = C_AIR * 3e-5 * np.sqrt(f) / a                          # perdite viscotermiche ~ sqrt(f)/a
    ka = 2 * np.pi * f / C_AIR * a
    g_rad = (2 - u) * np.minimum(ka ** 2, 1.0) * C_AIR / (4 * L_eff)  # perdita di radiazione alle estremita' aperte
    gamma = (max(float(p["loss"]), 1e-5) / 0.005) * (g_visc + g_rad)
    pw = 1.5 * float(np.clip(p["mode_falloff"], 0.0, 1.0)) - 0.25
    return f, _t60_to_tau(gamma), n.astype(np.float64) ** (-pw)


def _soundboard_params(p, n_modes):
    rho = 450.0
    a = max(float(p["size"]), 1e-2)
    asp = max(float(p["aspect"]), 0.1)
    h = max(float(p["thickness"]), 1e-4)
    E = max(float(p["stiffness"]), 1e6)
    ortho = max(float(p["ortho"]), 1.0)
    loss = max(float(p["loss"]), 1e-5)
    mf = float(np.clip(p["mode_falloff"], 0.0, 1.0))
    scale = 0.478 * h / a ** 2 * np.sqrt(E / rho)
    mm, nn = np.meshgrid(np.arange(1, 13), np.arange(1, 13), indexing="ij")
    M = (mm ** 2 + (nn / asp) ** 2 / np.sqrt(ortho)).ravel()
    order = np.argsort(M, kind="stable")
    mm, nn, M = mm.ravel()[order], nn.ravel()[order], M[order]
    f = scale * M
    g_shape = np.sin(mm * np.pi * 0.31) ** 2 * np.sin(nn * np.pi * 0.43) ** 2   # punto pilota/pickup fisso
    sel = f < 14000.0
    if sel.sum() < 3:
        sel = np.arange(len(f)) < 3
    nmax = 40 if n_modes is None else max(int(n_modes), 3)
    f, g_shape = f[sel][:nmax], g_shape[sel][:nmax]
    gains = g_shape * (f / f[0]) ** (-(1.5 * mf - 0.25))
    # cavita' di Helmholtz accoppiata al modo (1,1) (2 DOF piastra-aria)
    V = max(float(p["cavity"]), 1e-4)
    rh = max(float(p["hole"]), 1e-3)
    f_H = C_AIR / (2 * np.pi) * np.sqrt(np.pi * rh ** 2 / (V * (0.003 + 1.7 * rh)))
    wb2, wH2 = (2 * np.pi * f[0]) ** 2, (2 * np.pi * f_H) ** 2
    kk = 0.4 * 1.4e5 * a * (a * asp) / (V * rho * h)
    S2 = wb2 + kk + wH2
    disc = np.sqrt(max(S2 ** 2 - 4 * wb2 * wH2, 0.0))
    xs = np.array([(S2 - disc) / 2, (S2 + disc) / 2])
    f_c2 = np.sqrt(xs) / (2 * np.pi)
    k12 = kk * wH2
    pp = k12 / (k12 + (wb2 + kk - xs) ** 2 + 1e-30)                # partecipazione della piastra
    g_c2 = gains[0] * (pp + 0.6 * (1 - pp))
    freqs = np.concatenate([f_c2, f[1:]])
    gains = np.concatenate([g_c2, gains[1:]])
    # smorzamento ~ sqrt(f) (come i risonatori legacy; con Q costante i modi alti sparivano in fretta e il
    # centroide restava < 600 Hz, sanity 2026-09-21); il modo d'aria e' piu' smorzato
    gam = np.pi * 300.0 * loss * np.sqrt(np.maximum(freqs, 1.0) / 300.0)
    gam[:2] *= (1.0 + 2.0 * (1.0 - pp))
    return freqs, _t60_to_tau(gam), gains


def _chaotic_params(p, n_modes):
    K = 6 if n_modes is None else int(np.clip(n_modes, 1, 6))      # K doppietti (2K rotatori)
    f1 = min(_fundamental_freq("plate_circ", p["size"], p["thickness"], 7800.0, p["stiffness"]), 13000.0)
    f = f1 * _CHAOTIC_RATIOS[:K]
    keep = f * 1.6 < 0.5 * SR                                        # margine per glide + battimento
    f = f[keep] if keep.any() else f[:1]
    gains = 1.0 / (np.arange(1, len(f) + 1) ** (0.5 + 1.5 * float(np.clip(p["mode_falloff"], 0.0, 1.0))))
    return f, _mode_dampings(f, max(float(p["loss"]), 1e-4)), gains


_NEW_PARAMS = {"tube": _tube_params, "soundboard": _soundboard_params, "chaotic": _chaotic_params}


@_njit(cache=True)
def _lorenz_rk4(x, y, z, dt):
    k1x = 10.0 * (y - x)
    k1y = x * (28.0 - z) - y
    k1z = x * y - 2.6666666666666665 * z
    x2 = x + 0.5 * dt * k1x
    y2 = y + 0.5 * dt * k1y
    z2 = z + 0.5 * dt * k1z
    k2x = 10.0 * (y2 - x2)
    k2y = x2 * (28.0 - z2) - y2
    k2z = x2 * y2 - 2.6666666666666665 * z2
    x3 = x + 0.5 * dt * k2x
    y3 = y + 0.5 * dt * k2y
    z3 = z + 0.5 * dt * k2z
    k3x = 10.0 * (y3 - x3)
    k3y = x3 * (28.0 - z3) - y3
    k3z = x3 * y3 - 2.6666666666666665 * z3
    x4 = x + dt * k3x
    y4 = y + dt * k3y
    z4 = z + dt * k3z
    k4x = 10.0 * (y4 - x4)
    k4y = x4 * (28.0 - z4) - y4
    k4z = x4 * y4 - 2.6666666666666665 * z4
    return (x + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
            y + dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
            z + dt / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z))


@_njit(cache=True)
def _chaotic_core(x, wa, dw, r, ga, gb, nonlin, chaos, dt):
    """M doppietti di rotatori (fasori z = zr + j zi): A a w_m, B a w_m + dw_m*(1 + chaos*C_m(t)) (battimento);
    z <- r e^{j theta} z + g x, theta scalato da (1 + nonlin*En), En = E/E_picco_corrente (tension modulation
    invariante al livello). C_m(t) = proiezione decorrelata di Lorenz (stato iniziale fisso, riscaldato 20 unita'
    di tempo -> deterministico). Scambio di energia dentro il doppietto: rotazione di Givens (norma conservata)
    di angolo chaos*C2_m*0.001 per campione. Limitato: rotazioni ortogonali e r<1; nessun solve iterativo."""
    n = x.shape[0]
    M = wa.shape[0]
    zra = np.zeros(M)
    zia = np.zeros(M)
    zrb = np.zeros(M)
    zib = np.zeros(M)
    cp1 = np.zeros(M)
    sp1 = np.zeros(M)
    cp2 = np.zeros(M)
    sp2 = np.zeros(M)
    for m in range(M):
        cp1[m] = np.cos(0.7 * m)
        sp1[m] = np.sin(0.7 * m)
        cp2[m] = np.cos(0.7 * m + 1.3)
        sp2[m] = np.sin(0.7 * m + 1.3)
    y = np.zeros(n)
    lx, ly, lz = 1.0, 1.0, 1.0
    for _ in range(2000):
        lx, ly, lz = _lorenz_rk4(lx, ly, lz, 0.01)
    epk = 1e-30
    for i in range(n):
        lx, ly, lz = _lorenz_rk4(lx, ly, lz, dt)
        cx = lx / 20.0
        cy = ly / 28.0
        cz = (lz - 25.0) / 15.0
        E = 0.0
        for m in range(M):
            E += zra[m] * zra[m] + zia[m] * zia[m] + zrb[m] * zrb[m] + zib[m] * zib[m]
        if E > epk:
            epk = E
        gl = 1.0 + nonlin * (E / epk)
        xi = x[i]
        acc = 0.0
        for m in range(M):
            c = min(max(cp1[m] * cx + sp1[m] * cy, -1.0), 1.0)
            c2 = min(max(cp2[m] * cy + sp2[m] * cz, -1.5), 1.5)
            tha = wa[m] * gl
            thb = (wa[m] + dw[m] * (1.0 + chaos * c)) * gl
            csa = r[m] * np.cos(tha)
            sna = r[m] * np.sin(tha)
            csb = r[m] * np.cos(thb)
            snb = r[m] * np.sin(thb)
            a1 = csa * zra[m] - sna * zia[m] + ga[m] * xi
            b1 = sna * zra[m] + csa * zia[m]
            a2 = csb * zrb[m] - snb * zib[m] + gb[m] * xi
            b2 = snb * zrb[m] + csb * zib[m]
            ph = chaos * c2 * 0.001
            cg = 1.0 - 0.5 * ph * ph
            zra[m] = cg * a1 + ph * a2
            zrb[m] = -ph * a1 + cg * a2
            zia[m] = cg * b1 + ph * b2
            zib[m] = -ph * b1 + cg * b2
            acc += zra[m] + zrb[m]
        y[i] = acc
    return y


def _plate_rect_ratios(aspect, n_modes):
    """Rapporti dei modi di una piastra rettangolare semplicemente appoggiata:
    f_mn ~ m^2 + (n/aspect)^2, aspect = lato_b/lato_a. Qui la 'forma' entra davvero nel calcolo,
    non solo dimensione/materiale."""
    aspect = max(aspect, 0.1)
    pairs = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (3, 2), (2, 3)]
    vals = sorted((m ** 2 + (n / aspect) ** 2) for m, n in pairs)[:n_modes]
    vals = np.array(vals)
    return vals / vals[0]


def _fundamental_freq(shape, size, thickness, density, stiffness):
    """Legge di scala standard (non un calcolo FEM): barra/piastra sottile f1 ~ (h/L^2)*sqrt(E/rho)
    (Euler-Bernoulli/Kirchhoff-Love); membrana tesa f1 ~ (1/R)*sqrt(T/densita_areale) (onda su
    membrana, primo zero di Bessel). Le costanti sono illustrative."""
    size = max(size, 1e-3)
    thickness = max(thickness, 1e-5)
    if shape == "membrane":
        rho_areal = max(density * thickness, EPS)
        c = np.sqrt(max(stiffness, EPS) / rho_areal)
        f1 = 0.383 * c / size
    else:
        f1 = 0.161 * (thickness / size ** 2) * np.sqrt(max(stiffness, EPS) / max(density, EPS))
    return max(f1, 20.0)


def fundamental_freq(shape, size=None, thickness=None, density=None, stiffness=None, **params):
    """Wrapper pubblico su _fundamental_freq (2026-09-16d, fix stima f0): il modo
    fondamentale di un risonatore fisico e' gia' CALCOLATO ANALITICAMENTE qui (formule
    Fletcher & Rossing) -- per i dataset resonator_* dataset_gen.py usa QUESTO come
    etichetta "pitch" invece di stimarlo acusticamente sulla risposta all'impulso
    (bar/piastre/membrana sono fisicamente INARMONICHE: l'autocorrelazione/YIN su un
    materiale del genere resta intrinsecamente ambiguo -- vedi Terhardt 1974 sul pitch
    virtuale -- mentre qui il vero fondamentale e' noto per costruzione, gratis ed
    esatto). Mai usato per gli eccitatori (per loro il pitch resta acustico, via
    analyzer)."""
    if shape in _NEW_SHAPES:  # 2026-09-21: nuove forme, pitch = modo piu' basso del banco lineare
        freqs, _, _ = resonator_params(shape, size=size, thickness=thickness, density=density,
                                        stiffness=stiffness, **params)
        return float(np.min(freqs))
    return _fundamental_freq(shape, size, thickness, density, stiffness)


# Ricalibrato 2026-09-14: prima 'loss' non aveva alcun effetto (2.0/loss sempre >> tetto
# clip 8.0 su tutto PARAM_RANGES 0.0005-0.05 -> base sempre 8.0), e quell'8.0 comunque non
# era 8s veri: _resonant_mode usa r=exp(-1/(tau*sr)) -> inviluppo exp(-t/tau), quindi 'tau'
# e' un tempo caratteristico (1/e), non un -60dB. T60 reale = tau * ln(1000) ~= tau * 6.908.
# Target ora: loss=0.05 (max, piu' smorzato) -> T60=0.1s; loss=0.0005 (min) -> T60=10s
# (stesso rapporto 100x del range di loss), poi conversione T60->tau prima di _resonant_mode.
_T60_LOSS_K = 0.005
_T60_MIN, _T60_MAX = 0.05, 10.0  # s, tetto di sicurezza per loss fuori da PARAM_RANGES
_LN1000 = np.log(1000.0)  # ~6.908: T60 (-60dB) -> tau (1/e)


def _mode_dampings(freqs, loss):
    """Tempo di decadimento -60dB per modo (T60, poi convertito in tau per _resonant_mode):
    'loss' basso (metallo/vetro) -> decadimento lungo; i modi piu' acuti decadono piu' in
    fretta (comportamento tipico dei materiali reali)."""
    t60 = np.clip(_T60_LOSS_K / max(loss, 1e-4), _T60_MIN, _T60_MAX)
    tau = t60 / _LN1000
    return tau / np.sqrt(np.maximum(freqs / freqs[0], 1.0))


def resonator_params(shape="bar", size=None, aspect=None, thickness=None, density=None,
                      stiffness=None, loss=None, mode_falloff=0.5, n_modes=None,
                      radius=None, closure=None, flare=None, ortho=None, cavity=None, hole=None,
                      nonlin=None, beat=None, depth=None, chaos=None, speed=None):
    """
    Ritorna (freqs, damping_times, gains) per la topologia scelta, dati dimensione/forma/
    materiale. Parametri non passati (None) usano i default plausibili di _DEFAULTS[shape].

    shape: 'bar' | 'plate_rect' | 'plate_circ' | 'membrane' | 'tube' | 'soundboard' | 'chaotic'.
    size: lunghezza (bar, tube, m) o lato/raggio caratteristico (piastre/membrana, m) -> dimensione.
    aspect: rapporto larghezza/lunghezza, usato da plate_rect e soundboard -> forma.
    thickness, density, stiffness, loss: materiale (stiffness = E per bar/piastre, tensione
      superficiale T per membrane).
    mode_falloff: 0-1, quanto calano le ampiezze dei modi superiori.
    n_modes: quanti modi generare (default per forma: legacy 6 (max 6), tube <=48, soundboard 40, chaotic 8).
    Parametri solo delle nuove forme: radius/closure/flare (tube), ortho/cavity/hole (soundboard),
      nonlin/beat/depth/chaos/speed (chaotic, non lineari: usati solo da apply_resonator).
    """
    if shape not in _DEFAULTS:
        raise ValueError(f"forma sconosciuta: {shape} (valide: {list(_DEFAULTS)})")
    if shape in _NEW_SHAPES:
        given = dict(size=size, aspect=aspect, thickness=thickness, density=density, stiffness=stiffness,
                     loss=loss, mode_falloff=mode_falloff, radius=radius, closure=closure, flare=flare,
                     ortho=ortho, cavity=cavity, hole=hole, nonlin=nonlin, beat=beat, depth=depth,
                     chaos=chaos, speed=speed)
        pm = {**_DEFAULTS[shape], **{k: v for k, v in given.items() if v is not None}}
        return _NEW_PARAMS[shape](pm, n_modes)
    d = _DEFAULTS[shape]
    size = d["size"] if size is None else size
    aspect = d["aspect"] if aspect is None else aspect
    thickness = d["thickness"] if thickness is None else thickness
    density = d["density"] if density is None else density
    stiffness = d["stiffness"] if stiffness is None else stiffness
    loss = d["loss"] if loss is None else loss
    n_modes = int(np.clip(6 if n_modes is None else n_modes, 1, 6))

    ratios = _plate_rect_ratios(aspect, n_modes) if shape == "plate_rect" \
        else np.array(MODE_RATIOS[shape][:n_modes])
    f1 = _fundamental_freq(shape, size, thickness, density, stiffness)
    freqs = f1 * ratios
    damping_times = _mode_dampings(freqs, loss)
    gains = 1.0 / (np.arange(1, n_modes + 1) ** (0.5 + 1.5 * np.clip(mode_falloff, 0.0, 1.0)))
    return freqs, damping_times, gains


def _resonant_mode(x, freq, damping_time, sr=SR):
    """Filtro a 2 poli (un modo): eccitato da x, decade in damping_time secondi."""
    r = np.exp(-1.0 / (max(damping_time, 1e-4) * sr))
    theta = 2 * np.pi * min(max(freq, 1.0), sr / 2 - 1) / sr
    a = [1.0, -2 * r * np.cos(theta), r * r]
    return sg.lfilter([1.0], a, x)


# Cap sui tempi di decadimento dei modi (pitch per costruzione, audit 2026-09-24): con f0 dell'eccitatore noto,
# tau <= PITCH_CAP_K/f0 (al piu' K periodi di f0) e tau <= PITCH_CAP_Q/(pi*f_m) (Q massimo per modo): il risonatore
# colora (formanti/armoniche alle SUE frequenze assolute) senza poter "suonare" come nota propria e spostare il pitch.
PITCH_CAP_K = 2.0
PITCH_CAP_Q = 4.0


def apply_resonator(excitation, shape="bar", size=None, aspect=None, thickness=None,
                     density=None, stiffness=None, loss=None, mode_falloff=0.5,
                     n_modes=None, sr=SR, radius=None, closure=None, flare=None, ortho=None,
                     cavity=None, hole=None, nonlin=None, beat=None, depth=None, chaos=None, speed=None,
                     f0=None, cap_k=PITCH_CAP_K, cap_q=PITCH_CAP_Q):
    """
    Fa passare 'excitation' (es. l'output di un eccitatore in exciters.py) attraverso il banco
    modale definito da forma/dimensione/materiale. Ritorna audio float32 normalizzato (picco 0.9).
    Costo O(n_modes) sull'intero buffer: dentro il budget 25ms per la reimplementazione runtime.
    `chaotic` e' non lineare (loop per-campione, numba consigliato); invariante al livello d'ingresso.
    f0: Hz, `freq` dell'eccitatore (= pitch target). Se dato (> 0) applica il cap sui decadimenti dei modi
    (cap_k periodi di f0, cap_q come Q massimo per modo); None = comportamento precedente (nessun cap).
    """
    freqs, damping_times, gains = resonator_params(
        shape, size, aspect, thickness, density, stiffness, loss, mode_falloff, n_modes,
        radius, closure, flare, ortho, cavity, hole, nonlin, beat, depth, chaos, speed)
    if f0 is not None and f0 > 0:
        damping_times = np.minimum(np.asarray(damping_times, dtype=np.float64), float(cap_k) / float(f0))
        damping_times = np.minimum(damping_times, float(cap_q) / (np.pi * np.maximum(np.asarray(freqs, dtype=np.float64), 1.0)))
    if shape == "chaotic":
        pm = {**_DEFAULTS["chaotic"], **{k: v for k, v in dict(nonlin=nonlin, beat=beat, depth=depth,
              chaos=chaos, speed=speed).items() if v is not None}}
        M = len(freqs)
        w = 2 * np.pi * np.asarray(freqs, dtype=np.float64) / sr
        dw = 2 * np.pi * float(pm["beat"]) * (1.0 + 0.31 * np.arange(M)) / sr
        r = np.exp(-1.0 / (np.maximum(damping_times, 1e-4) * sr))
        g = np.asarray(gains, dtype=np.float64)
        y = _chaotic_core(np.ascontiguousarray(excitation, dtype=np.float64), w, dw, r, g,
                          g * float(pm["depth"]), float(pm["nonlin"]), float(pm["chaos"]),
                          float(pm["speed"]) / sr)
    else:
        y = np.zeros_like(excitation, dtype=np.float64)
        for f, dt, g in zip(freqs, damping_times, gains):
            y += g * _resonant_mode(excitation, f, dt, sr)
    peak = np.max(np.abs(y))
    return (y / peak * 0.9).astype(np.float32) if peak > EPS else y.astype(np.float32)


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    import soundfile as sf

    p = argparse.ArgumentParser(
        description="Demo: risposta all'impulso del risonatore per ciascuna delle 4 forme, parametri di default.")
    p.add_argument("--out-dir", default="renders_demo", help="cartella di output")
    p.add_argument("--duration", type=float, default=2.0, help="durata render (s)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(args.duration * SR)
    impulse = np.zeros(n)
    impulse[0] = 1.0

    for shape in _DEFAULTS:
        audio = apply_resonator(impulse, shape=shape)
        freqs, damping_times, gains = resonator_params(shape=shape)
        path = out_dir / f"resonator_{shape}.wav"
        sf.write(path, audio, SR)
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        print(f"[{shape}] f={np.round(freqs, 1).tolist()}  "
              f"decay={np.round(damping_times, 2).tolist()}  peak={peak:.3f}  rms={rms:.3f}  -> {path}")

    print(f"\nFatto: {len(_DEFAULTS)} render in {out_dir}/", file=sys.stderr)
