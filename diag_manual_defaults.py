"""diag_manual_defaults.py (2026-09-28) -- quale accoppiamento di default rende udibili i parametri in Mode=Manual.
Per ogni preset d'accoppiamento: A) ogni eccitatore, ogni parametro (freq esclusa) su 5 valori, via bar;
B) ogni forma, ogni parametro su 5 valori, eccitata da pluck 220 Hz. Metrica come diag_param_effect.py:
max distanza (dB RMS su bande di 1/3 d'ottava) tra le 5 versioni. Riepilogo: mediana per eccitatore/forma e globale.
Uso: python3 diag_manual_defaults.py 2> diag_manual_defaults.log"""
import sys
import numpy as np
import exciters as E
import resonator as R

SR = E.SR
EDGES = 50.0 * 2 ** (np.arange(0, 29) / 3.0)  # 50 Hz .. ~16 kHz (come diag_param_effect.py)


def grid(lo, hi, name):
    if name == "n_particles":
        return np.unique(np.round(np.geomspace(lo, hi, 5))).tolist()
    if lo > 0 and hi / lo >= 20:
        return np.geomspace(lo, hi, 5).tolist()
    return np.linspace(lo, hi, 5).tolist()


def bands_db(x):
    X = np.abs(np.fft.rfft(np.asarray(x, np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    e = np.array([X[(f >= a) & (f < b)].sum() for a, b in zip(EDGES[:-1], EDGES[1:])]) + 1e-20
    return np.maximum(10 * np.log10(e / e.sum()), -80.0)

PRESETS = {
    "attuale":  dict(harmonicity=0.5, pitch_focus=0.5, body=0.7, exc_attack=0.01, exc_hold=0.5, am_rate=5.0, am_depth=0.0),
    "h1_diretto": dict(harmonicity=1.0, pitch_focus=0.0, body=1.0, exc_attack=0.001, exc_hold=3.0, am_rate=5.0, am_depth=0.0),
    "h08":      dict(harmonicity=0.8, pitch_focus=0.0, body=1.0, exc_attack=0.001, exc_hold=3.0, am_rate=5.0, am_depth=0.0),
    "h0":       dict(harmonicity=0.0, pitch_focus=0.0, body=1.0, exc_attack=0.001, exc_hold=3.0, am_rate=5.0, am_depth=0.0),
}
DUR = 1.0


def dist(sigs):
    B = np.array([bands_db(s) for s in sigs])
    return max(np.sqrt(np.mean((B[i] - B[j]) ** 2)) for i in range(len(B)) for j in range(i + 1, len(B)))


def mid(lo, hi, log):
    return float(np.sqrt(lo * hi)) if log and lo > 0 else 0.5 * (lo + hi)


for pn, cp in PRESETS.items():
    per_exc, per_res = {}, {}
    for exc, pr in E.PARAM_RANGES.items():
        base = {k: mid(lo, hi, lo > 0 and hi / lo >= 20) for k, (lo, hi) in pr.items()}
        base["freq"] = float(np.clip(220.0, *pr["freq"]))
        if "n_particles" in base:
            base["n_particles"] = int(round(base["n_particles"]))
        ds = []
        for p, (lo, hi) in pr.items():
            if p == "freq":
                continue
            sigs = []
            for v in grid(lo, hi, p):
                xp = dict(base)
                xp[p] = int(round(v)) if p == "n_particles" else v
                raw, _ = E.generate(exc, duration=DUR, **xp)
                sigs.append(R.apply_resonator(raw, shape="bar", f0=xp["freq"], **cp))
            ds.append(dist(sigs))
        per_exc[exc] = (float(np.median(ds)), float(np.min(ds)))
    raw, _ = E.generate("pluck", duration=DUR, freq=220.0)
    for shape, pr in R.PARAM_RANGES.items():
        ds = []
        for p, (lo, hi) in pr.items():
            sigs = [R.apply_resonator(raw, shape=shape, f0=220.0, **{p: v}, **cp) for v in grid(lo, hi, p)]
            ds.append(dist(sigs))
        per_res[shape] = (float(np.median(ds)), float(np.min(ds)))
    ge = np.median([v[0] for v in per_exc.values()])
    gr = np.median([v[0] for v in per_res.values()])
    print(f"PRESET {pn}: eccitatori mediana {ge:.1f} dB | risonatori mediana {gr:.1f} dB", file=sys.stderr, flush=True)
    print("  ecc  " + " ".join(f"{k}={m:.1f}/{n:.1f}" for k, (m, n) in per_exc.items()), file=sys.stderr)
    print("  ris  " + " ".join(f"{k}={m:.1f}/{n:.1f}" for k, (m, n) in per_res.items()), file=sys.stderr, flush=True)
