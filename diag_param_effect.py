"""diag_param_effect.py (2026-09-25) -- quanto pesa ogni parametro di sintesi sul suono.
A: ogni eccitatore, ogni parametro (freq esclusa) su 5 valori: a secco, via bar CON cap (f0=freq), via bar SENZA cap.
B: ogni forma di risonatore, ogni parametro su 5 valori, eccitata da pluck 220 Hz: CON cap (f0=220) e SENZA.
Metrica: dist = max distanza (dB RMS su bande di 1/3 d'ottava 50 Hz-16 kHz) tra le 5 versioni; + rapporto max/min
di centroide e decay_time. Uscita: diag_param_effect.csv + riepilogo su stderr."""
import csv, inspect, sys
import numpy as np
import exciters as E
import resonator as R
from analyzer.descriptors import analyze_signal

SR = E.SR
DUR = 1.0
EDGES = 50.0 * 2 ** (np.arange(0, 29) / 3.0)  # 50 Hz .. ~16 kHz


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
    db = 10 * np.log10(e / e.sum())
    return np.maximum(db, -80.0)


def feats(x):
    d = analyze_signal(x, SR, extract_pitch=False, normalize=True)
    return bands_db(x), float(d.get("spectral_centroid", np.nan)), float(d.get("decay_time", np.nan))


def summarize(sigs):
    fs = [feats(s) for s in sigs]
    B = [b for b, _, _ in fs]
    dist = max(np.sqrt(np.mean((B[i] - B[j]) ** 2)) for i in range(len(B)) for j in range(i + 1, len(B)))
    def ratio(v):
        v = np.array([u for u in v if np.isfinite(u) and u > 0])
        return float(v.max() / v.min()) if len(v) > 1 else np.nan
    return dist, ratio([c for _, c, _ in fs]), ratio([d for _, _, d in fs])


rows = []
def emit(part, unit, param, cond, vals, sigs):
    d, rc, rd = summarize(sigs)
    rows.append(dict(part=part, unit=unit, param=param, cond=cond, values=" ".join(f"{v:.4g}" for v in vals),
                     dist_db=round(d, 2), centroid_ratio=round(rc, 3), decay_ratio=round(rd, 3)))
    print(f"{part} {unit:11s} {param:16s} {cond:6s} dist={d:6.2f} dB  centr x{rc:5.2f}  decay x{rd:5.2f}",
          file=sys.stderr, flush=True)


# ---------- A: eccitatori ----------
for name, fn in E.EXCITERS.items():
    sig = inspect.signature(fn).parameters
    f0 = float(sig["freq"].default) if "freq" in sig else 220.0
    for p, (lo, hi) in E.PARAM_RANGES[name].items():
        if p == "freq":
            continue
        vals = grid(lo, hi, p)
        dry = []
        for v in vals:
            kw = dict(duration=DUR, **({"freq": f0} if "freq" in sig else {}))
            kw[p] = int(v) if p == "n_particles" else v
            np.random.seed(0)
            a, _ = E.generate(name, **kw)
            dry.append(np.asarray(a, np.float64))
        emit("A", name, p, "dry", vals, dry)
        emit("A", name, p, "cap", vals, [R.apply_resonator(a, shape="bar", f0=f0) for a in dry])
        emit("A", name, p, "nocap", vals, [R.apply_resonator(a, shape="bar", f0=None) for a in dry])

# ---------- B: risonatori ----------
exc, _ = E.generate("pluck", duration=DUR, freq=220.0)
exc = np.asarray(exc, np.float64)
for shape, ranges in R.PARAM_RANGES.items():
    for p, (lo, hi) in ranges.items():
        vals = grid(lo, hi, p)
        for cond, f0 in (("cap", 220.0), ("nocap", None)):
            sigs = [R.apply_resonator(exc, shape=shape, f0=f0, **{p: v}) for v in vals]
            emit("B", shape, p, cond, vals, sigs)

with open("diag_param_effect.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print("\n=== RIEPILOGO: mediana dist_db per unita' ===", file=sys.stderr)
for part in ("A", "B"):
    for unit in dict.fromkeys(r["unit"] for r in rows if r["part"] == part):
        s = "  ".join(f"{c}={np.median([r['dist_db'] for r in rows if r['part']==part and r['unit']==unit and r['cond']==c]):5.2f}"
                      for c in ("dry", "cap", "nocap") if any(r["cond"] == c and r["unit"] == unit for r in rows))
        print(f"{part} {unit:11s} {s}", file=sys.stderr)
print("CSV: diag_param_effect.csv", file=sys.stderr)
