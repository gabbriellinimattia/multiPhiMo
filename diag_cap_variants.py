"""diag_cap_variants.py (2026-09-26) -- alternative al cap dei decadimenti: pitch vs peso del risonatore.
Varianti: cap (attuale K=2,Q=4) | nocap | soft (K=30, niente cap Q) | mix (50% eccitatore secco + 50% risonatore senza cap).
Per ogni eccitatore intonato: 5 config casuali (freq 110/220/440/880/330, clip al range) x 7 forme x 2 config risonatore.
pitch_ok = % con errore < 50 cent; oct = % errori d'ottava (|err| ~1200/1900/2400 +-80 c); nan = % pitch non rilevato.
res_db = distanza spettrale media (bande 1/3 ott., solo bande entro 40 dB dal massimo) tra i 14 risonatori
applicati allo STESSO eccitatore = quanto conta il risonatore. Uscita su stderr + diag_cap_variants.csv."""
import csv, sys
import numpy as np
import exciters as E
import resonator as R
from analyzer.descriptors import analyze_signal

SR = E.SR
DUR = 1.0
EDGES = 50.0 * 2 ** (np.arange(0, 29) / 3.0)
FREQS = [110.0, 220.0, 440.0, 880.0, 330.0]
PITCHED = ["bow", "blow", "strike", "pluck", "chaos", "mechanical", "bird", "vocal"]
rng = np.random.default_rng(1)


def rand_params(ranges, skip=()):
    out = {}
    for k, (lo, hi) in ranges.items():
        if k in skip:
            continue
        out[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if lo > 0 and hi / lo >= 20 else float(rng.uniform(lo, hi))
    return out


def bands(x):
    X = np.abs(np.fft.rfft(np.asarray(x, np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    e = np.array([X[(f >= a) & (f < b)].sum() for a, b in zip(EDGES[:-1], EDGES[1:])]) + 1e-20
    return 10 * np.log10(e / e.max())


def edist(a, b):
    m = np.maximum(a, b) > -40.0
    return float(np.sqrt(np.mean((np.maximum(a, -60) - np.maximum(b, -60))[m] ** 2)))


def unit(x):
    x = np.asarray(x, np.float64)
    return x / (np.sqrt(np.mean(x * x)) + 1e-12)


def variant(v, raw, shape, rp, f0):
    if v == "cap":
        return R.apply_resonator(raw, shape=shape, f0=f0, **rp)
    if v == "nocap":
        return R.apply_resonator(raw, shape=shape, f0=None, **rp)
    if v == "soft":
        return R.apply_resonator(raw, shape=shape, f0=f0, cap_k=30.0, cap_q=1e9, **rp)
    y = 0.5 * unit(raw) + 0.5 * unit(R.apply_resonator(raw, shape=shape, f0=None, **rp))
    return 0.9 * y / (np.max(np.abs(y)) + 1e-12)


VARS = ["cap", "nocap", "soft", "mix"]
rows = []
for exc in PITCHED:
    rng_ = E.PARAM_RANGES[exc]
    st = {v: dict(ok=0, oct=0, nan=0, n=0, d=[]) for v in VARS}
    for ci in range(5):
        f0 = float(np.clip(FREQS[ci], *rng_["freq"]))
        ep = rand_params(rng_, skip=("freq",))
        raw, _ = E.generate(exc, duration=DUR, freq=f0, **ep)
        combos = [(s, rand_params(R.PARAM_RANGES[s])) for s in R.PARAM_RANGES for _ in range(2)]
        for v in VARS:
            B = []
            for shape, rp in combos:
                y = variant(v, raw, shape, rp, f0)
                B.append(bands(y))
                p = analyze_signal(y, SR, extract_pitch=True, normalize=True).get("pitch", np.nan)
                s = st[v]
                s["n"] += 1
                if not (p is not None and np.isfinite(p) and p > 0):
                    s["nan"] += 1
                    continue
                c = abs(1200 * np.log2(p / f0))
                if c < 50:
                    s["ok"] += 1
                elif min(abs(c - t) for t in (1200, 1902, 2400)) < 80:
                    s["oct"] += 1
            st[v]["d"].append(np.mean([edist(B[i], B[j]) for i in range(len(B)) for j in range(i + 1, len(B))]))
    for v in VARS:
        s = st[v]
        r = dict(exciter=exc, variant=v, pitch_ok=round(100 * s["ok"] / s["n"], 1), oct=round(100 * s["oct"] / s["n"], 1),
                 nan=round(100 * s["nan"] / s["n"], 1), res_db=round(float(np.mean(s["d"])), 2))
        rows.append(r)
        print(f"{exc:11s} {v:6s} pitch_ok={r['pitch_ok']:5.1f}%  oct={r['oct']:5.1f}%  nan={r['nan']:5.1f}%  res_db={r['res_db']:5.2f}",
              file=sys.stderr, flush=True)

print("\n=== MEDIA SU TUTTI GLI ECCITATORI ===", file=sys.stderr)
for v in VARS:
    rr = [r for r in rows if r["variant"] == v]
    print(f"{v:6s} pitch_ok={np.mean([r['pitch_ok'] for r in rr]):5.1f}%  oct={np.mean([r['oct'] for r in rr]):5.1f}%  "
          f"nan={np.mean([r['nan'] for r in rr]):5.1f}%  res_db={np.mean([r['res_db'] for r in rr]):5.2f}", file=sys.stderr)
with open("diag_cap_variants.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
