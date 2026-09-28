"""diag_control.py (2026-09-26) -- controllabilita' dei descrittori dopo la revisione (cap eliminato, accoppiamento
harmonicity/pitch_focus/body). Per ogni eccitatore: N config casuali (parametri eccitatore + forma casuale + parametri
risonatore + accoppiamento, freq log-uniforme). Per ogni descrittore: cov = (p95-p5)/|mediana| (quanto varia),
poi i 2 parametri piu' legati (|Spearman|). Poi: % pitch entro 50 c per fasce di pitch_focus, inharmonicity e
harmonic_tension mediane per fasce di harmonicity. Uscita: stderr + diag_control.csv (tutte le righe, riusabile)."""
import csv, sys
import numpy as np
from scipy.stats import spearmanr
import exciters as E
import resonator as R
from analyzer.descriptors import analyze_signal

SR = E.SR
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
KEYS = ["spectral_centroid", "spectral_spread", "spectral_rolloff", "spectral_flatness", "roughness",
        "harmonic_tension", "inharmonicity", "formant_f1", "formant_f2", "formant_f3", "mod_rate", "mod_depth",
        "attack_time", "decay_time", "pitch"]
COMMON_RES = ("size", "loss", "mode_falloff")
rng = np.random.default_rng(7)


def rnd(lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if lo > 0 and hi / lo >= 20 else float(rng.uniform(lo, hi))


shapes = list(R.PARAM_RANGES)
all_rows = []
for exc, ranges in E.PARAM_RANGES.items():
    if ONLY and exc not in ONLY:
        continue
    rows = []
    for i in range(N):
        ep = {k: rnd(lo, hi) for k, (lo, hi) in ranges.items()}
        if exc == "shaker":
            ep["n_particles"] = int(round(ep["n_particles"]))
        shape = shapes[i % len(shapes)]
        rp = {k: rnd(lo, hi) for k, (lo, hi) in R.PARAM_RANGES[shape].items()}
        cp = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in R.COUPLING_RANGES.items()}
        raw, _ = E.generate(exc, duration=1.0, **ep)
        y = R.apply_resonator(raw, shape=shape, f0=ep.get("freq"), **rp, **cp)
        d = analyze_signal(y, SR, extract_pitch=True, normalize=False)
        row = dict(exciter=exc, shape=shape, **{f"x_{k}": v for k, v in ep.items()}, **{f"res_{k}": rp[k] for k in COMMON_RES}, **cp,
                   **{k: d.get(k, np.nan) for k in KEYS})
        rows.append(row)
    all_rows += rows
    pnames = [f"x_{k}" for k in ranges if k != "freq"] + [f"res_{k}" for k in COMMON_RES] + list(R.COUPLING_RANGES)
    print(f"\n=== {exc} (N={N}) ===", file=sys.stderr)
    for k in KEYS:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        m = np.isfinite(v)
        if m.sum() < 10:
            print(f"  {k:18s} NaN in {100 - 100 * m.mean():.0f}%", file=sys.stderr)
            continue
        p5, p50, p95 = np.percentile(v[m], [5, 50, 95])
        cov = (p95 - p5) / (abs(p50) + 1e-9)
        cs = []
        for p in pnames + (["x_freq"] if "freq" in ranges else []):
            x = np.array([r[p] for r in rows], dtype=np.float64)[m]
            c = spearmanr(x, v[m])[0] if np.std(x) > 0 and np.std(v[m]) > 0 else 0.0
            cs.append((abs(c) if np.isfinite(c) else 0.0, p, c))
        cs.sort(reverse=True)
        top = "  ".join(f"{p}={c:+.2f}" for _, p, c in cs[:2])
        print(f"  {k:18s} cov={cov:6.2f}  nan={100 - 100 * m.mean():3.0f}%  {top}", file=sys.stderr)
    if "freq" in ranges:
        pf = np.array([r["pitch_focus"] for r in rows])
        hm = np.array([r["harmonicity"] for r in rows])
        err = np.array([abs(1200 * np.log2(r["pitch"] / r["x_freq"])) if np.isfinite(r["pitch"]) and r["pitch"] > 0
                        else np.inf for r in rows])
        s = "  ".join(f"pf[{a:.2f}-{b:.2f}]={100 * np.mean(err[(pf >= a) & (pf < b)] < 50):3.0f}%"
                      for a, b in ((0, .33), (.33, .66), (.66, 1.01)))
        inh = np.array([r["inharmonicity"] for r in rows], dtype=np.float64)
        ht = np.array([r["harmonic_tension"] for r in rows], dtype=np.float64)
        s2 = "  ".join(f"h[{a:.2f}-{b:.2f}] inh={np.nanmedian(inh[(hm >= a) & (hm < b)]):.3f} ht={np.nanmedian(ht[(hm >= a) & (hm < b)]):.3f}"
                       for a, b in ((0, .33), (.33, .66), (.66, 1.01)))
        print(f"  PITCH ok<50c per pitch_focus: {s}\n  {s2}", file=sys.stderr, flush=True)

cols = list(dict.fromkeys(k for r in all_rows for k in r))
with open("diag_control.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(all_rows)
print("\nCSV: diag_control.csv", file=sys.stderr)
