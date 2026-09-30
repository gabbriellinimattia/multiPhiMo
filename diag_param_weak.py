"""diag_param_weak.py (2026-09-28) -- parametri deboli: effetto di OGNI parametro di sintesi, spettro + descrittori.
A) ogni eccitatore, ogni parametro (freq esclusa) su 5 valori (base = centro del range, freq 220 Hz):
   a secco (eccitatore grezzo) e via bar col preset d'accoppiamento di default del plugin;
B) ogni forma, ogni parametro su 5 valori, eccitata da pluck 220 Hz (default), stesso preset.
Render come dataset_v2/runtime: durata = R.note_duration(durata di default, forma, {}, exc_hold).
Metriche: dB = max distanza (dB RMS su bande di 1/3 d'ottava 50 Hz-16 kHz) tra le 5 versioni (come diag_manual_defaults);
desc = per ognuno dei 15 descrittori (analyze_signal) (max-min)/range di slider_ranges.json (log per attack/decay/pitch).
DEBOLE (via bar / forma) = dB < DB_THR e nessun descrittore > DESC_THR. "solo desc" = dB < DB_THR ma un descrittore si muove.
Uso: taskpolicy -b python3 diag_param_weak.py [--workers 2] [bow strike ...] 2> diag_param_weak.log   (CSV: diag_param_weak.csv)"""
import argparse
import csv
import inspect
import json
import sys
from multiprocessing import Pool

import numpy as np

import agents
import exciters as E
import resonator as R
from analyzer.descriptors import analyze_signal

SR = E.SR
EDGES = 50.0 * 2 ** (np.arange(0, 29) / 3.0)
DKEYS = list(agents.DESCRIPTOR_KEYS)
LOGD = set(agents.LOG_DESCRIPTORS)
DRANGE = json.load(open("slider_ranges.json"))
# = kCouplingManualDefaults (Coupling.hpp)
CP = dict(harmonicity=0.8, pitch_focus=0.0, body=1.0, exc_attack=0.001, exc_hold=3.0, am_rate=5.0, am_depth=0.0,
          form_f1=500.0, form_f2=1500.0, form_amt=0.0)
DB_THR, DESC_THR = 1.5, 0.10
F0 = 220.0


def grid(lo, hi, name):
    if name == "n_particles":
        return np.unique(np.round(np.geomspace(lo, hi, 5))).tolist()
    if lo > 0 and hi / lo >= 20:
        return np.geomspace(lo, hi, 5).tolist()
    return np.linspace(lo, hi, 5).tolist()


def mid(lo, hi):
    return float(np.sqrt(lo * hi)) if lo > 0 and hi / lo >= 20 else 0.5 * (lo + hi)


def bands_db(x):
    X = np.abs(np.fft.rfft(np.asarray(x, np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    e = np.array([X[(f >= a) & (f < b)].sum() for a, b in zip(EDGES[:-1], EDGES[1:])]) + 1e-20
    return np.maximum(10 * np.log10(e / e.sum()), -80.0)


def db_dist(sigs):
    B = np.array([bands_db(s) for s in sigs])
    return max(np.sqrt(np.mean((B[i] - B[j]) ** 2)) for i in range(len(B)) for j in range(i + 1, len(B)))


def desc_eff(sigs):
    """-> {descrittore: frazione del range spazzata dalle 5 versioni}"""
    D = [analyze_signal(np.asarray(s, np.float64), SR, extract_pitch=True, normalize=False) for s in sigs]
    out = {}
    for k in DKEYS:
        lo, hi = DRANGE[k]
        v = np.array([float(d.get(k, np.nan)) for d in D], np.float64)
        if k in LOGD:
            v, lo, hi = np.log(np.maximum(v, 1e-9)), np.log(lo), np.log(hi)
        v = v[np.isfinite(v)]
        out[k] = float((v.max() - v.min()) / (hi - lo)) if len(v) >= 2 else float("nan")
    return out


def dur(exc, shape):
    d0 = inspect.signature(E.EXCITERS[exc]).parameters["duration"].default
    return R.note_duration(d0, shape, {}, CP["exc_hold"])


def job_exc(exc):
    pr = E.PARAM_RANGES[exc]
    base = {k: mid(lo, hi) for k, (lo, hi) in pr.items()}
    base["freq"] = float(np.clip(F0, *pr["freq"]))
    if "n_particles" in base:
        base["n_particles"] = int(round(base["n_particles"]))
    T = dur(exc, "bar")
    rows = []
    for p, (lo, hi) in pr.items():
        if p == "freq":
            continue
        dry, wet = [], []
        for v in grid(lo, hi, p):
            xp = dict(base)
            xp[p] = int(round(v)) if p == "n_particles" else v
            raw, _ = E.generate(exc, duration=T, **xp)
            dry.append(raw)
            wet.append(R.apply_resonator(raw, shape="bar", f0=xp["freq"], **CP))
        rows.append(("exc", exc, p, db_dist(dry), desc_eff(dry), db_dist(wet), desc_eff(wet)))
    return rows


def job_res(shape):
    T = dur("pluck", shape)
    raw, _ = E.generate("pluck", duration=T, freq=F0)
    rows = []
    for p, (lo, hi) in R.PARAM_RANGES[shape].items():
        sigs = [R.apply_resonator(raw, shape=shape, f0=F0, **{p: v}, **CP) for v in grid(lo, hi, p)]
        rows.append(("res", shape, p, float("nan"), {}, db_dist(sigs), desc_eff(sigs)))
    return rows


def run(job):
    kind, name = job
    return job_exc(name) if kind == "exc" else job_res(name)


def top(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    if not d:
        return "-", float("nan")
    k = max(d, key=d.get)
    return k, d[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("names", nargs="*", help="solo questi eccitatori/forme (default: tutti)")
    a = ap.parse_args()
    jobs = [("exc", e) for e in E.PARAM_RANGES] + [("res", s) for s in R.PARAM_RANGES]
    if a.names:
        jobs = [j for j in jobs if j[1] in a.names]
    weak = []
    with open("diag_param_weak_sel.csv" if a.names else "diag_param_weak.csv", "w", newline="") as fh, Pool(a.workers) as pool:
        w = csv.writer(fh)
        w.writerow(["kind", "name", "param", "db_dry", "db_bar"] + [f"dry_{k}" for k in DKEYS] + [f"bar_{k}" for k in DKEYS])
        for rows in pool.imap(run, jobs):
            for kind, name, p, ddb, dd, wdb, wd in rows:
                w.writerow([kind, name, p, f"{ddb:.2f}", f"{wdb:.2f}"] + [f"{dd.get(k, np.nan):.3f}" for k in DKEYS]
                           + [f"{wd[k]:.3f}" for k in DKEYS])
                kd, vd = top(dd)
                kw, vw = top(wd)
                if wdb < DB_THR:
                    tag = "DEBOLE" if not vw > DESC_THR else "solo desc"
                else:
                    tag = ""
                if tag == "DEBOLE":
                    weak.append(f"{name}.{p}")
                sec = f"secco {ddb:5.1f} dB {kd}={vd:.2f} | " if kind == "exc" else ""
                print(f"{name:10s} {p:16s} {sec}bar {wdb:5.1f} dB {kw}={vw:.2f}  {tag}", file=sys.stderr, flush=True)
            fh.flush()
    print(f"DEBOLI ({len(weak)}): " + " ".join(weak), file=sys.stderr)


if __name__ == "__main__":
    main()
