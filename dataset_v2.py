"""dataset_v2.py (2026-09-26) -- dataset CONGIUNTO e coerente col runtime, per il surrogato parametri -> descrittori.

Differenze rispetto a dataset_gen.py (v1):
  - una riga = configurazione COMPLETA: parametri eccitatore (x_*, freq inclusa), forma (uniforme sulle 7), parametri
    del risonatore (r_*, vuoti per le forme che non li hanno), accoppiamento (harmonicity, pitch_focus, body);
  - render IDENTICO al runtime: exciters.generate (durata di default) -> resonator.apply_resonator(f0=freq, ...)
    SENZA cap (eliminato 2026-09-26); f0 = freq anche per noise/shaker (harmonicity/pitch_focus li intonano);
  - descrittori: analyze_signal(extract_pitch=True, normalize=False); pitch = stima acustica grezza (NaN se assente),
    pitch_err_c = errore in cent rispetto a freq;
  - campionamento uniforme (log per LOG_PARAMS/LOG_RESONATOR_PARAMS), RNG per riga = (seed, eccitatore, indice):
    RIPRENDIBILE -- se il CSV esiste continua dall'ultima riga scritta (Ctrl-C sicuro);
  - multiprocessing (--workers, default = numero di core).

Uso:
    python3 dataset_v2.py --n 30000                 # tutti gli eccitatori -> dataset_v5/<eccitatore>.csv
    python3 dataset_v2.py bow pluck --n 30000
"""
import argparse
import csv
import inspect
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import agents
import exciters as E
import resonator as R
from analyzer.descriptors import analyze_signal
from dataset_gen import SILENCE_PEAK

SHAPES = list(R.PARAM_RANGES)
RES_NAMES = sorted({k for r in R.PARAM_RANGES.values() for k in r})
LOG_X = set(agents.LOG_PARAMS)
LOG_R = set(agents.LOG_RESONATOR_PARAMS)
DKEYS = list(agents.DESCRIPTOR_KEYS)
PRINT_EVERY = 5000  # righe tra una stampa di avanzamento e l'altra


def _draw(rng, lo, hi, log):
    if log and lo > 0:
        return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
    return float(rng.uniform(lo, hi))


def columns(exc):
    return (DKEYS + ["pitch_err_c"] + [f"x_{k}" for k in E.PARAM_RANGES[exc]] + ["shape"]
            + [f"r_{k}" for k in RES_NAMES] + list(R.COUPLING_RANGES))


def render_row(args):
    exc, i, seed = args
    rng = np.random.default_rng([seed, list(E.PARAM_RANGES).index(exc), i])
    xp = {k: _draw(rng, lo, hi, k in LOG_X or k == "freq") for k, (lo, hi) in E.PARAM_RANGES[exc].items()}
    if "n_particles" in xp:
        xp["n_particles"] = int(round(xp["n_particles"]))
    shape = SHAPES[int(rng.integers(len(SHAPES)))]
    rp = {k: _draw(rng, lo, hi, k in LOG_R) for k, (lo, hi) in R.PARAM_RANGES[shape].items()}
    cp = {k: _draw(rng, lo, hi, k in R.COUPLING_LOG) for k, (lo, hi) in R.COUPLING_RANGES.items()}
    try:
        dur0 = inspect.signature(E.EXCITERS[exc]).parameters["duration"].default
        raw, sr = E.generate(exc, duration=R.note_duration(dur0, shape, rp, cp.get("exc_hold")), **xp)
        y = R.apply_resonator(raw, shape=shape, f0=xp.get("freq"), **rp, **cp)
        if not np.all(np.isfinite(y)) or float(np.max(np.abs(y))) < SILENCE_PEAK:
            return None
        d = analyze_signal(y, sr, extract_pitch=True, normalize=False)
    except Exception as e:
        return None
    row = {k: d.get(k, float("nan")) for k in DKEYS}
    p, f = row.get("pitch"), xp.get("freq")
    row["pitch_err_c"] = (1200.0 * np.log2(p / f)) if (p is not None and np.isfinite(p) and p > 0 and f) else float("nan")
    row.update({f"x_{k}": v for k, v in xp.items()})
    row["shape"] = shape
    row.update({f"r_{k}": v for k, v in rp.items()})
    row.update(cp)
    return row


BLOCK_STRIDE = 10_000_000  # spazi di indici RNG disgiunti per blocco (nessuna riga duplicata tra blocchi)


def run(exc, n, out_dir, seed, workers, block=0, blocks=1):
    """blocks > 1: questo processo genera solo la sua quota (n/blocks righe) in <eccitatore>.part<block>.csv;
    surrogate.py legge tutte le parti insieme. Ogni blocco e' riprendibile da solo."""
    if blocks > 1:
        n = n // blocks + (n % blocks if block == blocks - 1 else 0)
    path = Path(out_dir) / (f"{exc}.part{block}.csv" if blocks > 1 else f"{exc}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    if path.exists():
        with open(path) as fh:
            start = max(0, sum(1 for _ in fh) - 1)
    if start >= n:
        print(f"[{exc}] gia' completo ({start} righe)", file=sys.stderr)
        return
    mode = "a" if start > 0 else "w"
    t0 = time.time()
    written = discarded = 0
    idx = start + block * BLOCK_STRIDE
    with open(path, mode, newline="") as fh, Pool(workers) as pool:
        w = csv.DictWriter(fh, fieldnames=columns(exc), restval="")
        if start == 0:
            w.writeheader()
        # gli indici continuano da 'start': gli scarti vengono rimpiazzati con indici nuovi fino a n righe
        while start + written < n:
            need = n - start - written
            batch = [(exc, j, seed) for j in range(idx, idx + int(need * 1.05) + 8)]
            idx += len(batch)
            for row in pool.imap(render_row, batch, chunksize=4):
                if row is None:
                    discarded += 1
                    continue
                if start + written >= n:
                    continue
                w.writerow(row)
                written += 1
                if written % 500 == 0:
                    fh.flush()  # salvataggio frequente (ripresa sicura), stampa solo ogni PRINT_EVERY
                if written % PRINT_EVERY == 0:
                    el = time.time() - t0
                    eta = el / written * (n - start - written)
                    print(f"[{exc}] {start + written}/{n}  {written / el:.1f} righe/s  ETA {eta / 60:.0f} min",
                          file=sys.stderr, flush=True)
    print(f"[{exc}{'' if blocks == 1 else f' blocco {block}'}] fatto: {start + written} righe ({discarded} scartate) -> {path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exciters", nargs="*", default=list(E.PARAM_RANGES))
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out-dir", default="dataset_v5")  # v4 = + formanti, v5 = attacco/tenuta all'eccitatore, decadimento al risonatore
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--block", type=int, default=0, help="indice del blocco (0..blocks-1)")
    ap.add_argument("--blocks", type=int, default=1, help="in quanti blocchi/terminali dividere")
    a = ap.parse_args()
    for exc in a.exciters:
        run(exc, a.n, a.out_dir, a.seed, a.workers, a.block, a.blocks)
