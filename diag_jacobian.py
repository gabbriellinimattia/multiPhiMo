"""diag_jacobian.py (2026-09-26) -- CERTEZZA di controllo prima di rigenerare dataset: misura direttamente sul synth
(nessuna rete) se ogni descrittore si puo' muovere, anche insieme agli altri.

Per configurazioni casuali (eccitatore x forma): jacobiano J (15 descrittori x P parametri) per differenze centrali
sui render (passo 0.08 nello spazio normalizzato 0-1 dei parametri; descrittori in unita' di deviazione standard del
dataset_v3, stessa trasformazione del surrogato). Se J ha rango pieno, QUALSIASI combinazione di spostamenti dei
descrittori e' raggiungibile localmente. Per ogni descrittore:
  costo = norma minima di variazione dei parametri (0-1) per spostarlo di 0.5 std TENENDO FERMI TUTTI GLI ALTRI
          = 0.5*sqrt(((J J^T)^-1)_dd).  < 0.5: controllabile in modo indipendente; > 1: di fatto no.
Due varianti: sintesi attuale e + filtro formantico (prototipo, resonator.FORMANT_RANGES).
Uso: python3 diag_jacobian.py [n_config_per_eccitatore] [eccitatori,...]
"""
import sys
from multiprocessing import Pool
import os

import numpy as np
import torch

import exciters as E
import resonator as R
from surrogate import DKEYS, tdesc, LOG_X, LOG_R
from analyzer.descriptors import analyze_signal

H = 0.08
SHAPES = list(R.PARAM_RANGES)


def spec(exc, shape, formant):
    """[(gruppo, nome, lo, hi, log)] -- tutti i parametri liberi della configurazione."""
    s = [("x", k, lo, hi, k in LOG_X) for k, (lo, hi) in E.PARAM_RANGES[exc].items()]
    s += [("r", k, lo, hi, k in LOG_R) for k, (lo, hi) in R.PARAM_RANGES[shape].items()]
    s += [("c", k, lo, hi, k in R.COUPLING_LOG) for k, (lo, hi) in R.COUPLING_RANGES.items()]
    if formant:
        s += [("c", k, lo, hi, k in R.FORMANT_LOG) for k, (lo, hi) in R.FORMANT_RANGES.items()]
    return s


def val(u, lo, hi, log):
    return float(10 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo)))) if log and lo > 0 else float(lo + u * (hi - lo))


def desc(exc, shape, sp, u):
    xp, rp, cp = {}, {}, {}
    for (g, k, lo, hi, log), ui in zip(sp, u):
        {"x": xp, "r": rp, "c": cp}[g][k] = val(ui, lo, hi, log)
    if "n_particles" in xp:
        xp["n_particles"] = int(round(xp["n_particles"]))
    raw, sr = E.generate(exc, **xp)
    y = R.apply_resonator(raw, shape=shape, f0=xp.get("freq"), **rp, **cp)
    d = analyze_signal(y, sr, extract_pitch=True, normalize=False)
    return np.array([tdesc(d.get(k, np.nan), k) for k in DKEYS], dtype=np.float64)


def job(args):
    exc, shape, formant, seed = args
    rng = np.random.default_rng(seed)
    sp = spec(exc, shape, formant)
    u0 = rng.uniform(0.15, 0.85, len(sp))
    J = np.zeros((len(DKEYS), len(sp)))
    for i in range(len(sp)):
        up, um = u0.copy(), u0.copy()
        up[i] += H
        um[i] -= H
        J[:, i] = (desc(exc, shape, sp, up) - desc(exc, shape, sp, um)) / (2 * H)
    return exc, shape, formant, J


def cost(J, sd):
    Jz = J / sd[:, None]
    ok = np.all(np.isfinite(Jz), axis=1)
    out = np.full(len(DKEYS), np.nan)
    Jo = Jz[ok]
    G = Jo @ Jo.T + 1e-9 * np.eye(len(Jo))
    out[ok] = 0.5 * np.sqrt(np.clip(np.diag(np.linalg.inv(G)), 0, None))
    return out, np.linalg.svd(Jo, compute_uv=False).min() if len(Jo) else np.nan


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    excs = sys.argv[2].split(",") if len(sys.argv) > 2 else list(E.PARAM_RANGES)
    tasks = [(e, SHAPES[(i * 3 + k) % len(SHAPES)], f, 1000 * k + i) for i, e in enumerate(excs) for k in range(n)
             for f in (False, True)]
    with Pool(os.cpu_count()) as pool:
        res = pool.map(job, tasks, chunksize=1)
    for e in excs:
        sd = torch.load(f"weights_v3/{e}.pt", weights_only=False)["sd"]
        print(f"\n=== {e} ===  costo mediano per spostare di 0.5 std tenendo fermi gli altri (% config < 0.5)", file=sys.stderr)
        for f in (False, True):
            C, S = [], []
            for ee, sh, ff, J in res:
                if ee == e and ff == f:
                    c, smin = cost(J, sd)
                    C.append(c)
                    S.append(smin)
            C = np.array(C)
            line = " ".join(f"{k[:8]}={np.nanmedian(C[:, j]):4.2f}({100 * np.mean(C[:, j] < 0.5):3.0f}%)"
                            for j, k in enumerate(DKEYS))
            print(f"  {'+formanti' if f else 'attuale  '} sv_min={np.nanmedian(S):.2f}\n    {line}", file=sys.stderr, flush=True)
