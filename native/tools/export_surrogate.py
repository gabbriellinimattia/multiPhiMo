#!/usr/bin/env python3
"""Esporta i pesi del surrogato (surrogate.py, weights_v5/<eccitatore>.pt) per il plugin C++ (Surrogate.hpp).
Uso (dalla root): python3 native/tools/export_surrogate.py [--weights weights_v5] [--out native/data/surrogate]

Formato <eccitatore>.bin, little-endian ("PHSG", versione 1). Stringhe = uint16 lunghezza + utf8.
  magic "PHSG" | u32 version | str exc
  u32 n_x   ; per parametro eccitatore (ordine Codec.xnames): str nome, f32 lo, f32 hi, u8 log
  u32 n_s   ; per forma (ordine surrogate.SHAPES): str nome
  u32 n_r   ; per nome (ordine surrogate.RES_NAMES): str nome
  per forma : u32 n ; per parametro (ordine R.PARAM_RANGES[forma]): str nome, f32 lo, f32 hi, u8 log
  u32 n_c   ; per parametro d'accoppiamento (ordine ck["coup"]): str nome, f32 lo, f32 hi, u8 log
  u32 n_d   ; per descrittore (ordine DKEYS): str nome, u8 ha_offset, f32 offset, f32 mu, f32 sd
  u32 hidden | u32 n_out | u32 n_nets ; per rete 4 strati Linear: u32 out, u32 in, f32 W[out][in], f32 b[out]
  u32 rows | u32 dim ; f32 X[rows][dim], f32 Z[rows][n_d], u8 M[rows][n_d], u8 S[rows]
Le righe (punti di partenza KNN del solutore) sono un sottoinsieme: fino a ROWS_PER_SHAPE per forma, scelte con
subsample_rows() (seed fisso, riusata da ref_surrogate.py per confrontare il C++ con lo stesso sottoinsieme).
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import surrogate as S  # noqa: E402
import resonator as R  # noqa: E402
import exciters as E  # noqa: E402

ROWS_PER_SHAPE = 500


def subsample_rows(Sarr, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for s in range(len(S.SHAPES)):
        idx = np.where(Sarr == s)[0]
        if len(idx) > ROWS_PER_SHAPE:
            idx = np.sort(rng.choice(idx, ROWS_PER_SHAPE, replace=False))
        keep.append(idx)
    return np.concatenate(keep)


def _s(f, t):
    b = t.encode()
    f.write(struct.pack("<H", len(b)))
    f.write(b)


def export(exc, wdir, odir):
    ck = torch.load(Path(wdir) / f"{exc}.pt", weights_only=False)
    if ck.get("pf"):
        raise SystemExit(f"{exc}: pesi con feature di classe d'altezza (pf) non supportati dal C++")
    codec = S.Codec(exc, ck.get("coup"))
    states = ck.get("states", [ck["state"]])
    keep = subsample_rows(np.asarray(ck["S"]))
    X = np.asarray(ck["X"], np.float32)[keep]
    Z = np.asarray(ck["Z"], np.float32)[keep]
    M = np.asarray(ck["M"], np.uint8)[keep]
    Sa = np.asarray(ck["S"], np.uint8)[keep]
    odir.mkdir(parents=True, exist_ok=True)
    with open(odir / f"{exc}.bin", "wb") as f:
        f.write(b"PHSG")
        f.write(struct.pack("<I", 1))
        _s(f, exc)
        f.write(struct.pack("<I", len(codec.xnames)))
        for k in codec.xnames:
            lo, hi = E.PARAM_RANGES[exc][k]
            _s(f, k)
            f.write(struct.pack("<ffB", lo, hi, int(k in S.LOG_X)))
        f.write(struct.pack("<I", len(S.SHAPES)))
        for sh in S.SHAPES:
            _s(f, sh)
        f.write(struct.pack("<I", len(S.RES_NAMES)))
        for k in S.RES_NAMES:
            _s(f, k)
        for sh in S.SHAPES:
            f.write(struct.pack("<I", len(R.PARAM_RANGES[sh])))
            for k, (lo, hi) in R.PARAM_RANGES[sh].items():
                _s(f, k)
                f.write(struct.pack("<ffB", lo, hi, int(k in S.LOG_R)))
        f.write(struct.pack("<I", len(codec.coup)))
        for k in codec.coup:
            (lo, hi), lg = S._crange(k)
            _s(f, k)
            f.write(struct.pack("<ffB", lo, hi, int(lg)))
        f.write(struct.pack("<I", len(S.DKEYS)))
        for j, k in enumerate(S.DKEYS):
            o = S.OFFSET[k]
            _s(f, k)
            f.write(struct.pack("<Bfff", int(o is not None), o if o is not None else 0.0, ck["mu"][j], ck["sd"][j]))
        f.write(struct.pack("<III", ck["hidden"], len(S.DKEYS) + 1, len(states)))
        for st in states:
            for li in (0, 2, 4, 6):
                W = st[f"f.{li}.weight"].numpy().astype("<f4")
                b = st[f"f.{li}.bias"].numpy().astype("<f4")
                f.write(struct.pack("<II", W.shape[0], W.shape[1]))
                f.write(W.tobytes())
                f.write(b.tobytes())
        f.write(struct.pack("<II", len(keep), codec.dim))
        f.write(X.astype("<f4").tobytes())
        f.write(Z.astype("<f4").tobytes())
        f.write(M.tobytes())
        f.write(Sa.tobytes())
    print(f"{exc}: dim={codec.dim} coup={len(codec.coup)} reti={len(states)} righe={len(keep)} "
          f"-> {(odir / f'{exc}.bin').stat().st_size} byte")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights_v5"))
    ap.add_argument("--out", default=str(ROOT / "native" / "data" / "surrogate"))
    a = ap.parse_args()
    for p in sorted(Path(a.weights).glob("*.pt")):
        export(p.stem, a.weights, Path(a.out))
