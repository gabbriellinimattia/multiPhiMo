#!/usr/bin/env python3
"""Converte selector_dataset.csv (corpus di pair_selector.py) in un binario compatto per
il selettore C++ "Auto pair" (native/plugins/MultiPhiMo/PairSelector.hpp). Un solo uso,
non fa parte del plugin. Legge via pair_selector/agents (stesse costanti PAIRS/
DESCRIPTOR_KEYS/_read_csv del runtime Python, nessuna reimplementazione).

Uso (dalla root del progetto):
    python3 native/tools/build_selector_corpus.py [--csv selector_dataset.csv] [--out native/data/selector_corpus.bin]

Formato (little-endian):
    magic "PHSC" | version uint32
    desc_count uint16 + nomi (uint16 len + utf8)      -- DESCRIPTOR_KEYS
    exc_count  uint16 + nomi                            -- pair_selector.EXCITERS (ordinati)
    res_count  uint16 + nomi                            -- pair_selector.RESONATORS (ordinati)
    row_count  uint32
    per riga: desc_count float32 (grezzi, NaN preservato)
              exc_count*res_count float32 (score, ordine PAIRS: eccitatore esterno, forma interna)
              uint8 indice di best_exciter in EXCITERS (255 = assente/sconosciuto)
              uint8 indice di best_resonator in RESONATORS (255 = assente)          [v2]
    fasce (v2, 2026-09-25, "tabella dei margini"): per ogni eccitatore poi per ogni forma,
        desc_count coppie float32 (lo, hi) = percentili --lo-pct/--hi-pct del descrittore
        sulle righe in cui quell'eccitatore/forma e' il migliore (NaN = meno di
        --min-rows righe valide: nessuna fascia). Scritte anche in chiaro in --bands-csv.
Statistiche z-score e bias per eccitatore NON salvate: calcolate dal loader C++ (stessa
scelta di knn_corpus.bin, punto 6).
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # root progetto
from agents import DESCRIPTOR_KEYS, _read_csv
from pair_selector import EXCITERS, RESONATORS, PAIR_COLUMNS

MAGIC = b"PHSC"
VERSION = 2


def _write_str(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<H", len(b)))
    f.write(b)


def _write_names(f, names):
    f.write(struct.pack("<H", len(names)))
    for n in names:
        _write_str(f, n)


def main():
    p = argparse.ArgumentParser(description="Binario del corpus del selettore coppia per il motore C++.")
    p.add_argument("--csv", default="selector_dataset.csv")
    p.add_argument("--out", default="native/data/selector_corpus.bin")
    p.add_argument("--bands-csv", default="native/data/selector_bands.csv")
    p.add_argument("--lo-pct", type=float, default=25.0)
    p.add_argument("--hi-pct", type=float, default=75.0)
    p.add_argument("--min-rows", type=int, default=5)
    a = p.parse_args()

    rows = _read_csv(a.csv)
    if not rows:
        sys.exit("corpus vuoto")
    # stessa conversione di PairSelectorKnn.__init__ (float -> float32)
    X = np.array([[float(r[k]) for k in DESCRIPTOR_KEYS] for r in rows], dtype=np.float32)
    S = np.array([[float(r[c]) for c in PAIR_COLUMNS] for r in rows], dtype=np.float32)
    best = [EXCITERS.index(r.get("best_exciter", "")) if r.get("best_exciter", "") in EXCITERS else 255
            for r in rows]
    best_res = [RESONATORS.index(r.get("best_resonator", "")) if r.get("best_resonator", "") in RESONATORS else 255
                for r in rows]

    def bands(labels, idx):
        mask = np.array([b == idx for b in labels])
        out_b = np.full((len(DESCRIPTOR_KEYS), 2), np.nan, dtype=np.float32)
        counts = []
        for d in range(len(DESCRIPTOR_KEYS)):
            v = X[mask, d].astype(np.float64)
            v = v[~np.isnan(v)]
            counts.append(len(v))
            if len(v) >= a.min_rows:
                out_b[d] = np.percentile(v, [a.lo_pct, a.hi_pct])
        return out_b, int(mask.sum())

    exc_bands = [bands(best, i) for i in range(len(EXCITERS))]
    res_bands = [bands(best_res, i) for i in range(len(RESONATORS))]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        _write_names(f, DESCRIPTOR_KEYS)
        _write_names(f, EXCITERS)
        _write_names(f, RESONATORS)
        f.write(struct.pack("<I", len(rows)))
        for i in range(len(rows)):
            f.write(X[i].astype("<f4").tobytes())
            f.write(S[i].astype("<f4").tobytes())
            f.write(struct.pack("<B", best[i]))
            f.write(struct.pack("<B", best_res[i]))
        for b, _n in exc_bands + res_bands:
            f.write(b.astype("<f4").tobytes())
    bpath = Path(a.bands_csv)
    with open(bpath, "w") as f:
        f.write(f"# fasce p{a.lo_pct:g}-p{a.hi_pct:g} dei descrittori sulle righe in cui l'eccitatore/forma vince (selector_dataset.csv)\n")
        f.write("kind,name,n_rows," + ",".join(f"{k}_lo,{k}_hi" for k in DESCRIPTOR_KEYS) + "\n")
        for kind, names, bl in (("exciter", EXCITERS, exc_bands), ("resonator", RESONATORS, res_bands)):
            for name, (b, n) in zip(names, bl):
                f.write(f"{kind},{name},{n}," + ",".join(f"{x:.6g}" for x in b.flatten()) + "\n")
    print(f"fasce scritte in {bpath}: vittorie per eccitatore " +
          ", ".join(f"{e}={n}" for e, (_b, n) in zip(EXCITERS, exc_bands)) + "; per forma " +
          ", ".join(f"{r}={n}" for r, (_b, n) in zip(RESONATORS, res_bands)))
    n_nan = int(np.isnan(X).sum())
    print(f"NaN negli score: {int(np.isnan(S).sum())}")
    unknown = sum(1 for b in best if b == 255)
    print(f"{len(rows)} righe, {len(DESCRIPTOR_KEYS)} descrittori, {len(EXCITERS)} eccitatori x "
          f"{len(RESONATORS)} forme = {len(PAIR_COLUMNS)} coppie, NaN descrittori {n_nan}, "
          f"best_exciter sconosciuti {unknown}")
    print(f"scritto {out} ({out.stat().st_size} byte)")


if __name__ == "__main__":
    main()
