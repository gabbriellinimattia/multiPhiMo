#!/usr/bin/env python3
"""Converte i CSV del corpus KNN (pluck/chaos/bird/mechanical + resonator_*) in un
unico binario compatto per il loader C++ (punto 6 fase 2, porting nativo VST3).

Non tocca i CSV originali: legge via knn_corpus.py (stessa logica di filtro/concat
del runtime Python -- build_joint_locked_corpus/build_corpus, nessuna reimplementazione
del parsing), scrive un blob unico. Un solo uso, non fa parte del plugin.

Uso:
    python native/tools/build_knn_corpus.py [--dataset-dir dataset] [--out native/data/knn_corpus.bin]

Formato binario (little-endian, un blob unico -- vedi discussione punto 6):
    magic "PHKC" (4 byte) | version uint32 | section_count uint32
    poi, per ogni sezione, in quest'ordine:
        section_type   uint8   (0 = eccitatore KNN congiunto vincolato alla forma,
                                 1 = risonatore ibrido, corpus mono-agente)
        agent_name     stringa (uint16 len + utf8)  es. "pluck", "resonator_bar"
        shape_name     stringa (uint16 len + utf8)  forma risonatore accoppiata
                                 (== agent_name senza prefisso per section_type=1)
        desc_count     uint16  + desc_count stringhe (nomi descrittori, ordine di riga)
        exc_par_count  uint16  + nomi (0 per section_type=1)
        res_par_count  uint16  + nomi (parametri PROPRI della forma, mai l'unione)
        row_count      uint32
        dati grezzi: row_count * (desc_count+exc_par_count+res_par_count) float32,
            riga per riga, ordine per riga: descrittori, poi parametri eccitatore
            (se presenti), poi parametri risonatore. Valori NaN (letterale "nan" nel
            CSV, es. formanti non rilevate) preservati come IEEE754 NaN, non 0 -- lo
            zero è sostituito solo a runtime dopo lo z-score (nan_to_num), mai nel dato
            grezzo, per restare fedele a knn_corpus.py (mean/std con NaN ignorati).

    Nomi inclusi esplicitamente (non solo indici) cosi' il loader C++ puo' verificare
    per nome contro ParamRanges.hpp/Descriptors.hpp invece di fidarsi di un ordine
    implicito -- un disallineamento futuro (es. un parametro rinominato) fa fallire il
    caricamento invece di silenziosamente disallineare le colonne.

    Statistiche di normalizzazione (mean/std, NaN ignorati) NON incluse nel binario:
    calcolate dal loader C++ a runtime dagli stessi dati grezzi, replicando
    _nan_safe_stats() di agents.py (decisione punto 6: evita un secondo posto dove i
    due lati possono disallinearsi).
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # root progetto
import agents
import knn_corpus
from resonator import PARAM_RANGES as RESONATOR_PARAM_RANGES

MAGIC = b"PHKC"
VERSION = 1

# Deve restare sincronizzato con param_candidate.JOINT_LOCKED_EXCITERS (non importato
# direttamente per non tirare dentro l'intero modulo audio/SPSA in un tool offline).
JOINT_LOCKED_EXCITERS = ["bird", "chaos", "mechanical", "pluck"]


def _write_str(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<H", len(b)))
    f.write(b)


def _write_names(f, names):
    f.write(struct.pack("<H", len(names)))
    for n in names:
        _write_str(f, n)


def _write_rows(f, rows, desc_keys, exc_params, res_params, res_prefix=""):
    n = len(rows)
    ncols = len(desc_keys) + len(exc_params) + len(res_params)
    arr = np.empty((n, ncols), dtype=np.float32)
    for i, r in enumerate(rows):
        j = 0
        for k in desc_keys:
            arr[i, j] = float(r[k]); j += 1
        for p in exc_params:
            arr[i, j] = float(r[p]); j += 1
        for p in res_params:
            arr[i, j] = float(r[res_prefix + p]); j += 1
    f.write(struct.pack("<I", n))
    f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return n, ncols


def write_joint_section(f, dataset_dir, exciter_name, shape):
    corpus = knn_corpus.build_joint_locked_corpus(
        dataset_dir / f"{exciter_name}.csv", exciter_name, shape)
    f.write(struct.pack("<B", 0))
    _write_str(f, exciter_name)
    _write_str(f, shape)
    _write_names(f, corpus.desc_keys)
    _write_names(f, corpus.param_names)
    _write_names(f, corpus.res_param_names)
    n, ncols = _write_rows(f, corpus.rows, corpus.desc_keys, corpus.param_names, corpus.res_param_names, res_prefix="pair_res_")
    print(f"  [joint] {exciter_name}+{shape}: {n} righe x {ncols} colonne")
    return n * ncols * 4


def write_resonator_section(f, dataset_dir, shape):
    agent_name = f"resonator_{shape}"
    corpus = knn_corpus.build_corpus(dataset_dir / f"{agent_name}.csv", agent_name)
    f.write(struct.pack("<B", 1))
    _write_str(f, agent_name)
    _write_str(f, shape)
    _write_names(f, corpus.desc_keys)
    _write_names(f, [])
    _write_names(f, corpus.param_names)
    n, ncols = _write_rows(f, corpus.rows, corpus.desc_keys, [], corpus.param_names)
    print(f"  [resonator] {agent_name}: {n} righe x {ncols} colonne")
    return n * ncols * 4


def main():
    p = argparse.ArgumentParser(description="Costruisce il blob binario del corpus KNN per il motore C++.")
    p.add_argument("--dataset-dir", default="dataset")
    p.add_argument("--out", default="native/data/knn_corpus.bin")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shapes = sorted(RESONATOR_PARAM_RANGES)  # 7 forme
    sections = []  # lista di (kind, exciter_or_none, shape)
    for exc in JOINT_LOCKED_EXCITERS:
        for shape in shapes:
            sections.append(("joint", exc, shape))
    for shape in shapes:
        sections.append(("resonator", None, shape))

    import io
    body = io.BytesIO()
    ok_sections = 0
    total_bytes = 0
    skipped = []
    for kind, exc, shape in sections:
        try:
            if kind == "joint":
                total_bytes += write_joint_section(body, dataset_dir, exc, shape)
            else:
                total_bytes += write_resonator_section(body, dataset_dir, shape)
            ok_sections += 1
        except Exception as e:
            label = f"{exc}+{shape}" if exc else f"resonator_{shape}"
            skipped.append((label, str(e)))
            print(f"  [SKIP] {label}: {e}")

    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", ok_sections))
        f.write(body.getvalue())

    size = out_path.stat().st_size
    print(f"\nScritte {ok_sections}/{len(sections)} sezioni ({total_bytes} byte dati + header/nomi) -> {out_path}")
    print(f"Dimensione file: {size} byte ({size / 1024 / 1024:.2f} MB)")
    if skipped:
        print(f"Sezioni saltate ({len(skipped)}): {skipped}")


if __name__ == "__main__":
    main()
