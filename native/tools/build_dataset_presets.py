#!/usr/bin/env python3
"""Genera native/plugins/MultiPhiMo/DatasetPresets.hpp: 20 preset di descrittori presi a caso da dataset_v5
(2 per eccitatore, forme diverse), per il menu "Preset" della GUI (Mode=Agent: eccitatore, risonatore, 15 descrittori).
Righe ammesse: 15 descrittori finiti e intonata (|pitch_err_c| <= 50 cent) salvo noise/shaker (freq libera).
Uso (dalla root): python3 native/tools/build_dataset_presets.py [--seed 7]"""
import argparse
import csv
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXC = ["bow", "blow", "strike", "pluck", "shaker", "noise", "chaos", "mechanical", "bird", "vocal"]  # = kExciterEnum
RES = ["bar", "plate_rect", "plate_circ", "membrane", "tube", "soundboard", "chaotic"]              # = kResonatorEnum
D = ["spectral_centroid", "spectral_spread", "spectral_rolloff", "spectral_flatness", "roughness", "harmonic_tension",
     "inharmonicity", "formant_f1", "formant_f2", "formant_f3", "mod_rate", "mod_depth", "attack_time", "decay_time",
     "pitch"]


def fl(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def flit(x):
    t = "%.6g" % x
    return (t if any(c in t for c in ".e") else t + ".0") + "f"


ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--data", default=str(ROOT / "dataset_v5"))
a = ap.parse_args()
rng = random.Random(a.seed)
out = []
for exc in EXC:
    ok = []
    for f in sorted(Path(a.data).glob(f"{exc}.part*.csv")) or [Path(a.data) / f"{exc}.csv"]:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                v = [fl(r[k]) for k in D]
                if any(x is None for x in v):
                    continue
                pe = fl(r.get("pitch_err_c"))
                if exc not in ("noise", "shaker") and (pe is None or abs(pe) > 50):
                    continue
                ok.append((r["shape"], v))
    rng.shuffle(ok)
    used = set()
    for shape, v in ok:
        if shape in used:
            continue
        used.add(shape)
        out.append((exc, shape, v))
        if len(used) == 2:
            break
    print(f"{exc}: {len(ok)} righe ammesse, scelte {sorted(used)}")

lines = ["#pragma once",
         "// GENERATO da native/tools/build_dataset_presets.py (seed %d) -- non modificare a mano." % a.seed,
         "// 20 preset di descrittori presi a caso da dataset_v5 (2 per eccitatore): menu \"Preset\" della GUI.",
         "// exciter/resonator = indici di kExciterEnum/kResonatorEnum (ParamLayout.hpp); d = 15 descrittori,",
         "// ordine kDescriptorNames.",
         "#include <cstdint>", "", "namespace phimo {", "",
         "struct DatasetPreset { uint32_t exciter, resonator; float d[15]; };", "",
         "static constexpr uint32_t kDatasetPresetCount = %d;" % len(out), "",
         "static const char* const kDatasetPresetLabels[%d] = {" % len(out)]
for i, (exc, shape, _) in enumerate(out):
    lines.append('    "%02d %s + %s",' % (i + 1, exc, shape))
lines += ["};", "", "static const DatasetPreset kDatasetPresets[%d] = {" % len(out)]
for exc, shape, v in out:
    lines.append("    {%d, %d, {%s}}," % (EXC.index(exc), RES.index(shape), ", ".join(flit(x) for x in v)))
lines += ["};", "", "} // namespace phimo", ""]
p = ROOT / "native" / "plugins" / "MultiPhiMo" / "DatasetPresets.hpp"
p.write_text("\n".join(lines))
print(f"scritto {p} ({len(out)} preset)")
