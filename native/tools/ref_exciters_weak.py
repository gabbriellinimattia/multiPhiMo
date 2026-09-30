#!/usr/bin/env python3
"""Confronto per tests/test_exciters_weak.cpp (2026-09-28, parametri deboli).
bow/strike/blow0 (deterministici; blow0 = blow senza rumore): max |C++ - Python|. blow/shaker (RNG diversi): distanza dB (1/3 ott.) C++ vs Python per
ogni valore, ed effetto del parametro (max distanza tra i 5 valori, metrica di diag_param_weak.py) in entrambi.
Uso (dalla root): python3 native/tools/ref_exciters_weak.py native/plugins/MultiPhiMo/tests/out_weak"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import exciters as E
from diag_param_weak import bands_db, db_dist

d = Path(sys.argv[1] if len(sys.argv) > 1 else "native/plugins/MultiPhiMo/tests/out_weak")
D, F = 1.0, 220.0
t60 = np.geomspace(0.05, 5.0, 5)
py = {
    "bow": [E.bow(duration=D, freq=F, bow_force=0.525, bow_velocity=0.525, bow_position=0.235, brightness=0.525, t60=v)
            for v in t60],
    "blow": [E.blow(duration=D, freq=F, mouth_pressure=0.55, reed_stiffness=0.5, breath_noise=0.3, brightness=0.525,
                    t60=v) for v in t60],
    "blow0": [E.blow(duration=D, freq=F, mouth_pressure=0.55, reed_stiffness=0.5, breath_noise=0.0, brightness=0.525,
                     t60=v) for v in t60],
    "strike": [E.strike(duration=D, freq=F, impact_velocity=0.55, hammer_mass=0.01, hammer_stiffness=1e7,
                        nonlinearity=1.75, material=v, size_damping=0.5) for v in np.linspace(0, 1, 5)],
    "shaker": [E.shaker(duration=D, freq=F, n_particles=50, energy=v, decay_time=1.05, material=0.5)
               for v in np.linspace(0.1, 1.0, 5)],
}
for exc, ys in py.items():
    cs = [np.fromfile(d / f"{exc}_{i}.f32", dtype=np.float32).astype(np.float64) for i in range(5)]
    ys = [np.asarray(y, np.float64) for y in ys]
    if exc in ("bow", "strike", "blow0"):
        m = max(float(np.max(np.abs(c - y))) if len(c) == len(y) else np.inf for c, y in zip(cs, ys))
        print(f"{exc:7s} max|C++-Py| = {m:.2e}  (len {len(cs[0])}/{len(ys[0])})")
    else:
        dd = [float(np.sqrt(np.mean((bands_db(c) - bands_db(y)) ** 2))) for c, y in zip(cs, ys)]
        print(f"{exc:7s} dB C++ vs Py per valore: " + " ".join(f"{v:.1f}" for v in dd))
    print(f"{exc:7s} effetto parametro: C++ {db_dist(cs):.1f} dB | Py {db_dist(ys):.1f} dB")
