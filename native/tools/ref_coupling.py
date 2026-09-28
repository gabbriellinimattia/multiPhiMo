#!/usr/bin/env python3
"""Riferimento Python per tests/test_coupling.cpp (stesso formato di stampa).
Uso (dalla root): python3 native/tools/ref_coupling.py"""
import sys
from pathlib import Path

import numpy as np
import scipy.signal as sg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import exciters as E
import resonator as R

f32 = lambda d: {k: float(np.float32(v)) for k, v in d.items()}  # stessi valori float del C++
b, a = sg.butter(2, [0.02, 0.03], btype="band")
print("butter b=" + " ".join(f"{v:.12g}" for v in b) + " a=" + " ".join(f"{v:.12g}" for v in a))
full = dict(harmonicity=0.3, pitch_focus=0.5, body=0.8, exc_attack=0.02, exc_hold=0.4, am_rate=5.0, am_depth=0.3,
            form_f1=500.0, form_f2=1500.0, form_amt=0.5)
bar = dict(size=0.3, thickness=0.01, density=2700.0, stiffness=7e10, loss=0.005, mode_falloff=0.5)
cfgs = [
    ("bar", bar, full, 220.0),
    ("bar", bar, dict(harmonicity=1.0, pitch_focus=0.0, body=1.0), 220.0),
    ("plate_circ", dict(size=0.2, thickness=0.002, density=7800.0, stiffness=2e11, loss=0.002, mode_falloff=0.3),
     dict(harmonicity=0.0, pitch_focus=0.8, body=0.5, exc_hold=1.0, exc_attack=0.005), 330.0),
    ("tube", dict(size=0.6, radius=0.012, closure=0.5, flare=0.2, loss=0.005, mode_falloff=0.5),
     dict(harmonicity=0.7, pitch_focus=0.2, body=0.9, form_f1=700.0, form_f2=2200.0, form_amt=1.0), 147.0),
    ("soundboard", dict(size=0.4, aspect=0.75, thickness=0.0045, stiffness=1.2e10, ortho=15.0, cavity=0.005,
                        hole=0.04, loss=0.005, mode_falloff=0.5),
     dict(harmonicity=0.5, exc_hold=0.2, am_rate=12.0, am_depth=0.8), 440.0),
    ("chaotic", dict(size=0.1, thickness=0.006, stiffness=7e10, loss=0.005, mode_falloff=0.5, nonlin=0.2, beat=3.0,
                     depth=0.6, chaos=0.4, speed=5.0),
     dict(harmonicity=0.4, pitch_focus=0.3, body=0.7, exc_attack=0.05, exc_hold=0.6), 110.0),
    ("membrane", dict(size=0.3, thickness=0.0005, density=1000.0, stiffness=3000.0, loss=0.01, mode_falloff=0.6),
     dict(harmonicity=0.1, pitch_focus=0.9), 0.0),
]
for i, (shape, rp, cp, f0) in enumerate(cfgs):
    rp, cp = f32(rp), f32(cp)
    dur = R.note_duration(1.0, shape, rp, cp.get("exc_hold"))
    raw = E.strike(duration=dur, freq=f0 if f0 > 0 else 220.0, impact_velocity=0.8, hammer_mass=0.02,
                   hammer_stiffness=5e7, nonlinearity=1.5, material=0.3, size_damping=0.4)
    y = R.apply_resonator(raw, shape=shape, f0=f0 if f0 > 0 else None, **rp, **cp)
    yd = y.astype(np.float64)
    print(f"cfg {i} {shape} dur={dur:.10g} n={len(y)} sum={yd.sum():.8g} sumsq={(yd * yd).sum():.8g} "
          f"y1000={yd[1000]:.8g} y20000={yd[20000]:.8g}")
