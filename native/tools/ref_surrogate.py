#!/usr/bin/env python3
"""Riferimento Python per tests/test_surrogate.cpp. Uso (dalla root): python3 native/tools/ref_surrogate.py bow"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "native" / "tools"))
import surrogate as S  # noqa: E402
from export_surrogate import subsample_rows  # noqa: E402

exc = sys.argv[1] if len(sys.argv) > 1 else "bow"
inv = S.Inverter(exc)
keep = subsample_rows(np.asarray(inv.S))
inv.X, inv.Z, inv.M, inv.S = inv.X[keep], inv.Z[keep], inv.M[keep], inv.S[keep]
print(f"exc={exc} dim={inv.codec.dim} nets={len(inv.nets)} rows={len(keep)} hidden={inv.nets[0].f[0].out_features}")
with torch.no_grad():
    out = inv._fwd(torch.from_numpy(inv.X[:3].astype(np.float32))).numpy()
for r in range(3):
    o = out[r]
    print(f"fwd {r} {o[0]:.6g} {o[5]:.6g} {o[6]:.6g} {o[13]:.6g} {o[15]:.6g}")
vals = [1500, 1200, 3000, 0.05, 0.1, 0.5, 0.05, 500, 1500, 2500, 5, 0.1, 0.05, 0.8, 220]
tgt = {k: float(v) for k, v in zip(S.DKEYS, vals)}
xp, rp, cp, loss, _ = inv.solve(tgt, "bar", freq=220.0, n_knn=7, n_rand=0, steps=80)
x = inv.prev["bar"]
print(f"solve loss={loss:.5g} x0-7=" + "".join(f" {v:.4f}" for v in x[:8]))
for k in sorted(xp):
    print(f"exc {k}={xp[k]:.5g}")
for k in sorted(cp):
    print(f"coup {k}={cp[k]:.5g}")
