#!/usr/bin/env python3
"""Esporta i checkpoint .pt (agents.py) in un formato binario compatto per il motore C++.

Non modifica i pesi: solo serializzazione 1:1 dello state_dict + statistiche di
normalizzazione (desc_mean/desc_std), stesso ordine/layout usato da PyTorch
(nn.Linear.weight = [out_features, in_features], row-major). Riusa Agent.from_checkpoint
di agents.py per il caricamento (nessuna reimplementazione del parser .pt).

Uso:
    python native/tools/export_weights.py [--weights-dir weights] [--out-dir native/data/weights]

Formato binario (little-endian):
    magic "PHMD" (4 byte) | version uint32 | n_in,n_out,hidden,k uint32 x4
    poi, in quest'ordine, array float32 flat (dimensioni derivabili da n_in/n_out/hidden/k):
        desc_mean[n_in]  desc_std[n_in]
        backbone.0.weight[hidden*n_in]   backbone.0.bias[hidden]
        backbone.2.weight[hidden*hidden] backbone.2.bias[hidden]
        pi_head.weight[k*hidden]         pi_head.bias[k]
        mu_head.weight[(k*n_out)*hidden] mu_head.bias[k*n_out]
        log_sigma_head.weight[(k*n_out)*hidden] log_sigma_head.bias[k*n_out]
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # root progetto (dove sta agents.py)
from agents import Agent, AGENT_SPECS

MAGIC = b"PHMD"
VERSION = 1


def export_agent(name, ckpt_path, out_path):
    agent = Agent.from_checkpoint(ckpt_path)
    if agent.condition_on_resonator:
        raise RuntimeError(
            f"{name}: condition_on_resonator=True non supportato dal motore C++ "
            "(non adottato in produzione, vedi criticita_modelli.md)")
    sd = agent.model.state_dict()
    n_in = len(agent.descriptor_keys)
    n_out = len(agent.param_names)
    hidden = agent.hidden
    k = agent.model.k

    def w(key):
        return sd[key].detach().numpy().astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<IIII", n_in, n_out, hidden, k))
        for arr in (
            agent.desc_mean.astype(np.float32), agent.desc_std.astype(np.float32),
            w("backbone.0.weight"), w("backbone.0.bias"),
            w("backbone.2.weight"), w("backbone.2.bias"),
            w("pi_head.weight"), w("pi_head.bias"),
            w("mu_head.weight"), w("mu_head.bias"),
            w("log_sigma_head.weight"), w("log_sigma_head.bias"),
        ):
            f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    print(f"[{name}] n_in={n_in} n_out={n_out} hidden={hidden} k={k} -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Esporta i checkpoint .pt in binario per il motore C++.")
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--out-dir", default="native/data/weights")
    args = p.parse_args()

    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.out_dir)

    names = list(AGENT_SPECS)
    ok, missing = 0, []
    for name in names:
        ckpt = weights_dir / f"{name}.pt"
        if not ckpt.exists():
            missing.append(name)
            continue
        export_agent(name, ckpt, out_dir / f"{name}.bin")
        ok += 1
    print(f"\nEsportati {ok}/{len(names)} agenti in {out_dir}/")
    if missing:
        print(f"Mancanti (nessun checkpoint in {weights_dir}/): {missing}")


if __name__ == "__main__":
    main()
