"""diag_routing.py (2026-09-26) -- gli agenti seguono i descrittori? (routing reale di param_candidate, SPSA spenta)
Per ogni eccitatore, forma 'bar': 30 target "slider" (ogni descrittore indipendente, uniforme tra p5 e p95 del
dataset) e 30 target "raggiungibili" (righe vere del dataset). Per ciascuno: candidato -> render -> analisi.
distinct = n. di configurazioni diverse prodotte (su 30); corr = Spearman target/ottenuto per descrittore
(1 = segue, 0 = ignora); nmae = |ottenuto-target| / (p95-p5). MDN: mu_std = variazione delle medie della mistura
tra target (0 = uscita indipendente dall'ingresso), comps = componenti vincenti distinte."""
import csv, sys
import numpy as np
import torch
from scipy.stats import spearmanr
import agents
import param_candidate as PC
from exciters import generate
from resonator import apply_resonator
from analyzer.descriptors import analyze_signal

PC.SPSA_ELIGIBLE_EXCITERS = set()
SHAPE = "bar"
N = 30
rng = np.random.default_rng(3)
mgr = agents.AgentManager("weights")
out = []

for exc in agents.EXCITER_PARAM_RANGES:
    mgr.select_exciter(exc)
    mgr.select_resonator(SHAPE)
    pc = PC.ParamCandidateWorker(None, mgr, dataset_dir="dataset")
    with open(f"dataset/{exc}.csv") as fh:
        rows = list(csv.DictReader(fh))
    keys = [k for k in agents.descriptor_keys_for(exc) if k != "pitch"]
    D = np.array([[float(r[k]) if r[k] not in ("", "nan") else np.nan for k in keys] for r in rows])
    p5, p95 = np.nanpercentile(D, 5, axis=0), np.nanpercentile(D, 95, axis=0)
    pitches = np.array([float(r["pitch"]) for r in rows if r["pitch"] not in ("", "nan")])
    sets = {"slider": [], "reach": []}
    for _ in range(N):
        t = {k: float(rng.uniform(a, b)) for k, a, b in zip(keys, p5, p95)}
        t["pitch"] = float(rng.choice(pitches)) if len(pitches) else 220.0
        sets["slider"].append(t)
        r = rows[int(rng.integers(len(rows)))]
        t = {k: float(r[k]) for k in keys if r[k] not in ("", "nan")}
        t["pitch"] = float(r["pitch"]) if r["pitch"] not in ("", "nan") else 220.0
        sets["reach"].append(t)
    for name, targets in sets.items():
        cands, got, mus, comps = set(), [], [], set()
        for t in targets:
            pc._update_candidate(t)
            c = pc.candidate
            cands.add(tuple(round(v, 3) for v in list(c.exciter_params.values()) + list(c.resonator_params.values())))
            if exc not in PC.JOINT_LOCKED_EXCITERS:
                x = torch.from_numpy(mgr.exciter._desc_to_x(t)).float().unsqueeze(0)
                with torch.no_grad():
                    pi, mu, _ = mgr.exciter.model(x)
                comps.add(int(pi.argmax()))
                mus.append(mu.numpy()[0].ravel())
            raw, sr = generate(exc, duration=1.0, **c.exciter_params)
            y = apply_resonator(raw, shape=SHAPE, f0=c.f0, **c.resonator_params)
            g = analyze_signal(y, sr, extract_pitch=False, normalize=False)
            got.append([g.get(k, np.nan) for k in keys])
        T = np.array([[t.get(k, np.nan) for k in keys] for t in targets])
        G = np.array(got, dtype=np.float64)
        corr, nmae = [], []
        for j in range(len(keys)):
            m = np.isfinite(T[:, j]) & np.isfinite(G[:, j])
            cj = spearmanr(T[m, j], G[m, j])[0] if m.sum() > 5 and np.std(G[m, j]) > 0 else 0.0
            corr.append(0.0 if not np.isfinite(cj) else cj)
            nmae.append(np.nanmean(np.abs(G[m, j] - T[m, j])) / (p95[j] - p5[j] + 1e-12) if m.any() else np.nan)
        mu_std = float(np.mean(np.std(np.array(mus), axis=0))) if mus else np.nan
        line = (f"{exc:11s} {name:6s} distinct={len(cands):2d}/{N}  corr_med={np.median(corr):5.2f}  "
                f"nmae_med={np.nanmedian(nmae):5.2f}" + (f"  mu_std={mu_std:.3f} comps={len(comps)}" if mus else ""))
        print(line, file=sys.stderr, flush=True)
        print("    corr: " + " ".join(f"{k[:8]}={c:+.2f}" for k, c in zip(keys, corr)), file=sys.stderr, flush=True)
        out.append(line)

print("\n=== RIEPILOGO ===\n" + "\n".join(out), file=sys.stderr)
