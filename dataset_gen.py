"""
dataset_gen.py — Generazione dataset sintetici per i modelli inversi (agents.py, sez.5
pipeline_multiPhiMo.txt / sez.4 pipeline.md).

Un CSV per agente (11 totali: 7 eccitatori + 4 forme di risonatore — vedi nota in
agents.py sul perche' il risonatore e' 4 agenti, non 1), schema compatibile con
Agent.train() in agents.py: colonne = DESCRIPTOR_KEYS + nomi parametri dell'agente
(agents.AGENT_SPECS[nome]), valori RAW (non normalizzati: la normalizzazione/log e'
gia' fatta dentro Agent, sia in training che in inferenza).

Eccitatori: CAMPIONAMENTO CONGIUNTO con un risonatore (pipeline sez.4 punto 3: "i
descrittori finali dipendono dall'accoppiamento" — decisione confermata con l'utente:
dataset congiunto, agenti separati). Per ogni riga: parametri eccitatore da Sobol (griglia
principale) + forma e parametri risonatore campionati a parte (rumore di accoppiamento,
uniforme, non Sobol) da resonator.PARAM_RANGES; il render passa l'eccitazione attraverso
resonator.apply_resonator(). L'agente eccitatore predice solo i propri parametri: il
risonatore e' marginalizzato nel training (Agent.train() legge solo DESCRIPTOR_KEYS +
param_names propri, ignora le colonne extra), l'agente impara la distribuzione dei
parametri eccitatore che raggiungono un target al variare del risonatore accoppiato
(coerente col "configurazioni diverse possono dare lo stesso target" del progetto). La
coppia risonatore usata PERO' viene comunque loggata nel CSV in colonne dedicate
(pair_resonator_shape, pair_res_<nome>, prefisso per evitare la collisione con l'unico
nome di parametro condiviso tra vocabolari, "density" in noise vs. risonatore) — non per
il training dell'agente eccitatore, ma per abilitare a runtime una tecnica alternativa/
complementare (lookup KNN congiunto su tutto il dataset, vedi
claude/runtime_architettura_realtime.md) che recuperi una combinazione
(exciter_params, resonator_shape, resonator_params) gia' coerente invece che i soli
parametri eccitatore marginalizzati. Colonne assenti per la shape non campionata in quella
riga (es. pair_res_aspect fuori da plate_rect) restano vuote (restval='' di DictWriter).

Risonatore: ciascuna delle 4 forme e' eccitata da un IMPULSO UNITARIO (stessa scelta del
demo `python resonator.py`), NON da un eccitatore: l'agente risonatore riceve in input
SOLO parametri di forma/materiale, mai parametri di eccitazione, quindi il suo target deve
dipendere solo da quelli — usare un eccitatore reale introdurrebbe una variabile nascosta
non modellata nell'input dell'agente. La risposta all'impulso caratterizza il sistema
modale indipendentemente da chi lo eccitera' poi a runtime.

Sampling: Sobol (scipy.stats.qmc) per la griglia principale di ciascun agente, stesso
schema log/lineare di agents.LOG_PARAMS (freq, hammer_stiffness, coupling_rate,
n_particles, size, thickness, density, stiffness, loss in log10, resto lineare) —
coerente con come l'agente li vede in training/inferenza. Il risonatore di accoppiamento
(solo per i dataset eccitatore) usa invece un RNG uniforme indipendente per riga: e'
rumore da marginalizzare, non la variabile che il dataset deve coprire in modo uniforme.

Filtro qualita' (sez.4 pipeline, punto 6): scarta il campione se il render e' silenzioso
(peak < SILENCE_PEAK) o contiene NaN/Inf (instabilita' numerica di bow/blow/chaos a
parametri estremi, o di combinazioni eccitatore+risonatore estreme). NON scarta NaN nei
descrittori (pitch/formanti su suoni atonali: atteso — vedi README analyzer — gia'
gestito da Agent con nan_to_num).

Uso:
    python dataset_gen.py bow --n 5000 --out dataset/bow.csv
    python dataset_gen.py resonator_bar --n 3000 --out dataset/resonator_bar.csv
    python dataset_gen.py --all --n 5000 --out-dir dataset/

Non eseguito qui (vincolo di progetto): girarlo e mandare i log qui per la verifica.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from agents import (AGENT_SPECS, DESCRIPTOR_KEYS, EXCITER_PARAM_RANGES, LOG_PARAMS,
                    PITCH_LOCKED_EXCITERS, RESONATOR_SHAPES)
from analyzer import analyze_signal
from exciters import generate as exciter_generate, SR
from resonator import apply_resonator, fundamental_freq, PARAM_RANGES as RESONATOR_PARAM_RANGES

SILENCE_PEAK = 1e-4
PITCH_TOL_CENTS = 20.0  # 2026-09-24: etichetta pitch = misura MPM se entro +-20c da freq, altrimenti NaN (mascherata)
# forme di risonatore campionate per il pairing dei dataset ECCITATORE (2026-09-21): default = tutte;
# `--pair-shapes tube soundboard chaotic` genera i dataset supplementari `<eccitatore>_extra.csv`
# letti da knn_corpus.build_joint_locked_corpus accanto al dataset principale.
_PAIR_SHAPES = list(RESONATOR_SHAPES)
RESONATOR_DURATION = 12.0  # s, risposta all'impulso -- 2026-09-14: portato a 12s (margine sopra il T60 max 10s
# dopo il fix di resonator._mode_dampings) per i soli dataset resonator_*, non usato
# per il render coupled degli eccitatori (che ha una propria durata in exciters.py)


# ---------------- sampling (Sobol, log-space per LOG_PARAMS) ----------------

def _sample_params(param_ranges, n, seed):
    names = list(param_ranges)
    sampler = qmc.Sobol(d=len(names), scramble=True, seed=seed)
    unit = sampler.random(n)
    samples = []
    for row in unit:
        params = {}
        for name, u in zip(names, row):
            lo, hi = param_ranges[name]
            if name in LOG_PARAMS:
                lo_l, hi_l = np.log10(lo), np.log10(hi)
                params[name] = float(10 ** (lo_l + u * (hi_l - lo_l)))
            else:
                params[name] = float(lo + u * (hi - lo))
        samples.append(params)
    return samples


def _sample_one(param_ranges, rng):
    """Campiona UN punto uniforme (non Sobol: qui e' rumore di accoppiamento da
    marginalizzare, non la griglia principale del dataset), stesso schema log/lineare
    di LOG_PARAMS usato da _sample_params."""
    params = {}
    for name, (lo, hi) in param_ranges.items():
        if name in LOG_PARAMS:
            lo_l, hi_l = np.log10(lo), np.log10(hi)
            params[name] = float(10 ** rng.uniform(lo_l, hi_l))
        else:
            params[name] = float(rng.uniform(lo, hi))
    return params


# ---------------- render + validita' ----------------

def _render(agent_name, params, rng=None):
    if agent_name.startswith("resonator_"):
        shape = agent_name[len("resonator_"):]
        n = int(RESONATOR_DURATION * SR)
        impulse = np.zeros(n, dtype=np.float64)
        impulse[0] = 1.0
        return apply_resonator(impulse, shape=shape, **params), SR, None
    # eccitatore: campionamento congiunto col risonatore condiviso (vedi docstring in
    # cima al file). shape/res_params sono marginalizzati per il training dell'agente
    # eccitatore, ma restituiti come pairing per essere loggati (colonne pair_*).
    raw, sr = exciter_generate(agent_name, **params)  # params include freq (colonna, non parametro agente)
    shape = str(rng.choice(_PAIR_SHAPES))
    res_params = _sample_one(RESONATOR_PARAM_RANGES[shape], rng)
    pairing = {"pair_resonator_shape": shape}
    pairing.update({f"pair_res_{k}": v for k, v in res_params.items()})
    f0 = params["freq"] if agent_name in PITCH_LOCKED_EXCITERS else None
    return apply_resonator(raw, shape=shape, f0=f0, **res_params), sr, pairing


def _is_valid(audio):
    if not np.all(np.isfinite(audio)):
        return False
    return float(np.max(np.abs(audio))) >= SILENCE_PEAK


# ---------------- generazione di un dataset (un agente) ----------------

def generate_dataset(agent_name, n, out_path, seed):
    if agent_name not in AGENT_SPECS:
        raise ValueError(f"agente sconosciuto: {agent_name} (validi: {list(AGENT_SPECS)})")
    param_ranges = AGENT_SPECS[agent_name]
    param_names = list(param_ranges)
    locked = agent_name in PITCH_LOCKED_EXCITERS
    # 2026-09-24: freq degli eccitatori intonati = pitch target a runtime: campionata (log, Sobol) ma
    # NON e' un parametro dell'agente -> colonna a parte, fuori da param_names.
    samp_ranges = dict(param_ranges)
    if locked:
        samp_ranges["freq"] = EXCITER_PARAM_RANGES[agent_name]["freq"]
    samples = _sample_params(samp_ranges, n, seed)
    # stream separato dal Sobol dell'agente: campiona SOLO il risonatore di accoppiamento
    # per i dataset eccitatore (ignorato per i dataset resonator_*, vedi _render).
    pairing_rng = np.random.default_rng(seed + 1_000_000)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = DESCRIPTOR_KEYS + param_names + (["freq"] if locked else []) + (["pitch_raw"] if agent_name in EXCITER_PARAM_RANGES else [])
    is_exciter = not agent_name.startswith("resonator_")
    # 2026-09-16d (fix stima f0): per resonator_* l'etichetta "pitch" non arriva piu'
    # dalla stima acustica (bar/piastre/membrana sono inarmoniche per costruzione --
    # vedi resonator.fundamental_freq), ma dal fondamentale analitico, noto esatto.
    resonator_shape = None if is_exciter else agent_name[len('resonator_'):]
    if is_exciter:
        # unione dei nomi parametro su tutte le forme (es. "aspect" solo in plate_rect):
        # colonne fisse indipendenti dalla shape campionata riga per riga.
        all_res_names = sorted({n for r in RESONATOR_PARAM_RANGES.values() for n in r})
        fieldnames = fieldnames + ["pair_resonator_shape"] + [f"pair_res_{n}" for n in all_res_names]
    n_written = n_discarded = 0
    log_every = max(1, n // 20)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for i, params in enumerate(samples):
            try:
                audio, sr, pairing = _render(agent_name, params, pairing_rng)
                if not _is_valid(audio):
                    raise ValueError("render silenzioso o non finito (NaN/Inf)")
                d = analyze_signal(audio, sr, extract_pitch=True, normalize=False)
                if resonator_shape is not None:
                    d["pitch"] = fundamental_freq(resonator_shape, **params)
                elif is_exciter:
                    d["pitch_raw"] = d.get("pitch", float("nan"))
                    p = d["pitch_raw"]
                    ok = locked and p is not None and np.isfinite(p) and p > 0 and \
                        abs(1200.0 * np.log2(p / params["freq"])) <= PITCH_TOL_CENTS
                    d["pitch"] = float(p) if ok else float("nan")
            except Exception as e:
                print(f"SCARTATO [{agent_name}] sample {i}: {e}", file=sys.stderr)
                n_discarded += 1
                continue
            row = {k: d.get(k) for k in DESCRIPTOR_KEYS}
            if is_exciter:
                row["pitch_raw"] = d["pitch_raw"]
            row.update(params)
            if pairing:
                row.update(pairing)
            writer.writerow(row)
            n_written += 1
            if (i + 1) % log_every == 0:
                print(f"[{agent_name}] {i + 1}/{n}  scritti={n_written}  scartati={n_discarded}",
                      file=sys.stderr)

    print(f"[{agent_name}] fatto: {n_written} campioni validi, {n_discarded} scartati -> {out_path}",
          file=sys.stderr)
    return n_written, n_discarded


# ---------------- CLI ----------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Genera dataset sintetici (parametri -> descrittori) per uno o tutti gli 11 agenti.")
    p.add_argument("agent", nargs="?", choices=list(AGENT_SPECS),
                    help="nome agente (7 eccitatori o resonator_<bar|plate_rect|plate_circ|membrane>); omesso se --all")
    p.add_argument("--all", action="store_true", help="genera il dataset per tutti gli 11 agenti")
    p.add_argument("--n", type=int, default=5000, help="campioni per agente (default 5000)")
    p.add_argument("--out", default=None, help="CSV di output (solo per un singolo agente)")
    p.add_argument("--out-dir", default="dataset", help="cartella di output (con --all: <out-dir>/<agente>.csv)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pair-shapes", nargs="+", default=None, choices=list(RESONATOR_SHAPES),
                    help="solo dataset eccitatore: forme di risonatore accoppiate campionate (default tutte)")
    args = p.parse_args()
    if args.pair_shapes:
        _PAIR_SHAPES[:] = args.pair_shapes

    if args.all:
        for idx, name in enumerate(AGENT_SPECS):
            generate_dataset(name, args.n, Path(args.out_dir) / f"{name}.csv", args.seed + idx)
    else:
        if not args.agent:
            p.error("specificare un nome agente oppure --all")
        out = args.out or (Path(args.out_dir) / f"{args.agent}.csv")
        generate_dataset(args.agent, args.n, out, args.seed)
