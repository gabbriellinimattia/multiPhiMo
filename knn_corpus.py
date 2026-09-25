"""
knn_corpus.py -- corpus KNN in memoria per gli agenti la cui routing lo richiede
(punto 3 del prompt). In questo prototipo v1: SOLO i 4 agenti resonator_* (ibrido --
`loss` dalla MDN, tutti gli altri parametri dal KNN, vedi criticita_modelli.md sez.
"Routing finale inferenza per agente"). pluck/chaos restano MDN pura in v1 (nessun KNN
congiunto, vedi nota v2 nel prompt) -- non serve quindi costruire corpus per loro qui.

z-score sull'INTERO CSV dell'agente, niente split train/val (punto 3: "non c'e' piu'
bisogno dello split ... qui il corpus e' tutto il dataset"). cKDTree (scipy) per il
lookup del vicino piu' vicino, sub-millisecondo -- stessa trasformazione di
knn_baseline.py/knn_joint.py (usati in validazione offline con lo split), qui senza
split perche' a runtime non c'e' un held-out da rispettare.
"""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from agents import AGENT_SPECS, EPS, _nan_safe_stats, _read_csv, descriptor_keys_for
from resonator import PARAM_RANGES as RESONATOR_PARAM_RANGES


def _logp(arr, keys):
    """2026-09-24: pitch in log10 (come freq); NaN resta NaN, <=0 -> NaN (target assente = 0.0)."""
    if "pitch" in keys:
        i = keys.index("pitch")
        v = arr[..., i]
        arr[..., i] = np.where(v > 0, np.log10(np.maximum(v, EPS)), np.nan)
    return arr


class KnnCorpus:
    def __init__(self, agent_name, rows):
        self.agent_name = agent_name
        self.desc_keys = descriptor_keys_for(agent_name)
        self.param_names = list(AGENT_SPECS[agent_name])
        self.rows = rows
        X = _logp(np.array(
            [[float(r[k]) for k in self.desc_keys] for r in rows], dtype=np.float32), self.desc_keys)
        self.mean, self.std = _nan_safe_stats(X)
        self.tree = cKDTree(np.nan_to_num((X - self.mean) / self.std))

    def query(self, target: dict) -> dict:
        """target: dict nome_descrittore -> valore (chiavi mancanti trattate come 0 dopo
        z-score, coerente con Agent._desc_to_x). Ritorna i parametri PROPRI dell'agente
        (mai un risonatore/eccitatore del vicino -- qui il corpus e' sempre mono-agente,
        vedi nota v2 del prompt sul perche' il KNN congiunto non e' usato in v1)."""
        x = _logp(np.array([target.get(k, 0.0) for k in self.desc_keys], dtype=np.float32), self.desc_keys)
        xn = np.nan_to_num((x - self.mean) / self.std)
        _, idx = self.tree.query(xn)
        row = self.rows[int(idx)]
        return {p: float(row[p]) for p in self.param_names}


_cache = {}  # (csv_path, agent_name) -> KnnCorpus, costruito una sola volta per sessione


def build_corpus(csv_path, agent_name) -> KnnCorpus:
    """Costruisce (o ritorna dalla cache) il corpus KNN per agent_name, letto da
    csv_path. Il corpus intero resta in memoria per tutta la sessione (punto 3):
    chiamare piu' volte con la stessa coppia (csv_path, agent_name) e' economico."""
    key = (str(csv_path), agent_name)
    if key not in _cache:
        rows = _read_csv(csv_path)
        if not rows:
            raise ValueError(f"CSV vuoto o non trovato: {csv_path}")
        _cache[key] = KnnCorpus(agent_name, rows)
    return _cache[key]


class KnnJointLockedCorpus:
    """KNN congiunto vincolato a una forma di risonatore fissa (fase 2, 2026-09-15,
    risoluzione del confondimento pluck/chaos -- vedi criticita_modelli.md "Routing
    finale inferenza per agente" e knn_joint_locked.py per la validazione). Il vicino
    piu' vicino si cerca SOLO tra le righe del dataset dell'eccitatore che usano la
    forma richiesta, e restituisce sia i parametri PROPRI dell'eccitatore sia i
    parametri CONTINUI del risonatore accoppiato (pair_res_*) di quella riga -- mai la
    forma stessa, che resta sempre la selezione manuale di AgentManager (a differenza
    di knn_joint.py, usato solo in eval, dove la forma e' libera). Validato: vince
    nettamente su chaos (tutti i descrittori), migliora harmonic_tension/formanti su
    pluck con un costo modesto su altri descrittori. NON usare per noise (peggiora
    formanti/harmonic_tension senza risolvere il pitch, che e' un limite
    dell'analizzatore su materiale aperiodico, non del routing)."""

    def __init__(self, agent_name, resonator_shape, rows):
        if resonator_shape not in RESONATOR_PARAM_RANGES:
            raise ValueError(f"forma sconosciuta: {resonator_shape}")
        self.agent_name = agent_name
        self.resonator_shape = resonator_shape
        self.desc_keys = descriptor_keys_for(agent_name)
        self.param_names = list(AGENT_SPECS[agent_name])
        self.res_param_names = list(RESONATOR_PARAM_RANGES[resonator_shape])
        self.rows = [r for r in rows if r.get("pair_resonator_shape") == resonator_shape]
        if len(self.rows) < 10:
            raise ValueError(
                f"troppe poche righe per {agent_name}+{resonator_shape} (n={len(self.rows)})")
        X = _logp(np.array(
            [[float(r[k]) for k in self.desc_keys] for r in self.rows], dtype=np.float32), self.desc_keys)
        self.mean, self.std = _nan_safe_stats(X)
        self.tree = cKDTree(np.nan_to_num((X - self.mean) / self.std))

    def query(self, target: dict):
        """Ritorna (exciter_params, resonator_params) del vicino piu' vicino -- MAI la
        forma (fissa in ingresso)."""
        x = _logp(np.array([target.get(k, 0.0) for k in self.desc_keys], dtype=np.float32), self.desc_keys)
        xn = np.nan_to_num((x - self.mean) / self.std)
        _, idx = self.tree.query(xn)
        row = self.rows[int(idx)]
        exciter_params = {p: float(row[p]) for p in self.param_names}
        resonator_params = {p: float(row[f"pair_res_{p}"]) for p in self.res_param_names}
        return exciter_params, resonator_params


_joint_cache = {}  # (csv_path, agent_name, resonator_shape) -> KnnJointLockedCorpus


def build_joint_locked_corpus(csv_path, agent_name, resonator_shape) -> KnnJointLockedCorpus:
    """Costruisce (o ritorna dalla cache) il corpus KNN congiunto vincolato per
    (agent_name, resonator_shape), letto da csv_path (dataset dell'ECCITATORE, non del
    risonatore: serve pair_resonator_shape/pair_res_*)."""
    key = (str(csv_path), agent_name, resonator_shape)
    if key not in _joint_cache:
        rows = _read_csv(csv_path)
        if not rows:
            raise ValueError(f"CSV vuoto o non trovato: {csv_path}")
        # 2026-09-21: dataset supplementare con le forme nuove (tube/soundboard/chaotic), stesso schema
        extra = Path(csv_path).with_name(Path(csv_path).stem + "_extra.csv")
        if extra.exists():
            rows = rows + _read_csv(extra)
        _joint_cache[key] = KnnJointLockedCorpus(agent_name, resonator_shape, rows)
    return _joint_cache[key]
