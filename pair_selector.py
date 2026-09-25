"""
pair_selector.py -- selettore O(1) eccitatore+risonatore dato il target descrittori
(fase 2, priorita' 2, punto 3 del prompt: da eseguire PRIMA del routing MDN/KNN
esistente in param_candidate.py).

KNN sullo stesso principio di knn_corpus.py (z-score sull'intero corpus + cKDTree,
nessun training gradient-based -- "comincia semplice", scelto sul MLP alternativo
proposto dal prompt perche' zero training/iperparametri e riusa una macchina gia'
validata nel progetto). Il corpus e' il CSV generato da gen_selector_dataset.py: una
riga per TARGET (reale o sintetico) coi 13 descrittori + un punteggio di errore
aggregato per ciascuna delle 28 coppie eccitatore x forma-risonatore (vedi quel modulo
per come e' calcolato). Al query time: k vicini piu' vicini nello spazio descrittori,
MEDIA dei loro 28 vettori di punteggio (piu' robusto del solo 1-NN sulla label
"coppia migliore" della singola riga piu' vicina -- sfrutta tutta l'informazione
raccolta in fase di generazione, non solo l'argmin di una riga), poi argmin -> coppia.
"""
import numpy as np
from scipy.spatial import cKDTree

from agents import (DESCRIPTOR_KEYS, EPS, EXCITER_PARAM_RANGES, RESONATOR_SHAPES,
                    _nan_safe_stats, _read_csv)

EXCITERS = sorted(EXCITER_PARAM_RANGES)
RESONATORS = sorted(RESONATOR_SHAPES)
PAIRS = [(e, r) for e in EXCITERS for r in RESONATORS]  # ordine canonico, 28 coppie


def pair_column(exciter, resonator):
    return f"score_{exciter}_{resonator}"


PAIR_COLUMNS = [pair_column(e, r) for e, r in PAIRS]


class PairSelectorKnn:
    """rows: righe con DESCRIPTOR_KEYS + PAIR_COLUMNS (vedi gen_selector_dataset.py).
    k: vicini mediati di default (override per-query in query()).

    bias_alpha (2026-09-15, richiesto dall'utente dopo aver osservato una preferenza
    marcata per pluck/chaos in modalita' auto): correzione anti-sbilanciamento sulla
    frequenza di vittoria per ECCITATORE, non su un singolo descrittore. Analisi su
    selector_dataset.csv (eta^2 di ogni descrittore contro best_exciter, 1468 righe):
    il migliore, roughness, spiega solo il 13.8% della varianza tra eccitatori
    (eta^2=0.138, "medio" per Cohen, non "forte"); tutti gli altri sono <0.04 -- NESSUN
    descrittore singolo separa bene la scelta (e' una decisione multivariata sulle 13
    dimensioni insieme), quindi niente regola a soglie su un descrittore (es. su
    attack_time: pluck vince con mediana 0.95, bow con 1.22 -- non "basso" come un
    intuito fisico suggerirebbe). Correzione scelta invece: riequilibrio per frequenza
    di vittoria osservata nel corpus (tecnica standard per classi sbilanciate),
    indipendente dal DESCRITTORE responsabile. peso_e = (freq_osservata_e /
    freq_uniforme)^bias_alpha, moltiplicato sul punteggio (costo, minore=meglio) delle
    coppie con quell'eccitatore -- >1 penalizza gli eccitatori sovrarappresentati
    (pluck/chaos), <1 favorisce i sottorappresentati (bow/blow). bias_alpha=0 (default):
    nessuna correzione, comportamento originale puramente guidato dall'errore."""

    def __init__(self, rows, k=30, bias_alpha=0.3):
        if not rows:
            raise ValueError("corpus selettore vuoto")
        self.rows = rows
        self.k = max(1, min(k, len(rows)))
        self.bias_alpha = bias_alpha
        X = np.array(
            [[float(r[key]) for key in DESCRIPTOR_KEYS] for r in rows], dtype=np.float32)
        self.mean, self.std = _nan_safe_stats(X)
        self.tree = cKDTree(np.nan_to_num((X - self.mean) / self.std))
        self.scores = np.array(
            [[float(r[c]) for c in PAIR_COLUMNS] for r in rows], dtype=np.float32)
        self.exciter_bias = self._build_exciter_bias(rows, bias_alpha)

    @staticmethod
    def _build_exciter_bias(rows, bias_alpha):
        """Vettore (28,) allineato a PAIRS: peso moltiplicativo per punteggio, uno per
        eccitatore (uguale per le 4 coppie che lo condividono). Frequenza con Laplace
        smoothing (+1) sulle vittorie osservate (colonna best_exciter) tra le righe del
        corpus -- coerente con qualunque split (train/held-out) sia stato passato."""
        counts = {e: 1 for e in EXCITERS}  # Laplace smoothing, mai zero
        for r in rows:
            e = r.get("best_exciter", "")
            if e in counts:
                counts[e] += 1
        total = sum(counts.values())
        uniform = 1.0 / len(EXCITERS)
        weight = {e: (c / total / uniform) ** bias_alpha for e, c in counts.items()}
        return np.array([weight[e] for e, _r in PAIRS], dtype=np.float32)

    def query(self, target: dict, k=None):
        """target: dict descrittore -> valore (chiavi mancanti = 0 dopo z-score, come
        KnnCorpus/Agent). Ritorna (exciter_name, resonator_shape, score_aggregato_vicini).

        MEDIANA dei vettori di punteggio dei k vicini (non la MEDIA usata in una prima
        versione, 2026-09-15): la validazione held-out (eval_pair_selector.py) ha
        mostrato che la media e' fragile a un singolo vicino "patologico" (target con un
        descrittore vicino a zero, denom=max(abs(t),1e-6) in gen_selector_dataset.py fa
        esplodere il suo score su quasi tutte le coppie, stesso artefatto gia' noto su
        rel_error_pct in criticita_modelli.md) -- con la media, k piu' alto significava
        PIU' esposizione a un vicino cosi', non meno (k=15/30 peggio di k=5 nei test). La
        mediana e' insensibile a un singolo vicino fuori scala."""
        kk = max(1, min(k if k is not None else self.k, len(self.rows)))
        x = np.array([target.get(key, 0.0) for key in DESCRIPTOR_KEYS], dtype=np.float32)
        xn = np.nan_to_num((x - self.mean) / self.std)
        if kk == 1:
            _, idx = self.tree.query(xn)
            idx = np.array([idx])
        else:
            _, idx = self.tree.query(xn, k=kk)
            idx = np.atleast_1d(idx)
        agg_scores = np.median(self.scores[idx], axis=0)
        adjusted = agg_scores * self.exciter_bias if self.bias_alpha else agg_scores
        best = int(np.argmin(adjusted))
        exciter, resonator = PAIRS[best]
        return exciter, resonator, float(agg_scores[best])


_cache = {}  # (csv_path, k, bias_alpha) -> PairSelectorKnn


def build_selector(csv_path, k=30, bias_alpha=0.3) -> PairSelectorKnn:
    """Costruisce (o ritorna dalla cache) il selettore per csv_path, come
    knn_corpus.build_corpus: il corpus resta in memoria per tutta la sessione."""
    key = (str(csv_path), k, bias_alpha)
    if key not in _cache:
        rows = _read_csv(csv_path)
        _cache[key] = PairSelectorKnn(rows, k=k, bias_alpha=bias_alpha)
    return _cache[key]
