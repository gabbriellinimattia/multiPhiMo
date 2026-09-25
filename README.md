# multiPhiMo

Sintetizzatore a modelli fisici pilotato da agenti IA specializzati per tipo di eccitatore
e risonatore: in ingresso riceve valori di **descrittori spettrali** e gli agenti scelgono i
parametri fisici che li producono (matching sui descrittori, non timbrico).

- **Plugin VST3 (C++)**: in [`native/`](native/) — installazione, uso e compilazione in
  [`native/README.md`](native/README.md). Le release pronte (macOS Apple Silicon) sono nella
  pagina *Releases* del repository.
- **Prototipo Python**: in radice (`main.py`, `gui.py`, `agents.py`, `exciters.py`,
  `resonator.py`, `analyzer/`, …), usato per generare i dataset, addestrare gli agenti ed
  esportare pesi e corpus per il plugin (`native/tools/`). La cartella `dataset/` non è nel
  repository (si rigenera con `dataset_gen.py`); i pesi addestrati sono in `weights/`.

Licenza ISC (vedi [`LICENSE`](LICENSE)); licenze di terze parti del plugin in
[`native/THIRD_PARTY_LICENSES.md`](native/THIRD_PARTY_LICENSES.md).

---

## Analizzatore descrittori (Python)

Implementa la sez.1 di `pipeline_multiPhiMo.txt`: da un render audio estrae
13 scalari costanti (mediana sui frame attivi del segnale, non serie
temporali) da usare come target/output nel loop dataset ↔ agenti.

## Install

```
pip install -r requirements.txt
```

## Uso

File singolo (stampa JSON su stdout):
```
python -m analyzer render.wav
python -m analyzer render.wav --no-pitch --normalize --out out.json
```

Batch su una cartella (CSV, una riga per file, log di avanzamento su stderr):
```
python -m analyzer path/to/renders/ --batch --out dataset_descriptors.csv
```

## Vettore di output

`spectral_centroid, spectral_spread, spectral_flatness, roughness,
harmonic_tension, formant_f1, formant_f2, formant_f3, mod_rate, mod_depth,
attack_time, decay_time, decay_capped, pitch (opzionale, --no-pitch per
disattivarlo)`

## Note / approssimazioni consapevoli

- **roughness**: modello Vassilakis (2001) su coppie di picchi spettrali
  (top 40 per ampiezza) — stesso modello citato nel doc.
- **harmonic_tension**: non è lo spiral-array di Chew (richiede contesto
  tonale nota-per-nota, non applicabile a texture sintetiche arbitrarie).
  È la norma del 6D Tonal Centroid (Harte et al. 2006) calcolato sul chroma
  dello spettro medio — stessa trasformata di `librosa.feature.tonnetz`.
  Valore deterministico e stabile, non tarato percettivamente: coerente col
  vincolo di progetto "matching solo sui descrittori, non timbrico".
- **formanti**: LPC/Levinson-Durbin su segnale ricampionato a 10kHz
  (prassi standard per stabilità numerica), mediana sui frame vocati.
  Su eccitatori molto rumorosi/atonali (noise, shaker) i formanti possono
  risultare poco significativi — atteso, non è un bug.
- **pitch**: autocorrelazione normalizzata con soglia di voicing 0.3;
  ritorna NaN sui suoni senza intonazione chiara (es. noise, chaos).
- **modulazione**: cerca il picco nello spettro dell'inviluppo (Hilbert) in
  banda 0.5–20 Hz. Su render brevi (0.5–1s, sez.4) la risoluzione in
  frequenza è limitata: per rate sotto ~1-2 Hz servono render più lunghi.
- Gli scalari spettrali (centroid/spread/flatness/roughness/tension) sono la
  **mediana sui frame con energia sopra -40dB rispetto al picco** del
  render, per ignorare code di silenzio senza perdere il decadimento reale.

## Cosa manca / da validare con i log

- Non eseguito né testato qui (per richiesta del progetto). Prima verifica
  utile: far girare su 2-3 render noti per famiglia di eccitatore (es. un
  pluck armonico, uno shaker rumoroso, un bow con vibrato) e controllare che
  i valori abbiano il segno/ordine di grandezza atteso.
- Se i formanti su segnali molto brevi (<50ms) danno sempre NaN, è il
  `frame_ms=25` di LPC che non trova materiale sufficiente — segnalalo nei
  log, si abbassa la finestra.
- Se `roughness`/`harmonic_tension` sembrano schiacciati su un range troppo
  stretto tra famiglie diverse, valutiamo una normalizzazione post-hoc prima
  di darli in pasto agli agenti (sez.5).
