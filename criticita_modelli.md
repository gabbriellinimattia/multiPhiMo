# Criticità modelli — stato per agente

Aggiornato: 2026-09-16. Metrica di riferimento: **NMAE** (MedAE / deviazione standard del
descrittore sull'intero dataset) — 0.1-0.3 = buono, >~0.5 = da tenere d'occhio, >1 = tetto
serio. La % da sola è inaffidabile per descrittori spesso vicini a zero (roughness,
mod_rate/depth) e non è più il riferimento principale, solo un'informazione accessoria.

## bow — v6, 20k campioni, hidden=64/K=5
- Sano: centroid (0.003), spread (0.024), roughness (0.001), harmonic_tension (0.091),
  formanti (0.08-0.18), pitch (0.000)
- Debole: **mod_depth (0.35)** — MDN e KNN pari, limite residuo non ancora spiegato
- Accettabile: mod_rate (0.22), attack_time (0.25), flatness (0.20)

## noise — v1, ~20k campioni, hidden=64/K=5
- Sano: centroid, spread, flatness, roughness, harmonic_tension, formanti (NMAE ≤0.18)
- **Tetto fisico, non un bug**: mod_rate (0.62), mod_depth (0.57), attack_time (0.99) —
  MDN e KNN quasi identici (es. mod_rate 0.620 vs 0.620 esatto) -> non risolvibile con più
  dati/capacità. `noise` ha solo 3 parametri (color/density/correlation), nessuno controlla
  esplicitamente inviluppo o gate temporale.

## strike — v1, 20k campioni, hidden=64/K=5
- Sano: la maggior parte, incluso decay_time (0.023, ottimo) e mod_rate/mod_depth
  (0.005/0.038, quasi perfetti — corpo percosso senza vibrato)
- Debole: flatness (0.41, MDN peggio di KNN 0.34)
- Nota tecnica, non urgente: attack_time/decay_time hanno MedAE identico (0.004989s) su
  MDN e KNN — sospetto artefatto di quantizzazione sulla griglia 5ms di `envelope_times`
  (stessa famiglia del bug già risolto su mod_rate/bin FFT). NMAE già decente (0.156/0.023),
  da affinare con interpolazione sub-frame solo se un giorno serve più precisione lì.

## resonator_plate_rect — v1, hidden=64/K=5
- Sano: centroid, spread, roughness, harmonic_tension, mod_rate, pitch (NMAE ≤0.1)
- **decay_time ottimo, MDN >> KNN**: 0.004 vs 0.035 — conferma che reintrodurre decay_time
  sui risonatori era la scelta giusta
- Debole: **flatness, MDN peggio di KNN**: 0.277 vs 0.065 — pattern opposto a bow (lì MDN
  vinceva). Da capire se è densità dataset o interazione parametri-flatness dei risonatori.
- attack_time = 0 (MedAE=0, NMAE=0.000) su entrambi i metodi — atteso: eccitazione a impulso,
  salita quasi istantanea, non un bug.

## resonator_bar — v1, hidden=64/K=5
- Sano: centroid, spread, roughness, harmonic_tension, mod_rate, pitch (NMAE ≤0.1)
- **decay_time ottimo, MDN >> KNN**: 0.002 vs 0.020 — stesso pattern di plate_rect
- Debole: **flatness**: 0.234 vs KNN 0.074; **formanti f1**: MDN 0.043 vs KNN 0.009 — KNN
  meglio anche qui
- attack_time = 0 su entrambi i metodi, stesso motivo di plate_rect (eccitazione impulsiva)

## resonator_plate_circ — v1, hidden=64/K=5
- Sano: centroid, spread, harmonic_tension, mod_rate, mod_depth, pitch (NMAE ≤0.1)
- **decay_time ottimo, MDN >> KNN**: 0.002 vs 0.022 — conferma pattern bar/plate_rect
- Debole: **flatness**: 0.221 vs KNN 0.072; **formant_f1**: MDN 0.042 vs KNN 0.011 — KNN
  meglio anche qui, 3° risonatore su 3 con lo stesso pattern
- attack_time = 0 su entrambi i metodi, stesso motivo (eccitazione impulsiva)

## resonator_membrane — v1, hidden=64/K=5
- Sano: centroid, spread, formanti, mod_rate, mod_depth, decay_time, pitch (NMAE ≤0.1)
- **decay_time ottimo, MDN >> KNN**: 0.002 vs 0.043 — 4° risonatore su 4 con lo stesso
  pattern
- Debole: **flatness**: 0.171 vs KNN 0.095 (gap più stretto che su bar/plate); roughness
  0.100 vs KNN 0.056; harmonic_tension 0.131 vs KNN 0.066
- formant_f1 qui sostanzialmente pari (MDN 0.017 vs KNN 0.016) — primo risonatore dove il
  gap sui formanti sparisce
- attack_time = 0 su entrambi, stesso motivo (eccitazione impulsiva)

## shaker — v1, 5k campioni, hidden=64/K=5
- Sano: centroid, spread, pitch (NMAE ≤0.02); harmonic_tension (0.126)
- **MDN batte KNN sui formanti**: f1 0.033 vs 0.087, f2 0.074 vs 0.147, f3 0.071 vs 0.178
  — pattern eccitatore (come bow), non risonatore
- flatness 0.193 vs KNN 0.225 — MDN leggermente meglio
- **Tetto fisico, non un bug (come noise)**: mod_rate (0.182 vs 0.191), mod_depth
  (0.433 vs 0.458), attack_time (0.542 vs 0.542, MedAE identico 0.1447s) — MDN/KNN quasi
  pari, eccitazione stocastica granulare senza inviluppo/vibrato esplicito nei parametri
- roughness NMAE sano (0.032) ma % fuorviante (med 71%, p95 38035%) — artefatto metrico
  già noto su valori vicino a zero, non un problema del modello

## pluck — v1, 5k campioni, hidden=64/K=5
- Sano: centroid, spread, mod_rate, mod_depth, pitch (NMAE ≤0.08); MDN legg. meglio KNN
- **attack_time: MDN nettamente meglio**: 0.050 vs KNN 0.100
- Debole, KNN batte MDN: **harmonic_tension** 0.549 vs KNN 0.369; **formanti** f1 0.354 vs
  0.237, f2 0.478 vs 0.327, f3 0.477 vs 0.357
- **decay_time male su entrambi**: 0.850 vs KNN 0.849, quasi identici — tetto
  fisico/difficoltà intrinseca del descrittore su pluck, non vantaggio di un metodo
- Causa (confermata con `analisi_pluck.py` su pluck.csv, correlazioni + eta^2): NON è
  confondimento da *forma* del risonatore accoppiato (eta^2 shape 0.001-0.035,
  trascurabile). `freq` (proprio di pluck) è il driver principale di formanti/
  harmonic_tension (r 0.33-0.59) ma spiega solo fino a ~35% della varianza; una quota
  consistente residua è legata ai parametri fisici CONTINUI del risonatore accoppiato,
  invisibili all'agente (`pair_res_size` r=0.28 su formant_f3, 0.21 su formant_f2;
  `pair_res_thickness`/`stiffness` r=0.05-0.25). `spectral_centroid` è un caso limite:
  dominato quasi solo dal risonatore (r fino a 0.245) con contributo ~nullo dei parametri
  propri di pluck — il suo NMAE basso (0.020) è probabile falso-successo (in eval il
  risonatore è fisso per riga, quindi il centroid finale dipende poco dalla scelta del
  modello). Il resto della varianza (non spiegata da freq+risonatore in lineare) è
  verosimilmente l'effetto non lineare/non monotono di `pluck_position` (notch di comb
  filter) e `dispersion` (allpass) — comb filter e allpass producono pattern periodici
  che una MDN piccola (hidden=64/K=5, 5k campioni) fatica a fittare, mentre KNN sfrutta
  il match locale esatto nello spazio a 13 descrittori. Stesso meccanismo di
  accoppiamento casuale vale per tutti gli eccitatori (non solo pluck, vedi
  `dataset_gen.py`), ma qui pesa di più perché l'impronta armonica propria di pluck (via
  comb filter) è debole rispetto all'influenza del corpo risonante — da confermare su
  blow/chaos.
- **Test soluzioni (2026-09-15, `compare_confound.py`, stesso split)**: KNN CONGIUNTO
  aiuta ma solo parzialmente (a differenza di chaos, vedi sotto, dove risolve quasi
  tutto): harmonic_tension 0.525→0.253, formanti 0.367→0.212 / 0.484→0.263 / 0.461→0.332
  — miglioramento reale ma con un costo su descrittori che NON dipendono dal risonatore:
  pitch peggiora (0.038→0.057), mod_depth peggiora (0.073→0.116), centroid legg. peggio
  (0.021→0.025) — atteso: il vicino sceglie una combinazione con un risonatore diverso da
  quello "giusto" per quella riga, quindi ciò che già andava bene con l'accoppiamento
  originale può degradare leggermente. decay_time INVARIATO (0.849→0.844, conferma che è
  un tetto fisico del descrittore su pluck, non del metodo). MDN CONDIZIONATA: effetto
  quasi nullo su tutto (harmonic_tension 0.525→0.518, formant_f1 0.367→0.334,
  formant_f2 0.484→0.454, formant_f3 leggermente peggio 0.461→0.468) — non giustifica la
  complessità aggiunta qui, a differenza del guadagno (comunque non vinto) su chaos.

## chaos — v1, 5k campioni, hidden=64/K=5
- Sano: centroid, spread, pitch, roughness (NMAE ≤0.02)
- **KNN batte MDN su quasi tutto, più netto che su pluck**: harmonic_tension 0.308 vs
  0.261; formanti f1 0.316 vs 0.219, f2 0.473 vs 0.264, f3 0.496 vs 0.292; attack_time
  0.102 vs 0.051; mod_rate 0.016 vs 0.001; mod_depth 0.005 vs 0.003 — pattern coerente con
  la causa trovata su pluck (accoppiamento a risonatore casuale non visibile
  all'agente), qui più marcato: probabile per l'estrema sensibilità del generatore
  caotico (bifurcation/x0) a piccole variazioni di parametri, che rende la mappatura
  target->parametri ancora più non liscia di pluck
- flatness quasi pari (0.198 vs 0.203)
- formanti valutati solo su 1411/2000 righe (71%) — 29% dei render chaos non hanno
  formanti rilevabili via LPC, probabile spettro troppo denso/rumoroso in quei casi
- **Test soluzioni (2026-09-15, `compare_confound.py`, stesso split)**: KNN CONGIUNTO
  (`knn_joint.py`, opzione 3 — il vicino più vicino restituisce anche il SUO risonatore
  invece di essere forzato su quello della riga di test) vince nettamente su tutta la
  linea: flatness 0.198→0.067, harmonic_tension 0.308→0.068, formanti 0.316-0.496→
  0.108-0.135, mod_rate/mod_depth→~0.000, attack_time 0.102→0.034. MDN CONDIZIONATA
  (`--condition-on-resonator`, opzione 2 — shape+parametri del risonatore accoppiato
  come input extra) dà invece un risultato misto: harmonic_tension/formant_f2/f3
  migliorano ma formant_f1 peggiora (0.316→0.347) e soprattutto mod_rate/mod_depth/
  attack_time PEGGIORANO (0.016→0.105, 0.005→0.028, 0.102→0.119) — probabile perché
  l'input quasi raddoppia (11 feature extra) a parità di capacità (hidden=64/K=5, 5k
  campioni), sottraendo capacità ai descrittori già ben appresi.

## Decisione confondimento pluck/chaos (2026-09-15)
Confermato su entrambi: KNN congiunto (opzione 3) batte la MDN condizionata (opzione 2),
che aiuta poco o nulla e su chaos peggiora mod_rate/mod_depth/attack_time — non
implementarla oltre il prototipo. Il guadagno del KNN congiunto è molto più netto su
chaos (confondimento quasi totale) che su pluck (parziale, con piccolo costo su
pitch/mod_depth/centroid che il KNN congiunto sacrifica cambiando risonatore). Scelta
consigliata per il routing runtime su questi due agenti: KNN congiunto SOLO per
harmonic_tension/formant_f1-3 (dove vince chiaramente su entrambi), MDN base per tutto
il resto (dove è pari o migliore) — routing per descrittore, non un metodo unico per
agente. `agents.py --condition-on-resonator` resta nel codice ma non va usato in
produzione per pluck/chaos allo stato attuale.

## blow — v1, 5k campioni, hidden=64/K=5
- Sano: centroid, spread, harmonic_tension, formanti (f1-f3: 0.077-0.124), flatness
  (0.137) — MDN batte KNN su tutto, pattern eccitatore pulito (come bow/shaker)
- Debole: mod_rate (0.198), mod_depth (0.210) — MDN comunque leggermente meglio di KNN
  (0.195/0.223, quasi pari) — simile a bow, non un tetto fisico netto come noise/shaker
- attack_time 0.103, MDN meglio di KNN (0.113)

## Tutti gli 11 agenti allenati — quadro riassuntivo MDN vs KNN
Eccitatori "puliti" (MDN batte KNN su formanti/harmonic_tension): bow, shaker, blow (3/7).
Eccitatori anomali (KNN batte MDN su formanti/harmonic_tension, causa nota: accoppiamento
a risonatore casuale non visibile all'agente, vedi sezione pluck): pluck, chaos (2/7).
Tetto fisico su mod_rate/mod_depth/attack_time (MDN≈KNN, non risolvibile): noise, shaker
(2/7). Risonatori: pattern opposto e speculare su tutti e 4 — KNN batte MDN su
flatness/formanti, MDN >> KNN su decay_time -- risolto con l'ibrido per parametro (vedi sezione sotto).

## Osservazione aperta — CONCLUSA (4/4 risonatori)
flatness: KNN batte MDN su tutti e 4 i risonatori (bar, plate_rect, plate_circ,
membrane), gap decrescente con la complessità geometrica (plate_circ 3x, membrane 1.8x).
formant_f1: gap KNN>MDN su bar/plate_rect/plate_circ, ma pari su membrane. Pattern
opposto a bow (dove MDN vince su quasi tutto) sistematico sui risonatori, non un bug
isolato né densità dataset (si attenua ma non sparisce con più superfici testate) —
probabile limite strutturale della mistura MDN su descrittori a bassa dinamica sui
risonatori (flatness/formanti variano poco tra parametri vicini, KNN locale li cattura
meglio del regressore globale). decay_time è l'opposto speculare: MDN >> KNN su tutti e
4, netto e stabile — non serve altro intervento lì. Non bloccante per procedere con gli
eccitatori mancanti.

## Ibrido risonatori (loss da MDN, resto da KNN) -- validato 2026-09-15 (4/4)
Testato con `compare_hybrid_resonator.py`: parametri da KNN tranne `loss`, preso dalla
MDN (loss controlla decay_time/mod_depth, gli altri la forma spettrale -- valido qui,
non su pluck/chaos, perche' i parametri risonatore mappano 1:1 sul render). Risultato
uniforme sui 4 risonatori: l'ibrido eredita il meglio di entrambi -- decay_time e
mod_depth quasi identici a MDN (0.002-0.004 e 0.005-0.011, contro 0.020-0.060 di KNN
puro), harmonic_tension/formanti quasi identici a KNN (il migliore dei tre, MDN e'
2-5x peggio). Unico compromesso: flatness a meta' strada (0.117-0.161) tra MDN
(0.171-0.277, peggiore) e KNN puro (0.065-0.095, migliore) -- `loss` influenza anche la
flatness, quindi prenderlo da MDN reintroduce un po' del suo errore li', ma resta molto
meglio di MDN puro. Decisione: adottare l'ibrido in produzione su tutti e 4 gli agenti
resonator_* al posto sia di MDN puro che di KNN puro -- nessun descrittore peggiora
rispetto al migliore dei due metodi originali.

## Routing finale inferenza per agente (2026-09-15)
- bow, noise, strike, shaker, blow: MDN base su tutti i descrittori (eccitatori "puliti",
  MDN pari o migliore di KNN ovunque; mod_rate/mod_depth/attack_time su noise/shaker sono
  un tetto fisico condiviso da entrambi i metodi, non un argomento per KNN).
- pluck, chaos: routing per descrittore -- KNN congiunto (`knn_joint.py`, il vicino
  restituisce anche il proprio risonatore accoppiato) SOLO su harmonic_tension/
  formant_f1-3; MDN base su tutto il resto (pitch, mod_rate/depth, centroid/spread,
  attack_time). decay_time resta un tetto fisico su pluck indipendentemente dal metodo.
  `--condition-on-resonator` non adottato (vedi Decisione sopra).
- resonator_bar/plate_rect/plate_circ/membrane: ibrido per parametro -- `loss` dalla MDN,
  tutti gli altri parametri dal KNN (vedi sezione Ibrido sopra) -- sostituisce sia MDN
  puro che KNN puro su tutti e 4.

Nota indipendente da questa scelta (da `runtime_architettura_realtime.md`): a runtime,
`mod_rate`/`decay_time` non si stabilizzano entro il budget 0.2-0.5s di raffinamento
SPSA-live su nessun agente -- vanno serviti dalla stima one-shot sopra (MDN/ibrido/KNN a
seconda dell'agente) senza aspettarsi che il loop di raffinamento li migliori, qualunque
sia il metodo scelto per la stima iniziale.

## Fix sistemici già applicati (validi per tutti gli agenti, non da ripetere)
- `decay_time` escluso dal target tranne pluck/strike/resonator_* (ridondante con
  attack_time per eccitatori sostenuti: decay ≈ durata_render − attack quando il segnale
  non decade sotto floor_db entro la finestra).
- `resonator.py`: `loss` ricalibrato (prima inerte su tutto il suo range), durata render
  risonatori portata a 12s.
- `descriptors.py`: `spectral_flatness` gated in frequenza (-60dB dal picco per-frame,
  prima dominata dal floor numerico alle alte frequenze); `modulation()` analizza solo
  dopo il picco RMS dell'inviluppo (`envelope_times`), non più tutto il buffer incluso il
  transiente d'attacco, con interpolazione parabolica sub-bin per `mod_rate`; `roughness`
  gated sui frame attivi (`_active_frames`), coerente con centroid/spread/flatness.
- `agents.py`: `LOG_DESCRIPTORS` — `attack_time`/`decay_time` in log10 prima della
  normalizzazione, come già fatto per i parametri (`LOG_PARAMS`), mai per i descrittori.
- Metrica: NMAE aggiunta a `eval_agent.py`/`knn_baseline.py` accanto a %/MedAE.

## noise — aggiunta intonazione (fase 2, 2026-09-15)
Aggiunta risonanza tonale opzionale (freq/tone_amount/tone_q, riusa _resonant_filter) a
noise() per dare a quell'eccitatore un parametro di frequenza fondamentale, prima
assente. Verificato con test d'ascolto mirato (render_noise_tone_test.py) + verifica
strumentale (check_tone_pitch_detection.py) su render "puliti" (nessun risonatore
accoppiato): FUNZIONA acusticamente e strumentalmente, pitch rilevato quasi esatto
(220.5/882.0Hz attesi 220/880) quando tone_q e' alto; a tone_q=0 (risonanza larga) il
tracker fallisce e sbatte sul ceiling (~2000Hz).

Nel dataset reale pero' (con risonatore accoppiato casuale, dataset_gen.py) la
correlazione pitch<->freq resta ~0 (corr=-0.09/-0.11) ANCHE nel regime migliore
(tone_amount>=0.7, tone_q>=0.7: ancora 51% delle righe incollate al ceiling) --
check_tone_q_filter.py conferma che tone_q non e' la causa dominante: e' lo STESSO
confondimento da risonatore accoppiato casuale gia' diagnosticato su pluck/chaos (il
risonatore e' invisibile all'agente eccitatore), qui piu' severo perche' la
"intonazione" di noise e' rumore filtrato (periodicita' debole ciclo-per-ciclo), quindi
il risonatore casuale la sovrasta piu' facilmente di un vero fondamentale armonico.

Soluzione nota (KNN congiunto, gia' validata su pluck/chaos in eval) NON applicabile nel
runtime v1 attuale: param_candidate.py lo dice esplicitamente ("pluck/chaos NON usano
KNN congiunto in questo prototipo v1... rischierebbe l'indipendenza
eccitatore/risonatore di AgentManager, non risolto qui"). Quindi: il meccanismo e'
completo e corretto lato sintesi, ma il matching pitch via MDN one-shot resta
inaffidabile su noise nel runtime attuale, ESATTAMENTE come gia' vero per
harmonic_tension/formanti su pluck/chaos -- non un problema nuovo, la stessa
limitazione architetturale nota, che la selettrice di coppia (fase 2 priorita' 2)
affronterebbe direttamente scegliendo eccitatore+risonatore insieme invece di
marginalizzare il risonatore.

## Range freq allargati (fase 2, priorita' 1, 2026-09-15)
bow/blow/pluck/strike/chaos: range freq allargati sulla base dei percentili reali per
categoria (real_descriptors_full.csv, tutto il corpus ~/Desktop/sample, non un campione
casuale globale -- vedi analyze_real_ranges.py). shaker invariato (gia' copriva la
distribuzione osservata). Validato: nessuna regressione offline su bow/blow/pluck/strike
(eval_agent.py/knn_baseline.py sostanzialmente identici a prima); lieve regressione su
chaos mod_rate/mod_depth (MDN 0.016->0.083, 0.005->0.023, resta comunque <0.1, non
allarmante). Pavimento su suoni reali migliorato: best-of-28 su real_eval_allpairs.csv
sceso da 0.889 (riferimento pre-modifica) a 0.642 media / 0.524 mediana.

## Routing KNN congiunto vincolato per pluck/chaos (fase 2, priorita' 1, 2026-09-15)
AGGIORNAMENTO: il limite architetturale descritto sopra ("pluck/chaos NON usano KNN
congiunto in questo prototipo v1") e' ora RISOLTO in produzione. Implementata variante
"shape-locked" del KNN congiunto (knn_corpus.py: KnnJointLockedCorpus /
build_joint_locked_corpus): la ricerca del vicino e' ristretta alle righe di training
con la stessa forma di risonatore gia' selezionata manualmente dall'utente (rispetta
il contratto AgentManager: la forma resta sempre una scelta indipendente, mai
un'inferenza), e il vicino restituisce eccitatore+parametri continui del proprio
risonatore accoppiato (mai la forma).

param_candidate.py: nuova costante JOINT_LOCKED_EXCITERS = {"pluck", "chaos"}.
_update_candidate() instrada questi due eccitatori sul KNN vincolato, con fallback
automatico su MDN+ibrido (_mdn_hybrid_route, logica pre-esistente invariata) in caso
di eccezione. bow/blow/strike/shaker/noise restano su MDN+ibrido.

noise ESPLICITAMENTE ESCLUSO da JOINT_LOCKED_EXCITERS: test di validazione
(knn_joint_locked.py) mostra che il KNN congiunto peggiora formanti/harmonic_tension
di noise senza risolvere il problema di pitch (MedAE pitch resta 0 in entrambi i casi)
-- confermato che il problema di pitch di noise e' un limite dell'analyzer
(collasso del pitch tracker su materiale prevalentemente aperiodico), non il
confondimento da risonatore accoppiato casuale in dataset_gen.py.

Validazione pre-implementazione (knn_joint_locked.py, NMAE): chaos vince su tutti i
descrittori (harmonic_tension 0.315->0.093; formant_f1/f2/f3 0.302/0.410/0.448->
0.147/0.168/0.170). pluck vince nel complesso (harmonic_tension 0.464->0.286,
formanti migliorati) con costi modesti accettabili su pitch/mod_depth/attack_time.

Sincronizzato lo stesso routing nella logica di valutazione duplicata: eval_real_sounds.py
e eval_random_system.py ora importano JOINT_LOCKED_EXCITERS e usano lo stesso ramo
KNN-vincolato/fallback-MDN in _route() (nuova funzione _mdn_hybrid_route() estratta in
entrambi i file). eval_real_exhaustive.py eredita automaticamente (importa _route da
eval_real_sounds.py, nessuna modifica necessaria li'). Nessuna rigenerazione dataset
o retraining richiesta: e' solo una decisione di routing su dati/checkpoint esistenti.

## Validazione end-to-end routing KNN vincolato (fase 2, priorita' 1, 2026-09-15)
eval_real_exhaustive.py rigirato con lo stesso protocollo (70 file, 10/categoria,
28 coppie, seed 0, no SPSA) su real_eval_allpairs_jointknn.csv (routing KNN vincolato
pluck/chaos attivo) vs real_eval_allpairs.csv (routing MDN, gia' post-range-widening,
riferimento 0.642/0.524). Risultato: best-of-28 media 0.642->0.532, mediana 0.524->0.446
-- ulteriore miglioramento del pavimento su suoni reali, confermando a livello di
sistema end-to-end il guadagno gia' visto isolatamente su pluck/chaos con
knn_joint_locked.py. Priorita' 1 (range + routing pluck/chaos/noise) considerata chiusa.

(nota: la colonna "media-sulle-28" resta enormemente distorta, es. media~150000 --
artefatto noto di rel_error_pct su target vicini a zero, vedi nota metrica NMAE in
cima al documento; ignorare, guardare solo best-of-28 media/mediana)

## Selettore eccitatore+risonatore -- fase 2, priorita' 2 (2026-09-15, implementazione)

Nuovi file: `pair_selector.py` (modello + query O(1)), `gen_selector_dataset.py`
(genera il dataset di training), `eval_pair_selector.py` (validazione held-out).
Modificati: `param_candidate.py` (routing auto opzionale), `main.py`/`gui.py`
(modalita' auto in aggiunta alla selezione manuale, mai al posto di essa).

**Dataset**: una riga per TARGET (non per descrittore/coppia come
real_eval_allpairs.csv), reale (da `~/Desktop/sample`, tutte le categorie,
`--n-per-category` file ciascuna, riusa `eval_real_exhaustive._pick_per_category`) o
sintetico (`eval_random_system._random_target` su tutti i 13 descrittori, non solo i
rilevanti per una coppia). Per ogni target, punteggio calcolato per le 28 coppie con
la STESSA routing runtime (`eval_real_sounds._route`, mai SPSA: il selettore agisce a
monte del raffinamento SPSA-live, sceglie sul candidato one-shot/ibrido/KNN-vincolato).
Punteggio = MEDIA (non somma come `_spsa_loss`) degli errori relativi al quadrato sui
descrittori rilevanti per quella coppia -- la media e' necessaria perche' il numero di
descrittori rilevanti differisce di 1 (decay_time) tra coppie, la somma le renderebbe
non confrontabili. Coppie fallite -> punteggio `inf` (mai selezionabile).

**Modello**: KNN (cKDTree, z-score sull'intero corpus) sullo stesso principio di
`knn_corpus.py`, scelto sul MLP alternativo proposto dal prompt -- zero training/
iperparametri, riusa una macchina gia' validata nel progetto (resonator_*,
pluck/chaos), coerente con "comincia semplice". La query aggrega (vedi ESITO sotto per
la scelta MEDIANA vs media) i vettori di punteggio delle k righe piu' vicine prima
dell'argmin -- usa tutta l'informazione raccolta in generazione, non solo l'etichetta
di una riga, piu' robusto al rumore di un singolo campione.

**Integrazione**: `ParamCandidateWorker` ha ora `auto_pair` (default False,
retrocompatibile), `selector_csv`, `selector_k`, `auto_pair_min_hold` (2.0s, evita
sfarfallio tra coppie vicine in punteggio quando il target oscilla di poco). Se attivo,
`_maybe_select_pair()` interroga il selettore ad ogni ciclo di raffinamento e, se la
coppia cambia, chiama `manager.select_exciter`/`select_resonator` -- stessa API della
selezione manuale, contratto AgentManager invariato (coppia sempre esplicita una volta
scelta). `main.py`: `--exciter`/`--resonator` ora opzionali, omettendo entrambi si
entra in modalita' auto (input = tutti i 13 descrittori, non il sottoinsieme di una
coppia). `gui.py`: checkbox "Auto coppia" che disabilita i combobox manuali e mostra la
coppia attiva (puo' cambiare da sola); deselezionandola si torna immediatamente alla
coppia mostrata nei combobox.

**Nota laterale (non toccata qui, fuori scope)**: `_route()` in `eval_real_sounds.py`,
`eval_random_system.py` ed `eval_real_exhaustive.py` referenzia `exciter_agent`/
`resonator_agent` nel ramo `use_spsa=True` senza definirli in quello scope (definiti
solo dentro `_mdn_hybrid_route`) -- NameError se una qualunque di quelle CLI viene
lanciata con `--spsa`. Mai esercitato da `gen_selector_dataset.py` (chiama sempre
`use_spsa=False`), quindi non blocca questa priorita', ma da correggere se in futuro
serve `--spsa` in uno di quei tre script.

## Validazione selettore -- ESITO (fase 2, priorita' 2, 2026-09-15)

Dataset generato (gen_selector_dataset.py, --n-per-category 40 --n-synthetic 1200,
seed 0): 1468 target, 0 falliti, 9213.8s totali. eval_pair_selector.py (--test-frac
0.2, seed 0, 1175 train / 293 held-out):

**Bug metodologico trovato e corretto**: la prima query di PairSelectorKnn mediava
(np.mean) i punteggi dei k vicini -- fragile a un singolo vicino "patologico" (target
con un descrittore vicino a zero, denom=max(abs(t),1e-6) in gen_selector_dataset.py fa
esplodere il suo punteggio su quasi tutte le coppie, stesso artefatto gia' noto su
rel_error_pct, vedi nota in cima al documento). Con la media, k piu' alto = piu'
esposizione a un vicino cosi': risultato (invertito rispetto all'atteso) k=1 miglior
selettore, k=30 peggiore. La stessa fragilita' rendeva anche la % "gap casuale->oracolo"
in eval_pair_selector.py inutile se calcolata sulla media (dominata dagli stessi
outlier, ~100% per QUALUNQUE selettore per costruzione -- una media include sempre
anche le coppie peggiori, un selettore qualsiasi le evita quasi sempre, quindi batte
"casuale" quasi ovunque su questa metrica indipendentemente dalla sua qualita' reale).
Corretto in due punti: PairSelectorKnn.query() ora usa la MEDIANA dei punteggi dei k
vicini (insensibile a un singolo outlier), eval_pair_selector.py riporta il gap
recuperato sulla mediana come riferimento primario (quello sulla media resta stampato
per trasparenza, etichettato "ignorare").

**Risultato dopo il fix** (mediana, riferimento primario):
| k | achieved (mediana) | gap recuperato |
|---|---|---|
| casuale | 10.32 | -- |
| oracolo (28/28) | 0.588 | 100% |
| k=1 | 2.172 | 83.7% |
| k=5 | 1.776 | 87.8% |
| k=15 | 1.409 | 91.6% |
| **k=30** | **1.322** | **92.5%** |

Pattern ora monotono in k (piu' vicini = piu' robusto con la mediana, opposto al
comportamento con la media) -- il trend non e' ancora appiattito a k=30, k piu' alti
potrebbero recuperare ancora qualcosa, ma il guadagno per step si sta riducendo
(83.7->87.8->91.6->92.5) e 92.5% e' gia' un risultato solido; non esplorato oltre per
restare nello spirito "usa meno token possibile" -- riprovabile in qualunque momento
senza rigenerare il dataset (`eval_pair_selector.py --k 30 50 80`, secondi, nessuna
nuova sintesi, legge solo selector_dataset.csv gia' presente).

**Decisione**: k=30 adottato come default (pair_selector.py, param_candidate.py,
main.py, gui.py -- 5 punti). Priorita' 2 (selettore O(1) eccitatore+risonatore)
considerata chiusa: dataset generato, modello scelto e validato held-out (92.5% del gap
casuale->oracolo recuperato), integrato in param_candidate.py/main.py/gui.py in aggiunta
alla selezione manuale (mai al suo posto). Non ancora esercitato end-to-end in
main.py/gui.py dal vivo (nessuna esecuzione qui per vincolo di progetto) -- primo test
dal vivo lasciato all'utente quando vorra' provare la modalita' auto.

## Correzione sbilanciamento eccitatori nel selettore (fase 2, priorita' 2, 2026-09-15)

Segnalato dall'utente dopo un uso dal vivo della modalita' auto in gui.py: preferenza
marcata e non uniforme tra eccitatori.

**Diagnosi (nessuna nuova sintesi, solo analisi di selector_dataset.csv gia' generato)**:
distribuzione best_exciter (oracolo riga-per-riga, 1468 righe): pluck 28.1%, chaos
26.4%, shaker 14.9%, strike 12.3%, noise 9.6%, bow 4.6%, blow 4.2%. Ipotesi iniziale
dell'utente (regola su attack_time: basso->pluck/strike, alto->noise/bow, medio->
blow/chaos) verificata e SMENTITA dai dati: mediana attack_time tra le vittorie di
pluck e' 0.952 (non bassa), quella di chaos 1.200 (alta, non media), quella di bow
1.220 (la piu' alta, unico punto coerente con l'ipotesi). eta^2 di ogni descrittore
contro best_exciter (stessa metrica di `analisi_pluck.py`): il migliore, roughness,
spiega solo il 13.8% della varianza (eta^2=0.138, "medio" per Cohen); tutti gli altri
<0.04 -- NESSUN descrittore singolo separa bene la scelta, e' una decisione
multivariata sulle 13 dimensioni insieme. Scartata quindi una regola a soglie su un
descrittore (proposta iniziale dell'utente): approssimazione debole, rischio di
peggiorare il matching senza nemmeno risolvere lo sbilanciamento.

**Meccanismo adottato**: correzione per FREQUENZA di vittoria osservata per
eccitatore (non per descrittore) in `pair_selector.PairSelectorKnn`, nuovo parametro
`bias_alpha`. peso_e = (freq_osservata_e / freq_uniforme)^bias_alpha, moltiplicato sul
punteggio (costo) delle coppie con quell'eccitatore prima dell'argmin -- >1 penalizza
i sovrarappresentati (pluck/chaos), <1 favorisce i sottorappresentati (bow/blow).
bias_alpha=0 = comportamento originale invariato. Filettato come `--selector-bias-alpha`
in `param_candidate.py`/`main.py`/`gui.py`, stesso schema di `--selector-k`.

**Sweep held-out (eval_pair_selector.py --k 30 --bias-alpha 0 0.3 0.5 0.8 1.0)**:

| alpha | gap mediana | pluck | chaos | strike | bow | blow | shaker | noise |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 92.5% | 67.9% | 17.7% | 11.6% | 0.3% | 0.0% | 2.0% | 0.3% |
| 0.3 | 91.0% | 40.6% | 14.0% | 23.2% | 8.9% | 3.1% | 6.1% | 4.1% |
| 0.5 | 86.6% | 19.5% | 7.5% | 24.2% | 20.1% | 20.8% | 3.8% | 4.1% |
| 0.8 | 79.9% | 4.4% | 4.8% | 15.0% | 30.0% | 43.7% | 0.3% | 1.7% |
| 1.0 | 77.0% | 2.0% | 3.1% | 12.3% | 29.4% | 52.9% | 0.0% | 0.3% |

Osservazione: a bias_alpha=0 il selettore (mediana di k=30 vicini) sceglie pluck il
67.9% delle volte -- molto piu' della sua quota nell'oracolo riga-per-riga (28.1%): la
mediana su k=30 vicini premia l'eccitatore mediamente buono in un intorno, non quello
ottimo punto-per-punto, e amplifica il vantaggio di pluck/chaos (piu' flessibili per il
confondimento da risonatore accoppiato, vedi sezioni pluck/chaos sopra). La correzione
NON converge verso una distribuzione piatta oltre alpha~0.5: i pesi sono calibrati sulla
frequenza grezza riga-per-riga, non su quella che il selettore k=30 produce davvero, e
oltre quella soglia lo sbilanciamento si ROVESCIA (blow+bow all'82% combinato a
alpha=1.0, shaker mai scelto) invece di appiattirsi -- limite noto del meccanismo,
accettabile perche' l'intervallo utile (0-0.3-0.5) e' gia' stato individuato.

**Decisione**: bias_alpha=0.3 adottato come default (stesso pattern di k: 5 punti --
pair_selector.py x2, param_candidate.py, main.py, gui.py). Costo accuratezza minimo
(92.5%->91.0% di gap mediana recuperato), pluck scende da 68% a 41%, ogni eccitatore
compare almeno al 3%. Resta un parametro CLI regolabile a runtime senza rigenerare
nulla (`--selector-bias-alpha`), se in uso dal vivo la distribuzione risultasse ancora
poco soddisfacente.

## Fix troncamento in "Play continuo" -- durata render dinamica (2026-09-16)

Segnalato dall'utente: con "Play continuo" (trigger ogni 1s), ogni suono nuovo sembrava
troncare quello vecchio. Diagnosi (analisi statica, nessuna esecuzione): NON era un bug
del mixer polifonico (`play_engine.py._callback`, gia' corretto -- somma tutte le voci
attive, rimuove solo a fine buffer o oltre `max_voices`). La causa reale: ogni funzione
eccitatore in `exciters.py` ha una durata di default FISSA (bow/blow=1.5s,
strike/noise=1.0s, pluck=2.0s, shaker=1.5s, chaos=1.5s), mai passata esplicitamente da
`param_candidate.py`/`play_engine.py`; `apply_resonator()` produce un buffer della
STESSA lunghezza dell'eccitazione (`y = np.zeros_like(excitation)` in `resonator.py`).
Risultato: il decadimento del risonatore (`damping_times`, puo' essere di piu' secondi
a seconda di `loss`) o dell'eccitatore stesso (`decay_time` su pluck/shaker, fino a 4s)
veniva sempre tagliato di netto a quella durata fissa, indipendentemente dal target --
con trigger ogni 1s e durate di 1.0-2.0s, il taglio a fine buffer cade quasi in
coincidenza col trigger successivo, dando l'illusione che sia il nuovo suono a troncare
il vecchio.

**Fix** (`play_engine.py`, isolato, nessuna modifica a `exciters.py`/`resonator.py`):
nuova `_estimate_duration(cand)`, chiamata in `trigger()` prima del render. Stima la
durata minima necessaria come `max(default_dur_funzione, damping_times_risonatore * 3,
decay_time_eccitatore_se_presente)`, clippata tra il default originale (mai piu' corta
di prima) e un tetto `_MAX_DURATION=6.0s` (evita render/buffer eccessivi). Passata
esplicitamente come `duration=...` a `exciter_generate()` (nessun conflitto: `duration`
non fa mai parte di `cand.exciter_params`). Testato dal vivo dall'utente (2026-09-16):
funziona, troncamento risolto.

## Estrazione a finestre (curva nel tempo) -- fase 2, priorita' 3 punto 1 (2026-09-16)

Aggiunta `analyze_signal_windowed`/`analyze_file_windowed` a `analyzer/descriptors.py`
(riusa `analyze_signal` invariata su ogni finestra, nessuna funzione di analisi
duplicata) + `--windowed --win-s --hop-s` in `analyzer/__main__.py` (solo file singolo,
non `--batch`). Default win_s=0.5/hop_s=0.3, dentro il budget realtime 0.2-0.5s (vedi
`runtime_architettura_realtime.md`); `normalize` applicato una volta sul segnale intero,
non per finestra (altrimenti si perdono le differenze di livello relativo tra finestre);
ultima finestra scartata se piu' corta di `min_last_frac*win_s` (default 0.25s).

Validato dal vivo dall'utente (due file reali, `~/Desktop/sample`):
- pluck corto (~0.5-0.6s): 1 sola finestra (coda del file scartata, residuo <0.25s) --
  comportamento atteso, non un bug.
- bow sostenuto (~4.85s): 16 finestre, `t_start` 0.0->4.5 a passi di 0.3s esatti, ultima
  finestra parziale (0.35s) inclusa correttamente. Meccanismo di scorrimento confermato
  corretto.

Nella curva del bow sostenuto, visibili per-finestra gli stessi limiti gia' noti a
livello di intero-file (non nuovi, solo piu' evidenti finestra-per-finestra): `pitch`
satura al ceiling fmax=2000Hz in piu' finestre non consecutive (autocorrelazione che
fallisce localmente, non una vera nota a 2000Hz); `decay_time`/`attack_time` rumorosi e
quasi sempre `decay_capped=true` su un suono sostenuto (nessun vero decadimento dentro
una finestra di 0.5s) -- coerente con quanto gia' documentato sopra
(`runtime_architettura_realtime.md`) su mod_rate/decay_time che non si stabilizzano nel
budget 0.2-0.5s. Nessuna azione richiesta qui: e' il comportamento atteso dei
descrittori esistenti applicati a un segmento corto, il punto 1 di priorita' 3
riguardava solo il meccanismo di finestratura, non il fixing di questi limiti.

Priorita' 3, punto 1 (estrazione a finestre) considerato chiuso.

## Auto-trigger su cambio candidato -- fase 2, priorita' 3 punto 2 (2026-09-16)

Implementato in `param_candidate.py` (`ParamCandidateWorker._maybe_auto_trigger`,
`_candidate_param_distance`, costanti `AUTO_TRIGGER_THRESHOLD=0.15`,
`AUTO_TRIGGER_MIN_HOLD=1.0`), agganciato in `main.py` (`--auto-trigger`,
`--auto-trigger-threshold`, `--auto-trigger-min-hold`) e `gui.py` (checkbox "Auto play
(segue variazioni)", indipendente da "Play continuo" gia' esistente). Approssima
"sintesi che segue la curva" facendo scattare `play_engine.trigger()` automaticamente
quando il candidato cambia in modo sostanziale, aggiunta esplicita che si affianca al
trigger manuale (Invio/bottone Play), mai al suo posto -- zero modifiche a
`exciters.py`/`resonator.py`/`play_engine.py` (solo un timestamp nel log di
`play_engine.trigger()` per diagnosi, vedi sotto).

"Sostanziale" = primo candidato disponibile, o cambio di eccitatore/risonatore (sempre),
o distanza euclidea in spazio 0-1 per-parametro (stesso schema di `_param_to01`) sopra
`auto_trigger_threshold`.

**Bug trovato dal vivo dall'utente (prima versione, senza hold)**: con `bow` (eccitatore
SPSA-eligible) a target FERMO, il trigger scattava quasi ad ogni ciclo di raffinamento
(0.3s), non solo sui cambi reali del target -- log mostrava durate molto diverse tra
cicli consecutivi (1.50s vs 4.34s). Causa: `spsa_refine` riparte dal candidato
one-shot/ibrido (deterministico) ad ogni ciclo con perturbazioni CASUALI (nessun seed
fisso, `rng = rng or np.random.default_rng()`) e solo 4 iterazioni -- il rumore
stocastico del risultato finale, da solo, supera facilmente 0.15 di distanza normalizzata
anche senza alcun cambio di target (coerente con quanto gia' documentato in
`runtime_architettura_realtime.md` sulla varianza di SPSA a poche iterazioni).

**Fix**: `auto_trigger_min_hold` (default 1.0s, stesso pattern di `auto_pair_min_hold`
gia' usato per il selettore di coppia) -- un cambio "sostanziale" puo' tradursi in un
trigger solo se e' passato almeno questo tempo dall'ultimo auto-trigger. NON sostituisce
la soglia di distanza, la combina (entrambe le condizioni devono valere).

**Validazione dal vivo (log con timestamp aggiunto a `play_engine.trigger()`,
2026-09-16)**: a target fermo, intervalli tra trigger consecutivi 1.46-1.71s (mai sotto
1.0s) -- l'hold funziona, il rumore SPSA non puo' piu' far scattare piu' di un trigger al
secondo. L'intervallo supera leggermente 1.0s perche' il trigger successivo aspetta
anche il prossimo ciclo di raffinamento (0.3s) in cui la distanza torni sopra soglia, non
scatta esattamente all'istante in cui l'hold scade.

Non ancora chiuso: manca il riscontro percettivo dell'utente (la cadenza ~1.5s a target
fermo e la reattivita' su un cambio reale di slider) prima di considerare il punto 2
validato.

## Fix crackle audio (thread contesa GIL) + limiter + acquisizione audio-in (2026-09-16)

**Diagnosi crackle** (`test_audio_pipeline.py`, sez.0-7, preceduta da ricerca online sulle
cause note di crackle/gratta in audio realtime -- regole di Bencina su cosa evitare nel
callback realtime, denormali): confermato che gli eccitatori a loop per-campione
(bow/blow/strike/pluck/chaos) rendono fino a >1s e girano sullo stesso processo del
callback audio; `spsa_refine()` su `bow` (default) satura il 354.7% del suo periodo di
raffinamento (0.3s) -- contesa GIL quasi continua tra il thread di raffinamento e il
callback `sounddevice`, causa root del crackle. Denormali e costo del mixer esclusi
(sez.4/6, trascurabili).

**Fix**: `SPSA_PROBE_DURATION=0.5` in `param_candidate.py` (riduce il costo per singola
chiamata SPSA) + `candidate_process.py` (nuovo): tutto il calcolo candidato/SPSA/render
spostato su un processo OS separato (`multiprocessing`, contesto spawn) via
`CandidateProcessHandle`, comunicazione con `multiprocessing.Queue`; il processo del
callback audio non compete piu' per GIL/CPU con quel lavoro (il GIL e' per-processo, non
per-thread). Rewiring completo di `gui.py`/`main.py` su `CandidateProcessHandle` al posto
di `AgentManager`/`ParamCandidateWorker` diretti. **Validato dal vivo**: "ottimo, nessun
gratto" -- crackle risolto. Residuo: saturazione con 11 voci molto rapide (atteso, gia'
previsto da `clipping_test()` sez.5).

**Limiter**: feedforward, no lookahead, in `play_engine.py._callback` dopo il gain
manuale. `LIMITER_CEILING=0.85`, attack 5ms/release 60ms (smoothing esponenziale verso il
gain-target basato sul picco del buffer), clip finale [-1,1] come backstop. **Validato dal
vivo** con lo stesso scenario a 11 voci rapide: saturazione percepita eliminata.

**Acquisizione descrittori da audio in ingresso** (nuovo `audio_input.py`,
`AudioDescriptorSource`): due thread daemon indipendenti -- (1) cattura microfono
bloccante (`sd.InputStream`, no callback, nessun lock necessario: unico proprietario del
buffer scorrevole win_s=0.5/hop_s=0.3, stesse convenzioni di `analyze_signal_windowed`),
chiama `analyze_signal` ad ogni hop, swap lock-free su riferimento (`self._raw = ...`,
stesso idioma gia' in uso nel progetto); (2) interpolazione EMA piu' rapida
(`interp_period=0.03s`) verso `self._raw`, con costante di tempo `smoothing_ms`
regolabile da GUI/CLI (default 200ms), scrive su `descriptor_input.set_target(...)` --
il "fade smooth" richiesto tra un set di valori e il successivo. Scelto un thread (non un
processo separato) dopo aver misurato il costo reale di `analyze_signal` su una finestra
tipica (sez.8 di `test_audio_pipeline.py`, aggiunta per questo): 3.8-8.6% di un hop di
300ms, trascurabile, nessuna contesa GIL significativa attesa col callback audio.
Wiring: checkbox "Audio in (microfono)" + slider "Smoothing" in `gui.py` (indipendente da
OSC, non mutuamente esclusivo, stesso pattern di "Play continuo"/"Auto play"); flag
`--audio-in`/`--smoothing-ms` in `main.py`. **Validato dal vivo**: slider seguono
suono/voce con fade smooth atteso.

Tutte le modifiche di questa sezione verificate solo con `ast.parse` prima del test dal
vivo, come da vincolo di progetto (mai eseguito codice qui).

## inharmonicity — retrain 2026-09-16f + fix bias NaN->0 (2026-09-16g)

Primo training dopo l'aggiunta di `inharmonicity`/`spectral_rolloff` e la riscrittura
YIN del pitch (retrain completo degli 11 agenti, dataset rigenerati). `pitch` conferma
il fix (NMAE 0.002-0.06 su tutti gli agenti, contro il problema originale che aveva
motivato la riscrittura). `inharmonicity` risulta il descrittore piu' debole ovunque
(NMAE 0.18-0.60 a seconda dell'agente), anche nella colonna migliore (KNN sui
risonatori, gia' preferito dal routing di produzione — vedi "Routing finale inferenza
per agente" sopra, nessuna azione li': il gap MDN-vs-KNN su resonator_*/chaos non e'
una regressione, quei descrittori passano gia' da KNN in produzione).

**Causa trovata (analisi statica del codice, nessuna esecuzione)**: `agents.py`
(`Agent._desc_to_x`/`Agent.train`) e `knn_corpus.py` (`KnnCorpus`/
`KnnJointLockedCorpus`, sia costruzione che query) usavano `np.nan_to_num()` per i
descrittori NaN in ingresso, sostituendoli con 0.0. Innocuo per pitch/formanti (0 e'
gia' fuori range, sentinel chiaramente separabile dai valori reali) ma sbagliato per
inharmonicity: il suo range valido comincia proprio da 0.0 ("armonico perfetto", un
target legittimo e comune), quindi il sentinel collideva con un valore reale.
Ogni riga con inharmonicity NaN (40-65% dei campioni a seconda dell'agente, spesso
proprio i render PIU' inarmonici/rumorosi, dove f0 o i parziali non erano rilevabili)
veniva rietichettata come "armonico perfetto" — sia come input sia nel calcolo di
mean/std (fatto DOPO lo zero-fill in entrambi i file), sporcando la normalizzazione
esattamente nella zona di target piu' probabile. Bug condiviso da MDN e KNN
(nan_to_num identico in entrambi), quindi non spiega le differenze MDN-vs-KNN gia'
documentate sopra, ma probabilmente spiega perche' il pavimento di errore su
inharmonicity resta alto anche nel metodo migliore per ciascun agente.

**Fix**: `NAN_SENTINEL = {"inharmonicity": -1.0}` + `_fill_nan_descriptors()` in
`agents.py` (sentinel per-descrittore, -1.0 e' sempre fuori range per inharmonicity
che e' >=0; tutti gli altri descrittori restano a 0.0, comportamento invariato).
Sostituito `np.nan_to_num()` in tutti i punti di produzione che lo usavano: `agents.py`
(x2), `knn_corpus.py` (x4, entrambe le classi), `pair_selector.py` (x2, selettore
coppia auto). Aggiornato anche `knn_baseline.py` (x1, baseline di valutazione ancora in
uso) per coerenza nei confronti futuri. Lasciati intoccati gli script one-off gia'
conclusi (`compare_confound.py`, `compare_hybrid_resonator.py`, `knn_joint.py`,
`knn_joint_locked.py`, `_apply_joint_routing.py`): non importati da nessun codice di
produzione, le loro conclusioni documentate riguardano il set di 13 descrittori
precedente a inharmonicity. Verificato solo con `ast.parse`, come da vincolo di
progetto (mai eseguito codice qui).

**Da fare**: retrain degli 11 agenti (stessi comandi gia' usati, vedi sezioni sopra) +
`eval_agent.py`/`knn_baseline.py` per confermare che l'errore su inharmonicity scende
in modo consistente su tutti gli agenti, non solo su quelli dove gia' vinceva KNN.

## Retrain di conferma 2026-09-16h — il fix 2026-09-16g ha PEGGIORATO le cose

Retrain completo degli 11 agenti dopo il fix del sentinel NaN->-1.0 (2026-09-16g,
stessi comandi/dataset di prima). Risultato: `inharmonicity` e' peggiorato quasi
ovunque, sia MDN sia KNN, in alcuni casi in modo netto (resonator_bar KNN NMAE
0.082->0.170, piu' che raddoppiato; resonator_plate_rect MDN 0.409->0.431 e KNN
0.139->0.325). Solo poche eccezioni migliorano leggermente (strike MDN 0.422->0.340,
noise MDN 0.384->0.336) — non un pattern coerente, probabile rumore di training
(seed diverso lato inizializzazione pesi, il dataset e lo split sono identici).

**Causa della regressione** (rianalisi del codice): il fix 2026-09-16g iniettava il
sentinel (-1.0) PRIMA di calcolare mean/std di normalizzazione — esattamente lo
stesso errore strutturale del vecchio `np.nan_to_num()` che doveva risolvere, solo
spostato a un valore diverso. Con 40-65% delle righe a un valore fisso (fosse 0.0 o
-1.0 non cambia il meccanismo), mean/std calcolati su QUELLA colonna vengono
spostati/gonfiati dal sentinel — e questi mean/std sono condivisi da TUTTE le righe
del dataset (non solo quelle con NaN), quindi la normalizzazione peggiora anche per
le righe con inharmonicity gia' valida. Con -1.0 il problema e' semanticamente
risolto (nessuna collisione con "armonico perfetto") ma numericamente PEGGIORE del
vecchio 0.0: uno scarto piu' ampio dal grosso dei valori validi gonfia std ancora
di piu', schiacciando ulteriormente la risoluzione disponibile per i valori veri.

**Fix corretto (2026-09-16h)**: `agents.py` — sostituito `NAN_SENTINEL`/
`_fill_nan_descriptors` con `_nan_safe_stats()` (`np.nanmean`/`np.nanstd`, NaN
ignorati nel calcolo invece che sostituiti prima). Il fill avviene SOLO dopo la
normalizzazione (`np.nan_to_num` sul risultato di `(X - mean) / std`, mai su `X`
prima), sempre a 0.0 — ma qui 0.0 e' uno z-score neutro ("assumi la media"), non un
valore di dominio: nessuna collisione possibile con nessun descrittore, per
costruzione. Non serve piu' un sentinel per-descrittore: la stessa funzione e'
corretta per tutti, sostituita ovunque (`agents.py` x3, `knn_corpus.py` x5,
`pair_selector.py` x3, `knn_baseline.py` x2 — stessi punti del fix precedente).
Verificato solo con `ast.parse`.

**Da fare**: nuovo retrain (stessi comandi, prossima versione `_v4`/`_v9`) per
confermare che questa volta l'errore su inharmonicity scenda in modo consistente.
Se anche questo fix non basta, il problema potrebbe non essere la normalizzazione
ma la difficolta' intrinseca del descrittore (dipende da `material`/nonlinearita'
dei parziali in modo non lineare, simile al confondimento gia' noto su
pluck/chaos) — da considerare solo dopo aver escluso la normalizzazione come causa.

## Chiusura filone inharmonicity/NaN — ESITO (2026-09-16h, retrain v4 completo)

Retrain di conferma completato su tutti gli 11 agenti dopo il fix 2026-09-16h
(`_nan_safe_stats`, nanmean/nanstd invece di sentinel pre-normalizzazione).
Confronto NMAE `inharmonicity` v2 (originale, nan_to_num->0) vs v3 (sentinel -1.0,
regressione) vs v4 (fix corretto), colonna KNN (deterministica, nessuna
inizializzazione casuale, quindi il confronto piu' pulito):

blow 0.273->0.291->0.273; bow 0.182->0.182->0.182; strike 0.314->0.463->0.320;
pluck 0.329->0.462->0.337; shaker 0.353->0.357->0.358; noise 0.365->0.388->0.389;
chaos 0.433->0.579->0.518; resonator_bar 0.082->0.170->0.087; resonator_plate_rect
0.139->0.325->0.147; resonator_plate_circ 0.089->0.164->0.097; resonator_membrane
0.140->0.320->0.133.

v4 recupera la regressione di v3 su TUTTI gli 11 agenti, tornando vicinissimo (o
uguale) al baseline v2 — conferma che la diagnosi (mean/std contaminati dal
sentinel pre-normalizzazione) era corretta e il fix e' quello giusto. Colonna MDN
mista (meta' agenti leggermente meglio di v2, meta' leggermente peggio, sempre
entro variazione normale run-to-run: `Agent.train()` non fissa un seed PyTorch,
solo quello dello split dati) — nessun segnale sistematico ne' in un senso ne'
nell'altro, quindi non attribuibile al fix.

**Decisione**: chiuso. Il fix 2026-09-16h resta in produzione (nessun'altra
modifica alla normalizzazione). L'errore residuo su `inharmonicity` (NMAE
0.13-0.60 a seconda dell'agente, la colonna migliore per ciascuno — quasi sempre
KNN per resonator_*/chaos, MDN per gli eccitatori "puliti") e' la difficolta'
intrinseca del descrittore (dipendenza non lineare da `material`/nonlinearita' dei
parziali), NON un artefatto di normalizzazione o gestione NaN — stesso genere di
limite gia' documentato per mod_rate/mod_depth su noise/shaker o decay_time su
pluck. Nessun'altra azione pianificata su questo descrittore per ora.
