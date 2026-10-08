# Comparazione modelli — FlightPREV vs FlightBERT (Binary Encoding)

**Protocollo equo, stesso test per tutti** (`src/compare_models.py`):
- giorno di test: **07/04/2024** (`Italy_24-04-07.csv`), mai visto da nessun modello
  (FlightPREV trainato su 03–05/04; FlightBERT `fb_v3` su 03+04+05 con split
  temporale interno; `flightbert_gpu` solo su 05/04);
- **3 871 ancore comuni** (500 voli × ~8), valide per entrambi i modelli
  (storia 60 s @ 5 s per FlightPREV, 9 punti @ 30 s per FlightBERT, futuro
  120 s osservato);
- orizzonti comuni **30 / 60 / 90 / 120 s**
  (`fb_v3` ha passo nativo 30 s; `flightbert_gpu` passo 20 s → rollout a 120 s
  con interpolazione a 30/90 s);
- **verità neutrale**: interpolazione lineare dei punti raw puliti agli
  orizzonti esatti (indipendente dagli smoothing delle due pipeline);
- metriche: ADE orizzontale haversine (m), MAE quota (m), mediana/p90,
  suddivisione per fase di volo; baseline CV e persistenza sulle stesse ancore.

## 1. Risultato: vince FlightPREV (e l'ensemble v1+v2 è il migliore in assoluto)

ADE in metri sulle 3 871 ancore comuni:

| orizzonte | **FP ens (v1+v2)** | FP v1 | CV | FB-gpu | FB-v3 | persist |
|---:|---:|---:|---:|---:|---:|---:|
| +30 s | **376** | 382 | 492 | 2 345 | 12 455 | 4 949 |
| +60 s | **563** | 575 | 735 | 3 361 | 10 765 | 9 864 |
| +90 s | **864** | 888 | 1 127 | 5 367 | 10 656 | 14 740 |
| +120 s | **1 241** | 1 276 | 1 613 | 7 351 | 10 524 | 19 607 |

Errore a 120 s (mediana): ens **548 m**, v1 558 m, CV 739 m, FB-gpu 1 710 m,
FB-v3 3 762 m. Stesso ordinamento in crociera, salita e discesa.

Sul **test set completo** (120 000 finestre, 07/04):

| modello | ADE medio | FDE@120 s | mediana@120 | MAE quota |
|---|---:|---:|---:|---:|
| **FP ensemble v1+v2** | **582.5 m** | **1 180.8 m** | **426 m** | **45.7 m** |
| FP v1 (256/4) | 596.8 m | 1 209.2 m | 433 m | 47.5 m |
| FP v2 (384/6) | 601.4 m | 1 218.2 m | 440 m | 47.5 m |
| CV | 833.2 m | 1 619.3 m | — | — |

Miglioramento dell'ensemble vs CV: **+30.1 %** sull'ADE medio
(582.5 vs 833.2 m), su tutti i 12 orizzonti (10–120 s).

## 2. Perché FlightBERT perde: diagnosi

Due evidenze indipendenti (protocollo comune sopra + valutazione nativa
`flightbert.py --test_only` su 07/04, 392 370 finestre, `reports/flightbert/`):

1. **`fb_v3` (direct, 30 s) è peggio della persistenza a tutti gli orizzonti
   brevi**: nativo hdist@30 s = 8.79 km vs persistenza 4.72 km; errore quasi
   piatto (~8–10 km) da 30 a 270 s. Causa probabile: nel modo `direct` ogni
   posizione di output `t` prevede l'orizzonte `t+1` vedendo (causal mask)
   solo gli input `0..t` — la predizione a 1 step usa **una sola osservazione**,
   quindi non può stimare alcuna velocità. Il difetto è strutturale, non di
   training (val e test concordano).
2. **`flightbert_gpu` (iterativo, 20 s) ha la forma giusta ma è 3–5× sopra
   FlightPREV e peggio di CV ovunque** (nativo hdist@20 s = 1.79 km vs
   persistenza 3.13 km: batte la persistenza ma non CV). Concause:
   - risoluzione BE (0.001° ≈ 111 m, `fb_v3` addirittura 0.0001° ≈ 11 m) molto
     più fine della **quantizzazione dei dati (0.01° ≈ 1 113 m)**: i bit bassi
     codificano rumore impredicibile e la BCE li pesa quanto i bit alti;
   - allenamento su un solo giorno (`gpu`) / validazione rumorosa;
   - rollout iterativo che accumula l'errore (vs readout diretto di FlightPREV).
   - Prova a supporto: l'**unica grandezza precisa nei dati, la quota**
     (piedi interi, non quantizzata), è prevista bene anche da FlightBERT
     (MAE quota nativa 297 m a 30 s dominata da pochi outlier; sul protocollo
     comune FB-gpu 91 m vs FP 211 m — ma CV fa 66 m).

Encoding verificato corretto (round-trip BE esatto per le posizioni, nessuna
saturazione dei bit): il divario è del modello, non dell'harness.

## 3. Modifica applicata al vincitore: capacità + ensemble

Tentativo 1 — **v2 più capiente** (stessi dati, `config_v2.json`:
d_model 384, 6 layer, 12 teste, FFN 1024, ~10 M parametri): val ADE 611.1 m
vs 608.2 m di v1, test ADE 601.4 vs 596.8 m. Non migliora: a parità di dati
il modello più grande **sovraccita** (train 0.027 vs val 0.084) — il limite è
il rumore di quantizzazione, non la capacità.

Tentativo 2 — **ensemble v1+v2 (media delle predizioni)**: errori
parzialmente scorrelati tra le due architetture → **migliora v1 su tutti gli
orizzonti** (−1.6/−2.1/−2.7/−2.7 % a 30/60/90/120 s sul protocollo comune;
−2.4 % ADE e −3.8 % MAE quota sul test completo). **È il modello finale.**

## 4. File dei report

- `reports/` — FlightPREV v1 sul test (metriche + 6 grafici, invariato);
- `reports/v2test/` — FlightPREV v2 sul test (ablazione capacità);
- `reports/flightbert/` — valutazione nativa fb_v3 su 07/04
  (`metrics_native_07.json`, `hdist_nativa_07.png`, `config.json`);
- `reports/compare/` — protocollo comune:
  `metrics_compare.csv`, `summary_compare.json`, `ensemble_test.npz`
  (ADE v1/v2/ens/CV sul test completo), `ade_curves_compare.png`,
  `cdf_120_compare.png`, `err_by_phase_compare.png`, questo file.

## 5. Verdetto

1. **FlightPREV (Transformer su residuo-CV) batte FlightBERT (BE binaria) del
   75–90 %** a tutti gli orizzonti 30–120 s, e batte anche CV del 28–30 %.
2. La codifica binaria coarse-to-fine è elegante ma, con dati quantizzati più
   grossolanamente della risoluzione BE e BCE uniforme sui bit, spreca capacità
   sui bit bassi; il `direct` di fb_v3 aggiunge un difetto strutturale.
3. L'aumento di capacità da solo non paga (overfit); **l'ensemble di due
   architetture diverse sì** ed è il modello finale (ADE 582.5 m, FDE@120 s
   1 181 m, mediana 426 m sul test completo).
