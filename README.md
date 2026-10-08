# FlightPREV — Previsione a breve termine di traiettorie aeree (10–120 s)

Modello di deep learning in **PyTorch** per prevedere la traiettoria di un aereo
nei successivi **10–120 secondi** (12 orizzonti, ogni 10 s) a partire dai dati
ADS-B "OpenFlight" registrati sullo spazio aereo italiano in 5 giorni
(3–7 aprile 2024).

> **Nota sui file grandi**: i CSV originali (~6 GB) e il dataset processato
> (~360 MB) **non** sono inclusi nel repo (limite GitHub di 100 MB/file) ma si
> rigenerano con gli script sotto. Anche i checkpoint FlightBERT
> (`FlightBert++/runs/*/best.pt`, ~435 MB) sono esclusi — le loro
> configurazioni (`config.json`), metriche e report **sono** inclusi.
> I checkpoint finali FlightPREV (`runs/transformer_v1/best.pt`,
> `runs/transformer_v2/best.pt`, ~11 MB) sono inclusi e bastano per
> `src.predict` e `src.evaluate`.

Il modello prevede, per ogni orizzonte: **latitudine, longitudine e quota**.
Architettura "encoder Transformer + testa multi-orizzonte" con:
apprendimento del **residuo rispetto al modello balistico a velocità costante**,
**embedding** di tipo velivolo/compagnia/aeroporti, task ausiliario di
classificazione della **fase di volo**, incertezza opzionale, e confronto
sistematico con **baseline cinematiche classiche** (CV, CA, CT).

---

## 1. Il dataset

5 file CSV senza intestazione (19 colonne), uno per giorno:

| file | righe | copertura (UTC) |
|---|---|---|
| Italy_24-04-03.csv | 5.3 M | 03/04 09:35 → 15:59 |
| Italy_24-04-04.csv | 14.1 M | 03/04 15:59 → 04/04 15:59 |
| Italy_24-04-05.csv | 15.3 M | 04/04 15:59 → 05/04 15:59 |
| Italy_24-04-06.csv | 16.0 M | 05/04 15:59 → 06/04 15:59 |
| Italy_24-04-07.csv | 15.2 M | 06/04 15:59 → 07/04 15:59 |

Totale **~65.9 milioni di righe**, ~4.3 giorni continui, ~15 000 voli/giorno,
bounding box ~39.3–51.7°N / 2.5–15.9°E (Italia + arco alpino e Francia orientale).

### Schema delle colonne (identificato con analisi cinematica)

| col | nome | unità | note |
|---|---|---|---|
| 0 | icao24 | — | indirizzo ICAO 24-bit |
| 1 | lat | gradi | **quantizzata a 0.01°** (~1.1 km) |
| 2 | lon | gradi | **quantizzata a 0.01°** |
| 3 | track | gradi | rotta vera 0–360 (verificata sui rilevamenti) |
| 4 | alt | piedi | quota barometrica |
| 5 | gs | nodi | velocità al suolo (verificata: ratio 1.00 a 20 s) |
| 6 | — | — | colonna vuota |
| 7 | sector | — | settore ATC corrente (es. `F-LIRF4`), 4176 valori — *non usato* |
| 8 | ac_type | — | tipo ICAO (A320, B738, …) |
| 9 | reg | — | marche |
| 10 | ts | s | timestamp Unix UTC |
| 11 | dep | — | aeroporto di partenza (IATA) |
| 12 | arr | — | aeroporto di destinazione (IATA) |
| 13 | flight_no | — | numero di volo |
| 14 | on_ground | 0/1 | 1 = a terra (quota 0 nel 99.7% dei casi) |
| 15 | vrate | ft/min | rateo verticale, quantizzato a 64 ft/min |
| 16 | callsign | — | callsign radio |
| 17 | — | — | sempre 0 |
| 18 | operator | — | compagnia (ICAO) |

### Qualità dei dati (importante)

- **Posizioni quantizzate a 0.01°** (~1112 m in latitudine, ~750 m in longitudine):
  i rilevamenti sono "a scalini". Per questo:
  - la velocità implicita a 2–4 s è rumorosa (solo a ≥20 s converge a quella reale);
  - la suddivisione in segmenti non interpreta gli scalini (~1.1–1.6 km) come
    teleport (criterio: salto > 2 km **e** velocità implicita > 450 m/s);
  - i **target di training** vengono smorzati con una media mobile centrata
    (15 s) per ridurre il rumore di quantizzazione dei label.
- **Campionamento irregolare** per volo: 2–7 s (moda 3 s).
- Traiettorie a terra escluse (finestre interamente in volo).

---

## 2. Pipeline dati (sliding windows)

`src/build_dataset.py` per ogni file CSV:

1. **Pulizia**: scarto righe senza icao/lat/lon/ts, limiti di plausibilità
   (bbox, quota −600…60000 ft, gs 0…800 kt), deduplica `(icao24, ts)`,
   ordinamento per `(icao24, ts)`.
2. **Segmentazione**: un volo è diviso in segmenti continui (buco > 60 s o
   teleport); un segmento deve durare ≥ 200 s.
3. **Ricampionamento** su griglia regolare a **5 s** (interpolazione lineare)
   + filtro mediano a 3 punti; un punto è *valido* se l'osservazione più vicina
   dista ≤ 15 s ed è in volo.
4. **Finestre scorrevoli** (ancora = istante di predizione `t0`):
   - **input**: 60 s di storia → **13 punti × 5 s**, 8 feature per punto:
     offset ENU (E,N) rispetto all'ancora, quota relativa, velocità al suolo,
     rateo verticale, sin/cos della rotta, tempo relativo;
   - **output**: posizioni a **+10, +20, …, +120 s** (12 orizzonti);
   - **target = residuo rispetto al modello a velocità costante (CV)**
     calcolato con le grandezze misurate all'ancora:
     `y = (posizione_vera − posizione_CV) / scala`. La rete impara solo la
     *correzione* alla predizione balistica (vento, virate, accelerazioni).
   - **task ausiliario**: fase di volo (crociera/salita/discesa) a +60 s.
5. **Contesto statico** per finestra: quota e velocità dell'ancora +
   embedding categorici (tipo velivolo, compagnia, partenza, destinazione).
6. Passo tra ancore: 60 s, max 40 finestre per volo; split **per giorno**:
   - **train**: 03–05/04 (~53 h di traffico) → fino a 700 000 finestre
   - **val**: 06/04 → fino a 120 000
   - **test**: 07/04 → fino a 120 000

Nessuna statistica di normalizzazione è calcolata su train/test: si usano
**scale fisiche fisse** (10 km, 1 km, 250 m/s, …) → nessun data leakage.

---

## 3. Modello

```
input (B,13,8) ─► proiezione lineare + positional encoding appreso
                 + token [CLS] + token contesto (scalari + embedding)
                 │
                 ▼
        TransformerEncoder  (d=256, 4 layer, 8 teste, FFN 768, GELU, norm_first)
                 │
                 ▼
   pooling [CLS ‖ contesto ‖ media(storia)] ─► MLP ─► teste:
        ├─ xy   : 12×2 correzioni (E,N) vs CV      [Huber]
        ├─ z    : 12 correzioni di quota vs CV     [Huber]
        ├─ phase: 3 classi a +60 s                 [CrossEntropy, ausiliaria]
        └─ logvar: incertezza per orizzonte        [opzionale, NLL gaussiana]
```

- Variante `arch="gru"` con encoder BiGRU (`--arch gru`).
- Addestramento: AdamW, lr 3e-4, cosine + warmup, AMP bf16, grad clip,
  early stopping (patience 12) sul val ADE; checkpoint best/last + log CSV.
- Loss: `Huber(xy) + 0.3·Huber(z) + 0.1·CE(fase)` (pesi configurabili).

### Baseline di confronto (`src/baselines.py`)

| baseline | descrizione |
|---|---|
| **CV** | velocità costante (gs/rotta/rateo misurati all'ancora) |
| **CT** | velocità angolare costante (rateo di virata dagli ultimi 25 s) |
| **CA** | accelerazione costante (dalle ultime posizioni; sensibile al rumore) |
| **zero** | posizione costante |

---

## 4. Risultati

### Test set (07/04, 120 000 finestre) — ADE in metri (modello finale: ensemble)

| orizzonte | **ens v1+v2** | v1 | v2 | CV | CT |
|---:|---:|---:|---:|---:|---:|
| +10 s | **221** | 227 | 227 | 342 | 385 |
| +30 s | **269** | 276 | 278 | 423 | 469 |
| +60 s | **462** | 473 | 477 | 678 | 782 |
| +90 s | **773** | 792 | 799 | 1 084 | 1 320 |
| +120 s | **1 181** | 1 209 | 1 218 | 1 619 | 2 041 |
| **ADE medio** | **582.5** | 596.8 | 601.4 | 833.2 | 999.2 |
| **FDE@120 s** | **1 180.8** | 1 209.2 | 1 218.2 | 1 619.3 | 2 040.6 |

- **Miglioramento ADE vs baseline: CV +30.1 %, CT +41.7 %, zero +95.1 %.**
- Errore a 120 s dell'ensemble: mediana **426 m**, p90 3 191 m, p95 ~5 300 m.
- MAE quota (12 orizzonti): **45.7 m**.
- Per fase di volo (FDE@120 s, ens vs CV): crociera ~790 vs 1 063 m;
  salita ~1 800 vs 2 414 m; discesa ~1 740 vs 2 450 m.

### Addestramento

- v1 (d=256, 4 layer, 2.9 M param): 700 000 finestre, early stop epoca 41,
  best val ADE 608.2 m (epoca 29), ~23 s/epoca su RTX 4070.
- v2 (d=384, 6 layer, 12 teste, ~10 M param, `config_v2.json`): best val ADE
  611.1 m — il modello più grande sovraccita (train 0.027 vs val 0.084):
  il limite è il rumore di quantizzazione, non la capacità.
- Finale: **ensemble a media v1+v2**, +2.4 % ADE e +3.8 % MAE quota su v1.

Grafici in `reports/`: `ade_curves.png`, `error_cdf_120s.png`,
`trajectory_samples.png`, `error_by_phase.png`, `training_curves.png`;
ablazione v2 in `reports/v2test/`.

---

## 5. Come eseguire

Requisiti: Python ≥3.10, `pip install -r requirements.txt` (PyTorch con CUDA
consigliato). Eseguire **dalla cartella del progetto**.

```powershell
# 1) dataset (una volta sola, ~15-30 min)
python -m src.build_dataset --files Italy_24-04-03.csv Italy_24-04-04.csv Italy_24-04-05.csv --split train
python -m src.build_dataset --files Italy_24-04-06.csv --split val
python -m src.build_dataset --files Italy_24-04-07.csv --split test

# 2) addestramento (~45-60 min su RTX 4070)
python -m src.train --run runs/transformer_v1

# 3) valutazione + grafici
python -m src.evaluate --run runs/transformer_v1/best.pt --split test

# 4) demo di predizione (finestre del test set, ensemble finale)
python -m src.predict --run runs/transformer_v1/best.pt runs/transformer_v2/best.pt --n 6

# 4b) predizione su una traccia CSV nuova
python -m src.predict --run runs/transformer_v1/best.pt --csv Italy_24-04-07.csv --icao 4CAFFB
```

Tutto in un comando: `powershell -ExecutionPolicy Bypass -File run_all.ps1`.

Tutti i parametri stanno in `config.json` (griglia temporale, orizzonti,
architettura, loss, …); i principali sono sovrascrivibili da riga di comando.

---

## 6. Confronto con FlightBERT (codifica binaria)

In `FlightBert++/` è presente un secondo modello (implementazione del paper
FlightBERT, T-ITS 2023: codifica binaria degli attributi, classificazione
multi-bit, 38 M parametri, checkpoint `fb_v3` e `flightbert_gpu`).

**Verdetto** (dettagli in `reports/compare/COMPARISON.md`): su un protocollo
comune equo (3 871 ancore del 07/04, orizzonti 30/60/90/120 s, stessa verità),
**FlightPREV batte FlightBERT del 75–90 %** a tutti gli orizzonti, e anche la
semplice baseline CV batte entrambi i checkpoint FlightBERT:

| @120 s | ens v1+v2 | v1 | CV | FB-gpu | FB-v3 |
|---|---:|---:|---:|---:|---:|
| FDE (m) | **1 241** | 1 276 | 1 613 | 7 351 | 10 524 |

Cause: risoluzione BE (11–111 m) più fine della quantizzazione dati (1 113 m)
→ bit bassi = rumore; `fb_v3/direct` con difetto strutturale (step-1 da una
sola osservazione, peggio della persistenza); rollout iterativo e 1 solo
giorno di training per `gpu`. Script: `src/compare_models.py`.

## 7. Struttura del progetto

```
FlightPREV/
├── config.json              parametri globali
├── run_all.ps1              pipeline completa
├── requirements.txt
├── src/
│   ├── columns.py           schema delle colonne CSV
│   ├── config.py            default + merge config.json (config_v2.json: v2)
│   ├── utils.py             geometria ENU, filtri, I/O, tabelle
│   ├── build_dataset.py     pulizia + ricampionamento + sliding windows
│   ├── dataset.py           Dataset PyTorch (memory-mapped)
│   ├── models.py            Transformer / GRU + checkpoint
│   ├── baselines.py         CV, CT, CA, zero
│   ├── train.py             training loop (AMP, scheduler, early stop)
│   ├── evaluate.py          metriche per orizzonte/fase + grafici
│   ├── predict.py           inferenza singolo modello o ensemble (demo / CSV)
│   └── compare_models.py    confronto equo FlightPREV vs FlightBERT
├── data/processed/          dataset generato (train/val/test + vocab.json)
├── runs/                    checkpoint (transformer_v1, transformer_v2) e log
├── reports/                 report v1 sul test
├── reports/v2test/          report v2 sul test (ablazione)
├── reports/flightbert/      report nativo FlightBERT fb_v3 su 07/04
└── reports/compare/         protocollo comune + ensemble + COMPARISON.md
```

---

## 7. Limiti e sviluppi futuri

- La quantizzazione a 0.01° delle posizioni pone un **floor di errore** di
  ~200–400 m per orizzonte, comune a modello e baseline.
- Le quote barometriche non distinguono voli VFR di piccoli aerei da
  interrogazioni errate; i filtri di plausibilità aiutano ma non eliminano.
- Il modello non conosce l'**intento di atterraggio**: nei 120 s finali di un
  avvicinamento estrapola la discesa (può produrre quote sotto il suolo).
  Sviluppo futuro: feature di prossimità a pista / "atterrerà entro l'orizzonte".
- Sviluppi possibili: attenzione verso **aerei vicini** (context sociale),
  integrazione **meteo** (vento in quota), previsione **probabilistica**
  completa (mixture density / diffusion), fine-tuning per fase di volo,
  `predict_logvar: true` per l'incertezza per orizzonte già supportato.
