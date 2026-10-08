"""Configurazione centrale della pipeline.

Tutti i parametri hanno un default in DEFAULTS; un config.json nella directory
di lavoro (o passato con --config) puo' sovrascriverli parzialmente.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

VERSION = "1.0.0"

DEFAULTS: dict = {
    # ------------------------------------------------------------------ dati
    "data": {
        # griglia temporale di ricampionamento (secondi)
        "step_s": 5,
        # storia osservata: history_s secondi -> history_s/step_s + 1 punti
        "history_s": 60,
        # orizzonti di predizione (secondi): 10..120 ogni 10
        "horizons_s": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        # suddivisione in segmenti: nuovo segmento se buco > seg_gap_s
        # oppure salto netto (> seg_jump_m) a velocita' impossibile
        # (le posizioni sono quantizzate a 0.01 deg: i "gradini" della scala
        #  ~1.1-1.6 km NON devono essere interpretati come salti)
        "seg_gap_s": 60,
        "seg_max_speed_ms": 450.0,
        "seg_jump_m": 2000.0,
        # media mobile centrata sui target (punti della griglia) per ridurre
        # il rumore di quantizzazione dei label (3 = finestra 15 s)
        "target_smooth": 3,
        # un punto di griglia e' valido se l'osservazione piu' vicina dista
        # al massimo valid_gap_s secondi
        "valid_gap_s": 15.0,
        # filtro mediano (med3) sui segnali ricampionati per attenuare il
        # jitter ADS-B (3 = finestra 15 s)
        "smooth_k": 3,
        # durata minima di un segmento per estrarre finestre
        "min_segment_s": 200,
        # finestre: passo fra ancore successive e massimo per volo
        "stride_s": 60,
        "max_per_flight": 40,
        # limiti globali per split (sottocampionamento casuale con seed)
        "caps": {"train": 700000, "val": 120000, "test": 120000},
        # vocabolari embedding: frequenza minima nel train (altrimenti UNK=0)
        "min_count": {"ac_type": 20, "operator": 20, "dep": 10, "arr": 10},
        # filtri di plausibilita' in fase di pulizia
        "bbox": [38.5, 52.5, 1.5, 16.5],        # lat_min lat_max lon_min lon_max
        "alt_range_ft": [-600.0, 60000.0],
        "gs_range_kt": [0.0, 800.0],
    },
    # ------------------------------------------------- fattori di scala fissi
    # (niente statistiche di train/test -> nessun data leakage)
    "scales": {
        "xy": 10000.0,    # metri
        "z": 1000.0,      # metri (quota relativa all'ancora)
        "z0": 12000.0,    # metri (quota assoluta ancora, contesto)
        "gs": 250.0,      # m/s
        "vr": 25.0,       # m/s
    },
    # ---------------------------------------------------------------- modello
    "model": {
        "arch": "transformer",   # transformer | gru
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
        "ff": 768,
        "dropout": 0.1,
        "emb": {"ac_type": 16, "operator": 12, "dep": 8, "arr": 8},
        "predict_logvar": False,  # True = testa di incertezza (NLL gaussiana)
    },
    # ------------------------------------------------------------ addestramento
    "train": {
        "batch_size": 1024,
        "epochs": 50,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "warmup_steps": 500,
        "w_xy": 1.0,
        "w_z": 0.3,
        "w_phase": 0.1,
        "huber_delta": 0.1,
        "grad_clip": 1.0,
        "patience": 12,
        "num_workers": 4,
        "seed": 42,
        "amp": True,
        "phase_threshold_ms": 1.5,   # ~295 ft/min
    },
}


def _deep_update(base: dict, over: dict) -> dict:
    out = deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str | Path | None = "config.json", **overrides) -> dict:
    """Carica DEFAULTS + config.json (se esiste) + override espliciti."""
    cfg = deepcopy(DEFAULTS)
    p = Path(path) if path else None
    if p is not None and p.exists():
        cfg = _deep_update(cfg, json.loads(p.read_text(encoding="utf-8")))
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
