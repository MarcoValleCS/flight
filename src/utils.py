"""Funzioni di utilita' condivise."""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

R_EARTH = 6371000.0


# --------------------------------------------------------------------------- IO
def setup_console() -> None:
    """Output UTF-8 su console Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(p: str | Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(obj, p: str | Path) -> None:
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class Stopwatch:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0


# ------------------------------------------------------------- geometria/ENU
def enu_from_latlon(lat: np.ndarray, lon: np.ndarray,
                    lat0: np.ndarray, lon0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trasformazione equirettangolare locale (E, N) in metri, ancorata a (lat0, lon0).

    Abbastanza precisa per distanze < ~100 km alle latitudini europee.
    """
    x = np.radians(lon - lon0) * R_EARTH * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R_EARTH
    return x, y


def latlon_from_enu(x: np.ndarray, y: np.ndarray,
                    lat0: np.ndarray, lon0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = lat0 + np.degrees(y / R_EARTH)
    lon = lon0 + np.degrees(x / (R_EARTH * np.cos(np.radians(lat0))))
    return lat, lon


def dist_m(lat1: np.ndarray, lon1: np.ndarray,
           lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Distanza haversine in metri."""
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1: np.ndarray, lon1: np.ndarray,
                lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    x = np.sin(lo2 - lo1) * np.cos(la2)
    y = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


# ------------------------------------------------------------------ segnali
def med3(a: np.ndarray) -> np.ndarray:
    """Filtro mediano a 3 punti (bordi invariati)."""
    if len(a) < 3:
        return a.astype(np.float64, copy=True)
    out = a.astype(np.float64, copy=True)
    out[1:-1] = np.median(np.stack([a[:-2], a[1:-1], a[2:]], axis=0), axis=0)
    return out


def mov_avg(a: np.ndarray, k: int) -> np.ndarray:
    """Media mobile a k punti, bordi replicati (k dispari consigliato)."""
    a = a.astype(np.float64)
    if k <= 1 or len(a) < k:
        return a.copy()
    pad = k // 2
    ap = np.pad(a, pad, mode="edge")
    return np.convolve(ap, np.ones(k) / k, mode="valid")


def phase_from_vr(vr_ms: np.ndarray, thr_ms: float) -> np.ndarray:
    """Fase di volo dal rateo verticale: 0=crociera, 1=salita, 2=discesa."""
    out = np.zeros(np.shape(vr_ms), dtype=np.int8)
    out[vr_ms > thr_ms] = 1
    out[vr_ms < -thr_ms] = 2
    return out


PHASE_NAMES = ["cruise", "climb", "descent"]


# ------------------------------------------------------------------ tabelle
def fmt_seconds(s: float) -> str:
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s/60:.1f}min"
    return f"{s/3600:.2f}h"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Tabella ASCII semplice (senza dipendenze)."""
    widths = [len(str(h)) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
