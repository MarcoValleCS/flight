"""Costruzione del dataset a finestre scorrevoli ("sliding windows").

Pipeline per ogni file CSV giornaliero:
    1. lettura e pulizia (valori mancanti, limiti di plausibilita')
    2. ordinamento per (icao24, timestamp) e deduplica
    3. suddivisione di ogni volo in segmenti continui
    4. ricampionamento su griglia regolare di 5 s (interpolazione lineare)
       + filtro mediano a 3 punti per attenuare il jitter ADS-B
    5. estrazione di finestre scorrevoli:
         input  : 60 s di storia  (13 punti a 5 s)
         output : posizioni a +10, +20, ..., +120 s (12 orizzonti)
    6. i target sono espressi come *residui* rispetto al modello
       "velocita' costante" (CV) calcolato con la velocita' misurata
       (gs/rotta/rateo verticale dell'ancora): la rete impara solo la
       correzione alla predizione balistica.

Vengono salvati array .npy (apribili in memory-map) + meta.json per ogni split.

Uso:
    python -m src.build_dataset --files Italy_24-04-03.csv Italy_24-04-04.csv Italy_24-04-05.csv --split train
    python -m src.build_dataset --files Italy_24-04-06.csv --split val
    python -m src.build_dataset --files Italy_24-04-07.csv --split test
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import columns as C
from .config import load_config
from .utils import (R_EARTH, enu_from_latlon, ensure_dir, log, med3, mov_avg,
                    phase_from_vr, read_json, seed_all, setup_console,
                    write_json)

FEATURE_NAMES = ["x", "y", "z_rel", "gs", "vr", "sin_trk", "cos_trk", "t_rel"]

# chiavi degli accumulatori (per split)
ACC_KEYS = [
    "x", "y_xy", "y_z", "phase", "phase0", "lat0", "lon0", "z0", "gs0",
    "trk0", "vr0", "ts0", "ac_id", "typ", "op", "dep", "arr", "file_id",
]

DTYPES_ACC = {
    "x": np.float16,
    "y_xy": np.float32,
    "y_z": np.float32,
    "phase": np.int8,
    "phase0": np.int8,
    "lat0": np.float32,
    "lon0": np.float32,
    "z0": np.float32,
    "gs0": np.float32,
    "trk0": np.float32,
    "vr0": np.float32,
    "ts0": np.int64,
    "ac_id": np.int32,
    "typ": np.int16,
    "op": np.int16,
    "dep": np.int16,
    "arr": np.int16,
    "file_id": np.int8,
}


# --------------------------------------------------------------------------
def segment_flights(t: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                    gap_s: float, max_speed_ms: float,
                    jump_m: float = 2000.0) -> list[np.ndarray]:
    """Divide un volo in segmenti continui; ritorna liste di indici.

    Un nuovo segmento inizia se: buco temporale > gap_s, oppure salto di
    posizione > jump_m a velocita' implicita > max_speed_ms (teleport/errore).
    """
    if len(t) < 2:
        return [np.arange(len(t))]
    dt = np.diff(t).astype(np.float64)
    la1, lo1, la2, lo2 = map(np.radians, (lat[:-1], lon[:-1], lat[1:], lon[1:]))
    a = np.sin((la2 - la1) / 2) ** 2 + \
        np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    dist = 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    imp = dist / np.maximum(dt, 1e-6)
    cut = np.where((dt > gap_s) | ((imp > max_speed_ms) & (dist > jump_m)))[0] + 1
    return np.split(np.arange(len(t)), cut)


def resample_segment(t: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                     trk_deg: np.ndarray, alt_ft: np.ndarray, gs_kt: np.ndarray,
                     vr_fpm: np.ndarray, ground: np.ndarray, *,
                     step_s: float, smooth_k: int, valid_gap_s: float,
                     target_smooth: int = 3) -> dict:
    """Ricampiona un segmento su griglia regolare e marca i punti validi.

    - lat/lon/alt/gs/vr/rotta: med3 (attenua gli spike)
    - lat_t/lon_t/alt_t: media mobile centrata sui target (riduce il rumore
      di quantizzazione dei label; il futuro e' noto nei label, nessun leak)
    """
    t = t.astype(np.float64)
    trk_rad = np.unwrap(np.radians(trk_deg.astype(np.float64)))
    n = int(np.floor((t[-1] - t[0]) / step_s + 1e-6)) + 1
    tg = t[0] + np.arange(n, dtype=np.float64) * float(step_s)

    lat_g = np.interp(tg, t, lat)
    lon_g = np.interp(tg, t, lon)
    alt_g = np.interp(tg, t, alt_ft)
    gs_g = np.interp(tg, t, gs_kt)
    vr_g = np.interp(tg, t, vr_fpm)
    tr_g = np.interp(tg, t, trk_rad)
    gnd = np.interp(tg, t, ground.astype(np.float64)) >= 0.5

    if smooth_k and smooth_k >= 3:
        lat_g = med3(lat_g)
        lon_g = med3(lon_g)
        alt_g = med3(alt_g)
        gs_g = med3(gs_g)
        vr_g = med3(vr_g)
        tr_g = med3(tr_g)

    lat_t = mov_avg(lat_g, target_smooth)
    lon_t = mov_avg(lon_g, target_smooth)
    alt_t = mov_avg(alt_g, target_smooth)

    # distanza dall'osservazione piu' vicina (sinistra e destra)
    left = np.searchsorted(t, tg, side="right") - 1
    left_ok = left >= 0
    left_i = np.clip(left, 0, len(t) - 1)
    dist_l = tg - t[left_i]
    right = left + 1
    right_ok = right <= len(t) - 1
    right_i = np.clip(right, 0, len(t) - 1)
    dist_r = t[right_i] - tg
    gap = np.minimum(np.where(left_ok, dist_l, np.inf),
                     np.where(right_ok, dist_r, np.inf))
    valid = (gap <= valid_gap_s) & (~gnd)   # valido = aereo + osservazione vicina

    return dict(tg=tg, lat=lat_g, lon=lon_g, alt=alt_g, gs=gs_g,
                tr=tr_g, vr=vr_g, valid=valid,
                lat_t=lat_t, lon_t=lon_t, alt_t=alt_t)


def extract_windows(seg: dict, horizons_s: list[int], history_s: int,
                    step_s: float, stride_s: float, max_per_flight: int,
                    sc: dict, phase_thr_ms: float,
                    last_only: bool = False,
                    future_required: bool = True) -> dict | None:
    """Estrae le finestre (ancore) da un segmento ricampionato.

    last_only=True       -> usa solo l'ultima ancora valida (inferenza sul presente)
    future_required=False -> non richiede che il futuro sia presente nel file;
                             i target non disponibili vengono posti a NaN
                             (usato per l'inferenza su tracce troncate).
    """
    tg = seg["tg"]
    H = int(round(history_s / step_s))            # passi di storia (12)
    hoff = np.array([int(round(h / step_s)) for h in horizons_s], dtype=np.int64)
    max_off = int(hoff.max())                     # 24 (120 s)
    n = len(tg)
    i0 = H
    i1 = n - 1 - max_off if future_required else n - 1
    if i1 < i0:
        return None

    stride = max(1, int(round(stride_s / step_s)))
    inv = (~seg["valid"]).astype(np.int64)
    pref = np.concatenate(([0], np.cumsum(inv)))
    i_all = np.arange(i0, i1 + 1, stride, dtype=np.int64)
    if future_required:
        good = (pref[i_all + max_off + 1] - pref[i_all - H]) == 0
    else:
        good = (pref[i_all + 1] - pref[i_all - H]) == 0
    i_sel = i_all[good]
    if len(i_sel) == 0:
        return None
    if last_only:
        i_sel = i_sel[-1:]
    elif 0 < max_per_flight < len(i_sel):
        pick = np.unique(np.round(np.linspace(0, len(i_sel) - 1, max_per_flight)).astype(np.int64))
        i_sel = i_sel[pick]

    # ---- contesto per finestra --------------------------------------------
    lat0 = seg["lat"][i_sel]
    lon0 = seg["lon"][i_sel]
    alt0_ft = seg["alt"][i_sel]
    alt0_m = alt0_ft * C.FT_TO_M
    gs0_ms = seg["gs"][i_sel] * C.KT_TO_MS
    trk0 = seg["tr"][i_sel]                        # rad, non "wrappata"
    vr0_ms = seg["vr"][i_sel] * C.FPM_TO_MS

    # ---- input: 13 punti di storia ----------------------------------------
    ih = i_sel[:, None] + np.arange(-H, 1, dtype=np.int64)[None, :]
    x_m, y_m = enu_from_latlon(seg["lat"][ih], seg["lon"][ih], lat0[:, None], lon0[:, None])
    z_rel_m = (seg["alt"][ih] - alt0_ft[:, None]) * C.FT_TO_M
    gs_ms = seg["gs"][ih] * C.KT_TO_MS
    vr_ms = seg["vr"][ih] * C.FPM_TO_MS
    sin_tr = np.sin(seg["tr"][ih])
    cos_tr = np.cos(seg["tr"][ih])
    t_rel = (tg[ih] - tg[i_sel][:, None]) / float(history_s)

    X = np.stack([
        x_m / sc["xy"], y_m / sc["xy"], z_rel_m / sc["z"],
        gs_ms / sc["gs"], vr_ms / sc["vr"], sin_tr, cos_tr, t_rel,
    ], axis=-1)

    # ---- target: 12 orizzonti (posizioni smorzate per ridurre il rumore di
    #      quantizzazione dei label; nei label il futuro e' noto) -------------
    it = i_sel[:, None] + hoff[None, :]
    if not future_required:
        fut_ok = it <= (n - 1)
        itc = np.clip(it, 0, n - 1)
    else:
        fut_ok = None
        itc = it
    xt, yt = enu_from_latlon(seg["lat_t"][itc], seg["lon_t"][itc],
                             lat0[:, None], lon0[:, None])
    zt_m = (seg["alt_t"][itc] - alt0_ft[:, None]) * C.FT_TO_M

    # correzione balistica CV (velocita' misurata all'ancora)
    tau = np.asarray(horizons_s, dtype=np.float64)[None, :]
    vx = gs0_ms * np.sin(trk0)     # traccia oraria da Nord: E = sin, N = cos
    vy = gs0_ms * np.cos(trk0)
    cvx = vx[:, None] * tau
    cvy = vy[:, None] * tau
    cvz = vr0_ms[:, None] * tau

    y_xy = np.stack([(xt - cvx) / sc["xy"], (yt - cvy) / sc["xy"]], axis=-1)
    y_z = (zt_m - cvz) / sc["z"]
    if fut_ok is not None:
        y_xy = np.where(fut_ok[..., None], y_xy, np.nan).astype(np.float32)
        y_z = np.where(fut_ok, y_z, np.nan).astype(np.float32)

    # ---- fase di volo (ausiliaria) -----------------------------------------
    i60 = np.clip(i_sel + int(round(60 / step_s)), 0, n - 1)
    vr60 = seg["vr"][i60] * C.FPM_TO_MS
    phase = phase_from_vr(vr60, phase_thr_ms)
    if fut_ok is not None:
        phase = np.where(i_sel + int(round(60 / step_s)) <= n - 1, phase, -1)
    phase0 = phase_from_vr(vr0_ms, phase_thr_ms)

    return dict(
        x=X.astype(np.float16),
        y_xy=y_xy.astype(np.float32),
        y_z=y_z.astype(np.float32),
        phase=phase.astype(np.int8),
        phase0=phase0.astype(np.int8),
        lat0=lat0.astype(np.float32),
        lon0=lon0.astype(np.float32),
        z0=alt0_m.astype(np.float32),
        gs0=gs0_ms.astype(np.float32),
        trk0=(np.degrees(trk0) % 360.0).astype(np.float32),
        vr0=vr0_ms.astype(np.float32),
        ts0=tg[i_sel].astype(np.int64),
        n=len(i_sel),
    )


# --------------------------------------------------------------------------
class VocabBuilder:
    """Vocabolario incrementale per gli embedding categorici (solo train)."""

    def __init__(self, fields: list[str]):
        self.fields = fields
        self.cnt = {f: Counter() for f in fields}
        self.nxt = {f: 1 for f in fields}
        self._ids = {f: {} for f in fields}

    def add(self, field: str, value) -> int:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0
        value = str(value)
        self.cnt[field][value] += 1
        return self._id(field, value)

    def _id(self, field: str, value: str) -> int:
        m = self._ids[field]
        i = m.get(value)
        if i is None:
            i = self.nxt[field]
            m[value] = i
            self.nxt[field] += 1
        return i

    def finalize(self, min_count: dict) -> tuple[dict, dict]:
        """Prune e costruzione vocabolario finale (id stabili, 1..K)."""
        maps, sizes = {}, {}
        for f in self.fields:
            items = [(v, c) for v, c in self.cnt[f].items() if c >= min_count.get(f, 1)]
            items.sort(key=lambda kv: (-kv[1], kv[0]))
            maps[f] = {v: i + 1 for i, (v, _) in enumerate(items)}   # 0 = UNK
            sizes[f] = len(maps[f]) + 1
        return maps, sizes


def load_vocab(path: Path) -> tuple[dict, dict]:
    v = read_json(path)
    return v["map"], v["sizes"]


def _vlook(builder: VocabBuilder | None, vocab_map: dict | None,
           field: str, value) -> int:
    if builder is not None:
        return builder.add(field, value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    return vocab_map[field].get(str(value), 0)


def clean_dataframe(df: pd.DataFrame, d: dict) -> pd.DataFrame:
    """Pulizia di plausibilita' (stessa logica di process_file, per inferenza)."""
    df = df.dropna(subset=["icao24", "lat", "lon", "ts"])
    lat_min, lat_max, lon_min, lon_max = d["bbox"]
    df = df[df["lat"].between(lat_min, lat_max) & df["lon"].between(lon_min, lon_max)]
    df = df[df["alt"].between(*d["alt_range_ft"]) & df["gs"].between(*d["gs_range_kt"])]
    df = df.sort_values(["icao24", "ts"], kind="stable")
    df = df.drop_duplicates(["icao24", "ts"], keep="first")
    return df


# --------------------------------------------------------------------------
def process_file(csv_path: Path, split: str, cfg: dict, file_id: int,
                 builder: VocabBuilder | None, vocab_map: dict | None,
                 accum: dict, icao_ids: dict, stats: dict,
                 max_rows: int | None, limit_flights: int | None) -> None:
    d = cfg["data"]
    sc = cfg["scales"]
    log(f"  lettura {csv_path.name} ...")
    df = pd.read_csv(csv_path, header=None, usecols=C.USECOLS, dtype=C.DTYPES,
                     low_memory=False, nrows=max_rows)
    df.columns = C.NEW_NAMES
    stats["rows_raw"] += len(df)

    n0 = len(df)
    df = df.dropna(subset=["icao24", "lat", "lon", "ts"])
    stats["rows_missing"] += n0 - len(df)
    n_after_missing = len(df)

    lat_min, lat_max, lon_min, lon_max = d["bbox"]
    m_ok = (df["lat"].between(lat_min, lat_max) & df["lon"].between(lon_min, lon_max))
    df = df[m_ok]
    stats["rows_bbox"] += n_after_missing - len(df)

    n1 = len(df)
    df = df[df["alt"].between(*d["alt_range_ft"]) & df["gs"].between(*d["gs_range_kt"])]
    stats["rows_range"] += n1 - len(df)

    df = df.sort_values(["icao24", "ts"], kind="stable")
    df = df.drop_duplicates(["icao24", "ts"], keep="first")
    stats["rows_kept"] += len(df)

    ts_np = df["ts"].to_numpy(np.int64)
    lat_np = df["lat"].to_numpy(np.float32)
    lon_np = df["lon"].to_numpy(np.float32)
    trk_np = df["trk"].to_numpy(np.float32)
    alt_np = df["alt"].to_numpy(np.float32)
    gs_np = df["gs"].to_numpy(np.float32)
    vr_np = df["vrate"].to_numpy(np.float32)
    gr_np = df["ground"].to_numpy(np.int8)

    groups = df.groupby("icao24", sort=False, observed=True).indices
    step_s = float(d["step_s"])
    horizons = list(d["horizons_s"])
    phases = Counter()

    n_flights = 0
    n_windows = 0
    for icao, idx in groups.items():
        if limit_flights and n_flights >= limit_flights:
            break
        n_flights += 1
        first = df.iloc[idx[0]]
        typ = _vlook(builder, vocab_map, "ac_type", first["ac_type"])
        op = _vlook(builder, vocab_map, "operator", first["operator"])
        dep = _vlook(builder, vocab_map, "dep", first["dep"])
        arr = _vlook(builder, vocab_map, "arr", first["arr"])
        ac_id = icao_ids.setdefault(str(icao), len(icao_ids) + 1)

        segs = segment_flights(ts_np[idx], lat_np[idx], lon_np[idx],
                               d["seg_gap_s"], d["seg_max_speed_ms"],
                               d.get("seg_jump_m", 2000.0))
        for sidx in segs:
            if len(sidx) < 5:
                continue
            if ts_np[idx[sidx[-1]]] - ts_np[idx[sidx[0]]] < d["min_segment_s"]:
                continue
            seg = resample_segment(
                ts_np[idx][sidx], lat_np[idx][sidx], lon_np[idx][sidx],
                trk_np[idx][sidx], alt_np[idx][sidx], gs_np[idx][sidx],
                vr_np[idx][sidx], gr_np[idx][sidx],
                step_s=step_s, smooth_k=d["smooth_k"], valid_gap_s=d["valid_gap_s"],
                target_smooth=d.get("target_smooth", 3))
            win = extract_windows(
                seg, horizons, d["history_s"], step_s, d["stride_s"],
                d["max_per_flight"], sc, cfg["train"]["phase_threshold_ms"])
            if win is None:
                continue
            nw = win.pop("n")
            accum["x"].append(win["x"])
            for k in ACC_KEYS:
                if k in ("x", "typ", "op", "dep", "arr", "ac_id", "file_id"):
                    continue
                accum[k].append(win[k])
            accum["typ"].append(np.full(nw, typ, np.int16))
            accum["op"].append(np.full(nw, op, np.int16))
            accum["dep"].append(np.full(nw, dep, np.int16))
            accum["arr"].append(np.full(nw, arr, np.int16))
            accum["ac_id"].append(np.full(nw, ac_id, np.int32))
            accum["file_id"].append(np.full(nw, file_id, np.int8))
            phases.update(win["phase0"].tolist())
            n_windows += nw

    stats["flights"] += n_flights
    stats["windows"] += n_windows
    log(f"    -> {n_flights} voli, {n_windows} finestre")


# --------------------------------------------------------------------------
def assemble_split(accum: dict, split: str, cfg: dict, args, sources: list) -> tuple[dict, dict]:
    caps = cfg["data"]["caps"]
    dtypes = DTYPES_ACC
    arrays = {}
    for k in ACC_KEYS:
        if len(accum[k]) == 0:
            continue
        arrays[k] = np.concatenate(accum[k], axis=0)
    n = len(arrays.get("ts0", []))
    log(f"  totale finestre {split}: {n}")
    if n == 0:
        # salva uno split vuoto consistente
        shapes = {
            "x": (0, int(cfg["data"]["history_s"] / cfg["data"]["step_s"]) + 1, len(FEATURE_NAMES)),
            "y_xy": (0, len(cfg["data"]["horizons_s"]), 2),
            "y_z": (0, len(cfg["data"]["horizons_s"])),
        }
        for k in ACC_KEYS:
            arrays[k] = np.zeros(shapes.get(k, (0,)), dtype=dtypes[k])
        n = 0

    cap = int(caps.get(split, 0))
    if 0 < cap < n:
        rng = np.random.default_rng(int(cfg["train"]["seed"]))
        take = np.sort(rng.choice(n, size=cap, replace=False))
        arrays = {k: a[take] for k, a in arrays.items()}
        log(f"  sottocampionato a {cap} finestre (seed={cfg['train']['seed']})")
        n = cap

    return arrays, {}


def remap_and_prune(arrays: dict, builder: VocabBuilder, min_count: dict) -> tuple[dict, dict]:
    """Remap degli id categorici dopo il prune del vocabolario (train)."""
    maps, sizes = builder.finalize(min_count)
    field_to_key = {"ac_type": "typ", "operator": "op", "dep": "dep", "arr": "arr"}
    for f in builder.fields:
        old = arrays[field_to_key[f]]
        old_max = int(old.max()) + 1 if len(old) else 1
        lut = np.zeros(max(old_max, 1), dtype=np.int16)
        for v, new in maps[f].items():
            old_id = builder._ids[f].get(v)
            if old_id is not None and old_id < len(lut):
                lut[old_id] = new
        arrays[field_to_key[f]] = lut[old]
    return maps, sizes


# --------------------------------------------------------------------------
def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description="Costruzione dataset a finestre.")
    ap.add_argument("--files", nargs="+", required=True, help="file CSV (in ordine cronologico)")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out", default="data/processed", help="cartella di output")
    ap.add_argument("--vocab", default=None, help="percorso vocab.json (default: <out>/vocab.json)")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--max-rows", type=int, default=None, help="limite righe per file (debug)")
    ap.add_argument("--limit-flights", type=int, default=None, help="limite voli per file (debug)")
    ap.add_argument("--stride-s", type=float, default=None, help="override passo ancore (s)")
    ap.add_argument("--max-per-flight", type=int, default=None)
    args = ap.parse_args()

    over = {}
    if args.stride_s is not None:
        over.setdefault("data", {})["stride_s"] = args.stride_s
    if args.max_per_flight is not None:
        over.setdefault("data", {})["max_per_flight"] = args.max_per_flight
    cfg = load_config(args.config, **over)
    seed_all(int(cfg["train"]["seed"]))

    out_dir = ensure_dir(Path(args.out) / args.split)
    vocab_path = Path(args.vocab) if args.vocab else Path(args.out) / "vocab.json"

    builder, vocab_map = None, None
    if args.split == "train":
        builder = VocabBuilder(["ac_type", "operator", "dep", "arr"])
    else:
        if not vocab_path.exists():
            raise SystemExit(f"vocab.json non trovato in {vocab_path}: esegui prima lo split train.")
        vocab_map, _ = load_vocab(vocab_path)
        log(f"  vocabolario caricato: {arg_sizes(vocab_map)}")

    accum = defaultdict(list)
    icao_ids: dict = {}
    stats = defaultdict(int)
    sources = []
    for fid, f in enumerate(args.files):
        f = Path(f)
        before = defaultdict(int, stats)
        process_file(f, args.split, cfg, fid, builder, vocab_map,
                     accum, icao_ids, stats, args.max_rows, args.limit_flights)
        sources.append({
            "file": f.name,
            "windows": stats["windows"] - before["windows"],
            "flights": stats["flights"] - before["flights"],
        })

    arrays, _ = assemble_split(accum, args.split, cfg, args, sources)

    maps, sizes = None, None
    if args.split == "train":
        maps, sizes = remap_and_prune(arrays, builder, cfg["data"]["min_count"])
        write_json({"map": maps, "sizes": sizes}, vocab_path)
        log(f"  vocabolario salvato: {arg_sizes(maps)}")
    else:
        maps, sizes = load_vocab(vocab_path)

    # ---- salvataggio -------------------------------------------------------
    n = len(arrays["ts0"])
    for k in ACC_KEYS:
        if k in arrays:
            np.save(out_dir / f"{k}.npy", arrays[k])
    meta = {
        "version": 1,
        "split": args.split,
        "n_windows": int(n),
        "files": [s["file"] for s in sources],
        "sources": sources,
        "feature_names": FEATURE_NAMES,
        "history_s": cfg["data"]["history_s"],
        "step_s": cfg["data"]["step_s"],
        "horizons_s": cfg["data"]["horizons_s"],
        "scales": cfg["scales"],
        "vocab_sizes": {k: int(v) for k, v in (sizes or {}).items()},
        "phase_names": ["cruise", "climb", "descent"],
        "stats": dict(stats),
        "config": cfg,
    }
    write_json(meta, out_dir / "meta.json")
    log(f"  salvato in {out_dir}  (n_windows={n})")
    log(f"  righe: lette={stats['rows_raw']}  mantenute={stats['rows_kept']}  "
        f"scartate(missing={stats['rows_missing']}, bbox={stats['rows_bbox']}, range={stats['rows_range']})")


def arg_sizes(vocab_map: dict) -> str:
    return ", ".join(f"{k}={len(v) + 1}" for k, v in vocab_map.items())


if __name__ == "__main__":
    main()
