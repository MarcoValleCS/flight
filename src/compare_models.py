"""Confronto equo FlightPREV (Transformer, residuo-CV) vs FlightBERT (BE binaria).

Protocollo comune (stesso giorno di test, mai visto da nessun modello):
    - file: Italy_24-04-07.csv (fb_v3 e' allenato su 03+04+05, gpu solo su 05,
      FlightPREV su 03+04+05 con test=07 -> 07 e' unseen per tutti)
    - stesse ancore, stessi orizzonti comuni {30, 60, 90, 120} s
      (fb_v3 ha passo nativo 30 s; gpu passo 20 s -> valutato a 60/120 s)
    - stessa verita': interpolazione lineare dei punti raw puliti agli
      orizzonti esatti (neutrale rispetto agli smoothing delle due pipeline)
    - metriche: ADE orizzontale (haversine, m), MAE quota (m), mediana/p90,
      suddivisione per fase di volo + baseline CV e persistenza.

Uso:
    python -m src.compare_models --n-windows 3000 --out reports/compare
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import columns as C
from .build_dataset import (clean_dataframe, extract_windows,
                            resample_segment, segment_flights)
from .baselines import baseline_cv
from .config import load_config
from .models import load_model_from_checkpoint
from .utils import (dist_m, ensure_dir, latlon_from_enu, log, phase_from_vr,
                    print_table, read_json, setup_console)

FB_PATH = Path(__file__).resolve().parent.parent / "FlightBert++" / "flightbert.py"
COMMON_H = [30, 60, 90, 120]
KT_TO_KMH = 1.852
FPM_TO_KMH = 0.3048 / 60.0 * 3.6


def load_fb_module():
    spec = importlib.util.spec_from_file_location("fbcode", FB_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["fbcode"] = m
    spec.loader.exec_module(m)
    return m


def build_fb_model(fb, ckpt_path: str, device: torch.device):
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location="cpu")
    cfgd = dict(ck.get("config", {}))
    fields = set(fb.Config.__dataclass_fields__)
    init = {k: v for k, v in cfgd.items() if k in fields}
    if isinstance(init.get("eval_horizons"), list):
        init["eval_horizons"] = tuple(init["eval_horizons"])
    cfg = fb.Config(**init)
    fb.set_position_precision(int(cfgd.get("pos_prec", 1000)))
    enc = fb.BinaryEncoding({a: int(v) for a, v in ck.get("bits", cfg.bits).items()})
    model = fb.FlightBERT(cfg, enc)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    model.eval()
    info = {
        "name": Path(ckpt_path).parent.name,
        "period": int(cfg.period), "k": int(cfg.inp_seq_len),
        "n": int(cfg.max_horizon), "direct": bool(cfg.direct),
        "bits": dict(enc.bits), "epoch": ck.get("epoch"),
    }
    return model, enc, cfg, info


@torch.no_grad()
def fb_predict_direct(fb, model, enc, X: np.ndarray, n_steps: int,
                      device: torch.device, batch_size: int = 256) -> np.ndarray:
    """X: (N,k,D) uint8 -> (N,n_steps,6) valori reali (deg,deg,m,kmh)."""
    out = []
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[s:s + batch_size]).to(device).float()
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            probs = model(xb)
        if probs.dim() == 2:
            probs = probs.unsqueeze(1)
        bits = (probs[:, :n_steps] > 0.5).float().cpu().numpy()
        dec = enc.decode_matrix(bits.reshape(-1, enc.total_bits)
                                ).reshape(len(xb), n_steps, 6)
        for j, a in enumerate(fb.ATTRS):
            dec[:, :, j] /= fb.PRECISION[a]
        out.append(dec)
    return np.concatenate(out, axis=0)


@torch.no_grad()
def fb_predict_iterative(fb, model, enc, X: np.ndarray, n_steps: int,
                         device: torch.device, batch_size: int = 128) -> np.ndarray:
    out = []
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[s:s + batch_size]).to(device).float()
        rolled = fb.predict_horizons(model, xb, n_steps, False)
        bits = rolled.float().cpu().numpy()
        dec = enc.decode_matrix(bits.reshape(-1, enc.total_bits)
                                ).reshape(len(xb), n_steps, 6)
        for j, a in enumerate(fb.ATTRS):
            dec[:, :, j] /= fb.PRECISION[a]
        out.append(dec)
    return np.concatenate(out, axis=0)


def fb_history_be(raw: dict, t0: float, period: int, k: int, gap_allow: float,
                  fb, enc) -> np.ndarray | None:
    """Finestra step-hold (come _resample_leg) allineata all'ancora t0."""
    ts = raw["ts"]
    grid = t0 - (np.arange(k, dtype=np.float64)[::-1]) * 0 + 0.0  # placeholder
    grid = t0 - (k - 1 - np.arange(k, dtype=np.float64)) * period
    idx = np.searchsorted(ts, grid, side="right") - 1
    if (idx < 0).any():
        return None
    if ((grid - ts[idx]) > gap_allow).any():
        return None
    lat = raw["lat"][idx]; lon = raw["lon"][idx]; alt_ft = raw["alt"][idx]
    gs = raw["gs"][idx]; trk = raw["trk"][idx]; vr = raw["vr"][idx]
    rad = np.radians(trk)
    vx = gs * KT_TO_KMH * np.sin(rad)
    vy = gs * KT_TO_KMH * np.cos(rad)
    vz = vr * FPM_TO_KMH
    P = fb.PRECISION
    vals = np.stack([lon * P["lon"], lat * P["lat"], alt_ft * 0.3048 * P["alt"],
                     vx * P["vx"], vy * P["vy"], vz * P["vz"]], axis=1)
    return enc.encode_matrix(vals)


def interp_truth(raw: dict, t0: float, horizons: list[int],
                 tol_s: float = 15.0) -> dict | None:
    ts = raw["ts"]
    out = {}
    for c in ["lat", "lon", "alt"]:
        v = np.interp(t0 + np.asarray(horizons, dtype=np.float64), ts, raw[c])
        # validita': osservazione vicina
        for h in horizons:
            j = np.searchsorted(ts, t0 + h)
            d = min(abs(ts[min(j, len(ts) - 1)] - (t0 + h)),
                    abs(ts[max(j - 1, 0)] - (t0 + h)))
            if d > tol_s:
                return None
        out[c] = v
    return out


def collect_anchors(csv_path: str, cfg: dict, vocab_map: dict,
                    max_flights: int, per_flight: int,
                    horizons: list[int]) -> tuple[list, list, list, dict]:
    """Raccoglie ancore valide per ENTRAMBI i modelli + truth neutrale."""
    d = cfg["data"]
    log(f"lettura {Path(csv_path).name} ...")
    df = pd.read_csv(csv_path, header=None, usecols=C.USECOLS, dtype=C.DTYPES,
                     low_memory=False)
    df.columns = C.NEW_NAMES
    df = clean_dataframe(df, d)
    log(f"  righe pulite: {len(df)}")

    sc = cfg["scales"]
    step_s = float(d["step_s"])
    wins, infos, fb_raws = [], [], []
    counts = {"flights_seen": 0, "flights_used": 0, "anchors_fp": 0,
              "anchors_kept": 0}
    max_h = max(horizons)
    for icao, gdf in df.groupby("icao24", sort=False, observed=True):
        if counts["flights_used"] >= max_flights:
            break
        counts["flights_seen"] += 1
        sel = gdf.sort_values("ts")
        first = sel.iloc[0]
        ids = [vocab_map["ac_type"].get(str(first["ac_type"]), 0),
               vocab_map["operator"].get(str(first["operator"]), 0),
               vocab_map["dep"].get(str(first["dep"]), 0),
               vocab_map["arr"].get(str(first["arr"]), 0)]
        raw = dict(ts=sel["ts"].to_numpy(np.float64),
                   lat=sel["lat"].to_numpy(np.float64),
                   lon=sel["lon"].to_numpy(np.float64),
                   alt=sel["alt"].to_numpy(np.float64),
                   gs=sel["gs"].to_numpy(np.float64),
                   trk=sel["trk"].to_numpy(np.float64),
                   vr=sel["vrate"].to_numpy(np.float64))
        ts_np = sel["ts"].to_numpy(np.int64)
        segs = segment_flights(ts_np, sel["lat"].to_numpy(np.float32),
                               sel["lon"].to_numpy(np.float32),
                               d["seg_gap_s"], d["seg_max_speed_ms"],
                               d.get("seg_jump_m", 2000.0))
        kept = 0
        for sidx in segs:
            if kept >= per_flight:
                break
            if len(sidx) < 5:
                continue
            if ts_np[sidx[-1]] - ts_np[sidx[0]] < 240 + max_h:
                continue
            seg = resample_segment(
                ts_np[sidx], sel["lat"].to_numpy(np.float32)[sidx],
                sel["lon"].to_numpy(np.float32)[sidx],
                sel["trk"].to_numpy(np.float32)[sidx],
                sel["alt"].to_numpy(np.float32)[sidx],
                sel["gs"].to_numpy(np.float64)[sidx],
                sel["vrate"].to_numpy(np.float32)[sidx],
                sel["ground"].to_numpy(np.int8)[sidx],
                step_s=step_s, smooth_k=d["smooth_k"],
                valid_gap_s=d["valid_gap_s"],
                target_smooth=d.get("target_smooth", 3))
            win = extract_windows(seg, horizons, d["history_s"], step_s,
                                  d["stride_s"], 10 ** 9, sc,
                                  cfg["train"]["phase_threshold_ms"])
            if win is None:
                continue
            n = win.pop("n")
            counts["anchors_fp"] += n
            # need FB history: raw coverage back to t0-240-120
            for wi in range(n):
                if kept >= per_flight:
                    break
                t0 = float(win["ts0"][wi])
                if t0 - raw["ts"][0] < 360:
                    continue
                if raw["ts"][-1] - t0 < max_h:
                    continue
                truth = interp_truth(raw, t0, horizons)
                if truth is None:
                    continue
                # FB history check (period 30, k=9, gap 120) - fatto dopo
                # per non importare fb qui: controllo copertura grezza
                j = np.searchsorted(raw["ts"], t0 - 240)
                if j < 1 or (t0 - 240) - raw["ts"][j - 1] > 120:
                    continue
                w1 = {k2: (v[wi] if isinstance(v, np.ndarray) else v)
                      for k2, v in win.items()}
                w1["x"] = np.asarray(w1["x"], dtype=np.float32)
                wins.append(w1)
                infos.append({"t0": t0, "ids": ids,
                              "lat0": float(w1["lat0"]), "lon0": float(w1["lon0"]),
                              "z0": float(w1["z0"]), "gs0": float(w1["gs0"]),
                              "trk0": float(w1["trk0"]), "vr0": float(w1["vr0"]),
                              "truth": truth, "icao": str(icao)})
                fb_raws.append(raw)
                kept += 1
        if kept:
            counts["flights_used"] += 1
    counts["anchors_kept"] = len(wins)
    log(f"  voli visti: {counts['flights_seen']}, usati: {counts['flights_used']}, "
        f"ancore FP: {counts['anchors_fp']}, tenute: {counts['anchors_kept']}")
    return wins, infos, fb_raws, counts


def fp_predict_batch(wins: list, infos: list, model, dm: dict,
                     device: torch.device, horizons: list[int],
                     batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    sc = dm["scales"]
    H = len(horizons)
    model_h = [int(h) for h in dm["horizons_s"]]
    h_idx = [model_h.index(int(h)) for h in horizons]
    P_lat = np.zeros((len(wins), H)); P_lon = np.zeros_like(P_lat)
    P_alt = np.zeros_like(P_lat)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(wins), batch_size):
            sl = wins[s:s + batch_size]
            xb = torch.from_numpy(np.stack([w["x"] for w in sl])).to(device)
            st = torch.tensor([[w["z0"] / sc["z0"], w["gs0"] / sc["gs"]] for w in
                               [infos[i] for i in range(s, s + len(sl))]],
                              dtype=torch.float32).to(device)
            cb = torch.tensor([infos[i]["ids"] for i in range(s, s + len(sl))],
                              dtype=torch.long).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                out = model(xb, st, cb)
            xy_res = out["xy"].float().cpu().numpy()
            z_res = out["z"].float().cpu().numpy()
            for j, k in enumerate(range(s, s + len(sl))):
                info = infos[k]
                state = np.array([[info["lat0"], info["lon0"], info["z0"],
                                   info["gs0"], info["trk0"], info["vr0"]]])
                cv_xy, cv_z = baseline_cv(state, np.asarray(horizons, float))
                pxy = xy_res[j][h_idx] * sc["xy"] + cv_xy[0]
                plat, plon = latlon_from_enu(pxy[:, 0], pxy[:, 1],
                                             np.array([info["lat0"]]),
                                             np.array([info["lon0"]]))
                P_lat[k] = plat; P_lon[k] = plon
                P_alt[k] = info["z0"] + z_res[j][h_idx] * sc["z"]
    return P_lat, P_lon, P_alt


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description="Confronto FlightPREV vs FlightBERT.")
    ap.add_argument("--fp-run", nargs="+", default=["runs/transformer_v1/best.pt"],
                    help="uno o due checkpoint FlightPREV (con due: + ensemble media)")
    ap.add_argument("--fb-checkpoints", nargs="+",
                    default=["FlightBert++/runs/fb_v3/best.pt",
                             "FlightBert++/runs/flightbert_gpu/best.pt"])
    ap.add_argument("--csv", default="Italy_24-04-07.csv")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default="reports/compare")
    ap.add_argument("--max-flights", type=int, default=500)
    ap.add_argument("--per-flight", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    t_all = time.time()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"device: {device}")
    cfg = load_config(args.config)
    out = ensure_dir(args.out)

    vocab_map = read_json(Path(args.data) / "vocab.json")["map"]
    fp_models = []
    for rp in args.fp_run:
        m, ck = load_model_from_checkpoint(rp, device)
        fp_models.append((m, ck))
        log(f"FlightPREV caricato da {rp} "
            f"({ck['model_cfg']['arch']} d={ck['model_cfg']['d_model']} "
            f"L={ck['model_cfg']['n_layers']}, epoca {ck.get('epoch')})")
    dm = fp_models[0][1]["data_meta"]
    horizons = COMMON_H

    wins, infos, fb_raws, counts = collect_anchors(
        args.csv, cfg, vocab_map, args.max_flights, args.per_flight, horizons)
    if len(wins) == 0:
        raise SystemExit("nessuna ancora comune trovata")
    # sottocampiona a max 4000 per i tempi di rollout iterativo
    if len(wins) > 4000:
        rng = np.random.default_rng(0)
        take = np.sort(rng.choice(len(wins), 4000, replace=False))
        wins = [wins[i] for i in take]
        infos = [infos[i] for i in take]
        fb_raws = [fb_raws[i] for i in take]
    N = len(wins)
    log(f"finestre di confronto: {N}")

    T_lat = np.stack([infos[i]["truth"]["lat"] for i in range(N)])
    T_lon = np.stack([infos[i]["truth"]["lon"] for i in range(N)])
    T_alt_ft = np.stack([infos[i]["truth"]["alt"] for i in range(N)])
    T_alt = T_alt_ft * 0.3048

    results = {}

    # ---- FlightPREV (1 o 2 modelli + eventuale ensemble) -------------------
    t0 = time.time()
    fp_preds = []
    for m, _ in fp_models:
        fp_preds.append(fp_predict_batch(wins, infos, m, dm, device,
                                         horizons, args.batch_size))
    results["flightprev"] = {"lat": fp_preds[0][0], "lon": fp_preds[0][1],
                             "alt": fp_preds[0][2],
                             "seconds": time.time() - t0}
    log(f"  FlightPREV[0]: {N} predizioni in {results['flightprev']['seconds']:.1f}s")
    methods = ["flightprev"]
    if len(fp_preds) == 2:
        ens = tuple((a + b) / 2 for a, b in zip(fp_preds[0], fp_preds[1]))
        results["flightprev_ens"] = {"lat": ens[0], "lon": ens[1], "alt": ens[2]}
        methods.append("flightprev_ens")
        log("  ensemble flightprev(v1+v2): media delle predizioni")

    # ---- FlightBERT --------------------------------------------------------
    fb = load_fb_module()
    fb_results = {}
    for ckpt in args.fb_checkpoints:
        name = Path(ckpt).parent.name
        model, enc, fcfg, info = build_fb_model(fb, ckpt, device)
        period, k = info["period"], info["k"]
        log(f"FlightBERT[{name}]: period={period}s k={k} n={info['n']} "
            f"direct={info['direct']} bits={info['bits']} (epoca {info['epoch']})")
        t0 = time.time()
        X, keep = [], []
        for i in range(N):
            be = fb_history_be(fb_raws[i], infos[i]["t0"], period, k,
                               4.0 * period, fb, enc)
            if be is not None:
                X.append(be)
                keep.append(i)
        X = np.stack(X)
        log(f"  input BE: {X.shape} (tenute {len(keep)}/{N})")
        if info["direct"]:
            n_steps = max(int(h / period) for h in horizons)
            dec = fb_predict_direct(fb, model, enc, X, n_steps, device)
            step_of = {h: int(h / period) - 1 for h in horizons}
            idx = [step_of[h] for h in horizons]
            pred = dec[:, idx]
        else:
            # rollout iterativo fino al max orizzonte comune multiplo del periodo
            n_steps = max(int(round(h / period)) for h in horizons)
            dec = fb_predict_iterative(fb, model, enc, X, n_steps, device)
            pred = np.zeros((len(X), len(horizons), 6))
            for j, h in enumerate(horizons):
                s_exact = h / period
                s0 = int(np.floor(s_exact))
                if abs(s_exact - round(s_exact)) < 1e-9:
                    pred[:, j] = dec[:, int(round(s_exact)) - 1]
                else:  # interpolazione lineare tra step (es. 30 s con passo 20)
                    a = s_exact - s0
                    pred[:, j] = (1 - a) * dec[:, s0 - 1] + a * dec[:, s0]
        secs = time.time() - t0
        full = {"lat": np.full((N, len(horizons)), np.nan),
                "lon": np.full((N, len(horizons)), np.nan),
                "alt": np.full((N, len(horizons)), np.nan)}
        kk = np.asarray(keep)
        full["lon"][kk] = pred[:, :, 0]
        full["lat"][kk] = pred[:, :, 1]
        full["alt"][kk] = pred[:, :, 2]
        fb_results[name] = {**full, "seconds": secs, "kept": keep,
                            "info": info}
        log(f"  FlightBERT[{name}]: {len(keep)} predizioni in {secs:.1f}s")
    results.update({f"flightbert_{k}": v for k, v in fb_results.items()})

    # ---- baseline CV + persistenza -----------------------------------------
    cv_lat = np.zeros((N, len(horizons))); cv_lon = np.zeros_like(cv_lat)
    cv_alt = np.zeros_like(cv_lat)
    for i in range(N):
        info = infos[i]
        state = np.array([[info["lat0"], info["lon0"], info["z0"],
                           info["gs0"], info["trk0"], info["vr0"]]])
        cxy, cz = baseline_cv(state, np.asarray(horizons, float))
        la, lo = latlon_from_enu(cxy[0, :, 0], cxy[0, :, 1],
                                 np.array([info["lat0"]]), np.array([info["lon0"]]))
        cv_lat[i] = la; cv_lon[i] = lo; cv_alt[i] = info["z0"] + cz[0]
    results["cv"] = {"lat": cv_lat, "lon": cv_lon, "alt": cv_alt}
    results["persist"] = {
        "lat": np.stack([[infos[i]["lat0"]] * len(horizons) for i in range(N)]),
        "lon": np.stack([[infos[i]["lon0"]] * len(horizons) for i in range(N)]),
        "alt": np.stack([[infos[i]["z0"]] * len(horizons) for i in range(N)])}

    # ---- metriche -----------------------------------------------------------
    methods = methods + [f"flightbert_{k}" for k in fb_results] + ["cv", "persist"]
    ade = {}
    maez = {}
    for m in methods:
        r = results[m]
        e = np.full((N, len(horizons)), np.nan)
        ok = ~(np.isnan(r["lat"]).any(axis=1))
        e[ok] = dist_m(T_lat[ok], T_lon[ok], r["lat"][ok], r["lon"][ok])
        ade[m] = e
        maez[m] = np.abs(r["alt"] - T_alt)
    cov = {m: float((~np.isnan(ade[m][:, 0])).mean()) for m in methods}

    rows = []
    for j, h in enumerate(horizons):
        rows.append([f"{h}"] + [f"{np.nanmean(ade[m][:, j]):.0f}" for m in methods])
    print_table(["orizzonte_s"] + methods, rows)
    print_table(["metodo", "copertura", "FDE@120 (m)", "MAE quota (m)"],
                [[m, f"{cov[m]*100:.1f}%",
                  f"{np.nanmean(ade[m][:, -1]):.0f}",
                  f"{np.nanmean(maez[m]):.0f}"] for m in methods])
    for m in methods:
        e120 = ade[m][:, -1]
        e120 = e120[~np.isnan(e120)]
        if len(e120):
            print(f"{m}: @120s mediana {np.median(e120):.0f} m | "
                  f"p90 {np.percentile(e120, 90):.0f} m | n={len(e120)}")

    # per fase (da vr0)
    vr0 = np.array([infos[i]["vr0"] for i in range(N)])
    phase0 = phase_from_vr(vr0, float(cfg["train"]["phase_threshold_ms"]))
    phase_names = ["cruise", "climb", "descent"]
    phase_tab = []
    for p, pname in enumerate(phase_names):
        mask = phase0 == p
        phase_tab.append(
            [pname, str(int(mask.sum()))] +
            [f"{np.nanmean(ade[m][mask][:, -1]):.0f}" for m in methods])
    print_table(["fase", "n"] + [f"FDE120 {m}" for m in methods], phase_tab)

    # ---- salvataggi ----------------------------------------------------------
    with open(out / "metrics_compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "horizon_s", "ade_m", "mae_alt_m", "coverage"])
        for m in methods:
            for j, h in enumerate(horizons):
                w.writerow([m, h, f"{np.nanmean(ade[m][:, j]):.3f}",
                            f"{np.nanmean(maez[m][:, j]):.3f}", f"{cov[m]:.4f}"])
    summary = {
        "n_windows": N, "horizons_s": horizons,
        "counts": counts, "coverage": cov,
        "ade_per_horizon_m": {m: [float(np.nanmean(ade[m][:, j]))
                                  for j in range(len(horizons))] for m in methods},
        "fde120_m": {m: float(np.nanmean(ade[m][:, -1])) for m in methods},
        "mae_alt_m": {m: float(np.nanmean(maez[m])) for m in methods},
        "fb_info": {k: v["info"] for k, v in fb_results.items()},
        "fp_checkpoints": args.fp_run,
        "total_seconds": time.time() - t_all,
    }
    with open(out / "summary_compare.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"report salvati in {out.resolve()}")

    # ---- grafici --------------------------------------------------------------
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                         "font.size": 9})
    colors = {"flightprev": "#d62728", "flightprev_ens": "#ff7f0e",
              "flightbert_fb_v3": "#1f77b4",
              "flightbert_flightbert_gpu": "#17becf", "cv": "#7f7f7f",
              "persist": "#bcbd22"}
    H = np.asarray(horizons, float)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in methods:
        y = np.array([np.nanmean(ade[m][:, j]) for j in range(len(horizons))])
        ax.plot(H, y, marker="o", ms=3, color=colors.get(m), label=m)
    ax.set_yscale("log")
    ax.set_xlabel("orizzonte (s)")
    ax.set_ylabel("ADE (m, scala log)")
    ax.set_title(f"Confronto modelli — test 07/04 (n={N}, truth raw interpolata)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ade_curves_compare.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ref = ade.get("flightprev_ens", ade["flightprev"])[:, -1]
    ref = ref[~np.isnan(ref)]
    xmax = float(np.percentile(ref, 99.5)) if len(ref) else 20000.0
    xs = np.linspace(0, xmax, 400)
    for m in methods:
        e = np.sort(ade[m][:, -1])
        e = e[~np.isnan(e)]
        if len(e) == 0:
            continue
        ax.plot(xs, np.searchsorted(e, xs) / len(e), color=colors.get(m), label=m)
    ax.set_xlabel("errore a 120 s (m)")
    ax.set_ylabel("frazione finestre con errore < x")
    ax.set_title("CDF errore a 120 s (confronto)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cdf_120_compare.png")
    plt.close(fig)

    xpos = np.arange(len(phase_names))
    w = 0.16
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k, m in enumerate(methods):
        vals = [float(np.nanmean(ade[m][phase0 == p][:, -1])) for p in range(3)]
        ax.bar(xpos + (k - (len(methods) - 1) / 2) * w, vals, w,
               color=colors.get(m), label=m)
    ax.set_xticks(xpos, phase_names)
    ax.set_ylabel("errore a 120 s (m)")
    ax.set_title("Errore a 120 s per fase di volo (confronto)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "err_by_phase_compare.png")
    plt.close(fig)
    log("grafici salvati")


if __name__ == "__main__":
    main()
