"""Inferenza: previsione della traiettoria da +10 s a +120 s.

Modalita':
    demo : estrae finestre casuali dal test set, confronta previsione e reale
    csv  : legge un file CSV nel formato originale e predice il futuro
           dell'ultimo stato osservato di un volo

Uso:
    python -m src.predict --run runs/transformer_v1/best.pt --n 6
    python -m src.predict --run runs/transformer_v1/best.pt --csv Italy_24-04-07.csv --icao 4CAFFB
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import columns as C
from .baselines import baseline_cv
from .build_dataset import (clean_dataframe, extract_windows,
                            resample_segment, segment_flights)
from .config import load_config
from .dataset import TrajWindows
from .models import load_model_from_checkpoint
from .utils import (ensure_dir, latlon_from_enu, log, print_table, read_json,
                    setup_console)

PHASE_IT = ["crociera", "salita", "discesa"]


@torch.no_grad()
def model_forward(models, x: np.ndarray, static: np.ndarray, cats: np.ndarray,
                  device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward di uno o piu' modelli; con piu' modelli fa la media (ensemble)."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    st = torch.from_numpy(np.asarray(static, dtype=np.float32))
    ct = torch.from_numpy(np.asarray(cats, dtype=np.int64))
    xys, zs, phs = [], [], []
    for model in models:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(xt.to(device), st.to(device), ct.to(device))
        xys.append(out["xy"].float().cpu().numpy())
        zs.append(out["z"].float().cpu().numpy())
        phs.append(out["phase"].float().softmax(-1).cpu().numpy())
    xy = np.mean(xys, axis=0)
    z = np.mean(zs, axis=0)
    phase = np.mean(phs, axis=0)
    return xy, z, phase


def _first(v) -> float:
    return float(np.ravel(v)[0])


def predict_from_window(win: dict, model, dm: dict, vocab_map: dict | None,
                        ids: list[int] | None, device) -> dict:
    """Predice da un dizionario-finestra (come da build_dataset.extract_windows)."""
    sc = dm["scales"]
    horizons = np.asarray(dm["horizons_s"], dtype=np.float64)
    xin = win["x"]
    if xin.ndim == 2:
        xin = xin[None]
    x = xin.astype(np.float32)
    z0 = _first(win["z0"]); gs0 = _first(win["gs0"])
    static = np.array([[z0 / sc["z0"], gs0 / sc["gs"]]], dtype=np.float32)
    cats = np.array([ids if ids is not None else [0, 0, 0, 0]], dtype=np.int64)
    xy_res, z_res, phase_p = model_forward(model, x, static, cats, device)

    lat0 = _first(win["lat0"]); lon0 = _first(win["lon0"])
    trk0 = _first(win["trk0"]); vr0 = _first(win["vr0"])
    state = np.array([[lat0, lon0, z0, gs0, trk0, vr0]])
    cv_xy, cv_z = baseline_cv(state, horizons)
    pred_xy = xy_res[0] * sc["xy"] + cv_xy[0]
    pred_z = z_res[0] * sc["z"] + cv_z[0]

    plat, plon = latlon_from_enu(pred_xy[:, 0], pred_xy[:, 1],
                                 np.array([lat0]), np.array([lon0]))
    palt = z0 + pred_z

    y_xy = win.get("y_xy")
    y_z = win.get("y_z")
    err = None
    if y_xy is not None and np.isfinite(y_xy).all():
        true_xy = y_xy[0] * sc["xy"] + cv_xy[0]
        true_z = y_z[0] * sc["z"] + cv_z[0]
        tlat, tlon = latlon_from_enu(true_xy[:, 0], true_xy[:, 1],
                                     np.array([lat0]), np.array([lon0]))
        err = np.linalg.norm(pred_xy - true_xy, axis=-1)
        true_out = dict(lat=tlat, lon=tlon, alt=z0 + true_z)
    else:
        true_out = None

    return {
        "horizons": horizons, "lat": plat, "lon": plon, "alt": palt,
        "true": true_out, "err_m": err,
        "phase_probs": phase_p[0],
        "hist_xy": (win["x"][0] if win["x"].ndim == 3 else win["x"])[:, :2].astype(np.float64) * sc["xy"],
        "pred_xy": pred_xy, "cv_xy": cv_xy[0],
        "lat0": lat0, "lon0": lon0,
        "phase0": int(np.ravel(win.get("phase0", [0]))[0]),
    }


def print_prediction(res: dict, title: str) -> None:
    rows = []
    for i, h in enumerate(res["horizons"]):
        row = [f"+{int(h)}s", f"{res['lat'][i]:.4f}", f"{res['lon'][i]:.4f}",
               f"{res['alt'][i]:.0f}"]
        if res["true"] is not None:
            row += [f"{res['true']['lat'][i]:.4f}", f"{res['true']['lon'][i]:.4f}",
                    f"{res['true']['alt'][i]:.0f}", f"{res['err_m'][i]:.0f}"]
        rows.append(row)
    headers = ["t", "lat pred", "lon pred", "alt pred (m)"]
    if res["true"] is not None:
        headers += ["lat reale", "lon reale", "alt reale (m)", "errore (m)"]
    print(f"\n{title}")
    print_table(headers, rows)
    p = res["phase_probs"]
    print(f"fase prevista a +60 s: {PHASE_IT[int(np.argmax(p))]} "
          f"(p={p.max():.2f})")


def plot_predictions(results: list[dict], out_png: Path, title: str) -> None:
    n = len(results)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    for k, res in enumerate(results):
        ax = axes[k // cols][k % cols]
        la0 = np.array([res["lat0"]]); lo0 = np.array([res["lon0"]])
        hl, hg = latlon_from_enu(res["hist_xy"][:, 0], res["hist_xy"][:, 1], la0, lo0)
        pl, pg = latlon_from_enu(res["pred_xy"][:, 0], res["pred_xy"][:, 1], la0, lo0)
        cl, cg = latlon_from_enu(res["cv_xy"][:, 0], res["cv_xy"][:, 1], la0, lo0)
        ax.plot(hg, hl, "-", color="#1f77b4", lw=1.6, label="storia")
        ax.plot(pg, pl, "--", color="#d62728", lw=1.8, label="previsione")
        ax.plot(cg, cl, ":", color="#7f7f7f", lw=1.4, label="CV")
        if res["true"] is not None:
            ax.plot(res["true"]["lon"], res["true"]["lat"], "-", color="#2ca02c",
                    lw=1.6, label="reale")
        ax.scatter([hg[-1]], [hl[-1]], c="k", s=14, zorder=5)
        t = pd.to_datetime(int(res.get("ts0", 0)), unit="s", utc=True).strftime("%d/%m %H:%M")
        ax.set_title(f"{res.get('icao', '')} {t} " +
                     (f"| err@120 {res['err_m'][-1]:.0f} m" if res["err_m"] is not None else ""),
                     fontsize=8)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        if k == 0:
            ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    log(f"grafico salvato in {out_png}")


# --------------------------------------------------------------------------
def mode_demo(args, model, dm, device) -> None:
    ds = TrajWindows(Path(args.data) / args.split)
    if len(ds) == 0:
        raise SystemExit("dataset vuoto")
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    results = []
    for i in idxs:
        item = ds[int(i)]
        # ricostruisce un dizionario "finestra" compatto
        win = {
            "x": item["x"].numpy(), "y_xy": item["y_xy"].numpy()[None],
            "y_z": item["y_z"].numpy()[None],
            "lat0": float(item["state"][0]), "lon0": float(item["state"][1]),
            "z0": float(item["state"][2]), "gs0": float(item["state"][3]),
            "trk0": float(item["state"][4]), "vr0": float(item["state"][5]),
            "phase0": np.array([int(item["phase0"])]),
        }
        res = predict_from_window(win, model, dm, None,
                                  item["cats"].numpy().tolist(), device)
        res["ts0"] = int(item["ts0"]); res["icao"] = ""
        results.append(res)
    print_prediction(results[0], f"Esempio demo (finestra #{idxs[0]}): previsione 10-120 s")
    if all(r["err_m"] is not None for r in results):
        errs = np.stack([r["err_m"] for r in results])
        print(f"\nerrore medio sulle {len(results)} finestre demo: "
              f"+10s {errs[:,0].mean():.0f} m | +60s {errs[:,5].mean():.0f} m | "
              f"+120s {errs[:,-1].mean():.0f} m")
    out_png = ensure_dir(args.out) / "predict_demo.png"
    plot_predictions(results, out_png, "Previsioni demo dal test set")


def mode_csv(args, model, dm, device) -> None:
    cfg = load_config(args.config)
    log(f"lettura {args.csv} ...")
    df = pd.read_csv(args.csv, header=None, usecols=C.USECOLS, dtype=C.DTYPES,
                     low_memory=False, nrows=args.max_rows)
    df.columns = C.NEW_NAMES
    df = clean_dataframe(df, cfg["data"])
    if len(df) == 0:
        raise SystemExit("nessuna riga valida nel CSV")

    if args.icao:
        key = args.icao.upper()
        cats = [c for c in df["icao24"].cat.categories if str(c).upper() == key]
        sel = df[df["icao24"].isin(cats)] if cats else df.iloc[0:0]
        if len(sel) == 0:
            raise SystemExit(f"icao {args.icao} non trovato")
    else:
        counts = df.groupby("icao24", observed=True).size()
        icao = counts.idxmax()
        sel = df[df["icao24"] == icao]
        log(f"volo scelto automaticamente: {icao} ({len(sel)} punti)")
    sel = sel.sort_values("ts")
    icao = str(sel["icao24"].iloc[0])

    d = cfg["data"]
    segs = segment_flights(sel["ts"].to_numpy(np.int64), sel["lat"].to_numpy(np.float32),
                           sel["lon"].to_numpy(np.float32), d["seg_gap_s"],
                           d["seg_max_speed_ms"])
    sidx = segs[-1]                      # ultimo segmento continuo
    if len(sidx) < 5:
        raise SystemExit("segmento troppo corto")
    seg = resample_segment(
        sel["ts"].to_numpy(np.int64)[sidx], sel["lat"].to_numpy(np.float32)[sidx],
        sel["lon"].to_numpy(np.float32)[sidx], sel["trk"].to_numpy(np.float32)[sidx],
        sel["alt"].to_numpy(np.float32)[sidx], sel["gs"].to_numpy(np.float32)[sidx],
        sel["vrate"].to_numpy(np.float32)[sidx], sel["ground"].to_numpy(np.int8)[sidx],
        step_s=d["step_s"], smooth_k=d["smooth_k"], valid_gap_s=d["valid_gap_s"])
    win = extract_windows(seg, list(dm["horizons_s"]), int(dm["history_s"]),
                          float(dm["step_s"]), d["stride_s"], d["max_per_flight"],
                          dm["scales"], cfg["train"]["phase_threshold_ms"],
                          last_only=True, future_required=False)
    if win is None:
        raise SystemExit("storia insufficiente per la predizione (servono >=60 s)")

    vocab_map = None
    if (Path(args.data) / "vocab.json").exists():
        vocab_map = read_json(Path(args.data) / "vocab.json")["map"]
    first = sel.iloc[0]
    ids = [0, 0, 0, 0]
    if vocab_map is not None:
        ids = [
            vocab_map["ac_type"].get(str(first["ac_type"]), 0),
            vocab_map["operator"].get(str(first["operator"]), 0),
            vocab_map["dep"].get(str(first["dep"]), 0),
            vocab_map["arr"].get(str(first["arr"]), 0),
        ]
    res = predict_from_window(win, model, dm, vocab_map, ids, device)
    res["ts0"] = int(win["ts0"][0]); res["icao"] = icao
    dep, arr = first["dep"], first["arr"]
    route = f" [{dep}->{arr}]" if isinstance(dep, str) and isinstance(arr, str) else ""
    print_prediction(res, f"Volo {icao}{route}  tipo {first['ac_type']}  "
                          f"(ultimo stato {pd.to_datetime(res['ts0'], unit='s', utc=True):%Y-%m-%d %H:%M:%S} UTC)")

    out_png = ensure_dir(args.out) / f"predict_{icao}.png"
    plot_predictions([res], out_png, f"Previsione volo {icao} (+10..+120 s)")

    if args.export:
        out_csv = Path(args.export)
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("t_s,lat,lon,alt_m")
            if res["true"] is not None:
                f.write(",lat_true,lon_true,alt_true_m,err_m")
            f.write("\n")
            for i, h in enumerate(res["horizons"]):
                f.write(f"{int(h)},{res['lat'][i]:.6f},{res['lon'][i]:.6f},{res['alt'][i]:.1f}")
                if res["true"] is not None:
                    f.write(f",{res['true']['lat'][i]:.6f},{res['true']['lon'][i]:.6f},"
                            f"{res['true']['alt'][i]:.1f},{res['err_m'][i]:.1f}")
                f.write("\n")
        log(f"predizioni esportate in {out_csv}")


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description="Inferenza traiettoria 10-120 s.")
    ap.add_argument("--run", nargs="+", required=True,
                    help="uno o due checkpoint (con due: ensemble a media)")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n", type=int, default=6, help="finestre demo")
    ap.add_argument("--csv", default=None, help="modalita' CSV: file di traccia")
    ap.add_argument("--icao", default=None, help="icao24 del volo nel CSV")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--export", default=None, help="CSV di output delle predizioni")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models, dm = [], None
    for rp in args.run:
        model, ckpt = load_model_from_checkpoint(rp, device)
        models.append(model)
        dm = ckpt["data_meta"]
        log(f"modello {ckpt['model_cfg']['arch']} caricato da {rp} "
            f"(epoca {ckpt.get('epoch')})")
    model = models[0] if len(models) == 1 else models

    if args.csv:
        if args.export is None:
            stem = args.icao or "ultimo"
            args.export = str(Path(args.out) / f"predict_{stem}.csv")
        mode_csv(args, model, dm, device)
    else:
        mode_demo(args, model, dm, device)


if __name__ == "__main__":
    main()
