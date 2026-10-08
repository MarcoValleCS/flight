"""Valutazione del modello sul test set + confronto con baseline + grafici.

Uso:
    python -m src.evaluate --run runs/transformer_v1/best.pt --split test

Produce in reports/:
    metrics_<split>.csv     metriche per orizzonte e per metodo
    summary_<split>.json    riepilogo (ADE, FDE@120, MAE quota, percentili, fasi)
    ade_curves.png          errore ADE vs orizzonte (modello e baseline)
    error_cdf_120s.png      CDF dell'errore a 120 s
    trajectory_samples.png  esempi di traiettoria prevista vs reale
    error_by_phase.png      errore a 120 s per fase di volo
    training_curves.png     curve di addestramento (dal log.csv)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .baselines import baseline_ca, baseline_ct, baseline_cv, baseline_zero
from .dataset import TrajWindows
from .models import load_model_from_checkpoint
from .utils import (ensure_dir, latlon_from_enu, log, print_table, read_json,
                    setup_console)

METHODS = ["model", "cv", "ct", "ca", "zero"]
COLORS = {"model": "#d62728", "cv": "#1f77b4", "ct": "#2ca02c",
          "ca": "#9467bd", "zero": "#7f7f7f"}


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description="Valutazione modello.")
    ap.add_argument("--run", required=True, help="checkpoint (best.pt)")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--out", default="reports")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--samples", type=int, default=20000, help="finestre conservate per i grafici")
    ap.add_argument("--max-windows", type=int, default=0, help="limite finestre (0=tutte)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load_model_from_checkpoint(args.run, device)
    dm = ckpt["data_meta"]
    sc_xy = float(dm["scales"]["xy"])
    sc_z = float(dm["scales"]["z"])
    horizons = np.asarray(dm["horizons_s"], dtype=np.float64)
    step_s = float(dm["step_s"])
    phase_names = dm.get("phase_names", ["cruise", "climb", "descent"])
    log(f"modello caricato da {args.run} (epoca {ckpt.get('epoch')}, "
        f"val ADE {ckpt.get('val_ade_m', float('nan')):.1f} m)")

    split_dir = Path(args.data) / args.split
    meta = read_json(split_dir / "meta.json")
    ds = TrajWindows(split_dir)
    n_windows = len(ds) if args.max_windows <= 0 else min(len(ds), args.max_windows)
    log(f"split {args.split}: {n_windows} finestre")
    if n_windows == 0:
        raise SystemExit("nessuna finestra nello split richiesto")
    ds.n = n_windows
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    H = len(horizons)
    errors = {m: [] for m in METHODS}
    zerr = []
    phase0_all = []
    samples: dict = {k: [] for k in
                     ["hist", "truth", "pred", "cv", "lat0", "lon0", "phase0", "ts0"]}
    n_samp = 0
    rng = np.random.default_rng(0)

    for bi, batch in enumerate(loader):
        xb = batch["x"].to(device)
        sb = batch["static"].to(device)
        cb = batch["cats"].to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=(device.type == "cuda")):
            out = model(xb, sb, cb)
        pred_res = out["xy"].float().cpu().numpy()
        pred_z = out["z"].float().cpu().numpy()

        xn = batch["x"].numpy().astype(np.float64)
        y_xy = batch["y_xy"].numpy().astype(np.float64)
        y_z = batch["y_z"].numpy().astype(np.float64)
        state = batch["state"].numpy().astype(np.float64)

        cv_xy, _ = baseline_cv(state, horizons)
        truth_xy = y_xy * sc_xy + cv_xy
        pred_xy = pred_res * sc_xy + cv_xy
        ct_xy, _ = baseline_ct(xn, state, step_s, horizons)
        ca_xy, _ = baseline_ca(xn, state, step_s, horizons)
        zero_xy = np.zeros_like(truth_xy)

        errors["model"].append(np.linalg.norm(pred_xy - truth_xy, axis=-1))
        errors["cv"].append(np.linalg.norm(cv_xy - truth_xy, axis=-1))
        errors["ct"].append(np.linalg.norm(ct_xy - truth_xy, axis=-1))
        errors["ca"].append(np.linalg.norm(ca_xy - truth_xy, axis=-1))
        errors["zero"].append(np.linalg.norm(zero_xy - truth_xy, axis=-1))
        zerr.append(np.abs((pred_z - y_z) * sc_z))
        phase0_all.append(batch["phase0"].numpy().astype(np.int64))

        if n_samp < args.samples:
            k = min(args.samples - n_samp, len(xn))
            samples["hist"].append(xn[:k, :, :2] * sc_xy)
            samples["truth"].append(truth_xy[:k])
            samples["pred"].append(pred_xy[:k])
            samples["cv"].append(cv_xy[:k])
            samples["lat0"].append(state[:k, 0])
            samples["lon0"].append(state[:k, 1])
            samples["phase0"].append(batch["phase0"].numpy()[:k])
            samples["ts0"].append(batch["ts0"].numpy()[:k])
            n_samp += k
        if (bi + 1) % 20 == 0:
            log(f"  batch {bi+1} ...")

    errors = {m: np.concatenate(v, axis=0) for m, v in errors.items()}
    zerr = np.concatenate(zerr, axis=0)
    phase0_all = np.concatenate(phase0_all, axis=0)
    samples = {k: np.concatenate(v, axis=0) for k, v in samples.items()}
    N = errors["model"].shape[0]
    log(f"valutate {N} finestre")

    # ------------------------- metriche ------------------------------------
    ade_h = {m: errors[m].mean(axis=0) for m in METHODS}          # (H,)
    fde120 = {m: float(errors[m][:, -1].mean()) for m in METHODS}
    overall = {m: float(errors[m].mean()) for m in METHODS}
    z_mae = float(zerr.mean())

    rows = []
    for i, h in enumerate(horizons):
        rows.append([f"{int(h)}", *[f"{ade_h[m][i]:.1f}" for m in METHODS]])
    print_table(["orizzonte_s", *METHODS], rows)

    summary_rows = [
        ["ADE medio (m)", *[f"{overall[m]:.1f}" for m in METHODS]],
        ["FDE@120s (m)", *[f"{fde120[m]:.1f}" for m in METHODS]],
    ]
    print_table(["metrica", *METHODS], summary_rows)
    print(f"MAE quota (modello): {z_mae:.1f} m")

    e120 = errors["model"][:, -1]
    p50, p90, p95 = np.percentile(e120, [50, 90, 95])
    print(f"errore modello a 120 s: mediana {p50:.0f} m | p90 {p90:.0f} m | p95 {p95:.0f} m")
    imp = {m: 100.0 * (1 - overall["model"] / overall[m]) for m in METHODS if m != "model"}
    print("miglioramento ADE vs baseline: " +
          ", ".join(f"{m} {imp[m]:+.1f}%" for m in ["cv", "ct", "ca", "zero"]))

    # per fase
    phase_stats = {}
    for p, name in enumerate(phase_names):
        mask = phase0_all == p
        if mask.sum() == 0:
            continue
        phase_stats[name] = {
            "n": int(mask.sum()),
            "fde120_model_m": float(errors["model"][mask, -1].mean()),
            "fde120_cv_m": float(errors["cv"][mask, -1].mean()),
            "ade_model_m": float(errors["model"][mask].mean()),
            "ade_cv_m": float(errors["cv"][mask].mean()),
        }
    print_table(["fase", "n", "FDE120 modello", "FDE120 CV", "ADE modello", "ADE CV"],
                [[n, str(s["n"]), f"{s['fde120_model_m']:.0f}", f"{s['fde120_cv_m']:.0f}",
                  f"{s['ade_model_m']:.0f}", f"{s['ade_cv_m']:.0f}"]
                 for n, s in phase_stats.items()])

    # ------------------------- salvataggi ----------------------------------
    out = ensure_dir(args.out)
    with open(out / f"metrics_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "horizon_s", "ade_m"])
        for m in METHODS:
            for i, h in enumerate(horizons):
                w.writerow([m, int(h), f"{ade_h[m][i]:.3f}"])
    summary = {
        "run": str(args.run), "split": args.split, "n_windows": int(N),
        "overall_ade_m": overall, "fde120_m": fde120, "mae_z_m": z_mae,
        "model_120s_percentiles_m": {"p50": float(p50), "p90": float(p90), "p95": float(p95)},
        "improvement_vs": imp, "per_phase": phase_stats,
        "ade_per_horizon_m": {m: [float(v) for v in ade_h[m]] for m in METHODS},
        "horizons_s": [int(h) for h in horizons],
    }
    with open(out / f"summary_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ------------------------- grafici -------------------------------------
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True,
                         "grid.alpha": 0.3, "font.size": 9})

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in METHODS:
        ax.plot(horizons, ade_h[m], marker="o", ms=3, color=COLORS[m], label=m)
    ax.set_yscale("log")
    ax.set_xlabel("orizzonte di previsione (s)")
    ax.set_ylabel("ADE (m, scala log)")
    ax.set_title(f"Errore medio vs orizzonte ({args.split}, n={N})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "ade_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xmax = float(np.percentile(e120, 99.5))
    xs = np.linspace(0, xmax, 400)
    for m in ["model", "cv", "ct"]:
        e = np.sort(errors[m][:, -1])
        cdf = np.searchsorted(e, xs) / len(e)
        ax.plot(xs, cdf, color=COLORS[m], label=m)
    ax.set_xlabel("errore a 120 s (m)")
    ax.set_ylabel("frazione finestre con errore < x")
    ax.set_title("CDF dell'errore a 120 s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "error_cdf_120s.png")
    plt.close(fig)

    # esempi di traiettoria: 3 per fase
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for row, p in enumerate(range(len(phase_names))):
        idx_p = np.where(samples["phase0"] == p)[0]
        picks = rng.choice(idx_p, size=min(3, len(idx_p)), replace=False) if len(idx_p) else []
        for col in range(3):
            ax = axes[row, col]
            if col >= len(picks):
                ax.axis("off")
                continue
            i = picks[col]
            la0 = np.array([samples["lat0"][i]])
            lo0 = np.array([samples["lon0"][i]])
            hl, hg = latlon_from_enu(samples["hist"][i, :, 0], samples["hist"][i, :, 1], la0, lo0)
            tl, tg = latlon_from_enu(samples["truth"][i, :, 0], samples["truth"][i, :, 1], la0, lo0)
            pl, pg = latlon_from_enu(samples["pred"][i, :, 0], samples["pred"][i, :, 1], la0, lo0)
            cl, cg = latlon_from_enu(samples["cv"][i, :, 0], samples["cv"][i, :, 1], la0, lo0)
            fde_m = float(np.linalg.norm(samples["pred"][i, -1] - samples["truth"][i, -1]))
            fde_cv = float(np.linalg.norm(samples["cv"][i, -1] - samples["truth"][i, -1]))
            ax.plot(hg, hl, "-", color="#1f77b4", lw=1.6, label="history")
            ax.plot(tg, tl, "-", color="#2ca02c", lw=1.8, label="reale")
            ax.plot(pg, pl, "--", color="#d62728", lw=1.8, label="modello")
            ax.plot(cg, cl, ":", color="#7f7f7f", lw=1.5, label="CV")
            ax.scatter([hg[-1]], [hl[-1]], c="k", s=12, zorder=5)
            ax.set_title(f"{phase_names[p]} | err@120: modello {fde_m:.0f} m / CV {fde_cv:.0f} m",
                         fontsize=8)
            ax.set_xlabel("lon"); ax.set_ylabel("lat")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle("Esempi di previsione (12 orizzonti, 10-120 s)")
    fig.tight_layout()
    fig.savefig(out / "trajectory_samples.png")
    plt.close(fig)

    if phase_stats:
        names = list(phase_stats.keys())
        xpos = np.arange(len(names))
        w = 0.38
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.bar(xpos - w / 2, [phase_stats[n]["fde120_model_m"] for n in names], w,
               color=COLORS["model"], label="modello")
        ax.bar(xpos + w / 2, [phase_stats[n]["fde120_cv_m"] for n in names], w,
               color=COLORS["cv"], label="CV")
        ax.set_xticks(xpos, names)
        ax.set_ylabel("errore a 120 s (m)")
        ax.set_title("Errore a 120 s per fase di volo")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "error_by_phase.png")
        plt.close(fig)

    log_file = Path(args.run).parent / "log.csv"
    if log_file.exists():
        ep, tl, vl, ade = [], [], [], []
        with open(log_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ep.append(int(row["epoch"]))
                tl.append(float(row["train_loss"]))
                vl.append(float(row["val_loss"]))
                ade.append(float(row["val_ade_m"]))
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        axs[0].plot(ep, tl, label="train")
        axs[0].plot(ep, vl, label="val")
        axs[0].set_yscale("log")
        axs[0].set_xlabel("epoca"); axs[0].set_ylabel("loss")
        axs[0].legend(); axs[0].set_title("Loss")
        axs[1].plot(ep, ade, color=COLORS["model"])
        axs[1].set_xlabel("epoca"); axs[1].set_ylabel("val ADE (m)")
        axs[1].set_title("Val ADE")
        fig.tight_layout()
        fig.savefig(out / "training_curves.png")
        plt.close(fig)

    log(f"report salvati in {out.resolve()}")


if __name__ == "__main__":
    main()
