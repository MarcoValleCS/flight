"""Addestramento del modello di previsione traiettoria.

Uso tipico:
    python -m src.train --data data/processed --run runs/transformer_v1

Caratteristiche:
    - AMP bf16 su GPU (RTX 40xx), TF32
    - scheduler coseno con warmup, early stopping
    - loss Huber su (E,N) + quota + CE ausiliaria sulla fase di volo
    - checkpoint best/last + log CSV per epoca
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import load_config, save_config
from .dataset import TrajWindows
from .models import build_model, count_parameters
from .utils import ensure_dir, log, read_json, seed_all, setup_console


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def compute_loss(out: dict, batch: dict, tcfg: dict) -> tuple[torch.Tensor, dict]:
    e_xy = out["xy"].float() - batch["y_xy"]
    e_z = out["z"].float() - batch["y_z"]
    delta = float(tcfg["huber_delta"])
    if "logvar" in out:
        lv = out["logvar"].float()
        d2 = (e_xy ** 2).sum(dim=-1)
        l_xy = (0.5 * (d2 * torch.exp(-lv) + lv)).mean()
    else:
        l_xy = F.smooth_l1_loss(e_xy, torch.zeros_like(e_xy), beta=delta)
    l_z = F.smooth_l1_loss(e_z, torch.zeros_like(e_z), beta=delta)
    l_ph = F.cross_entropy(out["phase"].float(), batch["phase"])
    loss = tcfg["w_xy"] * l_xy + tcfg["w_z"] * l_z + tcfg["w_phase"] * l_ph
    return loss, {"xy": l_xy.item(), "z": l_z.item(), "ph": l_ph.item()}


@torch.no_grad()
def eval_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device,
               tcfg: dict, sc_xy: float, sc_z: float,
               max_steps: int | None = None) -> dict:
    model.eval()
    tot_loss = 0.0
    ade = 0.0
    fde = 0.0
    maez = 0.0
    n = 0
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            out = model(batch["x"], batch["static"], batch["cats"])
        loss, _ = compute_loss(out, batch, tcfg)
        exy = (out["xy"].float() - batch["y_xy"]) * sc_xy
        d = torch.linalg.norm(exy, dim=-1)          # (B, H) metri
        ez = (out["z"].float() - batch["y_z"]) * sc_z
        b = batch["x"].size(0)
        tot_loss += loss.item() * b
        ade += d.mean().item() * b
        fde += d[:, -1].mean().item() * b
        maez += ez.abs().mean().item() * b
        n += b
    n = max(n, 1)
    return {"loss": tot_loss / n, "ade_m": ade / n, "fde120_m": fde / n, "maez_m": maez / n}


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description="Addestramento modello traiettorie.")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--run", default="runs/transformer_v1")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--arch", default=None, choices=["transformer", "gru"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="limite step/epoca (debug)")
    args = ap.parse_args()

    over = {}
    if args.arch:
        over.setdefault("model", {})["arch"] = args.arch
    if args.epochs:
        over.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size:
        over.setdefault("train", {})["batch_size"] = args.batch_size
    if args.lr:
        over.setdefault("train", {})["lr"] = args.lr
    if args.num_workers is not None:
        over.setdefault("train", {})["num_workers"] = args.num_workers
    cfg = load_config(args.config, **over)
    tcfg = cfg["train"]

    seed_all(int(tcfg["seed"]))
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"device: {device}")

    data_dir = Path(args.data)
    meta_tr = read_json(data_dir / "train" / "meta.json")
    n_feat = len(meta_tr["feature_names"])
    n_hist = int(round(meta_tr["history_s"] / meta_tr["step_s"])) + 1
    n_h = len(meta_tr["horizons_s"])
    sc = meta_tr["scales"]
    vocab_sizes = meta_tr["vocab_sizes"]
    log(f"train: {meta_tr['n_windows']} finestre | feat={n_feat} hist={n_hist} "
        f"orizzonti={n_h} | vocab {vocab_sizes}")

    model = build_model(cfg["model"], vocab_sizes, n_feat, n_hist, n_h).to(device)
    log(f"modello {cfg['model']['arch']}: {count_parameters(model):,} parametri")

    ds_tr = TrajWindows(data_dir / "train")
    ds_va = TrajWindows(data_dir / "val")
    nw = int(tcfg["num_workers"])
    pin = device.type == "cuda"
    g = torch.Generator().manual_seed(int(tcfg["seed"]))
    ld_tr = DataLoader(ds_tr, batch_size=int(tcfg["batch_size"]), shuffle=True,
                       num_workers=nw, pin_memory=pin, drop_last=True,
                       persistent_workers=(nw > 0), generator=g)
    ld_va = DataLoader(ds_va, batch_size=max(1024, int(tcfg["batch_size"])), shuffle=False,
                       num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))

    steps_per_epoch = len(ld_tr)
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    total_steps = max(1, steps_per_epoch * int(tcfg["epochs"]))
    warmup = min(int(tcfg["warmup_steps"]), max(1, total_steps // 10))
    lr0 = float(tcfg["lr"])

    opt = torch.optim.AdamW(model.parameters(), lr=lr0,
                            weight_decay=float(tcfg["weight_decay"]))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    run_dir = ensure_dir(args.run)
    save_config(cfg, run_dir / "config.json")
    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "lr", "train_loss", "val_loss", "val_ade_m",
             "val_fde120_m", "val_maez_m", "seconds"])

    best_ade = float("inf")
    bad = 0
    for epoch in range(1, int(tcfg["epochs"]) + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for step, batch in enumerate(ld_tr):
            if args.max_steps and step >= args.max_steps:
                break
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                out = model(batch["x"], batch["static"], batch["cats"])
            loss, _ = compute_loss(out, batch, tcfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            sched.step()
            b = batch["x"].size(0)
            tot += loss.item() * b
            n += b
            if (step + 1) % 200 == 0:
                log(f"  ep {epoch} step {step+1}/{steps_per_epoch} "
                    f"loss {tot/max(n,1):.5f} lr {sched.get_last_lr()[0]:.2e}")
        train_loss = tot / max(n, 1)

        va = eval_epoch(model, ld_va, device, tcfg,
                        float(sc["xy"]), float(sc["z"]), args.max_steps)
        secs = time.time() - t0
        lr_now = sched.get_last_lr()[0]
        log(f"epoca {epoch:3d} | lr {lr_now:.2e} | train {train_loss:.5f} | "
            f"val {va['loss']:.5f} | ADE {va['ade_m']:7.1f} m | "
            f"FDE120 {va['fde120_m']:7.1f} m | zMAE {va['maez_m']:6.1f} m | {secs:.0f}s")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, f"{lr_now:.8f}", f"{train_loss:.6f}", f"{va['loss']:.6f}",
                 f"{va['ade_m']:.3f}", f"{va['fde120_m']:.3f}", f"{va['maez_m']:.3f}",
                 f"{secs:.1f}"])

        ckpt = {
            "model_state": model.state_dict(),
            "model_cfg": cfg["model"],
            "data_meta": {
                "feature_names": meta_tr["feature_names"],
                "scales": sc, "vocab_sizes": vocab_sizes,
                "horizons_s": meta_tr["horizons_s"],
                "history_s": meta_tr["history_s"], "step_s": meta_tr["step_s"],
                "phase_names": meta_tr["phase_names"],
            },
            "epoch": epoch,
            "val_ade_m": va["ade_m"],
            "train_cfg": tcfg,
        }
        torch.save(ckpt, run_dir / "last.pt")
        if va["ade_m"] < best_ade - 1e-4:
            best_ade = va["ade_m"]
            bad = 0
            torch.save(ckpt, run_dir / "best.pt")
            log(f"  -> nuovo best (ADE {best_ade:.1f} m) salvato in {run_dir/'best.pt'}")
        else:
            bad += 1
            if bad >= int(tcfg["patience"]):
                log(f"early stopping (nessun miglioramento da {bad} epoche)")
                break

    log(f"addestramento concluso. Miglior ADE (val): {best_ade:.1f} m")
    log(f"checkpoint: {run_dir/'best.pt'}")


if __name__ == "__main__":
    main()
