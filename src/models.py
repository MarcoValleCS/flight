"""Modelli PyTorch per la previsione multi-orizzonte della traiettoria.

Architettura principale (arch="transformer"):
    - proiezione delle feature per passo (13 passi di storia da 5 s)
    - positional encoding appreso + token [CLS] + token di contesto statico
      (quota ancora, velocita' ancora + embedding di tipo velivolo, compagnia,
      aeroporto di partenza e destinazione)
    - encoder Transformer (norm_first, GELU)
    - pooling (CLS + statico + media della storia) -> MLP
    - teste di uscita:
        xy     : 12x2 correzioni (E, N) in metri / 10000 rispetto al CV
        z      : 12 correzioni di quota relativa (m / 1000) rispetto al CV
        phase  : 3 classi (crociera/salita/discesa) a +60 s (task ausiliario)
        logvar : opzionale, incertezza per orizzonte (NLL gaussiana)

Variante arch="gru": encoder BiGRU al posto del Transformer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

EMB_KEYS = ["ac_type", "operator", "dep", "arr"]


def _heads(d_model: int, n_horizons: int, predict_logvar: bool,
           n_phase: int = 3) -> nn.ModuleDict:
    heads = nn.ModuleDict({
        "xy": nn.Linear(d_model, n_horizons * 2),
        "z": nn.Linear(d_model, n_horizons),
        "phase": nn.Linear(d_model, n_phase),
    })
    if predict_logvar:
        heads["logvar"] = nn.Linear(d_model, n_horizons)
    return heads


class TrajTransformer(nn.Module):
    def __init__(self, n_feat: int, n_hist: int, n_horizons: int,
                 vocab_sizes: dict, emb_dims: dict, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 8, ff: int = 768,
                 dropout: float = 0.1, predict_logvar: bool = False,
                 n_phase: int = 3, n_static: int = 2):
        super().__init__()
        self.n_hist = n_hist
        self.n_horizons = n_horizons
        self.emb_keys = list(EMB_KEYS)

        self.in_proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_hist, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

        self.emb = nn.ModuleDict({
            k: nn.Embedding(int(vocab_sizes[k]), int(emb_dims[k]))
            for k in self.emb_keys
        })
        static_in = n_static + sum(int(emb_dims[k]) for k in self.emb_keys)
        self.static_proj = nn.Sequential(
            nn.Linear(static_in, d_model), nn.GELU())

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)

        self.pool = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.GELU())
        self.heads = _heads(d_model, n_horizons, predict_logvar, n_phase)

    def forward(self, x: torch.Tensor, static: torch.Tensor,
                cats: torch.Tensor) -> dict:
        B = x.size(0)
        h = self.in_proj(x) + self.pos
        emb = torch.cat(
            [self.emb[k](cats[:, i]) for i, k in enumerate(self.emb_keys)], dim=-1)
        s = self.static_proj(torch.cat([static, emb], dim=-1)).unsqueeze(1)
        tokens = torch.cat([self.cls.expand(B, -1, -1), s, h], dim=1)
        z = self.encoder(tokens)
        pooled = torch.cat([z[:, 0], z[:, 1], z[:, 2:].mean(dim=1)], dim=-1)
        r = self.pool(pooled)
        out = {
            "xy": self.heads["xy"](r).view(B, self.n_horizons, 2),
            "z": self.heads["z"](r),
            "phase": self.heads["phase"](r),
        }
        if "logvar" in self.heads:
            out["logvar"] = self.heads["logvar"](r).clamp(-8.0, 8.0)
        return out


class TrajGRU(nn.Module):
    def __init__(self, n_feat: int, n_hist: int, n_horizons: int,
                 vocab_sizes: dict, emb_dims: dict, d_model: int = 256,
                 n_layers: int = 2, n_heads: int = 8, ff: int = 768,
                 dropout: float = 0.1, predict_logvar: bool = False,
                 n_phase: int = 3, n_static: int = 2):
        super().__init__()
        self.n_horizons = n_horizons
        self.emb_keys = list(EMB_KEYS)

        self.in_proj = nn.Linear(n_feat, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=max(1, n_layers),
                          batch_first=True, bidirectional=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.emb = nn.ModuleDict({
            k: nn.Embedding(int(vocab_sizes[k]), int(emb_dims[k]))
            for k in self.emb_keys
        })
        static_in = n_static + sum(int(emb_dims[k]) for k in self.emb_keys)
        self.static_proj = nn.Sequential(
            nn.Linear(static_in, 2 * d_model), nn.GELU())
        self.pool = nn.Sequential(
            nn.Linear(2 * d_model + 2 * d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.GELU())
        self.heads = _heads(d_model, n_horizons, predict_logvar, n_phase)

    def forward(self, x: torch.Tensor, static: torch.Tensor,
                cats: torch.Tensor) -> dict:
        B = x.size(0)
        h = self.in_proj(x)
        out, _ = self.gru(h)
        emb = torch.cat(
            [self.emb[k](cats[:, i]) for i, k in enumerate(self.emb_keys)], dim=-1)
        s = self.static_proj(torch.cat([static, emb], dim=-1))
        pooled = torch.cat([out[:, -1], out.mean(dim=1), s], dim=-1)
        r = self.pool(pooled)
        outd = {
            "xy": self.heads["xy"](r).view(B, self.n_horizons, 2),
            "z": self.heads["z"](r),
            "phase": self.heads["phase"](r),
        }
        if "logvar" in self.heads:
            outd["logvar"] = self.heads["logvar"](r).clamp(-8.0, 8.0)
        return outd


def build_model(cfg_model: dict, vocab_sizes: dict, n_feat: int,
                n_hist: int, n_horizons: int) -> nn.Module:
    arch = cfg_model.get("arch", "transformer")
    common = dict(
        n_feat=n_feat, n_hist=n_hist, n_horizons=n_horizons,
        vocab_sizes=vocab_sizes, emb_dims=cfg_model["emb"],
        d_model=cfg_model["d_model"], n_layers=cfg_model["n_layers"],
        ff=cfg_model["ff"], dropout=cfg_model["dropout"],
        predict_logvar=cfg_model.get("predict_logvar", False))
    if arch == "transformer":
        return TrajTransformer(n_heads=cfg_model["n_heads"], **common)
    if arch == "gru":
        return TrajGRU(**common)
    raise ValueError(f"arch sconosciuta: {arch}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model_from_checkpoint(ckpt_path: str | Path, device="cpu"):
    """Ricostruisce il modello salvato da train.py. Ritorna (model, ckpt)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    dm = ckpt["data_meta"]
    n_feat = len(dm["feature_names"])
    n_hist = int(round(dm["history_s"] / dm["step_s"])) + 1
    n_h = len(dm["horizons_s"])
    model = build_model(ckpt["model_cfg"], dm["vocab_sizes"], n_feat, n_hist, n_h)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt
