"""
FlightBERT: Binary Encoding Representation for Flight Trajectory Prediction
=============================================================================
Re-implementation of

    D. Guo, E. Q. Wu, Y. Wu, J. Zhang, R. Law, Y. Lin,
    "FlightBERT: Binary Encoding Representation for Flight Trajectory Prediction",
    IEEE Trans. Intelligent Transportation Systems, 24(2):1828-1842, Feb. 2023.

Architecture (Fig. 1-4, Sec. III-IV):

  BE representation  [Sec. III-C]  p_t = [Lon, Lat, Alt, Vx, Vy, Vz] -> n-hot binary
                      vectors: value -> round(value * 10^m) -> integer -> base 2.
                      Velocities are signed, the highest bit carrying the sign.
                      Bit widths (Sec. IV-B, exp. B1): Lon 18, Lat 16, Alt 11
                      [in 10 m], Vx/Vy/Vz 11  =>  78-dim vector.

  Trajectory module  [Sec. III-D1] AE block  : e_ak = FC(a_k^b)          (Eq. 4)
                                         ACoAtt   : for the correlated pairs
                                         (Lon,Vx), (Lat,Vy), (Alt,Vz), bidirectionally,
                                         e'_ak = softmax(QK^T/sqrt(d_k)) V (Eq. 5)
                                         e''_ak = concat(e_ak, e'_ak)    (Eq. 6)
                                         e_t  = concat_k e''_ak = 6 x 128 = 768

  Backbone          [Sec. III-D2] n stacked Transformer decoder blocks
                      (MMSA -> +LN -> FF -> +LN) with causal masking, plus a
                      sinusoidal position embedding.

  Predictor         [Sec. III-D3] FC(768 -> 78) + Sigmoid, i.e. the trajectory
                      prediction is cast as a Multi Binary Classification task.
                      Binary decoding: threshold the output at 0.5.

  HC loss           [Sec. III-E] L_HC = L_MSE(decoded decimals) + L_BCE(BE)  (Eq. 7);
                      the thresholding is discontinuous so a straight-through
                      estimator copies the gradients (Sec. IV-B2). See
                      --mse_mode for the scaling of the MSE term.

  Multi-step TP     [Sec. III-A] the net predicts the next point; horizons 2..n
                      are obtained iteratively, the predicted point being appended
                      as pseudo-observation while a sliding window keeps the input
                      length at k = 9 (3 min @ 20 s) and n = 15 (5 min).
                      Reported horizons: 1 / 3 / 9 / 15.

Dataset: standard 19-column ADS-B "OpenFlights"-style table, **CSV or Excel**
(.xlsx/.xlsm/.xls/.ods), without a header row:

    A(0) icao24 | B(1) lat | C(2) lon | D(3) heading(deg) | E(4) altitude(ft)
    F(5) ground_speed(kt) | G(6) transponder | H(7) registration | I(8) type
    J(9) callsign | K(10) timestamp(epoch s) | L(11) origin | M(12) destination
    N(13) flight_number | O(14) status | P(15) vertical_rate(ft/min)
    Q(16) callsign2 | R(17) flag | S(18) operator

(indices overridable with --*_col). The velocity components required by the
paper are NOT expected in the file: they are *derived* from what the feed
already carries (ground speed + heading -> x = East, y = North in km/h;
vertical rate in column P, or the altitude derivative, -> z = Up).

Run
---
    python flightbert.py --csv Italy_24-04-03.csv
    python flightbert.py --csv Italy_24-04-03.csv --limit_flights 60 --epochs 2
    python flightbert.py --test_only --resume runs/flightbert/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LOG = logging.getLogger("flightbert")


# =============================================================================
# 1. Trajectory attributes and binary encoding                    [Sec. III-C]
# =============================================================================

#: p_t = [Lon, Lat, Alt, Vx, Vy, Vz]                                        (Eq. 1)
ATTRS: Tuple[str, ...] = ("lon", "lat", "alt", "vx", "vy", "vz")

#: attributes whose BE vector starts with a sign bit                      (Sec. III-C)
SIGNED_ATTRS: Tuple[str, ...] = ("vx", "vy", "vz")

#: 10^m of the BE process: the number of decimals kept for every attribute.
#: Sec. IV-B: 0.001 deg for Lon/Lat (~110 m), 10 m for Alt, km/h for velocities.
PRECISION: Dict[str, float] = {
    "lon": 1000.0, "lat": 1000.0, "alt": 0.1, "vx": 1.0, "vy": 1.0, "vz": 1.0,
}

#: 1 degree of latitude, metres (the same 111.32 the metric report uses)
DEG_M = 111_320.0

#: cap on the per-sample errors kept for the percentile report: this many
#: batches verbatim, then an adaptive stride keeps the retained set ~constant
HDIST_BATCHES = 2048


def set_position_precision(decimals: int) -> None:
    """Set the lon/lat BE resolution to 10^-`decimals` degrees (--pos_prec).

    This is the *only* knob that changes the positional resolution of the
    binary encoding.  Resampling more coarsely (--period) does not: the
    quantisation stays exactly the same, it just covers a larger displacement
    per predicted step, which makes the task harder rather than more accurate.
    """
    s = str(int(decimals))
    if int(decimals) < 1 or s[0] != "1" or s.strip("0") != "1":
        raise ValueError("--pos_prec must be a power of ten >= 1000 "
                         f"(1000 = 0.001 deg ~ 111 m, 10000 ~ 11 m, "
                         f"100000 ~ 1 m), got {decimals}")
    if decimals < 1000:
        LOG.warning("--pos_prec %d is coarser than the paper's 0.001 deg; the "
                    "positional resolution will be worse than 111 m", decimals)
    PRECISION["lon"] = PRECISION["lat"] = float(decimals)
    d_lat, d_lon = DEG_M / decimals, DEG_M / decimals * math.cos(math.radians(45.0))
    LOG.info("BE positional resolution: 1/%d deg = %.2f m lat / %.2f m lon @45N "
             "(bit widths follow automatically)", decimals, d_lat, d_lon)


def attr_resolution() -> Dict[str, float]:
    """Smallest representable step of every attribute, in its own unit."""
    out: Dict[str, float] = {}
    for a in ATTRS:
        step = 1.0 / PRECISION[a]
        out[a] = step * DEG_M if a in ("lon", "lat") else step
    return out


def encoded_to_real(arr: np.ndarray) -> np.ndarray:
    """(T, 6) in encoded integer units -> (T, 6) deg/deg/m/kmh.

    The inverse of what `_to_encoded_units` does to the raw feed; the stored
    legs and the BE cache's `vals` are all in encoded units.
    """
    out = np.array(arr, dtype=np.float64, copy=True)
    for j, a in enumerate(ATTRS):
        out[:, j] /= PRECISION[a]
    return out

#: default number of bits per attribute                                    (Sec. IV-B)
DEFAULT_BITS: Dict[str, int] = {
    "lon": 18, "lat": 16, "alt": 11, "vx": 11, "vy": 11, "vz": 11,
}

#: kinematically correlated attribute pairs driving the ACoAtt block      (Fig. 4)
CORR_PAIRS: Tuple[Tuple[str, str], ...] = (("lon", "vx"), ("lat", "vy"), ("alt", "vz"))

#: unit conversions of the raw feed
KT_TO_KMH = 1.852
FT_TO_M = 0.3048
FTMIN_TO_KMH = 0.3048 / 60.0 * 3.6

#: human readable unit of every attribute (for the metric report)
ATTR_UNITS: Dict[str, str] = {
    "lon": "deg", "lat": "deg", "alt": "m", "vx": "km/h", "vy": "km/h", "vz": "km/h",
}

CSV_COLUMNS: Tuple[str, ...] = (
    "icao24", "latitude", "longitude", "heading", "altitude_ft", "ground_speed_kt",
    "transponder", "registration", "aircraft_type", "callsign", "timestamp",
    "origin", "destination", "flight_number", "status", "vertical_rate_ftmin",
    "callsign2", "flag", "operator",
)


def bits_needed(value: float, signed: bool) -> int:
    """Smallest bit width able to represent `value` exactly.

    Signed attributes reserve their highest bit for the sign, so only
    ``bits - 1`` bits are available for the magnitude.
    """
    m = int(math.ceil(abs(float(value)) - 1e-9))
    n = 2 if signed else 1
    while n < 32:
        cap = ((1 << (n - 1)) - 1) if signed else ((1 << n) - 1)
        if m <= cap:
            break
        n += 1
    return n


class BinaryEncoding:
    """Binary encoding / decoding of the trajectory attributes.

    The BE process of Sec. III-C:  decimal -> round(*10^m) -> integer -> base-2
    n-hot vector.  For the signed velocity attributes the first (highest) bit
    encodes the sign, the remaining bits encode |value|.
    """

    def __init__(self, bits: Optional[Dict[str, int]] = None):
        self.bits: Dict[str, int] = {a: int((bits or DEFAULT_BITS)[a]) for a in ATTRS}
        for a in ATTRS:
            if not 2 <= self.bits[a] <= 32:
                raise ValueError(f"attribute '{a}': implausible bit width {self.bits[a]}")
        off, o = {}, 0
        for a in ATTRS:
            off[a] = o
            o += self.bits[a]
        self._offsets = off
        self._total = o
        self._max_mag = {a: (1 << (self.bits[a] - 1)) - 1 if a in SIGNED_ATTRS
                         else (1 << self.bits[a]) - 1 for a in ATTRS}

    # ------------------------------------------------------------------ meta
    @property
    def offsets(self) -> Dict[str, int]:
        return dict(self._offsets)

    @property
    def total_bits(self) -> int:
        return self._total

    def max_magnitude(self, attr: str) -> int:
        """Largest magnitude (in encoded integer units) representable."""
        return self._max_mag[attr]

    def max_value(self, attr: str) -> float:
        return self._max_mag[attr] / PRECISION[attr]

    def min_value(self, attr: str) -> float:
        return -self.max_value(attr) if attr in SIGNED_ATTRS else 0.0

    def describe(self) -> str:
        return ", ".join(f"{a}:{self.bits[a]}b" for a in ATTRS)

    # ---------------------------------------------------------------- encode
    def encode_matrix(self, arr: np.ndarray) -> np.ndarray:
        """(T, 6) values in *encoded* integer units -> (T, D) uint8 BE vectors."""
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(ATTRS):
            raise ValueError(f"expected an (T, {len(ATTRS)}) array, got {arr.shape}")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.zeros((arr.shape[0], self._total), dtype=np.uint8)
        for j, a in enumerate(ATTRS):
            nb, o = self.bits[a], self._offsets[a]
            v = np.rint(arr[:, j]).astype(np.int64)
            if a in SIGNED_ATTRS:
                out[:, o] = (v < 0).astype(np.uint8)          # highest bit = sign
                mag, n_mag, first = np.minimum(np.abs(v), self._max_mag[a]), nb - 1, o + 1
            else:
                mag, n_mag, first = np.clip(v, 0, self._max_mag[a]), nb, o
            shifts = np.arange(n_mag - 1, -1, -1, dtype=np.int64)
            out[:, first:first + n_mag] = ((mag[:, None] >> shifts) & 1).astype(np.uint8)
        return out

    def encode(self, values: Dict[str, float]) -> np.ndarray:
        """Single point, values in real units -> (D,) uint8 BE vector."""
        vec = np.zeros((1, len(ATTRS)), dtype=np.float64)
        for j, a in enumerate(ATTRS):
            v = float(values[a]) * PRECISION[a]
            vec[0, j] = min(max(v, self.min_value(a) * PRECISION[a]),
                            self.max_value(a) * PRECISION[a])
        return self.encode_matrix(vec)[0]

    # ---------------------------------------------------------------- decode
    def decode_matrix(self, bits: np.ndarray) -> np.ndarray:
        """(B, D) BE vectors -> (B, 6) values in *encoded* integer units."""
        bits = np.asarray(bits)
        if bits.ndim == 1:
            bits = bits[None, :]
        if bits.shape[-1] != self._total:
            raise ValueError(f"expected {self._total} bits, got {bits.shape[-1]}")
        out = np.zeros((bits.shape[0], len(ATTRS)), dtype=np.float64)
        for j, a in enumerate(ATTRS):
            nb, o = self.bits[a], self._offsets[a]
            first = o + 1 if a in SIGNED_ATTRS else o
            n_mag = nb - 1 if a in SIGNED_ATTRS else nb
            body = bits[:, first:o + nb].astype(np.float64)
            out[:, j] = body @ (2.0 ** np.arange(n_mag - 1, -1, -1))
            if a in SIGNED_ATTRS:
                out[:, j] *= 1.0 - 2.0 * bits[:, o].astype(np.float64)
        return out

    def decode_real(self, bits: np.ndarray) -> np.ndarray:
        """(B, D) BE vectors -> (B, 6) values in real units (deg, m, km/h)."""
        out = self.decode_matrix(bits)
        for j, a in enumerate(ATTRS):
            out[:, j] /= PRECISION[a]
        return out

    def decode_dict(self, bits: np.ndarray) -> Dict[str, float]:
        v = self.decode_real(np.asarray(bits).reshape(1, -1))[0]
        return {a: float(v[j]) for j, a in enumerate(ATTRS)}


class _ThresholdSTE(torch.autograd.Function):
    """Straight-through estimator for the 0.5 binarization (Sec. IV-B2)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return (x > 0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad


def binarize_ste(x: torch.Tensor) -> torch.Tensor:
    return _ThresholdSTE.apply(x)


# =============================================================================
# 2. Configuration
# =============================================================================

@dataclass
class Config:
    # ------------------------------------------------------------------ data
    csv_path: str = ""
    out_dir: str = "runs/flightbert"
    id_col: int = 0
    lat_col: int = 1
    lon_col: int = 2
    hdg_col: int = 3
    alt_col: int = 4
    gs_col: int = 5
    ts_col: int = 10
    vs_col: int = 15
    sep: str = ""                    # "" = auto-detect (comma, then semicolon/tabs)
    decimal: str = ""                # "" = auto-detect
    resolved_dialect: Tuple[str, str] = (",", ".")   # filled in by the reader
    # Resampling period, s.  The paper used 20 s; 30 s is the default here because
    # a 20 s gap between ADS-B fixes is mostly track noise, and it buys a ~35%
    # smaller dataset.  Note this does NOT change the BE resolution: see
    # --pos_prec for that.
    period: int = 30                 # resampling period, s
    inp_seq_len: int = 9             # k observed points
    max_horizon: int = 15            # n predicted steps
    min_points: int = 12             # minimum length of a resampled leg
    gap_factor: float = 4.0          # a gap > factor*period breaks the trajectory
    split_mode: str = "time"         # "time" (paper: split by days) | "flight"
    train_frac: float = 0.70
    val_frac: float = 0.15
    limit_flights: int = 0            # >0: only the first N aircraft (debug)
    apply_sanity_filter: bool = True
    max_alt_ft: float = 60000.0
    max_gs_kt: float = 900.0
    max_vs_ftmin: float = 20000.0

    # -------------------------------------------------- BE representation
    bits: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_BITS))
    auto_bits: bool = True            # widen a bit width if the data requires it
    max_bits: int = 24
    #: decimals kept for lon/lat.  1000 = 0.001 deg ~ 111 m (paper, Sec. IV-B),
    #: 10000 = 0.0001 deg ~ 11 m, 100000 = 1 m.  This - not --period - is what
    #: sets the positional resolution; the bit widths follow automatically.
    pos_prec: int = 1000

    # --------------------------------------------------------------- model
    d_attr: int = 64                  # |e_ak| = |e'_ak|            (Sec. IV-B2)
    n_coatt_heads: int = 4            # heads of the CoAtt modules  (Sec. IV-B2)
    n_layers: int = 8                 # stacked Transformer blocks  (Sec. IV-B2)
    n_heads: int = 4
    dim_ff: int = 1536
    dropout: float = 0.1
    causal: bool = True
    learn_pos: bool = False
    #: predict every horizon in one forward (no autoregressive feedback loop)
    direct: bool = False

    # ---------------------------------------------------------------- loss
    bce_weight: float = 1.0
    mse_weight: Optional[float] = None    # None = calibrate the two terms at init
    mse_mode: str = "rel"            # "rel" (default, stable) | "phys" | "raw" (literal Eq. 8)

    # ------------------------------------------------------- optimisation
    epochs: int = 20
    batch_size: int = 512
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lr_factor: float = 0.5
    lr_patience: int = 2
    amp: bool = True
    amp_dtype: str = "fp16"           # "fp16" (GradScaler) | "bf16" (Ada+: no scaler)
    num_workers: int = 0
    prefetch_factor: int = 4          # batches prefetched per worker
    eval_batch_size: int = 512        # the 15-step rollout saturates the GPU at ~512
    eval_every: int = 1               # run validation every N epochs (1 = every epoch)
    seed: int = 42
    eval_horizons: Tuple[int, ...] = (1, 3, 9, 15)
    feed_soft: bool = False
    log_every: int = 50
    save_every: int = 1
    device: str = "auto"
    resume_path: str = ""
    map_step: float = 1.0
    map_flights: int = 40
    map_split: str = ""
    map_max_points: int = 300_000
    map_pred_max_km: float = 200.0

    def to_json(self) -> str:
        d = asdict(self)
        d["eval_horizons"] = list(self.eval_horizons)
        return json.dumps(d, indent=2, default=str)


# =============================================================================
# 3. Model                                                          [Sec. III-D]
# =============================================================================

class AttributeEmbedding(nn.Module):
    """AE block: one independent FC layer per attribute, e_ak = FC(a_k^b)  (Eq. 4)."""

    def __init__(self, enc: BinaryEncoding, d_attr: int = 64):
        super().__init__()
        self.enc = enc
        self.d_attr = d_attr
        self.proj = nn.ModuleDict(
            {a: nn.Linear(enc.bits[a], d_attr) for a in ATTRS})
        self.act = nn.GELU()

    def forward(self, be: torch.Tensor) -> Dict[str, torch.Tensor]:
        off = self.enc.offsets
        return {a: self.act(self.proj[a](be[..., off[a]:off[a] + self.enc.bits[a]]))
                for a in ATTRS}


class CoAtt(nn.Module):
    """Correlation attention for one directed pair a_k <- a_j (Eq. 5).

        e'_ak = softmax(Q K^T / sqrt(d_k)) V,   Q from e_ak, K and V from e_aj

    The paper writes the linear projections Q, K, V only; a LayerNorm is applied
    to both inputs before them, which keeps the attention logits bounded (it does
    not change the equations otherwise).
    """

    def __init__(self, d_attr: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if d_attr % n_heads:
            raise ValueError(f"d_attr={d_attr} is not divisible by n_heads={n_heads}")
        self.norm_q = nn.LayerNorm(d_attr)
        self.norm_kv = nn.LayerNorm(d_attr)
        self.attn = nn.MultiheadAttention(d_attr, n_heads, dropout=dropout,
                                          batch_first=True)

    def forward(self, e_self: torch.Tensor, e_other: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.attn(self.norm_q(e_self), self.norm_kv(e_other),
                           self.norm_kv(e_other), attn_mask=attn_mask,
                           need_weights=False)
        return out


class ACoAttBlock(nn.Module):
    """Attribute Correlation Attention block (Fig. 3 / Fig. 4).

    Every attribute is recalibrated with the embedding of its kinematically
    correlated attribute, bidirectionally (Lon<->Vx, Lat<->Vy, Alt<->Vz), and
    the final attribute embedding is the concatenation [e_ak, e'_ak] (Eq. 6).
    """

    def __init__(self, d_attr: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pairs = CORR_PAIRS
        self.co = nn.ModuleDict({
            f"{i}<-{j}": CoAtt(d_attr, n_heads, dropout) for i, j in CORR_PAIRS
        } | {f"{j}<-{i}": CoAtt(d_attr, n_heads, dropout) for i, j in CORR_PAIRS})

    def forward(self, embs: Dict[str, torch.Tensor],
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        corr: Dict[str, torch.Tensor] = {}
        for i, j in self.pairs:
            corr[i] = self.co[f"{i}<-{j}"](embs[i], embs[j], attn_mask)
            corr[j] = self.co[f"{j}<-{i}"](embs[j], embs[i], attn_mask)
        return torch.cat([torch.cat([embs[a], corr[a]], dim=-1) for a in ATTRS], dim=-1)


class TrajectoryEmbedding(nn.Module):
    """Trajectory embedding module = AE block + ACoAtt block.        [Sec. III-D1]"""

    def __init__(self, cfg: Config, enc: BinaryEncoding):
        super().__init__()
        self.ae = AttributeEmbedding(enc, cfg.d_attr)
        self.acoatt = ACoAttBlock(cfg.d_attr, cfg.n_coatt_heads, cfg.dropout)
        self.out_dim = len(ATTRS) * 2 * cfg.d_attr          # 6 x 2 x 64 = 768
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, be: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # NOTE the ACoAtt attention runs over the *time* axis, so it needs the
        # causal mask as much as the backbone does: without it every position
        # already sees the whole window and --direct would read the future.
        return self.norm(self.acoatt(self.ae(be), attn_mask))


class PositionalEncoding(nn.Module):
    """Sinusoidal position embedding added before the backbone (Fig. 1)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1,
                 learnable: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.learnable = learnable
        if learnable:
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.trunc_normal_(self.pe, std=0.02)
        else:
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence length {x.size(1)} > max_len {self.pe.size(1)}")
        return self.dropout(x + self.pe[:, :x.size(1)])


class SelfAttentionBlock(nn.Module):
    """One backbone block: MMSA -> residual+LayerNorm -> FF -> residual+LN.

    This is the post-LayerNorm Transformer decoder block of Fig. 1, without the
    cross-attention (the paper's backbone is a masked self-attention stack).
    """

    def __init__(self, cfg: Config, d_model: int):
        super().__init__()
        if d_model % cfg.n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={cfg.n_heads}")
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, cfg.n_heads, dropout=cfg.dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, cfg.dim_ff), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_ff, d_model), nn.Dropout(cfg.dropout))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


class FlightBERTBackbone(nn.Module):
    """Stacked Transformer blocks with causal masked multi-head self-attention."""

    def __init__(self, cfg: Config, d_model: int, max_len: Optional[int] = None):
        super().__init__()
        self.causal = cfg.causal
        self.max_len = int(max_len or cfg.inp_seq_len)
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(cfg, d_model) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.register_buffer(
            "tgt_mask",
            torch.triu(torch.ones(self.max_len, self.max_len, dtype=torch.bool), 1),
            persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        if t > self.max_len:
            raise ValueError(f"input length {t} exceeds the window ({self.max_len}); "
                             "the sliding window must keep the input length constant")
        mask = self.tgt_mask[:t, :t] if self.causal else None
        for blk in self.blocks:
            x = blk(x, mask)
        return self.norm(x)


class FlightBERT(nn.Module):
    """Flight trajectory prediction network (FTPNet) of Fig. 1.

    forward(be): (B, k, D) BE-encoded observation sequence
              -> (B, D) sigmoid probabilities of the next point's BE vector
                 (or (B, n, D) for every horizon when `direct` is set).

    `direct` (--direct)
        The paper trains the net on the *next* point only and rolls the horizon
        out by feeding its own hard 0/1 output back in (Sec. III-A).  That
        rollout compounds its own error: every horizon is conditioned on
        already-thresholded predictions.  With `direct` the predictor emits all
        n horizons in one shot from the same causal backbone - position t of the
        output is read from position t-1 of the *observations* only, so there
        is nothing to feed back and nothing to accumulate.  It also makes the
        evaluation n times cheaper (one forward instead of n).
    """

    def __init__(self, cfg: Config, enc: Optional[BinaryEncoding] = None):
        super().__init__()
        self.cfg = cfg
        self.enc = enc or BinaryEncoding(cfg.bits)
        cfg.bits = dict(self.enc.bits)
        self.in_dim = self.enc.total_bits
        self.direct = bool(cfg.direct)
        # how many readout positions the causal backbone has to offer
        self.n_out = max(cfg.max_horizon, cfg.inp_seq_len) if self.direct else 1
        self.embed = TrajectoryEmbedding(cfg, self.enc)
        d_model = self.embed.out_dim
        self.d_model = d_model
        seq_len = self.n_out if self.direct else cfg.inp_seq_len
        self.pos = PositionalEncoding(d_model, max_len=max(seq_len, 8),
                                      dropout=cfg.dropout, learnable=cfg.learn_pos)
        self.backbone = FlightBERTBackbone(cfg, d_model, max_len=seq_len)
        self.predictor = nn.Linear(d_model, self.in_dim)      # FC  (Sec. III-D3)
        self.sigmoid = nn.Sigmoid()                            # + Sigmoid
        # causal mask for the embedding stack (the backbone keeps its own)
        self.register_buffer(
            "tgt_mask",
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), 1),
            persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, be: torch.Tensor) -> torch.Tensor:
        if be.dim() != 3:
            raise ValueError(f"expected a (B, T, D) tensor, got {tuple(be.shape)}")
        if be.size(-1) != self.in_dim:
            raise ValueError(f"expected {self.in_dim} BE bits, got {be.size(-1)}")
        n = self.cfg.max_horizon
        if self.direct and be.size(1) < n:
            # not enough observed positions to read n horizons from: append empty
            # query positions.  Under the causal mask they see the whole window
            # and nothing else, so each of them is a read-only "query".
            pad = be.new_zeros(be.size(0), n - be.size(1), self.in_dim)
            be = torch.cat([be, pad], dim=1)
        t = be.size(1)
        # the embedding stack is causal in direct mode: every readout position
        # must depend on its own past only
        emask = self.tgt_mask[:t, :t] if (self.direct and self.cfg.causal) else None
        h = self.pos(self.embed(be, emask))     # trajectory embeddings
        h = self.backbone(h)                      # Transformer blocks
        if not self.direct:
            return self.sigmoid(self.predictor(h[:, -1, :]))   # FC + Sigmoid
        # one horizon per position: output t predicts t+1 steps ahead and, thanks
        # to the causal mask, has only ever seen observations 0..t
        return self.sigmoid(self.predictor(h[:, :n]))          # (B, n, D)


# =============================================================================
# 4. Hybrid Constrained loss                                       [Sec. III-E]
# =============================================================================

class HybridConstrainedLoss(nn.Module):
    """L_HC = L_MSE(p, y) + L_BCE(p^b, y^b)                                 (Eq. 7)

    * L_BCE : the multi binary classification objective on the BE codes.
    * L_MSE : the error of the *decoded decimals*. The 0.5 thresholding of the
      binary decoding is discontinuous, therefore a straight-through estimator
      copies the gradient (Sec. IV-B2).

    `mse_mode`
        "rel"  (default) : Eq. (8)-(9) with every attribute divided by its full
                 bit scale (2^bits - 1), so the term stays scale free. A high-bit
                 error still dominates (0.25 per attribute for the top bit of
                 lon) while the term stays comparable to the BCE, which is the
                 division of labour described in Sec. III-E: BCE minimises the
                 classification error of every bit, MSE constrains the decimal
                 error carried by the high bits.
        "raw"  : the literal sum of Eq. (8)-(9) in encoded integer units, where
                 one misclassified high bit of lon costs (2^17)^2 ~ 1.7e10. It is
                 faithful but numerically hostile at initialisation (the loss
                 starts around 1e9); use it only with a small --mse_weight.
        "phys" : the same bounded normalisation as "rel", but with a per-attribute
                 *re-weighting* that reflects physical significance.  In "rel"
                 the lon term is divided by 2^16 (~7000 km) while the vx term is
                 divided by 2^12 (2047 km/h), so a 1 km position mistake counts
                 ~1e5 times less than a 10 km/h velocity one - and a velocity
                 error is only ever the *consequence* of a bad position.  "phys"
                 multiplies each attribute by the squared ratio between the two
                 scales, renormalised to unit geometric mean, so the overall
                 magnitude (and therefore the automatic --mse_weight calibration)
                 is unchanged while the balance between attributes is physical.
    In all cases the per-attribute MSEs are averaged, and gradient clipping
    (--grad_clip, 1.0 by default) guards against the high-bit outliers.
    """

    #: plausible *error* per attribute, in its own real unit: 1 km of position,
    #: 100 m of altitude, 10 km/h of velocity
    PHYS_REF: Dict[str, float] = {"lon": 1.0, "lat": 1.0, "alt": 100.0,
                                  "vx": 10.0, "vy": 10.0, "vz": 10.0}

    def __init__(self, enc: BinaryEncoding, bce_weight: float = 1.0,
                 mse_weight: float = 1.0, mse_mode: str = "rel"):
        super().__init__()
        if mse_mode not in ("raw", "rel", "phys"):
            raise ValueError("mse_mode must be 'raw', 'rel' or 'phys'")
        self.enc = enc
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.mse_mode = mse_mode
        weights, meta = [], []
        for j, a in enumerate(ATTRS):
            nb, off = enc.bits[a], enc.offsets[a]
            signed = a in SIGNED_ATTRS
            n_body = nb - 1 if signed else nb
            start = sum(len(w) for w in weights)
            weights.append([float(2 ** (n_body - 1 - i)) for i in range(n_body)])
            meta.append((j, off, nb, start, signed))
        self.register_buffer("w", torch.tensor(sum(weights, []), dtype=torch.float32))
        self.register_buffer("full_scale",
                             torch.tensor([float(enc.max_magnitude(a)) for a in ATTRS]))
        # encoded units of one reference error, per attribute: lon/lat are
        # expressed in km -> degrees first, the rest are linear
        phys = torch.tensor(
            [(self.PHYS_REF[a] / 111.32 if a in ("lon", "lat") else self.PHYS_REF[a])
             * PRECISION[a] for a in ATTRS], dtype=torch.float32)
        # turning "rel" into "phys" is a per-attribute multiplier; renormalise it
        # to unit geometric mean so the loss keeps the same overall magnitude
        mult = (self.full_scale / phys) ** 2
        mult = mult / torch.exp(torch.log(mult).mean())
        self.register_buffer("phys_scale", phys)
        self.register_buffer("attr_weight", mult)
        self.meta = meta

    def _decode(self, bits: torch.Tensor) -> torch.Tensor:
        cols = []
        for j, off, nb, start, signed in self.meta:
            seg = bits[..., off:off + nb]
            body = seg[..., 1:] if signed else seg
            val = (body * self.w[start:start + body.size(-1)]).sum(-1)
            if signed:
                val = val * (1.0 - 2.0 * seg[..., 0])
            cols.append(val)
        return torch.stack(cols, dim=-1)

    def soft_decode(self, probs: torch.Tensor) -> torch.Tensor:
        """Differentiable decoding of the sigmoid output (STE + weighted sum)."""
        return self._decode(binarize_ste(probs))

    def _terms(self, probs: torch.Tensor, target_bits: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The two halves of Eq. (7), kept separate so they can be balanced."""
        probs = probs.float()
        target_bits = target_bits.float()
        bce = F.binary_cross_entropy(probs.clamp(1e-7, 1 - 1e-7), target_bits)
        diff = self.soft_decode(probs) - self._decode(target_bits)
        if self.mse_mode == "phys":
            # A physical reference (1 km of position) makes the term *unbounded*:
            # a 400 km mistake would contribute 1.6e5 and swamp the BCE, which is
            # what makes a plain squared error here unusable.  A Huber loss keeps
            # the physical weighting - still linear in km past ~1 km - but caps
            # the gradient, so one bad batch cannot blow the training up.
            z = diff / self.full_scale * self.attr_weight.sqrt()
            return bce, F.smooth_l1_loss(z, torch.zeros_like(z), beta=1.0)
        if self.mse_mode != "raw":
            diff = diff / self.full_scale
        return bce, (diff ** 2).mean()

    @torch.no_grad()
    def calibrate(self, model, device: torch.device, enc: BinaryEncoding,
                  k: int, batch: int = 64) -> float:
        """Pick mse_weight so that both terms of Eq. (7) start at the same size.

        The BCE is scale free (~0.69 at initialisation) while the MSE on freshly
        initialised bits is enormous, because decoding random bits spans the
        whole range.  Without this, any mse_mode other than the default silently
        either drowns the BCE or vanishes.
        """
        was_training = model.training
        model.eval()
        g = torch.Generator(device="cpu").manual_seed(0)
        x = torch.randint(0, 2, (batch, k, enc.total_bits), generator=g
                          ).float().to(device)
        y = torch.randint(0, 2, (batch, enc.total_bits), generator=g
                          ).float().to(device)
        if model.direct:                      # (B, D) targets -> (B, 1, D)
            y = y.unsqueeze(1)
            probs = model(x)[:, :1]
        else:
            probs = model(x)
        bce, mse = self._terms(probs, y)
        if was_training:
            model.train()
        if not torch.isfinite(mse) or float(mse) <= 0.0:
            return 1.0
        w = float(bce) / float(mse)
        return float(min(max(w, 1e-8), 1e4))

    def forward(self, probs: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
        # The loss is deliberately evaluated in float32, outside autocast:
        #   * binary_cross_entropy is on the autocast "unsafe" list, because the
        #     model already applies the sigmoid (it would reject the fp16 call);
        #   * the BE decode sums the bit weights, up to 2^18 - 1 for lon, which
        #     overflows the fp16 range (65504) and would turn the MSE term into
        #     inf/nan under AMP. The tensors involved are tiny (B x 78), so the
        #     fp32 fallback costs nothing next to the fp16 backbone.
        with torch.autocast(device_type=probs.device.type, enabled=False):
            bce, mse = self._terms(probs, target_bits)
            return self.bce_weight * bce + self.mse_weight * mse


# =============================================================================
# 5. Data pipeline
# =============================================================================

def _sniff_engine(path: str) -> Optional[str]:
    """Detect the real table format from the file content (extension-independent).

    Returns 'odf', 'openpyxl', 'xlrd', or None when it looks like a plain CSV.
    """
    import zipfile
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return None
    if head[:4] == b"\xD0\xCF\x11\xE0":          # OLE2 container -> old .xls
        return "xlrd"
    if head[:2] == b"PK":                         # zip container -> xlsx or ods
        try:
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
        except zipfile.BadZipFile:
            return None
        if "xl/workbook.xml" in names or any(n.startswith("xl/") for n in names):
            return "openpyxl"
        if "META-INF/manifest.xml" in names or "mimetype" in names:
            return "odf"
        return "openpyxl"
    return None


def resolve_csv(candidate: str) -> List[str]:
    """Resolve --csv into a list of files.

    --csv accepts a comma/semicolon separated list, and the files are
    concatenated *before* the chronological split.  Training on one day and
    testing on another is a far more honest protocol than slicing a single day
    into train/val/test, and the time-based split does that automatically.
    """
    wanted = [p.strip() for p in re.split(r"[;,]", candidate or "") if p.strip()]
    if wanted:
        missing = [w for w in wanted if not os.path.isfile(w)]
        if missing:
            raise FileNotFoundError("dataset not found: " + ", ".join(missing))
        return [os.path.abspath(w) for w in wanted]
    cands: List[str] = []
    cands += [os.path.join(os.getcwd(), "Italy_24-04-03.csv"),
              "/kaggle/input/dataset-voli/Italy_24-04-03.csv",
              "/kaggle/input/dataset-voli/flightbert/Italy_24-04-03.csv"]
    try:
        cands += [os.path.join(os.getcwd(), f) for f in sorted(os.listdir("."))
                  if f.lower().endswith((".csv", ".xlsx", ".xlsm", ".xls", ".ods"))]
    except OSError:
        pass
    for c in cands:
        if c and os.path.isfile(c):
            return [os.path.abspath(c)]
    raise FileNotFoundError(
        "dataset not found - pass --csv <path>[,<path>...]. Tried:\n  "
        + "\n  ".join(str(c) for c in cands))


def _to_numbers(df: pd.DataFrame, id_name: Optional[str] = None,
                 euro: bool = False) -> pd.DataFrame:
    """Convert every column except the aircraft id to float, failures -> NaN.

    With `euro` the decimal comma is turned into a decimal point first, which is
    what a file exported as "45,5;7,49" needs.
    """
    for name in df.columns:
        if name == id_name:
            continue
        s = df[name].astype("string")
        if euro:
            s = s.str.replace(",", ".", regex=False)
        df[name] = pd.to_numeric(s, errors="coerce")
    return df


def _coerce(df: pd.DataFrame, id_name: Optional[str] = None) -> pd.DataFrame:
    """Backwards-compatible alias of `_to_numbers` without decimal repair."""
    return _to_numbers(df, id_name)


def _numeric_ratio(df: pd.DataFrame, cfg: Config) -> float:
    """Fraction of rows where the telemetry is usable: drives format detection."""
    need = [cfg.lat_col, cfg.lon_col, cfg.hdg_col, cfg.alt_col, cfg.gs_col,
            cfg.ts_col]
    cols = [f"c{c}" for c in need if f"c{c}" in df.columns]
    if not cols or len(df) == 0:
        return 0.0
    sub = df[cols].notna().to_numpy(dtype=np.float32)
    return float((sub.mean(axis=1) >= 0.99).mean())


def _probe_ncols(path: str) -> int:
    """Number of fields of the first non-empty line (delimiter-agnostic)."""
    try:
        with open(path, "r", errors="replace") as f:
            for _ in range(50):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                best = max(line.count(","), line.count(";"), line.count("\t"),
                           line.count("|"))
                return best + 1 if best else 1
    except OSError:                                            # pragma: no cover
        pass
    return 0


def _read_csv_text(path: str, names: Sequence[str], cols: Sequence[int],
                   cfg: Config) -> pd.DataFrame:
    """Read a text CSV, auto-detecting the delimiter and the decimal separator.

    The number of columns of the file does not have to be 19: any column that is
    not present is simply dropped, as long as the telemetry columns are there.
    """
    n_cols = _probe_ncols(path)
    want = sorted(set(int(c) for c in cols))
    usable = [c for c in want if c < n_cols] if n_cols else want
    missing = [c for c in want if c not in usable]
    required = [cfg.lat_col, cfg.lon_col, cfg.hdg_col, cfg.alt_col, cfg.gs_col,
                cfg.ts_col]
    absent = [c for c in required if c not in usable]
    if absent:
        raise RuntimeError(
            f"the file exposes {n_cols} column(s) but the telemetry needs "
            f"columns {required}. Missing: {absent}.\n"
            "  If your layout differs, pass --lat_col/--lon_col/--hdg_col/"
            "--alt_col/--gs_col/--ts_col (0-based: A=0, B=1, ... K=10).")
    if missing:
        LOG.warning("columns %s are not present in the file (%d columns) - skipped",
                    missing, n_cols)

    dtypes: Dict[str, object] = {f"c{c}": np.float32 for c in usable
                                 if c != cfg.id_col}
    dtypes[f"c{cfg.ts_col}"] = np.float64
    if cfg.id_col in usable:
        dtypes[f"c{cfg.id_col}"] = "string"
    labels = [f"c{c}" for c in usable]
    # explicit --sep/--decimal win; otherwise try the fast comma+dot path first
    if not cfg.sep and not cfg.decimal:
        try:
            df = pd.read_csv(path, header=None, usecols=usable, dtype=dtypes,
                             low_memory=False)
            df.columns = labels
            ratio = _numeric_ratio(df, cfg)
            if ratio >= 0.5:
                return df
            LOG.warning("comma-separated read looks wrong (%d usable rows) - "
                        "trying other delimiters", int(ratio * len(df)))
        except Exception as exc:                                # pragma: no cover
            LOG.warning("typed CSV read failed (%s) - falling back", exc)

    id_name = f"c{cfg.id_col}" if cfg.id_col in usable else None
    candidates = [(cfg.sep or ";", cfg.decimal or ","), (";", "."), ("\t", "."),
                  (",", ","), ("|", "."), (cfg.sep or ",", cfg.decimal or ".")]
    for sep, decimal in candidates:
        try:
            raw = pd.read_csv(path, header=None, usecols=usable, sep=sep,
                              decimal=decimal, dtype="string", keep_default_na=False,
                              low_memory=False)
        except Exception:
            continue
        raw.columns = labels
        # `dtype="string"` keeps the raw text, so `decimal=` is not applied by
        # the parser: both readings are tried explicitly
        for euro in ((False, True) if sep != "," else (False, True)):
            df = _to_numbers(raw.copy(), id_name, euro)
            ratio = _numeric_ratio(df, cfg)
            if ratio >= 0.5:
                LOG.info("CSV parsed with separator %r, decimal %r "
                         "(%d usable rows)", sep, "," if euro else ".", int(ratio * len(df)))
                cfg.resolved_dialect = (sep, "," if euro else ".")
                return df
    raise RuntimeError(
        "could not parse the CSV: no delimiter/decimal combination produced "
        "usable telemetry. Expected columns B..F (lat, lon, heading, altitude, "
        "ground speed) and K (timestamp). Adjust --lat_col/--lon_col/--hdg_col/"
        "--alt_col/--gs_col/--ts_col if your layout differs.")


def _read_csv(path: str, cfg: Config) -> pd.DataFrame:
    """Read the raw feed (CSV or Excel), with a robust coercion fallback."""
    names = [f"c{i}" for i in range(len(CSV_COLUMNS))]
    cols = sorted({cfg.id_col, cfg.lat_col, cfg.lon_col, cfg.hdg_col, cfg.alt_col,
                   cfg.gs_col, cfg.ts_col}
                  | ({cfg.vs_col} if cfg.vs_col >= 0 else set()))
    ext = os.path.splitext(path)[1].lower()
    engine = _sniff_engine(path)
    if engine is None and ext in (".xlsx", ".xlsm", ".xls", ".ods"):
        engine = "odf" if ext == ".ods" else ("xlrd" if ext == ".xls" else None)
    if engine:
        try:
            df = pd.read_excel(path, header=None, usecols=cols, engine=engine)
            labels = [f"c{c}" for c in cols]
        except Exception:
            df = pd.read_excel(path, header=None, engine=engine)   # whole sheet
            labels = [f"c{c}" for c in range(df.shape[1])]
        df = df.dropna(how="all").reset_index(drop=True)
        df.columns = labels[:df.shape[1]]
        df = _coerce(df, f"c{cfg.id_col}" if f"c{cfg.id_col}" in df.columns else None)
    else:
        df = _read_csv_text(path, names, cols, cfg)
    idc = f"c{cfg.id_col}"
    if idc not in df.columns:
        LOG.warning("no column %d in the file - every row is treated as a "
                    "single flight", cfg.id_col)
        df["flight_id"] = "single"
    else:
        df["flight_id"] = df[idc].astype("string").str.strip()
    if df["flight_id"].isna().mean() > 0.5 or (df["flight_id"] == "").mean() > 0.5:
        LOG.warning("column %d carries no aircraft id - falling back to the "
                    "registration column 7", cfg.id_col)
        if engine:
            alt = pd.read_excel(path, header=None, usecols=[7], engine=engine)[7]
        else:
            sep, dec = getattr(cfg, "resolved_dialect", (cfg.sep or ",", cfg.decimal or "."))
            alt = pd.read_csv(path, header=None, names=names, usecols=[7],
                              sep=sep, decimal=dec, dtype="string",
                              keep_default_na=False, low_memory=False)["c7"]
        df["flight_id"] = alt.astype("string").str.strip()
    # rows with no id at all get a *unique* id: they can never form a
    # trajectory (a single point is filtered out), but they must NOT be
    # merged into one fake aircraft sharing the "unknown" label.
    missing = df["flight_id"].isna() | (df["flight_id"] == "")
    n_missing = int(missing.sum())
    if n_missing:
        LOG.info("rows without an aircraft id: %d (ignored as trajectories)", n_missing)
        df.loc[missing, "flight_id"] = ["__none_%d" % i for i in range(n_missing)]
    return df


def _to_encoded_units(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """ADS-B feed -> the six trajectory attributes in *encoded* integer units.

        lon / lat : degrees          -> deg * 1000            (0.001 deg, Sec. IV-B)
        alt       : feet             -> (ft * 0.3048) / 10    (10 m units)
        vx, vy    : kt + heading     -> East / North component in km/h
        vz        : ft/min           -> km/h

    The vertical component vz is read from the `vs_col` column (P, ft/min);
    when the column is absent (`--vs_col -1`) or carries no usable value it is
    derived from the altitude differences of every sorted track instead.
    """
    col = lambda i: df[f"c{i}"].to_numpy(dtype=np.float64)      # noqa: E731
    lat, lon = col(cfg.lat_col), col(cfg.lon_col)
    alt_ft, gs_kt, hdg, ts = (col(cfg.alt_col), col(cfg.gs_col), col(cfg.hdg_col),
                              col(cfg.ts_col))
    ts = np.nan_to_num(ts, nan=0.0)

    have_vs = (cfg.vs_col >= 0 and f"c{cfg.vs_col}" in df.columns and
               df[f"c{cfg.vs_col}"].notna().mean() > 0.01)
    if cfg.vs_col >= 0 and not have_vs:
        LOG.warning("vertical-rate column %d is empty - deriving Vz from the "
                    "altitude differences", cfg.vs_col)
    if cfg.vs_col >= 0 and have_vs:
        vs = df[f"c{cfg.vs_col}"].to_numpy(dtype=np.float64)
    else:
        vs = _derive_vz(df, alt_ft, ts)

    if cfg.apply_sanity_filter:
        ok = (np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt_ft) &
              np.isfinite(gs_kt) & np.isfinite(hdg) & np.isfinite(vs) & np.isfinite(ts) &
              (np.abs(lat) <= 90.0) & (np.abs(lon) <= 180.0) &
              (alt_ft >= 0.0) & (alt_ft <= cfg.max_alt_ft) &
              (gs_kt >= 0.0) & (gs_kt <= cfg.max_gs_kt) &
              (np.abs(hdg) <= 360.0) & (np.abs(vs) <= cfg.max_vs_ftmin) & (ts > 0.0))
        n0 = len(df)
        df = df[ok]
        lat, lon, alt_ft = lat[ok], lon[ok], alt_ft[ok]
        gs_kt, hdg, vs, ts = gs_kt[ok], hdg[ok], vs[ok], ts[ok]
        LOG.info("sanity filter: kept %d / %d rows (%.2f%%)", len(df), n0,
                 100.0 * len(df) / max(n0, 1))

    gs_kmh = gs_kt * KT_TO_KMH
    rad = np.radians(hdg)
    return pd.DataFrame({
        "flight_id": df["flight_id"].to_numpy(),
        "ts": ts,
        "lon": np.rint(lon * PRECISION["lon"]),
        "lat": np.rint(lat * PRECISION["lat"]),
        "alt": np.rint(alt_ft * FT_TO_M * PRECISION["alt"]),
        "vx": np.rint(gs_kmh * np.sin(rad) * PRECISION["vx"]),   # East
        "vy": np.rint(gs_kmh * np.cos(rad) * PRECISION["vy"]),   # North
        "vz": np.rint(vs * FTMIN_TO_KMH * PRECISION["vz"]),      # Up
    })


def _derive_vz(df: pd.DataFrame, alt_ft: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Vertical rate in ft/min, from the altitude differences of every track."""
    tmp = pd.DataFrame({"flight_id": df["flight_id"].to_numpy(), "ts": ts,
                        "alt": alt_ft})
    tmp = tmp.sort_values(["flight_id", "ts"], kind="mergesort")
    order = tmp.index
    dalt = tmp.groupby("flight_id", sort=False)["alt"].diff()
    dts = tmp.groupby("flight_id", sort=False)["ts"].diff()
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = (dalt / dts.replace(0, np.nan)) * 60.0      # ft/min
    rate = rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vz = np.zeros(len(df), dtype=np.float64)
    vz[df.index.get_indexer(order)] = rate.to_numpy(dtype=np.float64)
    return vz


def _resample_leg(values: np.ndarray, ts: np.ndarray, cfg: Config
                  ) -> Optional[Tuple[np.ndarray, float, float]]:
    """Evenly spaced leg on the grid t0 + j*period (step-hold resampling).

    Returns (leg, t_first, t_last) where the times refer to the resampled grid.
    """
    t0 = float(ts[0])
    n_grid = int((ts[-1] - ts[0]) // cfg.period) + 1
    if n_grid < cfg.min_points:
        return None
    grid = t0 + np.arange(n_grid, dtype=np.float64) * cfg.period
    idx = np.searchsorted(ts, grid, side="right") - 1
    np.clip(idx, 0, len(ts) - 1, out=idx)
    fresh = (grid - ts[idx]) <= cfg.gap_factor * cfg.period
    idx = idx[fresh]
    if len(idx) < cfg.min_points:
        return None
    leg = np.ascontiguousarray(values[idx], dtype=np.float32)
    return leg, t0, t0 + (len(leg) - 1) * cfg.period


def _split_trajectory(values: np.ndarray, ts: np.ndarray, cfg: Config
                      ) -> List[Tuple[np.ndarray, float, float]]:
    """Break a sorted track at large time gaps, then resample every piece."""
    legs: List[Tuple[np.ndarray, float, float]] = []
    if len(ts) < 2:
        return legs
    max_gap = cfg.gap_factor * cfg.period
    breaks = np.flatnonzero(np.diff(ts) > max_gap) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(ts)]))
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        out = _resample_leg(values[s:e], ts[s:e], cfg)
        if out is not None:
            legs.append(out)
    return legs


@dataclass
class DataBundle:
    legs: List[np.ndarray]                     # every resampled trajectory (T, 6)
    windows: Dict[str, List[Tuple[int, int]]]  # split -> (leg index, position)
    t_min: float = 0.0
    t_max: float = 0.0
    boundaries: Tuple[float, float] = (0.0, 0.0)
    t_start: List[float] = field(default_factory=list)   # epoch of leg[i][0]
    leg_flight: List[str] = field(default_factory=list)  # aircraft id of leg[i]
    leg_split: List[str] = field(default_factory=list)   # train/val/test of leg[i]


def _saturation_report(legs: List[np.ndarray], enc: BinaryEncoding
                       ) -> Dict[str, float]:
    """Fraction of values per attribute that the bit widths cannot represent.

    Anything above 0 means the BE is silently lossy for that attribute, so it
    belongs in the log next to the metric report rather than being discovered
    through an unexplained error floor.
    """
    over = {a: 0 for a in ATTRS}
    total = 0
    for leg in legs:
        total += len(leg)
        for j, a in enumerate(ATTRS):
            over[a] += int((np.abs(leg[:, j]) > enc.max_magnitude(a)).sum())
    if not total:
        return {}
    return {a: n / total for a, n in over.items() if n}


def build_dataset(cfg: Config) -> Tuple[DataBundle, BinaryEncoding]:
    """Load the CSV, derive the attributes, resample, split, index the windows."""
    paths = resolve_csv(cfg.csv_path)
    LOG.info("loading %d file(s): %s", len(paths), ", ".join(
        "%s (%.1f MB)" % (os.path.basename(p), os.path.getsize(p) / 1e6)
        for p in paths))
    t0 = time.time()
    frames = []
    for p in paths:
        LOG.info("reading %s", os.path.basename(p))
        frames.append(_to_encoded_units(_read_csv(p, cfg), cfg))
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    del frames
    LOG.info("%d rows parsed in %.1fs", len(df), time.time() - t0)

    df = df.drop_duplicates(subset=["flight_id", "ts"], keep="last")
    df = df.sort_values(["flight_id", "ts"], kind="mergesort").reset_index(drop=True)

    need = cfg.inp_seq_len + cfg.max_horizon
    legs: List[np.ndarray] = []
    t_start: List[float] = []
    t_end: List[float] = []
    leg_flight: List[str] = []
    n_flights = 0
    per_flight: List[int] = []
    for fid, g in df.groupby("flight_id", sort=False):
        if cfg.limit_flights and n_flights >= cfg.limit_flights:
            break
        n_flights += 1
        v = g[list(ATTRS)].to_numpy(dtype=np.float32)
        ts = g["ts"].to_numpy(dtype=np.float64)
        per_flight.append(len(ts))
        for leg, leg_t0, leg_t1 in _split_trajectory(v, ts, cfg):
            if len(leg) >= need:
                legs.append(leg)
                t_start.append(leg_t0)
                t_end.append(leg_t1)
                leg_flight.append(str(fid))
    del df
    if not legs:
        longest = max(per_flight) if per_flight else 0
        raise RuntimeError(
            "No usable trajectory: every aircraft needs at least "
            f"{need} samples at --period {cfg.period}s "
            f"(k={cfg.inp_seq_len} observed + {cfg.max_horizon} horizon) but the "
            f"longest track found has only {longest} row(s) across "
            f"{n_flights} aircraft.\n"
            "  * the file is probably a format sample rather than a real dataset;\n"
            "  * a smaller --period can help when the sampling rate is high;\n"
            "  * lower --inp_seq_len/--max_horizon if the tracks are very short;\n"
            "  * check --id_col (aircraft id) and --ts_col (timestamp).")
    LOG.info("%d aircraft -> %d trajectories of >= %d points (%.1fs)",
             n_flights, len(legs), need, time.time() - t0)

    # ---- bit widths: keep the paper's, widen them if this data needs more -----
    enc = BinaryEncoding(cfg.bits)
    if cfg.auto_bits:
        peak = np.zeros(len(ATTRS))
        for leg in legs:                       # running maximum over the whole set
            np.maximum(peak, np.abs(leg).max(axis=0), out=peak)
        widened, clamped = {}, {}
        for j, a in enumerate(ATTRS):
            need_bits = bits_needed(float(peak[j]), a in SIGNED_ATTRS)
            if need_bits > enc.bits[a]:
                widened[a] = min(need_bits, cfg.max_bits)
                if need_bits > cfg.max_bits:
                    clamped[a] = (need_bits, cfg.max_bits)
        if widened:
            LOG.warning("BE bit widths widened to cover the data range: %s",
                        ", ".join(f"{a}: {enc.bits[a]} -> {n}b" for a, n in widened.items()))
            if clamped:
                # the encoder clips at 2^bits-1, so a clamped width silently
                # saturates the extreme values instead of representing them
                LOG.error("max_bits=%d is too small: %s would need %s. Those "
                          "values are CLIPPED and the BE is lossy - raise "
                          "--max_bits or lower --pos_prec.",
                          cfg.max_bits,
                          ", ".join(a for a in clamped), ", ".join(
                              f"{a} {n}b" for a, (n, _) in clamped.items()))
            for a, n in widened.items():
                enc.bits[a] = n
            enc = BinaryEncoding(enc.bits)
        cfg.bits = dict(enc.bits)
        LOG.info("attribute peaks: %s", ", ".join(
            f"{a}={peak[j]:.0f} (cap {enc.max_magnitude(a)})" for j, a in enumerate(ATTRS)))
        sat = _saturation_report(legs, enc)
        if sat:
            LOG.warning("BE saturation (values clipped by the bit widths): %s",
                        ", ".join(f"{a} {v * 100:.3f}%" for a, v in sat.items()))
    LOG.info("BE representation: %s -> %d-dim vector (input = output)",
             enc.describe(), enc.total_bits)

    # ------------------------------------------------------------------ split
    t_start_arr = np.asarray(t_start, dtype=np.float64)
    t_end_arr = np.asarray(t_end, dtype=np.float64)
    t_min, t_max = float(t_start_arr.min()), float(t_end_arr.max())
    span = max(t_max - t_min, 1.0)
    t1 = t_min + span * cfg.train_frac
    t2 = t_min + span * (cfg.train_frac + cfg.val_frac)
    k, mh = cfg.inp_seq_len, cfg.max_horizon

    bundle = DataBundle(legs=legs,
                        windows={"train": [], "val": [], "test": []},
                        t_min=t_min, t_max=t_max, boundaries=(t1, t2),
                        t_start=list(t_start), leg_flight=leg_flight,
                        leg_split=[""] * len(legs))
    if cfg.split_mode == "flight":
        rng = np.random.RandomState(cfg.seed)
        order = rng.permutation(len(legs))
        owner = np.array(["test"] * len(legs), dtype=object)
        n_tr = int(len(legs) * cfg.train_frac)
        n_va = int(len(legs) * (cfg.train_frac + cfg.val_frac))
        owner[order[:n_tr]] = "train"
        owner[order[n_tr:n_va]] = "val"
        for i, split in enumerate(owner):
            horizon = 1 if split == "train" else mh
            n_pos = max(len(legs[i]) - k - horizon + 1, 0)
            bundle.windows[str(split)].extend((i, p) for p in range(n_pos))
    else:
        # The paper splits the data chronologically (by days). Every *window* is
        # assigned by the time of its last observation, so long trajectories
        # contribute to all three splits instead of being discarded:
        #   train : last observation (and its 1-step target) before t1
        #   val   : last observation after t1, whole horizon before t2
        #   test  : last observation after t2
        # no target ever crosses a split boundary.
        need_h = (mh - 1) * cfg.period
        for i, leg in enumerate(legs):
            n = len(leg)
            grid = t_start_arr[i] + np.arange(n, dtype=np.float64) * cfg.period
            pos = np.arange(0, max(n - k, 0))
            t_last = grid[pos + k - 1]
            pos_h = np.arange(0, max(n - k - mh + 1, 0))
            t_last_h = grid[pos_h + k - 1]
            bundle.windows["train"].extend(
                (i, int(p)) for p in pos[t_last <= t1])
            bundle.windows["val"].extend(
                (i, int(p)) for p in pos_h[(t_last_h >= t1) & (t_last_h + need_h <= t2)])
            bundle.windows["test"].extend(
                (i, int(p)) for p in pos_h[t_last_h >= t2])

    for s in ("train", "val", "test"):
        for li, _ in bundle.windows[s]:        # a leg can feed several splits;
            bundle.leg_split[li] = s            # keep the most specific one
        LOG.info("  %-5s windows=%-9d (trajectories: %d)", s,
                 len(bundle.windows[s]),
                 len({li for li, _ in bundle.windows[s]}))
    if not bundle.windows["train"]:
        raise RuntimeError("the training split is empty - adjust --period / "
                           "--min_points / --limit_flights or the split fractions")
    LOG.info("time span %s -> %s | train<%s val<%s test",
             pd.to_datetime(t_min, unit="s").strftime("%Y-%m-%d %H:%M"),
             pd.to_datetime(t_max, unit="s").strftime("%Y-%m-%d %H:%M"),
             pd.to_datetime(t1, unit="s").strftime("%H:%M"),
             pd.to_datetime(t2, unit="s").strftime("%H:%M"))
    return bundle, enc


# =============================================================================
# 6. Datasets and dataloaders
# =============================================================================

class BECache:
    """Binary-encodes every leg once; keeps the uint8 codes in memory."""

    def __init__(self, legs: List[np.ndarray], enc: BinaryEncoding):
        self.enc = enc
        self.be = [enc.encode_matrix(leg) for leg in legs]
        self.vals = [np.ascontiguousarray(leg, dtype=np.float32) for leg in legs]
        self.n_points = int(sum(len(b) for b in self.be))
        self.nbytes = int(sum(b.nbytes for b in self.be))


class TrainWindowDataset(Dataset):
    """k observed points -> the next point (the objective of Sec. III-A).

    With --direct the target is the whole horizon (n, D), because the model then
    reads one prediction per observed position instead of rolling forward.
    """

    def __init__(self, cache: BECache, windows: Sequence[Tuple[int, int]], cfg: Config):
        if not len(windows):
            raise RuntimeError("empty training set")
        self.cache, self.windows, self.k = cache, list(windows), cfg.inp_seq_len
        self.n = cfg.max_horizon
        self.direct = bool(cfg.direct)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        li, pos = self.windows[i]
        be = self.cache.be[li]
        x = torch.from_numpy(be[pos:pos + self.k])       # uint8 view, cast later
        if not self.direct:
            return x, torch.from_numpy(be[pos + self.k])
        tgt = be[pos + self.k:pos + self.k + self.n]
        if tgt.shape[0] < self.n:                        # never happens, stay safe
            tgt = np.concatenate([tgt, np.repeat(tgt[-1:], self.n - tgt.shape[0], 0)], 0)
        return x, torch.from_numpy(tgt)


class EvalWindowDataset(Dataset):
    """Input window + the next `max_horizon` ground truth points (encoded units)."""

    def __init__(self, cache: BECache, windows: Sequence[Tuple[int, int]], cfg: Config):
        self.cache, self.windows = cache, list(windows)
        self.k, self.n = cfg.inp_seq_len, cfg.max_horizon

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        li, pos = self.windows[i]
        be, val = self.cache.be[li], self.cache.vals[li]
        y = val[pos + self.k:pos + self.k + self.n]
        if y.shape[0] < self.n:                     # never happens, but stay safe
            pad = np.repeat(y[-1:], self.n - y.shape[0], axis=0)
            y = np.concatenate([y, pad], axis=0)
        return (torch.from_numpy(be[pos:pos + self.k]),   # uint8 view, cast later
                torch.from_numpy(np.ascontiguousarray(y)))


def build_dataloaders(bundle: DataBundle, cfg: Config, enc: BinaryEncoding):
    """One binary-encoded cache shared by the three datasets."""
    pin = torch.cuda.is_available()
    t0 = time.time()
    cache = BECache(bundle.legs, enc)
    LOG.info("BE cache: %d trajectories, %d points, %.1f MB (%.1fs)",
             len(bundle.legs), cache.n_points, cache.nbytes / 1e6, time.time() - t0)
    loaders: Dict[str, DataLoader] = {}
    kw = dict(num_workers=cfg.num_workers, pin_memory=pin,
              persistent_workers=cfg.num_workers > 0,
              prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None)
    for split in ("train", "val", "test"):
        if split == "train":
            ds: Dataset = TrainWindowDataset(cache, bundle.windows[split], cfg)
            # drop_last keeps the batch shape static and skips a tiny ragged
            # tail, but it must be off when the set is smaller than one batch,
            # otherwise the epoch would silently iterate over *nothing*.
            drop = len(ds) > cfg.batch_size
            loaders[split] = DataLoader(
                ds, batch_size=cfg.batch_size, shuffle=True, drop_last=drop, **kw)
        else:
            ds = EvalWindowDataset(cache, bundle.windows[split], cfg)
            # the 15-step rollout saturates the GPU around 512, and it is the
            # single most expensive thing this script does
            loaders[split] = DataLoader(
                ds, batch_size=max(cfg.eval_batch_size, 1), shuffle=False, **kw)
    return loaders


# =============================================================================
# 7. Training and evaluation
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        LOG.warning("PyTorch cannot see any CUDA device -> falling back to the CPU. "
                    "This build is %s (CUDA runtime: %s); the usual cause is a "
                    "CPU-only wheel installed from PyPI, whose default on Windows "
                    "is CPU. Install a CUDA build, e.g.\n"
                    "    pip install torch --index-url "
                    "https://download.pytorch.org/whl/cu128\n"
                    "or point --device cuda at an interpreter that already has one "
                    "(check with `python -c \"import torch; print(torch.__version__, "
                    "torch.cuda.is_available())\"`).",
                    torch.__version__, torch.version.cuda or "none (CPU-only build)")
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        LOG.warning("--device %s requested but PyTorch reports no CUDA device: "
                    "torch %s (CUDA runtime: %s). Training will fail or stay on "
                    "the CPU.", name, torch.__version__, torch.version.cuda or "none")
    return torch.device(name)


def make_scaler(enabled: bool) -> Optional[torch.amp.GradScaler]:
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):                            # pragma: no cover
        return torch.cuda.amp.GradScaler()


def amp_dtype(cfg: Config) -> torch.dtype:
    """Autocast dtype. bf16 needs no GradScaler and has the same tensor-core
    throughput as fp16 on Ada/Blackwell; fp16 is kept as the default because it
    is what the reported runs used."""
    return torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16


@torch.no_grad()
def predict_horizons(model: FlightBERT, x: torch.Tensor, n_steps: int,
                     feed_soft: bool = False) -> torch.Tensor:
    """(B, k, D) observations -> (B, n_steps, D) binary coded predictions.

    --direct : a single forward already returns every horizon, so there is
               nothing to iterate and no prediction is ever fed back.
    otherwise: the paper's iterative scheme (Eq. 3) - predict one step, append
               it (hard bits, or the soft probabilities with --feed_soft) and
               repeat, sliding the window.
    """
    was_training = model.training
    model.eval()
    if model.direct:
        probs = model(x)[:, :n_steps]
        out = probs if feed_soft else (probs > 0.5).to(x.dtype)
    else:
        window, outs = x, []
        for _ in range(n_steps):
            probs = model(window)
            nxt = probs if feed_soft else (probs > 0.5).to(window.dtype)
            outs.append(nxt)
            window = torch.cat([window[:, 1:], nxt.unsqueeze(1)], dim=1)
        out = torch.stack(outs, dim=1)
    if was_training:
        model.train()
    return out


#: backwards-compatible alias (the iterative scheme is the default)
autoregressive_rollout = predict_horizons


@torch.no_grad()
def evaluate(model: FlightBERT, loader: DataLoader, enc: BinaryEncoding, cfg: Config,
             device: torch.device, max_batches: int = 0) -> Dict[str, float]:
    """MAE / RMSE per attribute at the requested horizons (real units)."""
    model.eval()
    horizons = sorted({int(h) for h in cfg.eval_horizons if 1 <= int(h) <= cfg.max_horizon})
    horizons = horizons or [1]
    abs_sum = {h: np.zeros(len(ATTRS)) for h in horizons}
    sq_sum = {h: np.zeros(len(ATTRS)) for h in horizons}
    hdist = {h: 0.0 for h in horizons}
    cnt = {h: 0 for h in horizons}
    # per-sample horizontal errors, so the report can show the tail and not just
    # the mean: RMSE >> MAE throughout, i.e. a minority of windows dominates.
    # The mean below is exact; the percentiles run on a bounded sample so RAM
    # stays flat on huge splits.
    hdists: Dict[int, List[np.ndarray]] = {h: [] for h in horizons}
    hbatch: Dict[int, int] = {h: 0 for h in horizons}      # batches seen
    keep_every: Dict[int, int] = {h: 1 for h in horizons}  # adaptive stride
    thinned: set = set()
    # trivial reference: keep the last observed point for every horizon
    last_sum = {h: np.zeros(len(ATTRS)) for h in horizons}
    last_hdist = {h: 0.0 for h in horizons}
    t0, nb = time.time(), 0
    i_lon, i_lat = ATTRS.index("lon"), ATTRS.index("lat")
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()      # uint8 -> float on GPU
        b = x.size(0)
        with torch.autocast(device_type=device.type, dtype=amp_dtype(cfg),
                            enabled=bool(cfg.amp and device.type == "cuda")):
            rolled = predict_horizons(model, x, cfg.max_horizon, cfg.feed_soft)
        pred = enc.decode_matrix(
            rolled.reshape(-1, enc.total_bits).float().cpu().numpy()
        ).reshape(b, cfg.max_horizon, len(ATTRS))                     # (B, n, 6)
        last = enc.decode_matrix(
            x[:, -1, :].reshape(-1, enc.total_bits).float().cpu().numpy())
        truth = y.cpu().numpy()                                      # (B, n, 6)
        for h in horizons:
            d = np.abs(pred[:, h - 1] - truth[:, h - 1])             # (B, 6)
            abs_sum[h] += d.sum(axis=0)
            sq_sum[h] += (d ** 2).sum(axis=0)
            cnt[h] += b
            dl = np.abs(last - truth[:, h - 1])
            last_sum[h] += dl.sum(axis=0)
            # horizontal distance in km, a classical FTP metric
            lat_true = truth[:, h - 1, i_lat] / PRECISION["lat"]
            dy = d[:, i_lat] / PRECISION["lat"] * 111.32
            dx = d[:, i_lon] / PRECISION["lon"] * 111.32 * np.cos(np.radians(lat_true))
            hd = np.hypot(dx, dy)
            hdist[h] += float(hd.sum())
            hbatch[h] += 1
            if hbatch[h] <= HDIST_BATCHES or hbatch[h] % keep_every[h] == 0:
                hdists[h].append(hd.astype(np.float32))
                # tighten the stride if the retained sample is running away
                if len(hdists[h]) > 4 * HDIST_BATCHES and keep_every[h] < 1 << 20:
                    keep_every[h] *= 2
                    if h not in thinned:
                        thinned.add(h)
                        LOG.info("h=%d: the split has more than ~%d windows, the "
                                 "percentiles are estimated on a thinned subset "
                                 "(the mean stays exact)", h, HDIST_BATCHES * 512)
            last_hdist[h] += float(np.hypot(
                dl[:, i_lon] / PRECISION["lon"] * 111.32 * np.cos(np.radians(lat_true)),
                dl[:, i_lat] / PRECISION["lat"] * 111.32).sum())
        nb += 1
        if max_batches and nb >= max_batches:
            break
    if not cnt or all(c == 0 for c in cnt.values()):
        return {}

    res: Dict[str, float] = {"eval_seconds": time.time() - t0, "n_samples": float(cnt[horizons[0]])}
    lines = []
    for h in horizons:
        n = max(cnt[h], 1)
        mae, rmse = abs_sum[h] / n, np.sqrt(sq_sum[h] / n)
        score = 0.0
        for j, a in enumerate(ATTRS):
            res[f"mae@{h}_{a}"] = float(mae[j] / PRECISION[a])
            res[f"rmse@{h}_{a}"] = float(rmse[j] / PRECISION[a])
            score += mae[j] / enc.max_magnitude(a)
        res[f"score@{h}"] = float(score / len(ATTRS))
        res[f"hdist@{h}"] = hdist[h] / n
        pct = {}
        if hdists[h]:
            v = np.concatenate(hdists[h])
            for p in (50, 90, 95):
                pct[p] = float(np.percentile(v, p))
                res[f"hdist_p{p}@{h}"] = pct[p]
        for j, a in enumerate(ATTRS):
            res[f"persist_mae@{h}_{a}"] = float(last_sum[h][j] / n / PRECISION[a])
        res[f"persist_hdist@{h}"] = last_hdist[h] / n
        lines.append(
            "  h=%-3d " % h + "  ".join(
                "%s %.4f/%.4f" % (a, mae[j] / PRECISION[a], rmse[j] / PRECISION[a])
                for j, a in enumerate(ATTRS))
            + "  |  mean %.3f km  P90 %.3f km  (persistence %.3f km)"
            % (hdist[h] / n, pct.get(90, float("nan")), last_hdist[h] / n))
    res["report"] = "\n".join(lines)
    res_u = ", ".join("%s %s" % (a, ("%.0f m" % v) if a in ("lon", "lat", "alt")
                                 else ("%.0f km/h" % v))
                      for a, v in attr_resolution().items())
    LOG.info("horizon  " + "  ".join("%-13s" % a for a in ATTRS)
             + "  (MAE/RMSE, deg/deg/m/kmh)\n%s", "\n".join(lines))
    LOG.info("BE quantisation floor (smallest representable step): %s", res_u)
    return res


def train_one_epoch(model, loader, criterion, optimizer, scaler, cfg, device,
                    epoch: int) -> float:
    model.train()
    nb, t0 = 0, time.time()
    # the running loss stays on the device: float(loss) every step would force a
    # host sync per iteration and serialise the GPU pipeline (~5% at bs=1024)
    total = torch.zeros((), device=device)
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True).float()      # uint8 -> float on GPU
        y = y.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype(cfg),
                            enabled=bool(cfg.amp and device.type == "cuda")):
            loss = criterion(model(x), y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        total += loss.detach()
        nb += 1
        if cfg.log_every and (i + 1) % cfg.log_every == 0:
            LOG.info("  epoch %d [%d/%d] loss=%.4f (%.1f it/s)", epoch, i + 1,
                     len(loader), float(total) / nb,
                     (i + 1) / max(time.time() - t0, 1e-9))
    return float(total) / max(nb, 1)


def save_checkpoint(path: str, model, optimizer, scheduler, scaler, epoch, cfg,
                    enc, best: float, metrics: Dict[str, float]) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_val": best,
        "bits": dict(enc.bits),
        "config": asdict(cfg),
        "metrics": metrics,
    }, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None,
                    device: torch.device = torch.device("cpu")) -> Dict:
    LOG.info("loading checkpoint %s", path)
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except TypeError:                                              # torch < 1.13
        ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    for mod, key in ((optimizer, "optimizer_state_dict"),
                     (scheduler, "scheduler_state_dict"),
                     (scaler, "scaler_state_dict")):
        if mod is not None and ck.get(key):
            mod.load_state_dict(ck[key])
    return ck


# =============================================================================
# 8. Entry point
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FlightBERT (T-ITS 2023): binary encoding flight trajectory "
                    "prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--csv", default="",
                   help="path to the 19-column ADS-B CSV, or several separated "
                        "by ',' (they are concatenated before splitting, so "
                        "train on one day and test on another)")
    g.add_argument("--out_dir", default="runs/flightbert")
    g.add_argument("--period", type=int, default=30,
                   help="resampling period (s); the paper used 20")
    g.add_argument("--inp_seq_len", type=int, default=9, help="k observed points")
    g.add_argument("--max_horizon", type=int, default=15, help="n predicted steps")
    g.add_argument("--eval_horizons", type=int, nargs="+", default=[1, 3, 9, 15])
    g.add_argument("--min_points", type=int, default=12)
    g.add_argument("--split_mode", choices=["time", "flight"], default="time")
    g.add_argument("--train_frac", type=float, default=0.70)
    g.add_argument("--val_frac", type=float, default=0.15)
    g.add_argument("--limit_flights", type=int, default=0, help="debug: first N aircraft")
    g.add_argument("--id_col", type=int, default=0)
    g.add_argument("--lat_col", type=int, default=1)
    g.add_argument("--lon_col", type=int, default=2)
    g.add_argument("--hdg_col", type=int, default=3)
    g.add_argument("--alt_col", type=int, default=4)
    g.add_argument("--gs_col", type=int, default=5)
    g.add_argument("--ts_col", type=int, default=10)
    g.add_argument("--vs_col", type=int, default=15,
                   help="vertical-rate column (ft/min); -1 derives Vz from altitude")
    g.add_argument("--sep", default="", help="CSV separator, e.g. ';'. Empty = auto")
    g.add_argument("--decimal", default="", help="CSV decimal separator, e.g. ','. "
                                                 "Empty = auto")
    g.add_argument("--no_sanity_filter", action="store_true")
    g = p.add_argument_group("model")
    g.add_argument("--d_attr", type=int, default=64, help="size of e_ak and e'_ak")
    g.add_argument("--n_coatt_heads", type=int, default=4)
    g.add_argument("--n_layers", type=int, default=8)
    g.add_argument("--n_heads", type=int, default=4)
    g.add_argument("--dim_ff", type=int, default=1536)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--no_causal", action="store_true", help="drop the causal MMSA mask")
    g.add_argument("--direct", action="store_true",
                   help="predict all n horizons in a single forward instead of "
                        "the paper's iterative rollout: no prediction is fed "
                        "back, so the error cannot compound, and the evaluation "
                        "becomes n times cheaper. Requires --causal (the default).")
    g.add_argument("--no_auto_bits", action="store_true",
                   help="keep the paper's bit widths even if the data needs more")
    g.add_argument("--bits", default="",
                   help="force the BE widths, e.g. --bits 18,16,11,12,12,12 "
                        "(lon,lat,alt,vx,vy,vz); overrides --no_auto_bits and the "
                        "widths stored in a checkpoint")
    g.add_argument("--pos_prec", type=int, default=1000,
                   help="decimals kept for lon/lat: 1000 = 0.001 deg (~111 m, "
                        "the paper), 10000 = 0.0001 deg (~11 m), 100000 = 1 m. "
                        "This is what sets the positional resolution -- a larger "
                        "--period does NOT")
    g = p.add_argument_group("loss")
    g.add_argument("--bce_weight", type=float, default=1.0)
    g.add_argument("--mse_weight", type=float, default=None,
                   help="weight of the decoded-decimal MSE; default: calibrated "
                        "at startup so that it matches the BCE (the two terms "
                        "are otherwise incomparable in scale)")
    g.add_argument("--mse_mode", choices=["rel", "raw", "phys"], default="rel",
                   help="rel: MSE scaled by the bit capacity (stable, default); "
                        "phys: same scale, re-weighted so that a position error "
                        "counts as much as a velocity error; raw: literal Eq. (8) "
                        "in encoded units (loss ~1e9 at init)")
    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--batch_size", type=int, default=512)
    g.add_argument("--eval_batch_size", type=int, default=512,
                   help="the 15-step rollout saturates the GPU around 512")
    g.add_argument("--eval_every", type=int, default=1,
                   help="validate every N epochs (the last epoch always runs); "
                        "validation is the dominant per-epoch cost")
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight_decay", type=float, default=0.0)
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--lr_patience", type=int, default=2)
    g.add_argument("--lr_factor", type=float, default=0.5)
    g.add_argument("--num_workers", type=int, default=0)
    g.add_argument("--prefetch_factor", type=int, default=4)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", default="auto")
    g.add_argument("--no_amp", action="store_true")
    g.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="fp16",
                   help="fp16 needs the GradScaler; bf16 does not and is a bit "
                        "more robust on long runs")
    g.add_argument("--feed_soft", action="store_true",
                   help="feed the sigmoid probabilities instead of the hard bits")
    g.add_argument("--log_every", type=int, default=50)
    g.add_argument("--save_every", type=int, default=1)
    g = p.add_argument_group("misc")
    g.add_argument("--resume", default="", help="checkpoint to load")
    g.add_argument("--test_only", action="store_true")
    g.add_argument("--eval_batches", type=int, default=0, help="0 = all batches")
    g.add_argument("--plot", default="", help="write a trajectory plot to this path")
    g.add_argument("--map", default="",
                   help="write an interactive Leaflet map of the trajectories "
                        "replayed second by second to this .html path")
    g.add_argument("--map_step", type=float, default=1.0,
                   help="seconds between two fixes on the map (default 1 s; "
                        "anything finer than --period is interpolated)")
    g.add_argument("--map_flights", type=int, default=40,
                   help="how many aircraft to put on the map (longest first)")
    g.add_argument("--map_max_points", type=int, default=300000,
                   help="cap on the fixes embedded in the page; the grid is "
                        "coarsened automatically above it")
    g.add_argument("--map_split", choices=["train", "val", "test"], default="",
                   help="restrict the map to one split (default: any)")
    g.add_argument("--no_map_predict", action="store_true",
                   help="do not overlay the model forecast on the map")
    g.add_argument("--map_pred_max_km", type=float, default=200.0,
                   help="drop a mapped forecast that lands further than this "
                        "from the last fix (a model failure, not a track)")
    g.add_argument("--no_selftest", action="store_true")
    return p


def parse_bits(spec: str) -> Optional[Dict[str, int]]:
    if not spec:
        return None
    parts = [p for p in spec.replace(" ", "").split(",") if p]
    if len(parts) != len(ATTRS):
        raise ValueError(f"--bits expects {len(ATTRS)} comma separated integers "
                         f"({','.join(ATTRS)}), got {spec!r}")
    return {a: int(p) for a, p in zip(ATTRS, parts)}


def config_from_args(a: argparse.Namespace) -> Config:
    return Config(
        csv_path=a.csv, out_dir=a.out_dir, period=a.period, inp_seq_len=a.inp_seq_len,
        max_horizon=a.max_horizon, eval_horizons=tuple(a.eval_horizons),
        min_points=a.min_points, split_mode=a.split_mode, train_frac=a.train_frac,
        val_frac=a.val_frac, limit_flights=a.limit_flights, id_col=a.id_col,
        lat_col=a.lat_col, lon_col=a.lon_col, hdg_col=a.hdg_col, alt_col=a.alt_col,
        gs_col=a.gs_col, ts_col=a.ts_col, vs_col=a.vs_col, sep=a.sep, decimal=a.decimal,
        apply_sanity_filter=not a.no_sanity_filter, auto_bits=not a.no_auto_bits,
        d_attr=a.d_attr, n_coatt_heads=a.n_coatt_heads, n_layers=a.n_layers,
        n_heads=a.n_heads, dim_ff=a.dim_ff, dropout=a.dropout,
        causal=not a.no_causal, direct=a.direct,
        bce_weight=a.bce_weight, mse_weight=a.mse_weight,
        mse_mode=a.mse_mode, epochs=a.epochs, batch_size=a.batch_size,
        eval_batch_size=a.eval_batch_size, eval_every=a.eval_every, lr=a.lr,
        weight_decay=a.weight_decay, grad_clip=a.grad_clip, lr_patience=a.lr_patience,
        lr_factor=a.lr_factor, num_workers=a.num_workers,
        prefetch_factor=a.prefetch_factor, seed=a.seed, device=a.device,
        amp=not a.no_amp, amp_dtype=a.amp_dtype, pos_prec=a.pos_prec,
        feed_soft=a.feed_soft, log_every=a.log_every,
        save_every=a.save_every, resume_path=a.resume,
        map_step=a.map_step, map_flights=a.map_flights, map_split=a.map_split,
        map_max_points=a.map_max_points, map_pred_max_km=a.map_pred_max_km)


def warn_encoding_drift(stored: Dict, cfg: Config) -> None:
    """Refuse to resume silently under a different BE encoding.

    --period and --pos_prec change how the raw feed is quantised *before* the
    network sees it, so a checkpoint trained under one setting is meaningless
    under the other (the bit widths alone are not enough to catch it).
    """
    for key in ("period", "pos_prec", "auto_bits", "bits"):
        if key not in stored or key == "bits":
            continue
        old, new = stored[key], getattr(cfg, key)
        if old != new:
            LOG.error("%s differs from the checkpoint (%s -> %s). This changes "
                      "the BE encoding of the data, so this run is NOT "
                      "comparable to %s: resume with the original value or start "
                      "a new run in a different --out_dir.", key, old, new,
                      "the checkpoint")


def load_checkpoint_config(path: str) -> Dict:
    """Read the Config stored inside a checkpoint (empty dict if unavailable)."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:                                          # torch < 1.13
            ck = torch.load(path, map_location="cpu")
    except Exception as exc:                                       # pragma: no cover
        LOG.warning("could not read %s (%s)", path, exc)
        return {}
    cfgd = ck.get("config") if isinstance(ck, dict) else None
    return cfgd if isinstance(cfgd, dict) else {}


def bits_from_checkpoint(path: str) -> Optional[Dict[str, int]]:
    """Recover the BE widths a checkpoint was trained with."""
    bits = load_checkpoint_config(path).get("bits")
    if isinstance(bits, dict) and all(a in bits for a in ATTRS):
        return {a: int(bits[a]) for a in ATTRS}
    return None


def selftest(enc: BinaryEncoding, cfg: Config, model: FlightBERT,
             criterion: HybridConstrainedLoss,
             device: Optional[torch.device] = None) -> None:
    """Consistency checks: BE round-trip, tensor shapes, finiteness of a step."""
    vals = {"lon": 9.234, "lat": 45.678, "alt": 9800.0, "vx": -193.0, "vy": 86.0,
            "vz": 64.0}
    back = enc.decode_dict(enc.encode(vals))
    for a, v in vals.items():
        tol = abs(v) * 1e-6 + 1e-9
        if abs(back[a] - v) > tol:
            raise AssertionError(f"BE round-trip failed for {a}: {v} -> {back[a]}")
    if device is None:                       # follow the model, whatever it sits on
        device = next(model.parameters()).device
    rng = np.random.RandomState(0)
    x = torch.from_numpy(
        rng.randint(0, 2, size=(2, cfg.inp_seq_len, enc.total_bits)).astype(np.float32)
    ).to(device)
    shape = (2, cfg.max_horizon, enc.total_bits) if cfg.direct else (2, enc.total_bits)
    y = torch.from_numpy(rng.randint(0, 2, size=shape).astype(np.float32)).to(device)
    model.eval()
    out = model(x)
    if tuple(out.shape) != shape:
        raise AssertionError(f"unexpected model output {tuple(out.shape)}, "
                             f"expected {shape}")
    loss = criterion(out, y)
    loss.backward()
    if not torch.isfinite(loss):
        raise AssertionError("non-finite loss in the self-test")
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise AssertionError("non-finite gradients in the self-test")
    LOG.info("self-test OK | BE dim=%d | trajectory embedding=%d | loss=%.4f",
             enc.total_bits, model.d_model, float(loss.detach()))
    model.zero_grad(set_to_none=True)


def plot_prediction(model: FlightBERT, loader: DataLoader, enc: BinaryEncoding,
                    cfg: Config, device: torch.device, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                       # pragma: no cover
        LOG.warning("matplotlib unavailable (%s): skipping the plot", exc)
        return
    x, _ = next(iter(loader))
    x = x[:1].to(device).float()
    with torch.no_grad():
        bits = predict_horizons(model, x, cfg.max_horizon, cfg.feed_soft)
    pred = enc.decode_real(bits[0].cpu().numpy())
    hist = enc.decode_real(x[0].cpu().numpy())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, a in zip(axes, ("lon", "lat", "alt")):
        j = ATTRS.index(a)
        ax.plot(hist[:, j], "o-", label="observed")
        ax.plot(np.arange(len(hist), len(hist) + len(pred)), pred[:, j], "s--",
                label="predicted")
        ax.set_title(f"{a} ({ATTR_UNITS[a]})")
        ax.set_xlabel("time step")
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    LOG.info("plot written to %s", out_path)


# =============================================================================
# 7b. Interactive map                                       (visual inspection)
# =============================================================================

MAP_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>FlightBERT trajectory map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
 html,body{margin:0;height:100%;font:13px/1.4 system-ui,Segoe UI,sans-serif}
 #map{position:absolute;inset:0}
 #panel{position:absolute;left:10px;top:10px;z-index:1000;background:#fff;
   padding:10px 12px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.25);
   width:330px;max-height:92vh;overflow:auto}
 h1{font-size:14px;margin:0 0 6px}
 .row{display:flex;align-items:center;gap:6px;margin:6px 0}
 input[type=range]{flex:1}
 button,select{padding:3px 7px;border:1px solid #bbb;border-radius:5px;
   background:#fafafa;cursor:pointer}
 button:hover{background:#eee}
 table{border-collapse:collapse;font-size:12px;width:100%}
 td,th{padding:1px 3px;text-align:right}
 th{text-align:right;color:#555;font-weight:600}
 td:first-child,th:first-child{text-align:left}
 .muted{color:#666;font-size:11px}
 .swatch{display:inline-block;width:9px;height:9px;border-radius:2px;
   margin-right:4px}
</style></head><body>
<div id="map"></div>
<div id="panel">
 <h1>FlightBERT &mdash; __N__ aircraft, one fix every __STEP__ s</h1>
 <div class="row">
   <button id="play">&#9654; Play</button>
   <select id="speed">
     <option value="1">1&times;</option><option value="10">10&times;</option>
     <option value="60" selected>60&times;</option><option value="300">300&times;</option>
   </select>
   <span id="clock" class="muted"></span>
 </div>
 <div class="row"><input type="range" id="slider" min="0" max="__TMAX__" value="0"/></div>
 <div class="muted" id="hint"></div>
 <table id="tbl"><thead><tr><th>aircraft</th><th>lat</th><th>lon</th>
   <th>alt m</th><th>hdg</th><th>spd km/h</th></tr></thead><tbody></tbody></table>
 <p class="muted"><span class="swatch" style="background:#c0392b"></span>observed
   (fixes every __PERIOD__ s, linearly interpolated in between)
   __PREDNOTE__</p>
 <p class="muted">Basemap &copy; OpenStreetMap contributors. Generated by
   flightbert.py.</p>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DATA = __DATA__;
const T0 = DATA.t0, STEP = DATA.step;
const map = L.map('map').setView([DATA.lat0, DATA.lon0], 8);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom:19, attribution:'&copy; OpenStreetMap contributors'}).addTo(map);

const COLORS=['#c0392b','#1f77b4','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2',
 '#7f7f7f','#bcbd22','#17becf','#ff7f0e','#003f5c','#58508d','#bc5090','#a05195',
 '#ff6361','#7a5195','#ef5675','#ffa600'];
const iconCache={};
function icon(color, hdg){
  const key=color+'|'+Math.round(hdg/5)*5;
  if(iconCache[key]) return iconCache[key];
  const m=L.divIcon({className:'', iconSize:[16,16], iconAnchor:[8,8],
    html:'<svg width="16" height="16" viewBox="0 0 16 16" style="transform:rotate('
      +Math.round(hdg/5)*5+'deg)"><path d="M8 1 L12 14 L8 11 L4 14 Z" fill="'
      +color+'" stroke="#fff" stroke-width="0.8"/></svg>'});
  iconCache[key]=m; return m;
}
const tracks=DATA.tracks.map((t,i)=>({
  id:t.id, split:t.split, lat:t.lat, lon:t.lon, alt:t.alt, hdg:t.hdg, spd:t.spd,
  t0:t.t0, n:t.lat.length,
  color:COLORS[i%COLORS.length],
  marker:L.marker([t.lat[0],t.lon[0]],{icon:icon(COLORS[i%COLORS.length],t.hdg[0]),
    title:t.id, riseOnHover:true}).addTo(map),
  trail:L.polyline([],{color:COLORS[i%COLORS.length],weight:2,opacity:.55}).addTo(map),
  // pdat = the forecast coordinates, pred = the Leaflet line drawing them
  pdat: t.pred||null,
  pred: t.pred?L.polyline([],{color:'#111',weight:2.5,dashArray:'5,4',opacity:.8})
     .addTo(map):null,
  tbody:null
}));
// one row per aircraft in the table
const tb=document.querySelector('#tbl tbody');
tracks.forEach(tr=>{
  const row=document.createElement('tr');
  row.innerHTML='<td><span class="swatch" style="background:'+tr.color+'"></span>'
    +tr.id+'</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>';
  tb.appendChild(row); tr.tbody=row;
});

// position of aircraft i at absolute offset `u` seconds from T0
function sample(tr,u){
  if(u<tr.t0 || u>=tr.t0+tr.n*STEP) return null;
  const f=(u-tr.t0)/STEP, i=Math.floor(f), a=f-i;
  if(i>=tr.n-1) return {lat:tr.lat[tr.n-1],lon:tr.lon[tr.n-1],
                         alt:tr.alt[tr.n-1],hdg:tr.hdg[tr.n-1],spd:tr.spd[tr.n-1],i:i};
  const g=(k)=>tr.lat[i]+(tr.lat[i+1]-tr.lat[i])*a;
  const h=(k)=>tr.lon[i]+(tr.lon[i+1]-tr.lon[i])*a;
  return {lat:g(),lon:h(),alt:tr.alt[i]+(tr.alt[i+1]-tr.alt[i])*a,
           hdg:tr.hdg[i],spd:tr.spd[i],i:i};
}

let cur=0, playing=false, last=0;
const slider=document.getElementById('slider');
function draw(u){
  cur=u;
  slider.value=Math.round(u);
  document.getElementById('clock').textContent=new Date((T0+u)*1000)
    .toISOString().substr(11,8)+'Z';
  const seen=[];
  tracks.forEach(tr=>{
    const s=sample(tr,u);
    if(!s){ tr.marker.setOpacity(0); tr.trail.setLatLngs([]);
             if(tr.pred) tr.pred.setLatLngs([]);
             // keep the same 6 cells as the airborne case, so the row can be
             // filled in later without rebuilding it
             for(let q=1;q<tr.tbody.children.length;q++)
               tr.tbody.children[q].textContent='-';
             tr.tbody.children[0].innerHTML='<span class="swatch" style="background:'
               +tr.color+'"></span>'+tr.id+' <span class="muted">(ground)</span>';
             return; }
    tr.marker.setOpacity(1);
    tr.marker.setLatLng([s.lat,s.lon]);
    tr.marker.setIcon(icon(tr.color,s.hdg));
    seen.push([s.lat,s.lon]);
    tr.trail.setLatLngs(tr.lat.slice(0,s.i+1).map((la,i)=>[la,tr.lon[i]]));
    if(tr.pdat){
      // The forecast is anchored to the start of the observation window and is
      // revealed step by step, so the dashed line grows with the clock instead
      // of appearing in full from the beginning.  `pdt0` is the index of the
      // forecast point that sits at the current replay time.
      const P=tr.pdat, rel=u-P.pstart;
      if(rel<0){ tr.pred.setLatLngs([]); }
      else{
        const n=Math.min(P.lat.length, Math.floor(rel/STEP)+1);
        const seg=[];
        for(let q=0;q<n;q++) seg.push([P.lat[q],P.lon[q]]);
        if(n>0 && n<P.lat.length){                 // partial step: interpolate
          const a=(rel/STEP)-(n-1);
          seg.push([P.lat[n-1]+(P.lat[n]-P.lat[n-1])*a,
                    P.lon[n-1]+(P.lon[n]-P.lon[n-1])*a]);
        }
        tr.pred.setLatLngs(seg);
      }
    }
    const c=tr.tbody.children;
    c[0].innerHTML='<span class="swatch" style="background:'+tr.color
      +'"></span>'+tr.id;
    c[1].textContent=s.lat.toFixed(4); c[2].textContent=s.lon.toFixed(4);
    c[3].textContent=Math.round(s.alt); c[4].textContent=Math.round(s.hdg)+'\\u00b0';
    c[5].textContent=Math.round(s.spd);
  });
  if(seen.length) map.__fitted=map.__fitted||L.latLngBounds(seen).pad(0.25);
}
function loop(ts){
  if(playing){
    if(last) draw(cur + (ts-last)/1000*parseFloat(document.getElementById('speed').value));
    last=ts;
  } else last=0;
  requestAnimationFrame(loop);
}
document.getElementById('play').onclick=function(){
  playing=!playing; last=0;
  this.innerHTML=playing?'&#10073;&#10073; Pause':'&#9654; Play';
};
slider.oninput=function(){ playing=false;
  document.getElementById('play').innerHTML='&#9654; Play'; draw(+this.value); };
document.getElementById('hint').textContent =
  '__N__ aircraft | __NPTS__ interpolated fixes | drag the slider or press Play';
draw(0);
map.fitBounds(L.latLngBounds(tracks.flatMap(t=>t.lat.map((la,i)=>[la,t.lon[i]]))).pad(.15));
requestAnimationFrame(loop);
</script></body></html>
"""


def _track_arrays(leg: np.ndarray, t0: float, cfg: Config, enc: BinaryEncoding,
                  step: Optional[float] = None) -> Dict[str, np.ndarray]:
    """One resampled leg -> a track resampled at `step` seconds (default 1 s).

    The BE fix grid is `period` seconds wide (30 s by default), so anything
    finer is a *linear interpolation* between two fixes, not a measurement.
    """
    real = encoded_to_real(leg)                     # (T, 6) deg/deg/m/kmh...
    n = len(real)
    src_t = t0 + np.arange(n, dtype=np.float64) * cfg.period
    step = max(float(step or cfg.map_step), 1e-3)
    grid = np.arange(src_t[0], src_t[-1] + 1e-9, step)
    cols = {a: np.interp(grid, src_t, real[:, j]) for j, a in enumerate(ATTRS)}
    lat, lon = cols["lat"], cols["lon"]
    # bearing from the interpolated path, for the icon rotation
    dy = np.gradient(lat) * DEG_M
    dx = np.gradient(lon) * DEG_M * np.cos(np.radians(lat))
    hdg = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    spd = np.hypot(cols["vx"], cols["vy"])
    return {"lat": lat, "lon": lon, "alt": cols["alt"], "hdg": hdg, "spd": spd,
            "t": grid}


def _round_track(tr: Dict[str, np.ndarray], nd: int) -> Dict[str, list]:
    # 't' is recomputed in the browser from t0 + i*step, so it is not emitted
    return {k: [round(float(v), nd) for v in arr]
            for k, arr in tr.items() if k != "t"}


@torch.no_grad()
def write_map(path: str, bundle: DataBundle, enc: BinaryEncoding, cfg: Config,
              device: torch.device, model: Optional[FlightBERT] = None) -> str:
    """Write a self-contained Leaflet page replaying the trajectories second by
    second, with the model's forecast overlaid when a model is given.

    The page needs internet access for the OpenStreetMap tiles and the Leaflet
    library; the trajectory data itself is embedded in the file.
    """
    if not bundle.legs or not bundle.t_start:
        raise RuntimeError("no trajectories available for the map")
    order = list(range(len(bundle.legs)))
    if cfg.map_split:
        keep = [i for i in order if bundle.leg_split[i] == cfg.map_split]
        order = keep or order
    # longest legs first, so a --map_flights cap keeps the most informative ones
    order.sort(key=lambda i: -len(bundle.legs[i]))
    order = order[:max(1, cfg.map_flights)]

    # 1 s fixes over a whole day is ~10 MB of JSON: if the selection is too
    # large, coarsen the grid rather than emit an unusable page
    total_dur = sum((len(bundle.legs[i]) - 1) * cfg.period for i in order)
    step = max(float(cfg.map_step), 1e-3)
    if total_dur / step > cfg.map_max_points:
        step = total_dur / cfg.map_max_points
        LOG.warning("map: %d aircraft over %.0f s of track at %gs/fix would be "
                    "%.0f MB of JSON - coarsening to %.1f s/fix (--map_step %.3g "
                    "was asked). Raise --map_max_points to keep 1 s.",
                    len(order), total_dur, cfg.map_step, total_dur / cfg.map_step * 4e-5,
                    step, cfg.map_step)

    # the replay origin is the earliest mapped fix, so no track starts "before"
    # the slider and every t0 stays >= 0
    t_ref = min(float(bundle.t_start[i]) for i in order)
    tmax = 0.0
    tracks, total_pts = [], 0
    preds: Dict[int, Dict[str, list]] = {}
    if model is not None:
        preds = _map_predictions(model, bundle, enc, cfg, device, order, step, t_ref)

    for i in order:
        tr = _track_arrays(bundle.legs[i], bundle.t_start[i], cfg, enc, step)
        rid = bundle.leg_flight[i] if i < len(bundle.leg_flight) else str(i)
        entry = {"id": rid, "split": bundle.leg_split[i] or "train",
                 "t0": round(float(bundle.t_start[i]) - t_ref, 3)}
        entry.update(_round_track(tr, 5))
        if i in preds:
            entry["pred"] = preds[i]
        tracks.append(entry)
        total_pts += len(tr["lat"])
        tmax = max(tmax, float(tr["t"][-1] - t_ref))

    data = {"t0": t_ref, "step": step,
            "lat0": float(np.mean([t["lat"][0] for t in tracks])),
            "lon0": float(np.mean([t["lon"][0] for t in tracks])),
            "tracks": tracks}
    note = ("<br>&#9632; solid = observed, &#9632; dashed = model forecast"
            if preds else "")
    html = (MAP_TEMPLATE
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__N__", str(len(tracks)))
            .replace("__NPTS__", f"{total_pts:,}".replace(",", " "))
            .replace("__STEP__", f"{step:g}")
            .replace("__PERIOD__", f"{cfg.period}")
            .replace("__TMAX__", str(int(tmax)))
            .replace("__PREDNOTE__", note))
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    LOG.info("map written to %s (%d aircraft, %d fixes, %.1f MB)", path,
             len(tracks), total_pts, os.path.getsize(path) / 1e6)
    return path


@torch.no_grad()
def _map_predictions(model: FlightBERT, bundle: DataBundle, enc: BinaryEncoding,
                     cfg: Config, device: torch.device, order: Sequence[int],
                     step: float, t_ref: float) -> Dict[int, Dict[str, list]]:
    """Forecast every mapped trajectory from its last k observations."""
    k, n = cfg.inp_seq_len, cfg.max_horizon
    out: Dict[int, Dict[str, list]] = {}
    far: List[float] = []
    dropped = [0]
    batch = []
    for i in order:
        be = enc.encode_matrix(bundle.legs[i][-k:])
        batch.append(torch.from_numpy(be.astype(np.float32)))
    for s in range(0, len(batch), 256):
        chunk = batch[s:s + 256]
        x = torch.stack(chunk).to(device).float()
        with torch.autocast(device_type=device.type, dtype=amp_dtype(cfg),
                            enabled=bool(cfg.amp and device.type == "cuda")):
            pred = predict_horizons(model, x, n, cfg.feed_soft)
        bits = (pred > 0.5).float().cpu().numpy()
        for j, i in enumerate(order[s:s + 256]):
            real = enc.decode_real(bits[j])            # (n, 6) deg/deg/m/...
            n_pts = len(bundle.legs[i])
            t_last = bundle.t_start[i] + (n_pts - 1) * cfg.period
            # Anchor the forecast to the *start* of the observation window, not
            # to its end: the window is the last k fixes, so this is the moment
            # the model was actually asked to forecast. Anchoring at t_last put
            # the whole forecast past the end of the replay, where the aircraft
            # is already flagged "not airborne" and nothing is drawn.
            t_anchor = t_last - (k - 1) * cfg.period
            ft = t_last + np.arange(1, n + 1) * cfg.period
            grid = np.arange(t_anchor, ft[-1] + 1e-9, step)
            # a forecast that lands implausibly far away is a model failure, not
            # a trajectory: drop it instead of drawing a line across the sea
            last = encoded_to_real(bundle.legs[i][-1:])
            lat0, lon0 = last[0, 1], last[0, 0]
            dy = (real[-1, 1] - lat0) * DEG_M
            dx = (real[-1, 0] - lon0) * DEG_M * math.cos(math.radians(lat0))
            far.append(float(np.hypot(dx, dy)))
            if np.hypot(dx, dy) > cfg.map_pred_max_km:
                dropped[0] += 1
                continue
            out[i] = {"pstart": round(t_anchor - t_ref, 3),
                      "lat": [round(float(v), 5)
                              for v in np.interp(grid, ft, real[:, 1], left=lat0,
                                                 right=real[-1, 1])],
                      "lon": [round(float(v), 5)
                              for v in np.interp(grid, ft, real[:, 0], left=lon0,
                                                 right=real[-1, 0])]}
    if far:
        LOG.info("map: forecast displacement at h=%d - median %.0f km, max %.0f km; "
                 "%d/%d rejected as implausible (>%.0f km)",
                 n, float(np.median(far)), max(far), dropped[0], len(far),
                 cfg.map_pred_max_km)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    # parse_known_args keeps the script usable from a notebook/IPython cell,
    # where sys.argv also carries the kernel arguments.
    args, extra = build_argparser().parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    if extra:
        LOG.warning("ignoring unrecognised arguments: %s", " ".join(extra))

    cfg = config_from_args(args)
    if cfg.direct and not cfg.causal:
        raise SystemExit("--direct needs the causal mask: every horizon is read "
                         "from a position that must not see the future "
                         "(drop --no_causal)")
    # The BE vector width (and, in test mode, the whole setup) must be identical
    # to the one used for training, so it is taken from --bits, from the
    # checkpoint, or derived from the data - in that order of priority.
    if args.test_only and cfg.resume_path:
        stored = load_checkpoint_config(cfg.resume_path)
        if stored:
            changed = []
            for key, val in stored.items():
                if key in ("csv_path", "out_dir", "resume_path", "device", "test_only",
                           "amp", "num_workers", "log_every", "save_every", "seed",
                           "eval_batches", "plot", "limit_flights", "map",
                           "map_step", "map_flights", "map_split") or key not in asdict(cfg):
                    continue
                if getattr(cfg, key, None) != val:
                    changed.append(f"{key}: {getattr(cfg, key, None)} -> {val}")
                setattr(cfg, key, tuple(val) if isinstance(getattr(cfg, key), tuple)
                        else val)
            LOG.info("test mode: configuration restored from %s (%d field(s) changed%s)",
                     cfg.resume_path, len(changed),
                     ": " + ", ".join(changed) if changed else ")")
            cfg.eval_horizons = tuple(cfg.eval_horizons)
    forced = parse_bits(args.bits)
    stored_cfg: Dict = load_checkpoint_config(cfg.resume_path) if cfg.resume_path else {}
    if forced:
        cfg.bits = forced
        cfg.auto_bits = False
        LOG.info("BE bit widths forced by --bits: %s", cfg.bits)
    else:
        from_ckpt = bits_from_checkpoint(cfg.resume_path)
        if from_ckpt:
            cfg.bits = from_ckpt
            cfg.auto_bits = False
            LOG.info("BE bit widths taken from %s: %s", cfg.resume_path, cfg.bits)
    if stored_cfg:
        warn_encoding_drift(stored_cfg, cfg)
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    # the positional resolution must be set before anything reads PRECISION
    set_position_precision(cfg.pos_prec)
    if device.type == "cuda":
        # TF32 for the fp32 ops (the loss is deliberately fp32); the fp16
        # autocast path already runs on tensor cores at full rate
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    os.makedirs(cfg.out_dir, exist_ok=True)
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        LOG.info("device: %s (%s, sm_%d%d) | torch: %s | CUDA runtime: %s",
                 device, torch.cuda.get_device_name(device), cap[0], cap[1],
                 torch.__version__, torch.version.cuda)
    else:
        LOG.info("device: %s | torch: %s | CUDA runtime: %s", device,
                 torch.__version__, torch.version.cuda or "none (CPU-only build)")

    bundle, enc = build_dataset(cfg)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        f.write(cfg.to_json())            # written after the widths are settled
    loaders = build_dataloaders(bundle, cfg, enc)

    model = FlightBERT(cfg, enc).to(device)
    LOG.info("FlightBERT: %.2fM parameters | d_model=%d | %d blocks x %d heads | "
             "BE dim=%d | %s", sum(p.numel() for p in model.parameters()) / 1e6,
             model.d_model, cfg.n_layers, cfg.n_heads, enc.total_bits,
             "DIRECT multi-step (no feedback loop)" if cfg.direct
             else "iterative rollout (paper Sec. III-A)")
    # the loss owns buffers (bit weights, per-attribute scales): they must follow
    # the model onto the GPU, otherwise _decode multiplies CUDA probs by CPU weights
    criterion = HybridConstrainedLoss(enc, cfg.bce_weight, 1.0, cfg.mse_mode).to(device)
    if cfg.mse_weight is None:
        cfg.mse_weight = criterion.calibrate(model, device, enc, cfg.inp_seq_len)
        LOG.info("mse_mode=%s: mse_weight calibrated to %.3g so that the decoded-"
                 "decimal term starts at the same size as the BCE", cfg.mse_mode,
                 cfg.mse_weight)
    else:
        criterion.mse_weight = cfg.mse_weight
        LOG.info("mse_mode=%s: mse_weight %.3g (from the command line)",
                 cfg.mse_mode, cfg.mse_weight)
    if not args.no_selftest:
        selftest(enc, cfg, model, criterion, device)

    # fused Adam keeps the optimizer step in one multi-tensor kernel: fewer
    # launches and no per-parameter temporaries (falls back silently on CPU)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay,
                                 fused=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience)
    # bf16 has the same dynamic range as fp32, so it needs no loss scaling
    scaler = make_scaler(bool(cfg.amp and device.type == "cuda"
                              and amp_dtype(cfg) == torch.float16))

    best = float("inf")
    start_epoch = 0
    ckpt_best = os.path.join(cfg.out_dir, "best.pt")
    if cfg.resume_path and os.path.isfile(cfg.resume_path):
        meta = load_checkpoint(cfg.resume_path, model, optimizer, scheduler, scaler, device)
        start_epoch = int(meta.get("epoch", -1)) + 1
        best = float(meta.get("best_val", float("inf")))
        LOG.info("resumed from %s (epoch %d, best val %.5f)", cfg.resume_path,
                 start_epoch - 1, best)

    if args.test_only:
        if not (cfg.resume_path and os.path.isfile(cfg.resume_path)):
            raise SystemExit("--test_only needs --resume <checkpoint>")
        evaluate(model, loaders["test"], enc, cfg, device, args.eval_batches)
        if args.map:
            write_map(args.map, bundle, enc, cfg, device,
                      None if args.no_map_predict else model)
        return 0

    LOG.info("training: %d epochs | batch=%d | lr=%.2e | %d windows | "
             "val every %d epoch(s) on %d windows",
             cfg.epochs, cfg.batch_size, cfg.lr, len(loaders["train"].dataset),
             cfg.eval_every, len(loaders["val"].dataset))
    # the checkpoint does not store the per-epoch records, so a resume would
    # otherwise rewrite history.json with the new epochs only
    history: List[Dict[str, float]] = []
    hist_path = os.path.join(cfg.out_dir, "history.json")
    if start_epoch and os.path.isfile(hist_path):
        try:
            with open(hist_path) as f:
                prev = json.load(f)
            if isinstance(prev, list) and all(isinstance(e, dict) and "epoch" in e
                                             for e in prev):
                history = [e for e in prev if int(e["epoch"]) < start_epoch]
                LOG.info("history: keeping %d pre-resume epoch(s) from %s",
                         len(history), hist_path)
        except Exception as exc:                                    # pragma: no cover
            LOG.warning("could not read %s (%s)", hist_path, exc)
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer,
                                     scaler, cfg, device, epoch)
        last_epoch = epoch + 1 >= cfg.epochs
        val_metrics: Dict[str, float] = {}
        val_loss = train_loss
        if len(loaders["val"].dataset) and (
                cfg.eval_every <= 1 or last_epoch
                or (epoch + 1) % cfg.eval_every == 0):
            val_metrics = evaluate(model, loaders["val"], enc, cfg, device,
                                   args.eval_batches)
            val_loss = float(val_metrics.get("score@1", train_loss))
        else:
            LOG.info("epoch %d: validation skipped (--eval_every %d)", epoch + 1,
                     cfg.eval_every)
        scheduler.step(val_loss)
        LOG.info("epoch %2d/%d | train %.5f | val %.5f | lr %.2e | %.1fs",
                 epoch + 1, cfg.epochs, train_loss, val_loss,
                 optimizer.param_groups[0]["lr"], time.time() - t0)
        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                 **{k: v for k, v in val_metrics.items() if k != "report"}}
        history.append(entry)
        if val_loss < best:
            best = val_loss
            save_checkpoint(ckpt_best, model, optimizer, scheduler, scaler, epoch,
                            cfg, enc, best, val_metrics)
            LOG.info("  new best (%.5f) -> %s", best, ckpt_best)
        if cfg.save_every and (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(os.path.join(cfg.out_dir, "last.pt"), model, optimizer,
                            scheduler, scaler, epoch, cfg, enc, best, val_metrics)
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2, default=float)

    LOG.info("training finished - reloading the best checkpoint for the final report")
    if os.path.isfile(ckpt_best):
        load_checkpoint(ckpt_best, model, None, None, None, device)
    test_metrics: Dict[str, float] = {}
    if len(loaders["test"].dataset):
        LOG.info("========== TEST SET (%s) ==========", enc.describe())
        test_metrics = evaluate(model, loaders["test"], enc, cfg, device, args.eval_batches)
    if args.plot:
        plot_prediction(model, loaders["test"] if len(loaders["test"].dataset)
                        else loaders["train"], enc, cfg, device, args.plot)
    if args.map:
        write_map(args.map, bundle, enc, cfg, device,
                  None if args.no_map_predict else model)
    with open(os.path.join(cfg.out_dir, "test_metrics.json"), "w") as f:
        json.dump({k: v for k, v in test_metrics.items() if k != "report"}, f,
                  indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
