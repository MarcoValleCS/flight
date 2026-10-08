"""Baseline cinematiche classiche per il confronto (in coordinate ENU, metri).

Tutte le funzioni ricevono/ritornano array numpy (batch, orizzonti, 2) per la
posizione ENU relativa all'ancora e (batch, orizzonti) per la quota relativa.

Metodi:
    zero : l'aereo resta fermo
    cv   : velocita' costante (velocita' misurata all'ancora)
    ca   : accelerazione costante (dalle ultime posizioni della storia)
    ct   : velocita' angolare costante (integrazione numerica con il rateo
           di virata stimato dagli ultimi 25 s di rotta)
"""
from __future__ import annotations

import numpy as np

SC_XY = 10000.0   # fattori usati in build_dataset (le feature sono normalizzate)
SC_Z = 1000.0
SC_GS = 250.0
SC_VR = 25.0


def decode_history(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """x: (B, T, 8) -> posizioni ENU (m), gs (m/s), vr (m/s), rotta (rad)."""
    pos = x[..., :2].astype(np.float64) * SC_XY
    gs = x[..., 3].astype(np.float64) * SC_GS
    vr = x[..., 4].astype(np.float64) * SC_VR
    trk = np.arctan2(x[..., 5].astype(np.float64), x[..., 6].astype(np.float64))
    return pos, gs, vr, trk


def baseline_zero(state: np.ndarray, horizons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    B = len(state)
    H = len(horizons)
    return np.zeros((B, H, 2)), np.zeros((B, H))


def baseline_cv(state: np.ndarray, horizons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """state: (B,6) = [lat0, lon0, z0_m, gs0_ms, trk0_deg, vr0_ms]."""
    gs0 = state[:, 3].astype(np.float64)
    trk0 = np.radians(state[:, 4].astype(np.float64))
    vr0 = state[:, 5].astype(np.float64)
    tau = horizons.astype(np.float64)[None, :]
    vx = gs0 * np.sin(trk0)
    vy = gs0 * np.cos(trk0)
    xy = np.stack([vx[:, None] * tau, vy[:, None] * tau], axis=-1)
    z = vr0[:, None] * tau
    return xy, z


def baseline_ca(x: np.ndarray, state: np.ndarray, step_s: float,
                horizons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos, _, _, _ = decode_history(x)
    v_last = (pos[:, -1] - pos[:, -2]) / step_s
    v_prev = (pos[:, -2] - pos[:, -3]) / step_s
    acc = (v_last - v_prev) / step_s
    tau = horizons.astype(np.float64)[None, :]
    xy = v_last[:, None, :] * tau[..., None] + 0.5 * acc[:, None, :] * (tau ** 2)[..., None]
    z = state[:, 5].astype(np.float64)[:, None] * tau
    return xy, z


def baseline_ct(x: np.ndarray, state: np.ndarray, step_s: float,
                horizons: np.ndarray, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    _, _, _, trk = decode_history(x)
    trk = np.unwrap(trk, axis=1)
    k = min(6, trk.shape[1])               # ultimi ~25 s
    omega = (trk[:, -1] - trk[:, -k]) / ((k - 1) * step_s)   # rad/s
    gs0 = state[:, 3].astype(np.float64)
    trk0 = trk[:, -1]

    tmax = float(np.max(horizons))
    tg = np.arange(0.0, tmax + 1e-9, dt)
    phi = trk0[:, None] + omega[:, None] * tg[None, :]
    dx = gs0[:, None] * np.sin(phi) * dt
    dy = gs0[:, None] * np.cos(phi) * dt
    ex = np.cumsum(dx, axis=1)
    ey = np.cumsum(dy, axis=1)
    idx = np.round(horizons / dt).astype(int)
    xy = np.stack([ex[:, idx], ey[:, idx]], axis=-1)
    z = state[:, 5].astype(np.float64)[:, None] * horizons.astype(np.float64)[None, :]
    return xy, z
