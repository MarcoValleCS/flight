"""Dataset PyTorch per le finestre pre-calcolate (memory-mapped)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import read_json

ARRAY_KEYS = [
    "x", "y_xy", "y_z", "phase", "phase0", "lat0", "lon0", "z0", "gs0",
    "trk0", "vr0", "ts0", "ac_id", "typ", "op", "dep", "arr", "file_id",
]


class TrajWindows(Dataset):
    """Finestre di traiettoria; gli array .npy sono aperti in mmap (lazy)."""

    def __init__(self, split_dir: str | Path):
        self.dir = Path(split_dir)
        self.meta = read_json(self.dir / "meta.json")
        self.n = int(self.meta["n_windows"])
        sc = self.meta["scales"]
        self.sc_xy = float(sc["xy"])
        self.sc_z = float(sc["z"])
        self.sc_z0 = float(sc["z0"])
        self.sc_gs = float(sc["gs"])
        self.sc_vr = float(sc["vr"])
        self._a: dict | None = None

    # mmap aperto pigramente: per-processo (evita problemi con i worker)
    def _open(self) -> None:
        self._a = {
            k: np.load(self.dir / f"{k}.npy", mmap_mode="r")
            for k in ARRAY_KEYS if (self.dir / f"{k}.npy").exists()
        }

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        if self._a is None:
            self._open()
        a = self._a
        x = torch.from_numpy(np.ascontiguousarray(a["x"][i], dtype=np.float32))
        z0 = float(a["z0"][i])
        gs0 = float(a["gs0"][i])
        static = torch.tensor([z0 / self.sc_z0, gs0 / self.sc_gs], dtype=torch.float32)
        cats = torch.tensor(
            [int(a["typ"][i]), int(a["op"][i]), int(a["dep"][i]), int(a["arr"][i])],
            dtype=torch.long)
        return {
            "x": x,
            "static": static,
            "cats": cats,
            "y_xy": torch.from_numpy(np.array(a["y_xy"][i], dtype=np.float32)),
            "y_z": torch.from_numpy(np.array(a["y_z"][i], dtype=np.float32)),
            "phase": torch.tensor(int(a["phase"][i]), dtype=torch.long),
            "phase0": torch.tensor(int(a["phase0"][i]), dtype=torch.long),
            "state": torch.tensor(
                [float(a["lat0"][i]), float(a["lon0"][i]), z0, gs0,
                 float(a["trk0"][i]), float(a["vr0"][i])], dtype=torch.float32),
            "ts0": torch.tensor(int(a["ts0"][i]), dtype=torch.long),
            "ac_id": torch.tensor(int(a["ac_id"][i]), dtype=torch.long),
            "file_id": torch.tensor(int(a["file_id"][i]), dtype=torch.long),
        }
