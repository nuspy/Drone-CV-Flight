"""On-disk capture dataset and torch views with spatial holdout split.

Layout of a dataset directory:
    meta.jsonl              one JSON record per capture (pose, sun, agl, ...)
    shard_0000.npz          images uint8 [N, H, W, 3] (key "rgb")
    info.json               env name, anchor, camera, shard index

The train/val split is SPATIAL: the world is voxelized into val_cell_m cells
and whole cells are held out. Random splits would leak near-duplicate
neighboring views into validation and wildly overstate APR accuracy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

SHARD_SIZE = 512


@dataclass
class CaptureRecord:
    index: int
    shard: int
    pos_in_shard: int
    pos_sim: list[float]
    pos_enu: list[float]
    yaw_sim_deg: float
    heading_deg: float  # true-north heading of the camera forward axis
    pitch_deg: float
    agl_m: float | None
    ground_y_sim: float | None
    utc: str
    sun_azimuth_deg: float | None
    sun_elevation_deg: float | None


class DatasetWriter:
    def __init__(self, out_dir: Path, info: dict[str, Any]):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.info = info
        self._records: list[CaptureRecord] = []
        self._pending: list[np.ndarray] = []
        self._shard = 0
        existing = sorted(self.out_dir.glob("shard_*.npz"))
        if existing:  # append mode (active loop rounds)
            self._shard = int(existing[-1].stem.split("_")[1]) + 1
            with (self.out_dir / "meta.jsonl").open() as fh:
                self._count = sum(1 for _ in fh)
        else:
            self._count = 0

    def add(self, rgb: np.ndarray, meta: dict[str, Any]) -> None:
        record = CaptureRecord(
            index=self._count,
            shard=self._shard,
            pos_in_shard=len(self._pending),
            **meta,
        )
        self._pending.append(rgb)
        self._records.append(record)
        self._count += 1
        if len(self._pending) >= SHARD_SIZE:
            self._flush_shard()

    def _flush_shard(self) -> None:
        if not self._pending:
            return
        np.savez_compressed(
            self.out_dir / f"shard_{self._shard:04d}.npz", rgb=np.stack(self._pending)
        )
        with (self.out_dir / "meta.jsonl").open("a") as fh:
            for rec in self._records:
                fh.write(json.dumps(asdict(rec)) + "\n")
        self._pending = []
        self._records = []
        self._shard += 1

    def close(self) -> None:
        self._flush_shard()
        (self.out_dir / "info.json").write_text(json.dumps(self.info, indent=2))


class CaptureDataset:
    """Read-side: loads meta, memory-maps shards lazily."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.records = [
            json.loads(line) for line in (self.root / "meta.jsonl").read_text().splitlines()
        ]
        self.info = json.loads((self.root / "info.json").read_text())
        self._shard_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def image(self, index: int) -> np.ndarray:
        rec = self.records[index]
        shard = rec["shard"]
        if shard not in self._shard_cache:
            if len(self._shard_cache) > 4:
                self._shard_cache.pop(next(iter(self._shard_cache)))
            self._shard_cache[shard] = np.load(self.root / f"shard_{shard:04d}.npz")["rgb"]
        return self._shard_cache[shard][rec["pos_in_shard"]]

    # ---------------------------------------------------------------- split

    def cell_of(self, index: int, cell_m: float) -> tuple[int, int]:
        e, n, _ = self.records[index]["pos_enu"]
        return (int(np.floor(e / cell_m)), int(np.floor(n / cell_m)))

    def spatial_split(self, cell_m: float, val_fraction: float, seed: int = 0) -> tuple[list[int], list[int]]:
        """Deterministic whole-cell holdout: a cell is val iff a stable hash
        of (cell, seed) falls below val_fraction."""
        train_idx, val_idx = [], []
        for i in range(len(self)):
            cell = self.cell_of(i, cell_m)
            key = f"{cell[0]}:{cell[1]}:{seed}".encode()
            bucket = int.from_bytes(hashlib.sha1(key).digest()[:4], "big") / 2**32
            (val_idx if bucket < val_fraction else train_idx).append(i)
        return train_idx, val_idx

    def positions_enu(self) -> np.ndarray:
        return np.array([r["pos_enu"] for r in self.records])

    def headings_deg(self) -> np.ndarray:
        return np.array([r["heading_deg"] for r in self.records])


class TorchCaptureView(Dataset):
    """Torch view over a subset of a CaptureDataset.

    Yields (image tensor CHW float [0,1], pos_enu[3], heading sin/cos[2], index).
    With augment=True applies photometric jitter (brightness/contrast/channel
    gain + noise) — pose-preserving, so labels are untouched; it markedly
    reduces APR overfitting on small capture sets.
    """

    def __init__(self, ds: CaptureDataset, indices: list[int], augment: bool = False, seed: int = 0):
        self.ds = ds
        self.indices = list(indices)
        self.augment = augment
        self._gen = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        rec = self.ds.records[idx]
        arr = self.ds.image(idx).astype(np.float32) / 255.0
        if self.augment:
            g = self._gen
            arr = arr * g.uniform(0.8, 1.2) + g.uniform(-0.08, 0.08)
            arr = (arr - 0.5) * g.uniform(0.85, 1.15) + 0.5
            arr = arr * g.uniform(0.92, 1.08, size=(1, 1, 3))
            arr = arr + g.normal(0.0, 0.02, size=arr.shape)
            arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
        img = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        pos = torch.tensor(rec["pos_enu"], dtype=torch.float32)
        h = np.radians(rec["heading_deg"])
        heading = torch.tensor([np.sin(h), np.cos(h)], dtype=torch.float32)
        return img, pos, heading, idx
