"""Terrain elevation prior, built from capture ground probes.

Every capture knows the ground elevation under it (probed via depth during
collection), so the dataset doubles as a sparse terrain map: median ground
elevation (ENU up, meters) per horizontal cell. The localizer combines it
with the lidar AGL range to get an ABSOLUTE altitude measurement:
    altitude = terrain_elevation(estimated E, N) + lidar_range
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TerrainPrior:
    def __init__(self, cell_m: float, cells: dict[tuple[int, int], float]):
        self.cell_m = cell_m
        self.cells = cells
        self.global_median = float(np.median(list(cells.values()))) if cells else 0.0

    @classmethod
    def from_records(cls, pos_enu: np.ndarray, ground_up: np.ndarray, cell_m: float) -> "TerrainPrior":
        buckets: dict[tuple[int, int], list[float]] = {}
        for (e, n, _u), g in zip(pos_enu, ground_up, strict=True):
            key = (int(np.floor(e / cell_m)), int(np.floor(n / cell_m)))
            buckets.setdefault(key, []).append(float(g))
        return cls(cell_m, {k: float(np.median(v)) for k, v in buckets.items()})

    def elevation(self, e: float, n: float) -> float:
        """Ground elevation (ENU up) at a horizontal position; falls back to
        ring search then the global median."""
        ce, cn = int(np.floor(e / self.cell_m)), int(np.floor(n / self.cell_m))
        if (ce, cn) in self.cells:
            return self.cells[(ce, cn)]
        for ring in (1, 2):
            vals = [
                self.cells[(ce + dx, cn + dy)]
                for dx in range(-ring, ring + 1)
                for dy in range(-ring, ring + 1)
                if (ce + dx, cn + dy) in self.cells
            ]
            if vals:
                return float(np.median(vals))
        return self.global_median

    def save(self, path: Path) -> None:
        if self.cells:
            keys = np.array(list(self.cells.keys()), dtype=np.int64)
            vals = np.array(list(self.cells.values()), dtype=np.float32)
        else:
            keys = np.zeros((0, 2), dtype=np.int64)
            vals = np.zeros((0,), dtype=np.float32)
        np.savez_compressed(path, cell_m=self.cell_m, keys=keys, vals=vals)

    @classmethod
    def load(cls, path: Path) -> "TerrainPrior":
        data = np.load(path)
        cells = {
            (int(k[0]), int(k[1])): float(v)
            for k, v in zip(data["keys"], data["vals"], strict=True)
        }
        return cls(float(data["cell_m"]), cells)
