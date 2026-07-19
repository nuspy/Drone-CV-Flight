"""GIS world store: memory-mapped raster mosaics + metadata.

Layout of `artifacts/gis/<env>/`:

    meta.json       resolution, ENU origin/extent, anchor, stats, orbit points
    ground.npy      float32 (rows=N south->north, cols=E west->east) terrain m MSL-rel
    build_h.npy     float32 additive building height (0 where none)
    class_id.npy    uint8 ground/feature class (see landcover.GROUND_CLASSES + buildings)
    albedo.npy      optional uint8 HxWx3 draped orthophoto

Memmaps keep RAM bounded for large AOIs (a 40x40 km area at 1 m/px is ~13 GB
on disk but only touched pages in memory); the build writes them window by
window. Heights are stored relative to the anchor's ground_alt0 so the sim
frame starts near y=0 like every other dronecv world.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

BUILDING_CLASS_IDS = {
    "generic": 10,
    "residential": 11,
    "industrial": 12,
    "commercial": 13,
    "landmark": 14,
}


@dataclass
class GisMeta:
    res_m: float
    e0: float  # ENU easting of column 0 (west edge)
    n0: float  # ENU northing of row 0 (south edge)
    width: int
    height: int
    anchor: dict[str, Any]
    ground_alt0: float  # MSL elevation subtracted from the ground raster
    max_height: float
    orbit_points: list[list[float]] = field(default_factory=list)  # [[e, n], ...]
    stats: dict[str, Any] = field(default_factory=dict)
    attribution: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        from dataclasses import asdict

        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> GisMeta:
        return cls(**json.loads(text))


class GisStore:
    def __init__(self, root: Path, meta: GisMeta, mode: str = "r"):
        self.root = Path(root)
        self.meta = meta
        shape = (meta.height, meta.width)
        self.ground = np.lib.format.open_memmap(
            self.root / "ground.npy", mode=mode, dtype=np.float32, shape=shape
        )
        self.build_h = np.lib.format.open_memmap(
            self.root / "build_h.npy", mode=mode, dtype=np.float32, shape=shape
        )
        self.class_id = np.lib.format.open_memmap(
            self.root / "class_id.npy", mode=mode, dtype=np.uint8, shape=shape
        )
        albedo_path = self.root / "albedo.npy"
        self.albedo = (
            np.lib.format.open_memmap(albedo_path, mode="r")
            if mode == "r" and albedo_path.exists()
            else None
        )

    @classmethod
    def create(cls, root: Path, meta: GisMeta) -> GisStore:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        store = cls(root, meta, mode="w+")
        return store

    @classmethod
    def open(cls, root: Path) -> GisStore:
        root = Path(root)
        meta = GisMeta.from_json((root / "meta.json").read_text())
        return cls(root, meta, mode="r")

    def save_meta(self) -> None:
        (self.root / "meta.json").write_text(self.meta.to_json())

    def flush(self) -> None:
        for arr in (self.ground, self.build_h, self.class_id):
            if hasattr(arr, "flush"):
                arr.flush()
        self.save_meta()

    # ------------------------------------------------------------- coordinates

    def enu_to_rc(self, e: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ENU meters -> fractional (row, col)."""
        c = (np.asarray(e) - self.meta.e0) / self.meta.res_m
        r = (np.asarray(n) - self.meta.n0) / self.meta.res_m
        return r, c
