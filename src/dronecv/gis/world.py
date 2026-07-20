"""GisWorld: a real-world area as a dronecv simulator world.

Implements the same interface as the procedural `sim.headless.world.World`
(`height_at / normal_at / albedo_at / bounds / spawn_position / landmarks`),
so the existing raycaster, protocol server, capture pipeline, active loop and
harness run UNCHANGED on real geography.

Shape-first by design: geometry (terrain + extruded buildings) carries the
localization signal; colors are class bands with deterministic per-texel
speckle (feature detectors need corners — flat colors starve ORB), optionally
replaced by a draped orthophoto where one was provided at build time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dronecv.gis.store import GisStore

# class id -> RGB float. Ground classes 0-5, buildings 10-14.
CLASS_COLORS = {
    0: (0.42, 0.40, 0.33),  # bare ground
    1: (0.30, 0.44, 0.24),  # green
    2: (0.16, 0.30, 0.50),  # water
    3: (0.35, 0.35, 0.37),  # road
    4: (0.30, 0.26, 0.24),  # rail
    5: (0.45, 0.44, 0.42),  # parking
    6: (0.20, 0.35, 0.16),  # broadleaf forest canopy
    7: (0.14, 0.28, 0.18),  # conifer forest canopy
    10: (0.62, 0.55, 0.48),  # generic building
    11: (0.66, 0.52, 0.42),  # residential
    12: (0.55, 0.56, 0.60),  # industrial
    13: (0.60, 0.58, 0.52),  # commercial
    14: (0.72, 0.65, 0.55),  # landmark
}


class GisWorld:
    def __init__(self, store: GisStore):
        self.store = store
        m = store.meta
        self.res = m.res_m
        extent_e = m.width * m.res_m
        extent_n = m.height * m.res_m
        # The pipeline centers the mosaic on the anchor: e0 = -extent/2.
        self.size = float(max(extent_e, extent_n))
        self.half = self.size / 2.0
        self._e0, self._n0 = m.e0, m.n0
        self._w, self._h = m.width, m.height
        self._color_lut = np.zeros((32, 3), dtype=np.float32)
        for cid, rgb in CLASS_COLORS.items():
            self._color_lut[cid] = rgb
        self.landmarks: list = []  # no analytic primitives: buildings live in the heightfield

    @classmethod
    def open(cls, root: Path) -> GisWorld:
        return cls(GisStore.open(root))

    # -------------------------------------------------------------- heights

    def _rc(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = (np.asarray(x, dtype=np.float64) - self._e0) / self.res
        r = (np.asarray(z, dtype=np.float64) - self._n0) / self.res
        return (
            np.clip(r, 0.0, self._h - 1.001),
            np.clip(c, 0.0, self._w - 1.001),
        )

    def height_at(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Composite height: bilinear terrain + NEAREST building extrusion
        (nearest keeps walls vertical instead of smearing them)."""
        r, c = self._rc(x, z)
        r0, c0 = r.astype(np.int64), c.astype(np.int64)
        fr, fc = r - r0, c - c0
        g = self.store.ground
        ground = (
            g[r0, c0] * (1 - fr) * (1 - fc)
            + g[r0, c0 + 1] * (1 - fr) * fc
            + g[r0 + 1, c0] * fr * (1 - fc)
            + g[r0 + 1, c0 + 1] * fr * fc
        )
        rn, cn = np.floor(r).astype(np.int64), np.floor(c).astype(np.int64)
        above = np.maximum(self.store.build_h[rn, cn], self.store.veg_h[rn, cn])
        return (ground + above).astype(np.float64)

    def normal_at(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        eps = max(self.res, 1.0)
        hx = self.height_at(np.asarray(x) + eps, z) - self.height_at(np.asarray(x) - eps, z)
        hz = self.height_at(x, np.asarray(z) + eps) - self.height_at(x, np.asarray(z) - eps)
        n = np.stack([-hx, np.full_like(hx, 2 * eps), -hz], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    # --------------------------------------------------------------- albedo

    def albedo_at(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        r, c = self._rc(x, z)
        rn, cn = np.floor(r).astype(np.int64), np.floor(c).astype(np.int64)
        if self.store.albedo is not None:
            base = self.store.albedo[rn, cn].astype(np.float32) / 255.0
        else:
            base = self._color_lut[self.store.class_id[rn, cn]]
        # Deterministic per-texel speckle: gives feature detectors texture
        # without storing anything (hash of the integer texel coordinates).
        hashes = (rn * np.int64(73856093)) ^ (cn * np.int64(19349663))
        noise = ((hashes & 0xFFFF).astype(np.float32) / 65535.0 - 0.5) * 0.12
        return np.clip(base + noise[..., None], 0.03, 0.97)

    # --------------------------------------------------------------- bounds

    @property
    def max_height(self) -> float:
        return float(self.store.meta.max_height)

    @property
    def min_height(self) -> float:
        return float(self.store.meta.stats.get("min_ground", 0.0))

    @property
    def bounds_min(self) -> np.ndarray:
        return np.array([self._e0, 0.0, self._n0])

    @property
    def bounds_max(self) -> np.ndarray:
        return np.array(
            [self._e0 + self._w * self.res, self.max_height + 160.0, self._n0 + self._h * self.res]
        )

    @property
    def orbit_points(self) -> list[tuple[float, float]]:
        return [(float(e), float(n)) for e, n in self.store.meta.orbit_points]

    def spawn_position(self, gen: np.random.Generator, agl_m: float = 50.0) -> np.ndarray:
        bmin, bmax = self.bounds_min, self.bounds_max
        x = float(gen.uniform(bmin[0] * 0.7 + bmax[0] * 0.15, bmax[0] * 0.7 + bmin[0] * 0.15))
        z = float(gen.uniform(bmin[2] * 0.7 + bmax[2] * 0.15, bmax[2] * 0.7 + bmin[2] * 0.15))
        y = float(self.height_at(np.array(x), np.array(z))) + agl_m
        return np.array([x, y, z])
