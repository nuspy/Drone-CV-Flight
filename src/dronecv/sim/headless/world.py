"""Seeded procedural world for the headless simulator.

Sim frame convention matches Unity: X right, Y up, Z forward; the world is a
square [-size/2, +size/2] in X/Z. Terrain is a value-noise heightmap with a
procedurally colored albedo; landmarks are distinctive primitives (boxes and
cylinders with unique colors, height stripes and face shading) so that both
place recognition and yaw estimation have something to bite on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dronecv.config import WorldConfig
from dronecv.util.seeding import rng as make_rng


def _value_noise(gen: np.random.Generator, grid: int, octaves: int = 4, persistence: float = 0.55) -> np.ndarray:
    """Multi-octave bilinear value noise in [0, 1], shape (grid, grid)."""
    out = np.zeros((grid, grid), dtype=np.float64)
    amp, total = 1.0, 0.0
    for octave in range(octaves):
        cells = 2 ** (octave + 2)  # 4, 8, 16, ...
        coarse = gen.random((cells + 1, cells + 1))
        # Bilinear upsample to grid x grid.
        xs = np.linspace(0, cells, grid)
        x0 = np.clip(xs.astype(int), 0, cells - 1)
        fx = xs - x0
        rows = coarse[x0] * (1 - fx)[:, None] + coarse[x0 + 1] * fx[:, None]
        vals = rows[:, x0] * (1 - fx)[None, :] + rows[:, x0 + 1] * fx[None, :]
        out += amp * vals
        total += amp
        amp *= persistence
    return (out / total).astype(np.float32)


@dataclass
class Landmark:
    kind: str  # "box" | "cylinder"
    x: float
    z: float
    base_y: float
    half_x: float  # box half extent (or radius for cylinder)
    half_z: float
    height: float
    color_a: np.ndarray  # (3,) float in [0,1]
    color_b: np.ndarray
    stripe_m: float  # height-stripe period


class World:
    def __init__(self, cfg: WorldConfig, seed: int):
        self.cfg = cfg
        self.size = float(cfg.size_m)
        self.half = self.size / 2.0
        gen = make_rng(seed * 7919 + 13)

        g = cfg.grid
        self.heightmap = _value_noise(gen, g, octaves=5) * cfg.height_scale_m
        # Gentle bowl: keep edges lower so the flyable volume center is interesting.
        yy, xx = np.meshgrid(np.linspace(-1, 1, g), np.linspace(-1, 1, g), indexing="ij")
        self.heightmap *= (1.0 - 0.35 * (xx**2 + yy**2)).astype(np.float32)

        # Albedo: two independent noise fields drive a palette between
        # green/brown/grey bands plus a low-frequency regional tint, so distant
        # regions of the map look different (localizable ground texture).
        # The albedo texture is 4x the terrain grid and carries HIGH-FREQUENCY
        # per-texel detail (speckle, sharp patches). Smooth noise alone gives
        # feature detectors nothing to grip: ORB found literally zero corners
        # on an early smooth-only version of this texture.
        tg = g * 4
        n1 = _value_noise(gen, tg, octaves=6)
        n2 = _value_noise(gen, tg, octaves=3)
        palette_a = np.array([0.30, 0.42, 0.22])  # vegetation
        palette_b = np.array([0.52, 0.42, 0.30])  # soil
        palette_c = np.array([0.55, 0.55, 0.55])  # rock
        base = palette_a[None, None] * (1 - n1[..., None]) + palette_b[None, None] * n1[..., None]
        hm_up = np.kron(self.heightmap, np.ones((4, 4), dtype=np.float32))
        rocky = np.clip((hm_up / max(cfg.height_scale_m, 1e-6) - 0.55) * 3.0, 0, 1)[..., None]
        albedo = base * (1 - rocky) + palette_c[None, None] * rocky
        tint = (n2[..., None] - 0.5) * np.array([0.20, 0.10, 0.25])[None, None]
        speckle = gen.uniform(-0.09, 0.09, (tg, tg, 1))
        patches = (gen.random((tg, tg, 1)) < 0.04) * gen.uniform(-0.35, 0.35, (tg, tg, 1))
        self.albedo = np.clip(albedo + tint + speckle + patches, 0.05, 0.95).astype(np.float32)
        self._tex_grid = tg

        self.landmarks = self._place_landmarks(gen)

    # ------------------------------------------------------------------ terrain

    def _grid_coords(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g = self.cfg.grid
        u = (np.asarray(x) + self.half) / self.size * (g - 1)
        v = (np.asarray(z) + self.half) / self.size * (g - 1)
        return np.clip(u, 0, g - 1.000001), np.clip(v, 0, g - 1.000001)

    def height_at(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Bilinear terrain height, vectorized. Sim frame (x, z) in meters."""
        u, v = self._grid_coords(x, z)
        u0, v0 = u.astype(int), v.astype(int)
        fu, fv = u - u0, v - v0
        h = self.heightmap
        return (
            h[v0, u0] * (1 - fu) * (1 - fv)
            + h[v0, u0 + 1] * fu * (1 - fv)
            + h[v0 + 1, u0] * (1 - fu) * fv
            + h[v0 + 1, u0 + 1] * fu * fv
        )

    def normal_at(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Terrain normal (unit, sim frame), vectorized -> (..., 3)."""
        eps = self.size / self.cfg.grid
        hx = self.height_at(x + eps, z) - self.height_at(x - eps, z)
        hz = self.height_at(x, z + eps) - self.height_at(x, z - eps)
        n = np.stack([-hx, np.full_like(hx, 2 * eps), -hz], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def albedo_at(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        tg = self._tex_grid
        u = np.clip((np.asarray(x) + self.half) / self.size * (tg - 1), 0, tg - 1.000001)
        v = np.clip((np.asarray(z) + self.half) / self.size * (tg - 1), 0, tg - 1.000001)
        return self.albedo[v.astype(int), u.astype(int)]

    # ---------------------------------------------------------------- landmarks

    def _place_landmarks(self, gen: np.random.Generator) -> list[Landmark]:
        landmarks: list[Landmark] = []
        # Distinct, well-separated hues: golden-angle around the color wheel.
        for i in range(self.cfg.n_landmarks):
            for _attempt in range(60):
                x = float(gen.uniform(-0.85, 0.85) * self.half)
                z = float(gen.uniform(-0.85, 0.85) * self.half)
                min_d = self.size * 0.06
                if all((x - lm.x) ** 2 + (z - lm.z) ** 2 > min_d**2 for lm in landmarks):
                    break
            hue = (i * 137.508) % 360.0
            color_a = _hsv(hue, 0.85, 0.95)
            color_b = _hsv((hue + 180.0) % 360.0, 0.75, 0.75)
            kind = ["box", "cylinder", "box"][i % 3]
            height = float(gen.uniform(0.25, 0.9) * max(self.cfg.height_scale_m, 20.0)) + 12.0
            half_x = float(gen.uniform(4.0, 10.0))
            half_z = half_x if kind == "cylinder" else float(gen.uniform(4.0, 10.0))
            landmarks.append(
                Landmark(
                    kind=kind,
                    x=x,
                    z=z,
                    base_y=float(self.height_at(np.array(x), np.array(z))),
                    half_x=half_x,
                    half_z=half_z,
                    height=height,
                    color_a=color_a,
                    color_b=color_b,
                    stripe_m=float(gen.uniform(4.0, 9.0)),
                )
            )
        return landmarks

    # ------------------------------------------------------------------- bounds

    @property
    def max_height(self) -> float:
        return float(self.heightmap.max())

    @property
    def min_height(self) -> float:
        return float(self.heightmap.min())

    @property
    def bounds_min(self) -> np.ndarray:
        return np.array([-self.half, 0.0, -self.half])

    @property
    def bounds_max(self) -> np.ndarray:
        return np.array([self.half, self.max_height + 160.0, self.half])

    def spawn_position(self, gen: np.random.Generator, agl_m: float = 50.0) -> np.ndarray:
        x = float(gen.uniform(-0.7, 0.7) * self.half)
        z = float(gen.uniform(-0.7, 0.7) * self.half)
        y = float(self.height_at(np.array(x), np.array(z))) + agl_m
        return np.array([x, y, z])


def _hsv(h_deg: float, s: float, v: float) -> np.ndarray:
    h = (h_deg % 360.0) / 60.0
    i = int(h)
    f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return np.array(rgb, dtype=np.float32)
