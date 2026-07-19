"""Capture pose plans: where to teleport the camera to build the dataset.

- GridPlan: lawnmower grid over the flyable area, several altitude bands,
  several yaw angles per station -> broad, uniform coverage.
- OrbitPlan: rings around interest points (landmarks when known) -> dense
  multi-perspective views of recognizable structures.
- TargetedPlan: extra poses inside specified high-error cells -> used by the
  active loop to spend budget exactly where the model is weak.

Plans are pure generators of CapturePose in the sim frame; ground heights per
station are probed by the collector (the protocol has no height query).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dronecv.config import CaptureConfig
from dronecv.util.seeding import rng as make_rng


@dataclass(frozen=True)
class CapturePose:
    x: float
    z: float
    agl_m: float  # altitude above ground at (x, z); collector resolves to y
    yaw_deg: float
    pitch_deg: float


@dataclass(frozen=True)
class Station:
    x: float
    z: float


def grid_plan(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    cfg: CaptureConfig,
    margin_frac: float = 0.08,
    density=None,
) -> list[CapturePose]:
    """Lawnmower grid. `density(x, z) -> multiplier` (e.g. the GIS saliency
    prior: repetitive areas ~2x, distinctive ones ~0.5x) modulates the station
    density: the grid is generated finer and stations are kept with
    probability proportional to the local multiplier (deterministic hash, so
    plans are reproducible)."""
    x0, x1 = float(bounds_min[0]), float(bounds_max[0])
    z0, z1 = float(bounds_min[2]), float(bounds_max[2])
    mx, mz = (x1 - x0) * margin_frac, (z1 - z0) * margin_frac
    d_max = 2.2 if density is not None else 1.0
    spacing = cfg.grid_spacing_m / np.sqrt(d_max)
    xs = np.arange(x0 + mx, x1 - mx + 1e-6, spacing)
    zs = np.arange(z0 + mz, z1 - mz + 1e-6, spacing)
    yaws = np.linspace(0.0, 360.0, cfg.yaw_bins, endpoint=False)
    poses = []
    for x in xs:
        for z in zs:
            if density is not None:
                keep_p = float(density(x, z)) / d_max
                h = (int(x * 7.31) * 73856093) ^ (int(z * 7.31) * 19349663)
                if ((h & 0xFFFF) / 65535.0) > keep_p:
                    continue
            for agl in cfg.altitudes_agl_m:
                for yaw in yaws:
                    poses.append(CapturePose(float(x), float(z), float(agl), float(yaw), 0.0))
    return poses


def orbit_plan(
    points: list[tuple[float, float]],
    cfg: CaptureConfig,
) -> list[CapturePose]:
    """Rings around interest points, camera yawed to face the point."""
    poses: list[CapturePose] = []
    for px, pz in points:
        for i in range(cfg.orbit_points):
            ang = 2 * np.pi * i / cfg.orbit_points
            x = px + cfg.orbit_radius_m * np.sin(ang)
            z = pz + cfg.orbit_radius_m * np.cos(ang)
            # Face the point: sim yaw 0 = +Z, positive toward +X.
            yaw = (np.degrees(np.arctan2(px - x, pz - z))) % 360.0
            for agl in cfg.altitudes_agl_m[:2]:
                poses.append(CapturePose(float(x), float(z), float(agl), float(yaw), 0.0))
    return poses


def targeted_plan(
    cells: list[tuple[float, float, float]],  # (center_x, center_z, weight)
    cell_size_m: float,
    n_captures: int,
    cfg: CaptureConfig,
    seed: int,
) -> list[CapturePose]:
    """Sample poses inside weighted cells, with yaw/altitude diversity."""
    if not cells or n_captures <= 0:
        return []
    gen = make_rng(seed)
    weights = np.array([max(w, 1e-6) for _, _, w in cells])
    weights = weights / weights.sum()
    counts = gen.multinomial(n_captures, weights)
    poses: list[CapturePose] = []
    for (cx, cz, _w), count in zip(cells, counts, strict=True):
        for _ in range(count):
            x = cx + gen.uniform(-0.5, 0.5) * cell_size_m
            z = cz + gen.uniform(-0.5, 0.5) * cell_size_m
            agl = float(gen.choice(cfg.altitudes_agl_m)) * gen.uniform(0.8, 1.2)
            yaw = float(gen.uniform(0, 360))
            poses.append(CapturePose(float(x), float(z), agl, yaw, 0.0))
    return poses


def random_plan(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    n: int,
    cfg: CaptureConfig,
    seed: int,
    margin_frac: float = 0.1,
) -> list[CapturePose]:
    """Uniform random probe poses — used for held-out evaluation that shares
    no structure with the training plans."""
    gen = make_rng(seed * 7211 + 17)
    x0, x1 = float(bounds_min[0]), float(bounds_max[0])
    z0, z1 = float(bounds_min[2]), float(bounds_max[2])
    mx, mz = (x1 - x0) * margin_frac, (z1 - z0) * margin_frac
    lo, hi = min(cfg.altitudes_agl_m), max(cfg.altitudes_agl_m)
    return [
        CapturePose(
            float(gen.uniform(x0 + mx, x1 - mx)),
            float(gen.uniform(z0 + mz, z1 - mz)),
            float(gen.uniform(lo * 0.9, hi * 1.1)),
            float(gen.uniform(0, 360)),
            0.0,
        )
        for _ in range(n)
    ]


def shuffle_and_cap(poses: list[CapturePose], cap: int | None, seed: int) -> list[CapturePose]:
    gen = make_rng(seed * 977 + 3)
    idx = gen.permutation(len(poses))
    if cap is not None:
        idx = idx[:cap]
    return [poses[i] for i in idx]
