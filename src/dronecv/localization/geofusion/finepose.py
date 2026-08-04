"""M2 — fine pose by render-and-compare on invariant channels.

LoD-Loc-style, simplified: starting from a coarse candidate, hill-climb the
pose (x, z, yaw) maximizing an alignment score between the query view and a
re-rendered view at the hypothesis:

    score = 0.6 * building-silhouette IoU + 0.4 * depth agreement

Both terms ignore color and lighting entirely. Multi-scale steps (coarse ->
fine) keep the render count small; altitude is taken from the query AGL over
the model terrain (the EKF/baro fixes it in flight).
"""

from __future__ import annotations

import numpy as np

from dronecv.localization.geofusion.invariant import InvariantView


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else (1.0 if inter == 0 else 0.0)


def view_alignment(a: InvariantView, b: InvariantView) -> float:
    """Alignment in [0,1] between two invariant views (same size)."""
    iou = _iou(a.building, b.building)
    # a river/lake in frame anchors the layout; only scored when the QUERY
    # knows about water at all
    has_water = a.water is not None and a.water_mask.any()
    water = _iou(a.water_mask, b.water_mask) if has_water else 0.0

    both = np.isfinite(a.depth) & np.isfinite(b.depth)
    # sky must agree too (a wrong pose often puts buildings where sky was)
    sky_match = float((a.sky == b.sky).mean())
    if not both.any():
        # no shared depth (real photo without a depth model): silhouette +
        # sky (+ water when known), weights redistributed
        if has_water:
            return 0.45 * iou + 0.25 * sky_match + 0.30 * water
        return 0.65 * iou + 0.35 * sky_match
    da, db = a.depth[both], b.depth[both]
    if a.depth_relative or b.depth_relative:
        # monocular relative depth: median-normalize both sides so the
        # unknown scale cancels and only the depth STRUCTURE is compared
        da = da / max(float(np.median(da)), 1e-6)
        db = db / max(float(np.median(db)), 1e-6)
        rel = np.abs(da - db) / np.maximum(da, 1e-3)
    else:
        rel = np.abs(da - db) / np.maximum(da, 1.0)
    depth_ok = float(np.clip(1.0 - np.median(rel) * 2.0, 0.0, 1.0))
    if has_water:
        return 0.4 * iou + 0.25 * depth_ok + 0.15 * sky_match + 0.2 * water
    return 0.5 * iou + 0.3 * depth_ok + 0.2 * sky_match


def refine_pose(
    world,
    query: InvariantView,
    pos0: np.ndarray,
    yaw0: float,
    agl_m: float | None = None,
    steps: tuple[tuple[float, float], ...] = ((48.0, 20.0), (16.0, 8.0),
                                              (5.0, 3.0), (2.0, 1.5)),
    max_iters_per_scale: int = 12,
) -> tuple[np.ndarray, float, float]:
    """Hill-climb (x, z, yaw) from (pos0, yaw0). Returns (pos, yaw, score).
    `steps` is a coarse-to-fine list of (translation_m, yaw_deg) step sizes."""
    h, w = query.depth.shape

    def at(x: float, z: float, yaw: float) -> tuple[InvariantView, float]:
        ground = float(world.height_at(np.array([x]), np.array([z]))[0])
        alt = ground + (agl_m if agl_m is not None else max(10.0, pos0[1] - ground))
        pos = np.array([x, alt, z])
        v = InvariantView.from_pose(world, pos, yaw, query.pitch_down_deg,
                                    w, h, query.fov_deg)
        return v, view_alignment(query, v)

    x, z, yaw = float(pos0[0]), float(pos0[2]), float(yaw0)
    _, best = at(x, z, yaw)
    for t_step, y_step in steps:
        for _ in range(max_iters_per_scale):
            improved = False
            for dx, dz, dy in ((t_step, 0, 0), (-t_step, 0, 0), (0, t_step, 0),
                               (0, -t_step, 0), (0, 0, y_step), (0, 0, -y_step)):
                _, s = at(x + dx, z + dz, yaw + dy)
                if s > best + 1e-4:
                    x, z, yaw, best = x + dx, z + dz, (yaw + dy) % 360.0, s
                    improved = True
                    break
            if not improved:
                break
    ground = float(world.height_at(np.array([x]), np.array([z]))[0])
    alt = ground + (agl_m if agl_m is not None else max(10.0, pos0[1] - ground))
    return np.array([x, alt, z]), yaw, best
