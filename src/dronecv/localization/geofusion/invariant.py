"""Invariant views: what a pose looks like with color and light removed.

An `InvariantView` holds per-pixel depth (meters, inf = sky), a building
mask, a vegetation mask and the hit points in world coordinates. Rendered
from the GIS world these channels depend only on GEOMETRY and SEMANTICS —
sun position, palette colors, shadows and facade textures cannot change
them, which is exactly why the fusion localizer compares these instead of
RGB. For real photos the same structure is filled from a monocular-depth +
segmentation model (plug-in; not needed in simulation).

`descriptor()` compresses a view into a compact vector for retrieval:
log-depth histogram + class fractions + a per-column skyline signature
(elevation of the highest building pixel). All parts are ratios/angles, so
the descriptor is invariant to illumination and robust to small pose noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dronecv.sim.headless.rasterizer import NO_HIT, ray_directions, render


@dataclass
class InvariantView:
    depth: np.ndarray          # (H, W) float32, meters; inf where sky
    building: np.ndarray       # (H, W) bool
    vegetation: np.ndarray     # (H, W) bool
    points: np.ndarray         # (H, W, 3) sim-frame hit points (x=E, y=up, z=N)
    pos: np.ndarray            # camera position used to render (or estimate)
    yaw_deg: float
    pitch_down_deg: float
    fov_deg: float

    @property
    def sky(self) -> np.ndarray:
        return ~np.isfinite(self.depth)

    @classmethod
    def from_pose(
        cls,
        world,
        pos_sim: np.ndarray,
        yaw_deg: float,
        pitch_down_deg: float = 35.0,
        width: int = 64,
        height: int = 48,
        fov_deg: float = 70.0,
    ) -> InvariantView:
        """Render the invariant channels at a pose. The sun passed to the
        renderer is fixed and irrelevant: only the range image is kept."""
        pos = np.asarray(pos_sim, dtype=np.float64)
        _, rng = render(world, pos, yaw_deg, pitch_down_deg, width, height,
                        fov_deg, 160.0, 55.0)
        dirs = ray_directions(width, height, fov_deg, yaw_deg, pitch_down_deg)
        hit = rng < NO_HIT / 2
        depth = np.where(hit, rng, np.inf).astype(np.float32)
        pts = pos[None, None, :] + dirs * np.where(hit, rng, 0.0)[..., None]

        store = world.store
        m = store.meta
        c = np.clip(((pts[..., 0] - m.e0) / m.res_m).astype(np.int64), 0, m.width - 1)
        r = np.clip(((pts[..., 2] - m.n0) / m.res_m).astype(np.int64), 0, m.height - 1)
        build = (np.asarray(store.build_h)[r, c] > 0.5) & hit
        veg = (np.asarray(store.veg_h)[r, c] > 0.5) & hit & ~build
        return cls(depth=depth, building=build, vegetation=veg, points=pts,
                   pos=pos, yaw_deg=float(yaw_deg),
                   pitch_down_deg=float(pitch_down_deg), fov_deg=float(fov_deg))


N_DEPTH_BINS = 12
N_SKY_COLS = 16


def descriptor(view: InvariantView) -> np.ndarray:
    """Compact, illumination-invariant descriptor of a view (L2-normalized)."""
    h, w = view.depth.shape
    finite = np.isfinite(view.depth)
    parts: list[np.ndarray] = []

    # log-depth histogram (shape of the visible ranges; scale-compressed)
    d = view.depth[finite]
    hist, _ = np.histogram(np.log1p(d), bins=N_DEPTH_BINS, range=(0.0, 8.5))
    parts.append(hist / max(1, finite.sum()))

    # class fractions, split top/bottom half (coarse layout)
    for mask in (view.building, view.vegetation, view.sky):
        top, bot = mask[: h // 2], mask[h // 2:]
        parts.append(np.array([top.mean(), bot.mean()]))

    # skyline: highest building pixel per column band (elevation fraction)
    sky_sig = np.zeros(N_SKY_COLS)
    cols = np.array_split(np.arange(w), N_SKY_COLS)
    for i, cs in enumerate(cols):
        band = view.building[:, cs]
        rows = np.nonzero(band.any(axis=1))[0]
        sky_sig[i] = 1.0 - rows[0] / h if len(rows) else 0.0
    parts.append(sky_sig)

    v = np.concatenate(parts).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v
