"""M1 — coarse retrieval: "where am I", from geometry alone.

A pose grid is laid over the world (fixed AGL above ground, several yaws per
cell); each pose is rendered once as an `InvariantView` and compressed to a
descriptor. A query descriptor is matched by cosine similarity, optionally
restricted to a prior circle (the EKF/odometry support: M0 shrinking M1's
search). The result is a shortlist of candidate poses for M2 to refine.

This is the training-free stand-in for a DINOv2/AnyLoc head: same interface,
so a foundation-model descriptor can replace `descriptor()` without touching
the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dronecv.localization.geofusion.invariant import InvariantView, descriptor
from dronecv.util.logging import get_logger

log = get_logger("dronecv.geofusion.retrieval")


@dataclass
class Candidate:
    pos: np.ndarray
    yaw_deg: float
    score: float


@dataclass
class GeoRetrievalIndex:
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    yaws: np.ndarray = field(default_factory=lambda: np.zeros(0))
    descs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))

    @classmethod
    def build(
        cls,
        world,
        grid_step_m: float = 150.0,
        n_yaws: int = 6,
        agl_m: float = 60.0,
        margin_m: float = 100.0,
        width: int = 64,
        height: int = 48,
        pitch_down_deg: float = 35.0,
        fov_deg: float = 70.0,
    ) -> GeoRetrievalIndex:
        lo, hi = world.bounds_min, world.bounds_max
        xs = np.arange(lo[0] + margin_m, hi[0] - margin_m + 1e-6, grid_step_m)
        zs = np.arange(lo[2] + margin_m, hi[2] - margin_m + 1e-6, grid_step_m)
        yaw_list = np.arange(n_yaws) * (360.0 / n_yaws)
        poses, yaws, descs = [], [], []
        for z in zs:
            for x in xs:
                ground = float(world.height_at(np.array([x]), np.array([z]))[0])
                pos = np.array([x, ground + agl_m, z])
                for yaw in yaw_list:
                    view = InvariantView.from_pose(
                        world, pos, yaw, pitch_down_deg, width, height, fov_deg)
                    poses.append(pos)
                    yaws.append(yaw)
                    descs.append(descriptor(view))
        log.info(f"retrieval index: {len(xs)}x{len(zs)} cells x {n_yaws} yaws "
                 f"= {len(descs)} invariant views")
        return cls(positions=np.asarray(poses), yaws=np.asarray(yaws),
                   descs=np.asarray(descs, np.float32))

    def query(
        self,
        view: InvariantView,
        k: int = 8,
        prior_xy: np.ndarray | None = None,
        prior_radius_m: float | None = None,
    ) -> list[Candidate]:
        """Top-k candidate poses by descriptor similarity, optionally only
        within the prior circle (M0 -> M1 support)."""
        q = descriptor(view)
        sims = self._similarities(q)
        mask = np.ones(len(sims), bool)
        if prior_xy is not None and prior_radius_m is not None:
            dx = self.positions[:, 0] - prior_xy[0]
            dz = self.positions[:, 2] - prior_xy[1]
            mask = dx * dx + dz * dz <= prior_radius_m ** 2
            if not mask.any():  # prior outside the grid: fall back to global
                mask[:] = True
        sims = np.where(mask, sims, -np.inf)
        order = np.argsort(-sims)[:k]
        return [Candidate(pos=self.positions[i].copy(), yaw_deg=float(self.yaws[i]),
                          score=float(sims[i])) for i in order if np.isfinite(sims[i])]

    def _similarities(self, q: np.ndarray) -> np.ndarray:
        """Cosine similarities; when the query has NO depth information (real
        photos: the depth-histogram dims are all zero) both sides are sliced to
        the depth-free dims and renormalized, so rankings compare like with
        like instead of penalizing the missing modality."""
        from dronecv.localization.geofusion.invariant import N_DEPTH_BINS

        if q[:N_DEPTH_BINS].any():
            return self.descs @ q
        qs = q[N_DEPTH_BINS:]
        qn = float(np.linalg.norm(qs))
        ds = self.descs[:, N_DEPTH_BINS:]
        dn = np.linalg.norm(ds, axis=1)
        return (ds @ (qs / max(qn, 1e-9))) / np.maximum(dn, 1e-9)

    def similarity_at(self, q_desc: np.ndarray, xy: np.ndarray, yaw_deg: float) -> float:
        """Descriptor similarity of the index pose nearest to (xy, yaw) — the
        cheap per-particle weight used by M4 (no rendering per particle)."""
        dx = self.positions[:, 0] - xy[0]
        dz = self.positions[:, 2] - xy[1]
        dyaw = np.abs((self.yaws - yaw_deg + 180.0) % 360.0 - 180.0)
        cost = dx * dx + dz * dz + (dyaw * 3.0) ** 2
        i = int(np.argmin(cost))
        return float(self._similarities(q_desc)[i]) if len(self.descs) else 0.0
