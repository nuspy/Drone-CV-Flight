"""The cooperating cascade: M0 prior -> M1 retrieval -> M2 refine -> M3 veto
-> M4 particle fusion.

Single shot (`localize`): shortlist candidates by invariant descriptor,
refine the best few by render-and-compare, then let the constellation judge
veto look-alikes; confidence = alignment * constellation.

Sequence (`localize_sequence`): particles seeded from the first frame's
shortlist, propagated by body-frame odometry, weighted every frame by the
cheap descriptor field and periodically snapped by a fine-pose fix on the
current best hypothesis. Ambiguity that survives one frame dies over the
trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dronecv.localization.geofusion.constellation import (
    constellation_score,
    extract_buildings,
    load_model_buildings,
)
from dronecv.localization.geofusion.finepose import refine_pose
from dronecv.localization.geofusion.invariant import InvariantView, descriptor
from dronecv.localization.geofusion.particle import ParticleFuser
from dronecv.localization.geofusion.retrieval import GeoRetrievalIndex
from dronecv.util.logging import get_logger

log = get_logger("dronecv.geofusion")


@dataclass
class GeoFusionConfig:
    grid_step_m: float = 150.0
    n_yaws: int = 6
    agl_m: float = 60.0
    view_width: int = 64
    view_height: int = 48
    pitch_down_deg: float = 35.0
    fov_deg: float = 70.0
    topk: int = 8
    refine_top: int = 3
    veto_threshold: float = 0.15   # constellation score below this kills a fix
    n_particles: int = 400
    refine_every: int = 3          # sequence: fine-pose snap cadence (frames)


@dataclass
class FusionFix:
    pos: np.ndarray                # sim frame (x=E, y=up, z=N)
    yaw_deg: float
    confidence: float              # alignment * constellation, [0, 1]
    diagnostics: dict = field(default_factory=dict)


class GeoFusionLocalizer:
    def __init__(self, world, gis_dir: Path | None = None,
                 cfg: GeoFusionConfig | None = None,
                 index: GeoRetrievalIndex | None = None):
        self.world = world
        self.cfg = cfg or GeoFusionConfig()
        c = self.cfg
        self.index = index or GeoRetrievalIndex.build(
            world, c.grid_step_m, c.n_yaws, c.agl_m,
            width=c.view_width, height=c.view_height,
            pitch_down_deg=c.pitch_down_deg, fov_deg=c.fov_deg)
        self.model_buildings = (load_model_buildings(gis_dir)
                                if gis_dir is not None else np.zeros((0, 2)))

    # ------------------------------------------------------------ single shot

    def localize(self, view: InvariantView,
                 prior_xy: np.ndarray | None = None,
                 prior_radius_m: float | None = None) -> FusionFix:
        c = self.cfg
        candidates = self.index.query(view, k=c.topk, prior_xy=prior_xy,
                                      prior_radius_m=prior_radius_m)
        if not candidates:
            return FusionFix(pos=np.zeros(3), yaw_deg=0.0, confidence=0.0,
                             diagnostics={"reason": "no candidates"})

        observed = extract_buildings(view)
        best: FusionFix | None = None
        for cand in candidates[: c.refine_top]:
            pos, yaw, align = refine_pose(self.world, view, cand.pos,
                                          cand.yaw_deg, agl_m=None)
            const = constellation_score(observed, self.model_buildings,
                                        near_xy=pos[[0, 2]])
            conf = align * max(const, 1e-3)
            if const < c.veto_threshold and len(observed) >= 3:
                conf = 0.0  # the judge vetoes: layout is not there
            fix = FusionFix(pos=pos, yaw_deg=yaw, confidence=conf,
                            diagnostics={"alignment": align, "constellation": const,
                                         "retrieval_score": cand.score,
                                         "n_observed_buildings": int(len(observed))})
            if best is None or fix.confidence > best.confidence:
                best = fix
        return best

    # -------------------------------------------------------------- sequence

    def localize_sequence(self, views: list[InvariantView],
                          odometry: list[tuple[float, float, float]],
                          rng: np.random.Generator | None = None) -> list[FusionFix]:
        """`odometry[i]` = body-frame (forward_m, right_m, dyaw_deg) from
        frame i-1 to i (odometry[0] is ignored). Returns one fix per frame."""
        c = self.cfg
        pf = ParticleFuser(c.n_particles, rng=rng)
        pf.init_from_candidates(self.index.query(views[0], k=c.topk))

        fixes: list[FusionFix] = []
        last_fix_conf = 0.0
        for i, view in enumerate(views):
            if i > 0:
                fwd, right, dyaw = odometry[i]
                pf.predict(fwd, right, dyaw)
            # weak every-frame evidence: the descriptor field
            q = descriptor(view)
            pf.weight(lambda xy, yaw, _q=q: max(
                0.0, self.index.similarity_at(_q, xy, yaw)), beta=4.0)

            # strong periodic evidence: a full verified single-shot fix, using
            # the particle cloud itself as the prior (M4 -> M1 support)
            if i % c.refine_every == 0:
                xy, yaw, spread = pf.estimate()
                fix = self.localize(view, prior_xy=xy,
                                    prior_radius_m=max(200.0, 2.0 * spread))
                last_fix_conf = fix.confidence
                if fix.confidence > 0.2:
                    pf.weight_gaussian(fix.pos[[0, 2]], fix.yaw_deg)

            pf.resample_if_needed()
            xy, yaw, spread = pf.estimate()
            ground = float(self.world.height_at(np.array([xy[0]]), np.array([xy[1]]))[0])
            conf = float(np.clip(1.0 - spread / 100.0, 0.0, 1.0)) * max(0.3, last_fix_conf)
            fixes.append(FusionFix(pos=np.array([xy[0], ground + c.agl_m, xy[1]]),
                                   yaw_deg=yaw, confidence=conf,
                                   diagnostics={"spread_m": spread,
                                                "last_fix_conf": last_fix_conf}))
        return fixes
