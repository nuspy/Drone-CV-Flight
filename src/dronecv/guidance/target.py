"""Flight targets: geographic coordinates or a visual reference image.

A VisualTarget is resolved by querying the landmark database (and the APR as
a cross-check) with the target image: the matched capture's position becomes
the target position and its altitude the target altitude ("the altitude
indicated by the objective"). Resolution carries its own confidence so the
caller can refuse a shaky match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dronecv.geo.anchor import GeoAnchor
from dronecv.training.bundle import ModelBundle


@dataclass(frozen=True)
class CoordinateTarget:
    lat: float
    lon: float
    alt_msl: float


@dataclass(frozen=True)
class VisualTarget:
    image: np.ndarray  # HxWx3 uint8, same camera geometry as training


@dataclass(frozen=True)
class TargetFix:
    pos_enu: np.ndarray
    confidence: float
    source: str  # "coordinates" | "visual"


def resolve_target(
    target: CoordinateTarget | VisualTarget,
    anchor: GeoAnchor,
    bundle: ModelBundle | None = None,
    topk: int = 5,
) -> TargetFix:
    if isinstance(target, CoordinateTarget):
        pos = anchor.geodetic_to_enu(target.lat, target.lon, target.alt_msl)
        return TargetFix(pos_enu=pos, confidence=1.0, source="coordinates")

    if bundle is None:
        raise ValueError("resolving a visual target requires a model bundle")
    img = torch.from_numpy(target.image.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        desc = bundle.embed_net(img)[0].numpy()
        apr = bundle.pose_net.predict(img)
    fix = bundle.landmark_db.query(desc, topk=topk)
    pos = np.asarray(fix["pos_enu"], dtype=float)
    # Cross-check with the APR: wild disagreement lowers confidence.
    apr_pos = apr["pos_enu"][0].numpy()
    disagreement = float(np.linalg.norm(apr_pos[:2] - pos[:2]))
    spread = fix["spread_m"]
    confidence = float(np.clip(1.0 / (1.0 + spread / 30.0 + disagreement / 100.0), 0.0, 1.0))
    return TargetFix(pos_enu=pos, confidence=confidence, source="visual")
