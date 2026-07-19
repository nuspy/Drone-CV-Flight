"""3D guidance toward a target with a fixed safety standoff.

The goal point is the target position pushed `standoff_m` back along the
horizontal target->drone bearing, at the target's altitude: the drone
approaches, stops 10 m short on its own side, and matches the target's
altitude. Output bearing is degrees clockwise from TRUE NORTH (the drone's
required course), plus the altitude change and arrival/violation flags.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dronecv.config import GuidanceConfig
from dronecv.geo.frames import heading_deg_from_enu_vector


@dataclass(frozen=True)
class Guidance:
    bearing_deg_true: float  # course to the goal point
    horizontal_dist_m: float  # to the goal point
    altitude_delta_m: float  # goal alt - drone alt (positive = climb)
    dist_to_target_m: float  # horizontal, to the target itself
    at_goal: bool
    standoff_violation: bool
    goal_enu: np.ndarray


def compute_guidance(
    drone_pos_enu: np.ndarray,
    target_pos_enu: np.ndarray,
    cfg: GuidanceConfig,
    fallback_bearing_deg: float = 0.0,
) -> Guidance:
    drone = np.asarray(drone_pos_enu, dtype=float)
    target = np.asarray(target_pos_enu, dtype=float)
    to_target_h = target[:2] - drone[:2]
    dist_target = float(np.linalg.norm(to_target_h))

    if dist_target > 1e-6:
        away = -to_target_h / dist_target  # unit vector target -> drone
    else:
        # Directly above/below the target: define the standoff direction from
        # the fallback bearing (e.g. current heading, reversed) — any
        # horizontal direction satisfies the constraint.
        rad = math.radians((fallback_bearing_deg + 180.0) % 360.0)
        away = np.array([math.sin(rad), math.cos(rad)])

    goal = np.array(
        [target[0] + away[0] * cfg.standoff_m, target[1] + away[1] * cfg.standoff_m, target[2]]
    )
    to_goal = goal - drone
    dist_goal = float(np.linalg.norm(to_goal[:2]))
    alt_delta = float(goal[2] - drone[2])
    bearing = (
        heading_deg_from_enu_vector(np.array([to_goal[0], to_goal[1], 0.0]))
        if dist_goal > 1e-6
        else fallback_bearing_deg
    )
    return Guidance(
        bearing_deg_true=bearing,
        horizontal_dist_m=dist_goal,
        altitude_delta_m=alt_delta,
        dist_to_target_m=dist_target,
        at_goal=dist_goal < cfg.arrival_tol_m and abs(alt_delta) < cfg.alt_tol_m,
        standoff_violation=dist_target < cfg.standoff_m * 0.98,
        goal_enu=goal,
    )
