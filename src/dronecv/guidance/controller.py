"""Minimal flight controller: turns (estimate, guidance) into velocity
commands. Used by the autonomous target-reach test; a real autopilot would
replace it, consuming the same Guidance outputs.

Safety behavior: below min confidence or in lost mode the drone holds
position (zero velocity); approach speed shrinks with distance and with
confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dronecv.config import GuidanceConfig
from dronecv.geo.anchor import GeoAnchor
from dronecv.guidance.guidance import Guidance
from dronecv.localization.localizer import Estimate


@dataclass
class ControlCommand:
    vel_sim: np.ndarray  # simulator-frame velocity command
    yaw_rate_dps: float


class FlightController:
    def __init__(self, cfg: GuidanceConfig, anchor: GeoAnchor):
        self.cfg = cfg
        self.anchor = anchor

    def command(self, est: Estimate, guidance: Guidance) -> ControlCommand:
        if not est.initialized or est.lost or est.confidence < self.cfg.min_confidence:
            return ControlCommand(vel_sim=np.zeros(3), yaw_rate_dps=0.0)

        # Approach speed: proportional near the goal, cruise far away, scaled
        # by confidence so a shaky estimate flies gently.
        speed = min(self.cfg.cruise_speed_ms, 0.4 * guidance.horizontal_dist_m + 0.3)
        speed *= float(np.clip(est.confidence, 0.3, 1.0))
        if guidance.at_goal:
            speed = 0.0

        rad = math.radians(guidance.bearing_deg_true)
        vel_enu = np.array([math.sin(rad) * speed, math.cos(rad) * speed, 0.0])
        vel_enu[2] = float(np.clip(0.6 * guidance.altitude_delta_m, -3.0, 3.0))
        vel_sim = self.anchor.enu_to_sim(vel_enu)

        # Yaw the camera toward the course (keeps the view informative).
        yaw_err = (guidance.bearing_deg_true - est.heading_deg + 180.0) % 360.0 - 180.0
        yaw_rate = float(np.clip(1.5 * yaw_err, -60.0, 60.0))
        return ControlCommand(vel_sim=vel_sim, yaw_rate_dps=yaw_rate)
