"""Kinematic drone body: velocity-command tracking with a first-order lag.

The Unity DroneBody.cs implements exactly the same model so closed-loop
behavior transfers between the two simulators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dronecv.config import DroneConfig


@dataclass
class DroneState:
    pos_sim: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel_sim: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_deg: float = 0.0
    collided: bool = False


class DroneBody:
    def __init__(self, cfg: DroneConfig):
        self.cfg = cfg
        self.state = DroneState()
        self.cmd_vel_sim = np.zeros(3)
        self.cmd_yaw_rate_dps = 0.0

    def reset(self, pos_sim: np.ndarray, yaw_deg: float = 0.0) -> None:
        self.state = DroneState(pos_sim=np.asarray(pos_sim, dtype=float).copy(), yaw_deg=yaw_deg)
        self.cmd_vel_sim = np.zeros(3)
        self.cmd_yaw_rate_dps = 0.0

    def set_command(self, vel_sim: np.ndarray | None, yaw_rate_dps: float | None) -> None:
        if vel_sim is not None:
            v = np.asarray(vel_sim, dtype=float)
            horiz = np.array([v[0], 0.0, v[2]])
            speed = np.linalg.norm(horiz)
            if speed > self.cfg.max_speed_ms:
                horiz *= self.cfg.max_speed_ms / speed
            climb = float(np.clip(v[1], -self.cfg.max_climb_ms, self.cfg.max_climb_ms))
            self.cmd_vel_sim = np.array([horiz[0], climb, horiz[2]])
        if yaw_rate_dps is not None:
            self.cmd_yaw_rate_dps = float(
                np.clip(yaw_rate_dps, -self.cfg.max_yaw_rate_dps, self.cfg.max_yaw_rate_dps)
            )

    def step(self, dt: float, terrain_height_at) -> DroneState:
        s = self.state
        # First-order lag toward the commanded velocity.
        alpha = 1.0 - np.exp(-dt / max(self.cfg.response_tau_s, 1e-3))
        s.vel_sim = s.vel_sim + (self.cmd_vel_sim - s.vel_sim) * alpha
        s.pos_sim = s.pos_sim + s.vel_sim * dt
        s.yaw_deg = (s.yaw_deg + self.cmd_yaw_rate_dps * dt) % 360.0
        ground = float(terrain_height_at(s.pos_sim[0], s.pos_sim[2]))
        if s.pos_sim[1] <= ground + 0.5:
            s.pos_sim[1] = ground + 0.5
            s.collided = True
        return s
