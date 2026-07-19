"""Error-state EKF fusing all localization cues.

State (7): position ENU (3), velocity ENU (3), yaw = true-north heading of
the body forward axis (1, degrees internally radians). Attitude is yaw-only
by design: the drone body is kinematic and the camera tilt is fixed and
known; roll/pitch observability is not needed by any consumer.

Process model: constant velocity with white-accel noise; yaw random walk.
Measurements (each chi-square gated):
    - absolute position fixes (APR, retrieval consensus)  -> update_position
    - absolute yaw (sun/moon heading, APR/retrieval yaw)  -> update_yaw
    - velocity (VO ground-flow displacement / dt)         -> update_velocity
    - absolute altitude (terrain prior + lidar AGL)       -> update_altitude

Why an (error-state) EKF and not a particle filter: after consensus + gating
our absolute fixes are effectively unimodal; the multimodal/kidnapped case is
handled explicitly by lost-mode detection (a streak of rejected absolute
fixes) followed by re-initialization from the next retrieval consensus. A
`ParticleFuser` can implement this same interface later for genuinely
ambiguous environments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dronecv.config import LocalizationConfig


def _wrap_rad(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# 99% chi-square quantiles by degrees of freedom. cfg.gate_chi2 is the 2-dof
# base value; other dofs scale proportionally to their quantile.
CHI2_99 = {1: 6.63, 2: 9.21, 3: 11.34}
# During the initial transient (or right after re-init) the state can be far
# from truth with a covariance that has not caught up; gating there just
# starves the filter of exactly the fixes it needs.
GATE_MIN_POS_SIGMA_M = 40.0
GATE_MIN_YAW_SIGMA_DEG = 25.0


@dataclass
class FuserDiagnostics:
    accepted: dict[str, int] = field(default_factory=dict)
    rejected: dict[str, int] = field(default_factory=dict)
    last_nis: dict[str, float] = field(default_factory=dict)

    def note(self, cue: str, accepted: bool, nis: float) -> None:
        bucket = self.accepted if accepted else self.rejected
        bucket[cue] = bucket.get(cue, 0) + 1
        self.last_nis[cue] = nis


class EkfFuser:
    def __init__(self, cfg: LocalizationConfig):
        self.cfg = cfg
        self.x = np.zeros(7)  # [E, N, U, vE, vN, vU, yaw_rad]
        self.P = np.eye(7)
        self.initialized = False
        self.reject_streak = 0
        self.diag = FuserDiagnostics()

    # ------------------------------------------------------------------ state

    @property
    def pos(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def vel(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.x[6]) % 360.0

    @property
    def pos_sigma_h(self) -> float:
        """Horizontal 1-sigma (m) from the covariance."""
        return float(np.sqrt(max(0.5 * (self.P[0, 0] + self.P[1, 1]), 0.0)))

    def initialize(self, pos: np.ndarray, pos_cov: np.ndarray, yaw_deg: float, yaw_sigma_deg: float = 30.0) -> None:
        self.x[:] = 0.0
        self.x[:3] = pos
        self.x[6] = math.radians(yaw_deg % 360.0)
        self.P = np.zeros((7, 7))
        self.P[:3, :3] = pos_cov
        self.P[3:6, 3:6] = np.eye(3) * 4.0**2
        self.P[6, 6] = math.radians(yaw_sigma_deg) ** 2
        self.initialized = True
        self.reject_streak = 0

    @property
    def lost(self) -> bool:
        return self.initialized and self.reject_streak >= self.cfg.lost_reject_streak

    def _in_transient(self, cue: str) -> bool:
        if cue.endswith("yaw"):
            return math.degrees(math.sqrt(max(self.P[6, 6], 0.0))) > GATE_MIN_YAW_SIGMA_DEG
        return self.pos_sigma_h > GATE_MIN_POS_SIGMA_M

    # ---------------------------------------------------------------- predict

    def predict(self, dt: float) -> None:
        if not self.initialized or dt <= 0:
            return
        f = np.eye(7)
        f[0, 3] = f[1, 4] = f[2, 5] = dt
        self.x = f @ self.x
        q = np.zeros((7, 7))
        sa = self.cfg.process.accel_sigma
        # White-accel model: pos/vel blocks per axis.
        q_pp = sa**2 * dt**3 / 3.0
        q_pv = sa**2 * dt**2 / 2.0
        q_vv = sa**2 * dt
        for i in range(3):
            q[i, i] = q_pp
            q[i, i + 3] = q[i + 3, i] = q_pv
            q[i + 3, i + 3] = q_vv
        q[6, 6] = math.radians(self.cfg.process.yaw_rate_sigma_dps) ** 2 * dt
        self.P = f @ self.P @ f.T + q

    # ---------------------------------------------------------------- updates

    def _linear_update(self, h: np.ndarray, z: np.ndarray, r: np.ndarray, cue: str, gate_dof: int | None) -> bool:
        innov = z - h @ self.x
        if cue.endswith("yaw"):
            innov[0] = _wrap_rad(innov[0])
        s = h @ self.P @ h.T + r
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return False
        nis = float(innov @ s_inv @ innov)
        if gate_dof is not None and not self._in_transient(cue):
            gate = CHI2_99[gate_dof] * (self.cfg.gate_chi2 / CHI2_99[2]) * 2.0
            if nis > gate:
                self.diag.note(cue, False, nis)
                return False
        k = self.P @ h.T @ s_inv
        self.x = self.x + k @ innov
        self.x[6] = _wrap_rad(self.x[6])
        ikh = np.eye(7) - k @ h
        self.P = ikh @ self.P @ ikh.T + k @ r @ k.T  # Joseph form
        self.diag.note(cue, True, nis)
        return True

    def update_position(self, pos: np.ndarray, cov: np.ndarray, cue: str) -> bool:
        if not self.initialized:
            return False
        h = np.zeros((3, 7))
        h[0, 0] = h[1, 1] = h[2, 2] = 1.0
        ok = self._linear_update(h, np.asarray(pos, float), np.asarray(cov, float), cue, gate_dof=3)
        self.reject_streak = 0 if ok else self.reject_streak + 1
        return ok

    def update_yaw(self, yaw_deg: float, sigma_deg: float, cue: str) -> bool:
        if not self.initialized:
            return False
        h = np.zeros((1, 7))
        h[0, 6] = 1.0
        z = np.array([math.radians(yaw_deg % 360.0)])
        r = np.array([[math.radians(sigma_deg) ** 2]])
        return self._linear_update(h, z, r, cue + "_yaw", gate_dof=1)

    def update_velocity(self, vel: np.ndarray, cov: np.ndarray, cue: str = "vo") -> bool:
        if not self.initialized:
            return False
        h = np.zeros((3, 7))
        h[0, 3] = h[1, 4] = h[2, 5] = 1.0
        return self._linear_update(h, np.asarray(vel, float), np.asarray(cov, float), cue, gate_dof=3)

    def update_altitude(self, alt: float, sigma: float, cue: str = "lidar_alt") -> bool:
        if not self.initialized:
            return False
        h = np.zeros((1, 7))
        h[0, 2] = 1.0
        return self._linear_update(h, np.array([alt]), np.array([[sigma**2]]), cue, gate_dof=1)

    # -------------------------------------------------------------- confidence

    def confidence(self, target_err_m: float, cue_health: float = 1.0) -> float:
        """P(horizontal error < target) under the filter's own covariance
        (Rayleigh), scaled by cue health in [0, 1]. Uninitialized -> 0."""
        if not self.initialized:
            return 0.0
        sigma = max(self.pos_sigma_h, 1e-3)
        p = 1.0 - math.exp(-(target_err_m**2) / (2.0 * sigma**2))
        return float(np.clip(p * np.clip(cue_health, 0.0, 1.0), 0.0, 1.0))
