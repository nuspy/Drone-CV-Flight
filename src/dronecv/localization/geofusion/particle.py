"""M4 — particle fusion over a trajectory.

Look-alike poses that fool one frame cannot fool a whole flight: particles
carry (x, z, yaw), odometry (body-frame delta + yaw rate) propagates them,
and any per-pose score function (descriptor similarity, fine-pose fixes,
constellation) weights them. Systematic resampling keeps the survivors.
The estimate is the weighted mean with a circular yaw mean, plus a spread
that doubles as the confidence.
"""

from __future__ import annotations

import math

import numpy as np


class ParticleFuser:
    def __init__(self, n: int = 400, rng: np.random.Generator | None = None):
        self.n = n
        self.rng = rng or np.random.default_rng(0)
        self.xy = np.zeros((n, 2))
        self.yaw = np.zeros(n)
        self.w = np.full(n, 1.0 / n)

    def init_from_candidates(self, candidates, spread_m: float = 60.0,
                             spread_yaw_deg: float = 15.0) -> None:
        """Seed particles around the retrieval shortlist, proportional to
        each candidate's score (multi-hypothesis start)."""
        scores = np.array([max(1e-3, c.score) for c in candidates])
        counts = np.maximum(1, (scores / scores.sum() * self.n).astype(int))
        i = 0
        for cand, cnt in zip(candidates, counts, strict=False):
            cnt = min(cnt, self.n - i)
            if cnt <= 0:
                break
            self.xy[i:i + cnt, 0] = cand.pos[0] + self.rng.normal(0, spread_m, cnt)
            self.xy[i:i + cnt, 1] = cand.pos[2] + self.rng.normal(0, spread_m, cnt)
            self.yaw[i:i + cnt] = cand.yaw_deg + self.rng.normal(0, spread_yaw_deg, cnt)
            i += cnt
        if i < self.n:  # leftovers on the best candidate
            best = candidates[0]
            self.xy[i:, 0] = best.pos[0] + self.rng.normal(0, spread_m, self.n - i)
            self.xy[i:, 1] = best.pos[2] + self.rng.normal(0, spread_m, self.n - i)
            self.yaw[i:] = best.yaw_deg + self.rng.normal(0, spread_yaw_deg, self.n - i)
        self.w[:] = 1.0 / self.n

    def predict(self, forward_m: float, right_m: float, dyaw_deg: float,
                trans_noise_m: float = 3.0, yaw_noise_deg: float = 2.0) -> None:
        """Propagate with a BODY-frame odometry delta (VO output)."""
        yaw_rad = np.radians(self.yaw)
        # body forward = sim +Z rotated by yaw (Unity-style: +yaw turns Z->X)
        fx, fz = np.sin(yaw_rad), np.cos(yaw_rad)
        rx, rz = np.cos(yaw_rad), -np.sin(yaw_rad)
        self.xy[:, 0] += forward_m * fx + right_m * rx + self.rng.normal(0, trans_noise_m, self.n)
        self.xy[:, 1] += forward_m * fz + right_m * rz + self.rng.normal(0, trans_noise_m, self.n)
        self.yaw = (self.yaw + dyaw_deg + self.rng.normal(0, yaw_noise_deg, self.n)) % 360.0

    def weight(self, score_fn, beta: float = 8.0) -> None:
        """Multiply weights by exp(beta * score(x, z, yaw)); score in [0,1]."""
        s = np.array([score_fn(self.xy[i], self.yaw[i]) for i in range(self.n)])
        self.w *= np.exp(beta * np.clip(s, 0.0, 1.0))
        tot = self.w.sum()
        self.w = self.w / tot if tot > 0 else np.full(self.n, 1.0 / self.n)

    def weight_gaussian(self, fix_xy: np.ndarray, fix_yaw_deg: float,
                        sigma_m: float = 25.0, sigma_yaw_deg: float = 12.0) -> None:
        """Multiply weights by a Gaussian around an absolute fix (the strong
        measurement from a verified fine-pose)."""
        d2 = ((self.xy - np.asarray(fix_xy)[None, :]) ** 2).sum(axis=1)
        dyaw = (self.yaw - fix_yaw_deg + 180.0) % 360.0 - 180.0
        self.w *= np.exp(-d2 / (2 * sigma_m ** 2) - dyaw ** 2 / (2 * sigma_yaw_deg ** 2))
        tot = self.w.sum()
        self.w = self.w / tot if tot > 0 else np.full(self.n, 1.0 / self.n)

    def resample_if_needed(self, ratio: float = 0.5) -> None:
        neff = 1.0 / float((self.w ** 2).sum())
        if neff > ratio * self.n:
            return
        # systematic resampling
        positions = (np.arange(self.n) + self.rng.random()) / self.n
        idx = np.searchsorted(np.cumsum(self.w), positions)
        idx = np.clip(idx, 0, self.n - 1)
        self.xy = self.xy[idx].copy()
        self.yaw = self.yaw[idx].copy()
        self.w[:] = 1.0 / self.n

    def estimate(self) -> tuple[np.ndarray, float, float]:
        """(xy, yaw_deg, spread_m). Spread = weighted std of position."""
        xy = (self.w[:, None] * self.xy).sum(axis=0)
        s = (self.w * np.sin(np.radians(self.yaw))).sum()
        c = (self.w * np.cos(np.radians(self.yaw))).sum()
        yaw = math.degrees(math.atan2(s, c)) % 360.0
        var = (self.w[:, None] * (self.xy - xy) ** 2).sum(axis=0)
        return xy, yaw, float(np.sqrt(var.sum()))
