"""Classical relative visual odometry between consecutive frames.

ORB features + ratio-test matching, then a *ground-plane rigid transform*:
matched pixels are unprojected to rays, de-tilted (camera tilt is known and
fixed), intersected with the ground plane `agl` meters below the camera, and
a robust 2D Kabsch fit between the two projected point sets yields the yaw
change AND the metric drone displacement (in the previous body frame)
simultaneously. This resolves the classic rotation-vs-lateral-translation
ambiguity that defeats both azimuth-median heuristics and the essential
matrix (which additionally degenerates on rotation-only motion — two earlier
designs of this module failed exactly there).

Landmarks and sloped terrain violate the flat-ground model; iterative
residual trimming keeps only consistent ground features. When AGL is unknown
the caller passes agl_m=None and gets a yaw-only estimate (median azimuth
shift, exact for pure rotation).

Untrained by design: works identically in the headless sim, Unity, and on
real footage — a redundancy path independent of the learned models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VoResult:
    dyaw_deg: float
    disp_body_m: np.ndarray | None  # drone displacement, previous-body frame (x, y=0, z)
    inliers: int


def camera_matrix(width: int, height: int, fov_deg: float) -> np.ndarray:
    f = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def _level_rays(pts: np.ndarray, K_inv: np.ndarray, tilt_deg: float) -> np.ndarray:
    """Pixels -> unit rays in the body-level frame (x right, y up, z forward)."""
    ones = np.ones((len(pts), 1))
    cam = (K_inv @ np.hstack([pts, ones]).T).T  # x right, y down, z forward
    cam = np.stack([cam[:, 0], -cam[:, 1], cam[:, 2]], axis=1)  # y up
    pitch = math.radians(tilt_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rot_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])  # un-tilt
    d = cam @ rot_x.T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def _kabsch2d(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform with a ≈ R b + t (2D)."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    h = (b - cb).T @ (a - ca)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, d]) @ u.T
    return r, ca - r @ cb


class VisualOdometry:
    def __init__(self, width: int, height: int, fov_deg: float, tilt_deg: float, n_features: int = 800):
        self.K = camera_matrix(width, height, fov_deg)
        self.K_inv = np.linalg.inv(self.K)
        self.tilt_deg = tilt_deg
        self.orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _match(self, gray_prev: np.ndarray, gray_curr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        kp1, des1 = self.orb.detectAndCompute(gray_prev, None)
        kp2, des2 = self.orb.detectAndCompute(gray_curr, None)
        if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
            return None
        pairs = self.matcher.knnMatch(des1, des2, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < 10:
            return None
        pts1 = np.float32([kp1[g.queryIdx].pt for g in good])
        pts2 = np.float32([kp2[g.trainIdx].pt for g in good])
        return pts1, pts2

    def compute(self, gray_prev: np.ndarray, gray_curr: np.ndarray, agl_m: float | None) -> VoResult | None:
        matched = self._match(gray_prev, gray_curr)
        if matched is None:
            return None
        pts1, pts2 = matched
        r1 = _level_rays(pts1, self.K_inv, self.tilt_deg)
        r2 = _level_rays(pts2, self.K_inv, self.tilt_deg)

        if agl_m is None or agl_m < 2.0:
            return self._yaw_only(r1, r2)

        # Ground-plane projection in each body-level frame.
        down1, down2 = -r1[:, 1], -r2[:, 1]
        ok = (down1 > 0.12) & (down2 > 0.12)
        if ok.sum() < 8:
            return self._yaw_only(r1, r2)
        g1 = r1[ok] * (agl_m / down1[ok])[:, None]
        g2 = r2[ok] * (agl_m / down2[ok])[:, None]
        a = np.stack([g1[:, 0], g1[:, 2]], axis=1)  # (x, z) in prev frame
        b = np.stack([g2[:, 0], g2[:, 2]], axis=1)

        keep = np.ones(len(a), dtype=bool)
        r = np.eye(2)
        t = np.zeros(2)
        for _ in range(3):
            if keep.sum() < 6:
                return self._yaw_only(r1, r2)
            r, t = _kabsch2d(a[keep], b[keep])
            resid = np.linalg.norm((b @ r.T + t) - a, axis=1)
            cut = max(1.0, 2.5 * float(np.median(resid[keep])))
            keep = resid < cut
        n_inl = int(keep.sum())
        if n_inl < 6:
            return self._yaw_only(r1, r2)

        # r maps curr-frame ground coords into prev-frame: r = R_yaw(dyaw) in
        # the (x, z) plane with the Unity yaw convention (+Z toward +X).
        dyaw = math.degrees(math.atan2(r[0, 1], r[0, 0]))
        disp = np.array([t[0], 0.0, t[1]])
        pix1 = pts1[ok][keep]
        return VoResult(dyaw_deg=dyaw, disp_body_m=disp, inliers=n_inl) if len(pix1) else None

    def _yaw_only(self, r1: np.ndarray, r2: np.ndarray) -> VoResult | None:
        az1 = np.degrees(np.arctan2(r1[:, 0], r1[:, 2]))
        az2 = np.degrees(np.arctan2(r2[:, 0], r2[:, 2]))
        daz = _wrap(az1 - az2)
        med = float(np.median(daz))
        mad = float(np.median(np.abs(daz - med))) + 1e-6
        inl = np.abs(daz - med) < max(0.8, 3.0 * 1.4826 * mad)
        if inl.sum() < 8:
            return None
        return VoResult(dyaw_deg=float(np.median(daz[inl])), disp_body_m=None, inliers=int(inl.sum()))


def body_to_world_disp(disp_body: np.ndarray, yaw_prev_deg: float) -> np.ndarray:
    """Rotate a previous-body-frame displacement into the sim/world frame."""
    yaw = math.radians(yaw_prev_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rot_y @ disp_body
