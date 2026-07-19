"""Turns raw model outputs and pixels into gated EKF measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from dronecv.geo import celestial
from dronecv.geo.anchor import GeoAnchor


@dataclass
class SunDetection:
    azimuth_body_deg: float  # sun azimuth relative to the body forward axis
    elevation_deg: float


def detect_sun_disc(rgb: np.ndarray, k_inv: np.ndarray, tilt_deg: float) -> SunDetection | None:
    """Find the rendered/photographed sun disc and return its direction in
    the body-level frame. Bright, warm, compact blob near-saturated in R+G."""
    mask = (rgb[..., 0] > 235) & (rgb[..., 1] > 225) & (rgb[..., 2] < 245)
    n = int(mask.sum())
    if n < 3 or n > rgb.shape[0] * rgb.shape[1] * 0.05:  # absent or washed out
        return None
    ys, xs = np.nonzero(mask)
    # Compactness check: a disc, not scattered highlights.
    if xs.std() > rgb.shape[1] * 0.08 or ys.std() > rgb.shape[0] * 0.08:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    ray = k_inv @ np.array([cx, cy, 1.0])
    ray = np.array([ray[0], -ray[1], ray[2]])  # y up
    pitch = math.radians(tilt_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rot_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    d = rot_x @ ray
    d = d / np.linalg.norm(d)
    az = math.degrees(math.atan2(d[0], d[2]))
    el = math.degrees(math.asin(np.clip(d[1], -1, 1)))
    return SunDetection(azimuth_body_deg=az, elevation_deg=el)


def sun_heading_measurement(
    det: SunDetection,
    utc: datetime,
    anchor: GeoAnchor,
    approx_lat: float | None = None,
    approx_lon: float | None = None,
    min_elevation_deg: float = 5.0,
    max_elevation_mismatch_deg: float = 12.0,
) -> tuple[float, float] | None:
    """Absolute heading (deg true, sigma) from a detected sun disc.

    heading = true sun azimuth (ephemeris at the approximate position and
    time) - sun azimuth relative to the body. The elevation consistency check
    rejects bright blobs that are not the sun.
    """
    lat = approx_lat if approx_lat is not None else anchor.lat0
    lon = approx_lon if approx_lon is not None else anchor.lon0
    eph = celestial.sun_position(utc, lat, lon)
    if eph.elevation_deg < min_elevation_deg:
        return None
    if abs(eph.elevation_deg - det.elevation_deg) > max_elevation_mismatch_deg:
        return None
    heading = (eph.azimuth_deg - det.azimuth_body_deg) % 360.0
    # Small discs at low resolution: ~1-2 px centroid error -> a few degrees.
    sigma = 4.0
    return heading, sigma


def apr_position_measurement(pred: dict, calibration_scale: float, sigma_floor_m: float) -> tuple[np.ndarray, np.ndarray]:
    """APR prediction (single sample, numpy-ready) -> (pos, cov)."""
    pos = np.asarray(pred["pos_enu"], dtype=float)
    sigma = max(float(pred["sigma_pos_m"]) * calibration_scale, sigma_floor_m)
    cov = np.eye(3) * sigma**2
    cov[2, 2] *= 2.0  # altitude is the APR's weakest axis
    return pos, cov


def retrieval_position_measurement(fix: dict, sigma_floor_m: float) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(fix["pos_enu"], dtype=float)
    cov = np.asarray(fix["cov"], dtype=float) + np.eye(3) * sigma_floor_m**2
    return pos, cov


def vo_velocity_measurement(
    disp_world: np.ndarray, dt: float, inliers: int
) -> tuple[np.ndarray, np.ndarray]:
    vel = np.asarray(disp_world, dtype=float) / max(dt, 1e-3)
    # More inliers -> tighter; always generous (flat-ground assumption).
    sigma = float(np.clip(6.0 / np.sqrt(max(inliers, 1)), 0.6, 3.0))
    cov = np.diag([sigma**2, sigma**2, (2.0 * sigma) ** 2])
    return vel, cov
