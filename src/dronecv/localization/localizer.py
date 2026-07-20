"""The Localizer: sensor frames in, geo-referenced estimates out.

Consumes ONLY what a real drone would have: camera frames, the mock lidar
range, and the UTC clock. Never sees simulator poses (the protocol enforces
it: the truth channel is not even subscribable with role="localizer").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import torch

from dronecv.config import Config
from dronecv.geo.anchor import GeoAnchor
from dronecv.localization import measurements as meas
from dronecv.localization.fusion import EkfFuser
from dronecv.models.vo import VisualOdometry, camera_matrix
from dronecv.training.bundle import ModelBundle
from dronecv.util.logging import get_logger

log = get_logger("dronecv.localizer")


@dataclass
class Estimate:
    initialized: bool
    lat: float
    lon: float
    alt_msl: float
    alt_agl: float | None
    pos_enu: np.ndarray
    vel_enu: np.ndarray
    speed_ms: float
    heading_deg: float
    confidence: float
    lost: bool
    cues: dict[str, Any] = field(default_factory=dict)


class Localizer:
    def __init__(self, cfg: Config, bundle: ModelBundle, anchor: GeoAnchor | None = None):
        self.cfg = cfg
        self.bundle = bundle
        self.anchor = anchor or bundle.anchor
        self._check_anchor()
        cam = bundle.manifest.get("camera") or {}
        self.width = int(cam.get("width", cfg.sim.image_width))
        self.height = int(cam.get("height", cfg.sim.image_height))
        self.fov = float(cam.get("fov_deg", cfg.sim.fov_deg))
        self.tilt = float(cam.get("tilt_deg", cfg.sim.camera_tilt_deg))
        self.K = camera_matrix(self.width, self.height, self.fov)
        self.K_inv = np.linalg.inv(self.K)
        self.vo = VisualOdometry(self.width, self.height, self.fov, self.tilt)
        self.fuser = EkfFuser(cfg.localization)
        self.target_err_m = cfg.active_loop.thresholds.p95_pos_err_m
        # The filter covariance collapses when many cues agree — but their
        # errors are CORRELATED (same model bias), so the true error floor is
        # the model's measured generalization error, not the filter's P.
        metrics = bundle.manifest.get("metrics") or {}
        probe_p95 = min(
            (metrics.get("apr", {}) or {}).get("pos_err_h_p95_m", float("inf")),
            (metrics.get("retrieval", {}) or {}).get("fix_err_h_p95_m", float("inf")),
        )
        self.sigma_floor_m = max(3.0, 0.35 * probe_p95) if np.isfinite(probe_p95) else 3.0
        self._prev_gray: np.ndarray | None = None
        self._prev_heading: float | None = None
        self._prev_time: float | None = None
        self._recent_absolute: list[bool] = []

    def _check_anchor(self) -> None:
        b = self.bundle.anchor
        a = self.anchor
        if abs(b.lat0 - a.lat0) > 1e-6 or abs(b.lon0 - a.lon0) > 1e-6:
            raise ValueError(
                f"model bundle was trained under anchor ({b.lat0}, {b.lon0}, source={b.source}) "
                f"but the sim resolves to ({a.lat0}, {a.lon0}, source={a.source}) — refusing to mix"
            )

    # ------------------------------------------------------------------- step

    def step(self, rgb: np.ndarray, lidar_range_m: float | None, utc: str, t: float) -> Estimate:
        dt = 0.0 if self._prev_time is None else max(t - self._prev_time, 0.0)
        self._prev_time = t
        utc_dt = datetime.fromisoformat(utc)
        cues: dict[str, Any] = {}

        from dronecv.vision.preprocess_filter import apply_filter

        arr = apply_filter(rgb.astype(np.float32) / 255.0, self.bundle.filter_spec)
        img = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            desc = self.bundle.embed_net(img)[0].numpy()
            apr = {k: v[0].numpy() if v.ndim else v for k, v in self.bundle.pose_net.predict(img).items()}
        fix = self.bundle.landmark_db.query(desc, topk=self.cfg.localization.retrieval_topk)
        cues["retrieval_spread_m"] = fix["spread_m"]
        cues["retrieval_similarity"] = fix["similarity"]

        # ---- (re)initialization from the retrieval consensus ----
        if not self.fuser.initialized or self.fuser.lost:
            if self.fuser.lost:
                log.warning("lost mode: re-initializing from retrieval consensus")
            pos, cov = meas.retrieval_position_measurement(fix, self.cfg.localization.apr_sigma_floor_m)
            self.fuser.initialize(pos, cov * 2.0, fix["heading_deg"], yaw_sigma_deg=35.0)
            cues["initialized_from"] = "retrieval"

        self.fuser.predict(dt)

        # ---- absolute position fixes ----
        pos_r, cov_r = meas.retrieval_position_measurement(fix, self.cfg.localization.apr_sigma_floor_m)
        ok_r = self.fuser.update_position(pos_r, cov_r, "retrieval")
        pos_a, cov_a = meas.apr_position_measurement(
            apr, self.bundle.calibration_scale, self.cfg.localization.apr_sigma_floor_m
        )
        ok_a = self.fuser.update_position(pos_a, cov_a, "apr")
        self._recent_absolute.append(ok_r or ok_a)
        self._recent_absolute = self._recent_absolute[-20:]
        cues["retrieval_accepted"] = ok_r
        cues["apr_accepted"] = ok_a

        # ---- absolute yaw: APR, retrieval, sun ----
        self.fuser.update_yaw(float(apr["heading_deg"]), float(apr["sigma_heading_deg"]), "apr")
        self.fuser.update_yaw(fix["heading_deg"], 20.0, "retrieval")
        det = meas.detect_sun_disc(rgb, self.K_inv, self.tilt)
        if det is not None:
            lat, lon, _ = self.anchor.enu_to_geodetic(self.fuser.pos)
            sun = meas.sun_heading_measurement(
                det, utc_dt, self.anchor, lat, lon,
                min_elevation_deg=self.cfg.localization.sun_min_elevation_deg,
            )
            if sun is not None:
                heading, sigma = sun
                cues["sun_heading_deg"] = heading
                self.fuser.update_yaw(heading, sigma, "sun")

        # ---- altitude: terrain prior + lidar ----
        agl = None
        if lidar_range_m is not None:
            agl = float(lidar_range_m)
            terrain_up = self.bundle.terrain.elevation(self.fuser.pos[0], self.fuser.pos[1])
            self.fuser.update_altitude(terrain_up + agl, sigma=3.0)
            cues["lidar_agl_m"] = agl

        # ---- VO velocity ----
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self._prev_gray is not None and dt > 1e-3 and self._prev_heading is not None:
            vo_agl = agl if agl is not None else max(
                self.fuser.pos[2] - self.bundle.terrain.elevation(self.fuser.pos[0], self.fuser.pos[1]), 5.0
            )
            res = self.vo.compute(self._prev_gray, gray, vo_agl)
            if res is not None and res.inliers >= self.cfg.localization.vo_min_inliers and res.disp_body_m is not None:
                h = math.radians(self._prev_heading)
                disp_enu = np.array(
                    [
                        res.disp_body_m[0] * math.cos(h) + res.disp_body_m[2] * math.sin(h),
                        -res.disp_body_m[0] * math.sin(h) + res.disp_body_m[2] * math.cos(h),
                        0.0,
                    ]
                )
                vel, cov = meas.vo_velocity_measurement(disp_enu, dt, res.inliers)
                self.fuser.update_velocity(vel, cov)
                cues["vo_inliers"] = res.inliers
                cues["vo_dyaw_deg"] = res.dyaw_deg
        self._prev_gray = gray
        self._prev_heading = self.fuser.yaw_deg

        # ---- output ----
        health = float(np.mean(self._recent_absolute)) if self._recent_absolute else 0.0
        sigma_h = max(self.fuser.pos_sigma_h, self.sigma_floor_m)
        p_within = 1.0 - math.exp(-(self.target_err_m**2) / (2.0 * sigma_h**2))
        confidence = float(np.clip(p_within * (0.3 + 0.7 * health), 0.0, 1.0))
        if not self.fuser.initialized:
            confidence = 0.0
        pos = self.fuser.pos
        lat, lon, alt = self.anchor.enu_to_geodetic(pos)
        vel = self.fuser.vel
        cues.update({k: v for k, v in self.fuser.diag.last_nis.items()})
        return Estimate(
            initialized=self.fuser.initialized,
            lat=lat,
            lon=lon,
            alt_msl=alt,
            alt_agl=agl,
            pos_enu=pos,
            vel_enu=vel,
            speed_ms=float(np.linalg.norm(vel[:2])),
            heading_deg=self.fuser.yaw_deg,
            confidence=confidence,
            lost=self.fuser.lost,
            cues=cues,
        )
