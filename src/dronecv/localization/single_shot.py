"""Single-photo localization: one image in, geographic fix out.

Used by the quick-test path ("does an aerial photo of the trained
environment localize correctly?"), the inference server behind the Android
app, and — ported to Kotlin — by the app's on-device mode. No EKF here: with
a single frame there is no temporal state, so the retrieval consensus and the
APR prediction are fused statically by inverse-variance weighting, and their
mutual agreement feeds the confidence.

Preprocessing contract (mirrored EXACTLY by the Kotlin implementation):
center-crop to the model aspect ratio -> bilinear resize to (W, H) -> RGB
float [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from dronecv.geo.anchor import GeoAnchor
from dronecv.training.bundle import ModelBundle

RAYLEIGH_90 = 2.1460


@dataclass
class PhotoFix:
    lat: float
    lon: float
    alt_msl: float
    pos_enu: np.ndarray
    heading_deg: float
    confidence: float
    sigma_h_m: float
    diagnostics: dict[str, Any]


def preprocess(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """RGB uint8 (any size) -> model input HxWx3 float32 [0,1]."""
    h, w = image.shape[:2]
    target_ar = width / height
    ar = w / h
    if ar > target_ar:  # too wide: crop left/right
        new_w = int(round(h * target_ar))
        x0 = (w - new_w) // 2
        image = image[:, x0 : x0 + new_w]
    elif ar < target_ar:  # too tall: crop top/bottom
        new_h = int(round(w / target_ar))
        y0 = (h - new_h) // 2
        image = image[y0 : y0 + new_h]
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) / 255.0


class SingleShotLocalizer:
    def __init__(self, bundle: ModelBundle, topk: int = 5, target_err_m: float = 30.0):
        self.bundle = bundle
        self.anchor: GeoAnchor = bundle.anchor
        cam = bundle.manifest.get("camera") or {}
        self.width = int(cam.get("width", 96))
        self.height = int(cam.get("height", 96))
        self.topk = topk
        self.target_err_m = target_err_m

    def localize(self, image_rgb: np.ndarray) -> PhotoFix:
        from dronecv.vision.preprocess_filter import apply_filter

        arr = preprocess(image_rgb, self.width, self.height)
        arr = apply_filter(arr, self.bundle.filter_spec)
        img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            desc = self.bundle.embed_net(img)[0].numpy()
            apr = self.bundle.pose_net.predict(img)

        fix = self.bundle.landmark_db.query(desc, topk=self.topk)
        pos_r = np.asarray(fix["pos_enu"], dtype=float)
        sigma_r = max(float(fix["spread_m"]), 2.0)

        pos_a = apr["pos_enu"][0].numpy().astype(float)
        sigma_a = max(
            float(apr["sigma_pos_m"][0]) * self.bundle.calibration_scale, 2.0
        )

        # Inverse-variance fusion of the two absolute fixes.
        w_r, w_a = 1.0 / sigma_r**2, 1.0 / sigma_a**2
        pos = (pos_r * w_r + pos_a * w_a) / (w_r + w_a)
        sigma_fused = math.sqrt(1.0 / (w_r + w_a))

        # Agreement: if the two cues disagree far beyond their claimed
        # sigmas, the fix is suspect regardless of either sigma.
        disagreement = float(np.linalg.norm(pos_r[:2] - pos_a[:2]))
        sigma_pair = math.sqrt(sigma_r**2 + sigma_a**2)
        agreement = math.exp(-0.5 * (disagreement / max(sigma_pair, 1e-6)) ** 2)

        sigma_eff = sigma_fused / max(agreement, 0.05)
        confidence = (1.0 - math.exp(-(self.target_err_m**2) / (2.0 * sigma_eff**2))) * (
            0.5 + 0.5 * agreement
        )

        heading = fix["heading_deg"]
        lat, lon, alt = self.anchor.enu_to_geodetic(pos)
        return PhotoFix(
            lat=lat,
            lon=lon,
            alt_msl=alt,
            pos_enu=pos,
            heading_deg=float(heading),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            sigma_h_m=float(sigma_eff),
            diagnostics={
                "retrieval_pos_enu": pos_r.tolist(),
                "retrieval_spread_m": sigma_r,
                "retrieval_similarity": fix["similarity"],
                "apr_pos_enu": pos_a.tolist(),
                "apr_sigma_m": sigma_a,
                "disagreement_m": disagreement,
                "agreement": agreement,
            },
        )
