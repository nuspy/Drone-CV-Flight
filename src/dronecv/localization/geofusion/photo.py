"""Real-photo front-end: fill an `InvariantView` from a photograph.

Turns an RGB photo into the geometry/semantics channels the fusion localizer
consumes: sky / vegetation / water / building masks (classical, dependency
-free heuristics) and an estimated camera pitch from the horizon line. Depth
stays UNKNOWN (NaN): the retrieval and alignment stages detect that and score
on silhouette + skyline + sky agreement only, and the constellation judge
degrades to neutral — graceful, never wrong-confident.

This is deliberately the weakest link with the clearest upgrade path: swap
`segment_photo` for a segmentation foundation model and fill `depth` from a
monocular-depth model (Depth Anything / Metric3D) and every downstream stage
sharpens without any other change.
"""

from __future__ import annotations

import math

import numpy as np

from dronecv.localization.geofusion.invariant import InvariantView


def _resize_mean(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area resize; cv2 when available (fast), numpy box-mean fallback."""
    try:
        import cv2

        return cv2.resize(img.astype(np.float32), (width, height),
                          interpolation=cv2.INTER_AREA)
    except ImportError:
        h, w = img.shape[:2]
        rows = (np.arange(height + 1) * h // height).astype(int)
        cols = (np.arange(width + 1) * w // width).astype(int)
        out = np.zeros((height, width) + img.shape[2:], np.float32)
        for r in range(height):
            for c in range(width):
                out[r, c] = img[rows[r]:max(rows[r] + 1, rows[r + 1]),
                                cols[c]:max(cols[c] + 1, cols[c + 1])].mean(axis=(0, 1))
        return out


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :])
    return gx + gy


def _flood_from_top(cand: np.ndarray) -> np.ndarray:
    """Connected region of candidate pixels reachable from the top row."""
    h, w = cand.shape
    out = np.zeros_like(cand)
    stack = [(0, c) for c in range(w) if cand[0, c]]
    for rc in stack:
        out[rc] = True
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and cand[rr, cc] and not out[rr, cc]:
                out[rr, cc] = True
                stack.append((rr, cc))
    return out


def segment_photo(rgb: np.ndarray, width: int = 64, height: int = 48,
                  work_scale: int = 4) -> dict:
    """Heuristic sky/vegetation/water/building masks.

    Segmentation runs at `work_scale` times the output resolution so thin
    structures (spires, bridges, tree lines) survive the thresholds, then the
    masks are area-downsampled and re-composed by priority (sky > vegetation >
    water > building) so they stay disjoint."""
    wf, hf = width * work_scale, height * work_scale
    img = _resize_mean(rgb.astype(np.float32) / 255.0, wf, hf)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    v = img.max(axis=-1)
    grad = _grad_mag(0.299 * r + 0.587 * g + 0.114 * b)

    # sky: smooth+bright regions connected to the top edge (handles blue,
    # overcast and sunset skies alike)
    sky_cand = (grad < 0.08) & ((v > 0.45) | ((b > r) & (b > g) & (v > 0.3)))
    sky_f = _flood_from_top(sky_cand)

    veg_f = (~sky_f) & (g > r * 1.04) & (g > b * 1.04) & (g > 0.12)

    horizon_f = _horizon_row(sky_f)
    rows = np.arange(hf)[:, None].repeat(wf, 1)
    water_f = (~sky_f) & (~veg_f) & (b >= r * 0.95) & (grad < 0.06) & (rows > horizon_f)

    def down(mask: np.ndarray) -> np.ndarray:
        return _resize_mean(mask.astype(np.float32), width, height)

    sky_s, veg_s, water_s = down(sky_f), down(veg_f), down(water_f)
    sky = sky_s > 0.5
    veg = (~sky) & (veg_s > 0.35)
    water = (~sky) & (~veg) & (water_s > 0.35)
    building = ~(sky | veg | water)
    return {"sky": sky, "vegetation": veg, "water": water, "building": building,
            "horizon_row": int(round(horizon_f / work_scale))}


def _horizon_row(sky: np.ndarray) -> int:
    h, w = sky.shape
    lasts = []
    for c in range(w):
        rows = np.nonzero(sky[:, c])[0]
        lasts.append(rows[-1] if len(rows) else 0)
    return int(np.median(lasts))


def estimate_pitch_down_deg(horizon_row: int, height: int, fov_deg: float,
                            width: int) -> float:
    """Camera pitch from where the horizon sits in the frame: a camera pitched
    DOWN moves the horizon ABOVE the image center."""
    tan_v = math.tan(math.radians(fov_deg) / 2.0) * height / width
    frac = 0.5 - horizon_row / max(1, height - 1)
    return math.degrees(math.atan(2.0 * frac * tan_v))


def view_from_photo(rgb: np.ndarray, width: int = 64, height: int = 48,
                    fov_deg: float = 65.0,
                    pitch_down_deg: float | None = None,
                    depth_fn="auto") -> InvariantView:
    """Photo -> InvariantView. Depth: RELATIVE from the monocular plug-in
    when one is available (`depth_fn="auto"`), else unknown (NaN; inf on
    sky). Pass `depth_fn=None` to force mask-only."""
    seg = segment_photo(rgb, width, height)
    if pitch_down_deg is None:
        pitch_down_deg = float(np.clip(
            estimate_pitch_down_deg(seg["horizon_row"], height, fov_deg, width),
            2.0, 60.0))

    if depth_fn == "auto":
        from dronecv.localization.geofusion.depth_plugin import get_depth_fn

        depth_fn = get_depth_fn()
    relative = False
    if depth_fn is not None:
        d = _resize_mean(np.asarray(depth_fn(rgb), np.float32), width, height)
        depth = d.astype(np.float32)
        depth[seg["sky"]] = np.inf
        relative = True
    else:
        depth = np.full((height, width), np.nan, np.float32)
        depth[seg["sky"]] = np.inf
    return InvariantView(depth=depth, building=seg["building"],
                         vegetation=seg["vegetation"],
                         points=np.full((height, width, 3), np.nan),
                         pos=np.zeros(3), yaw_deg=0.0,
                         pitch_down_deg=pitch_down_deg, fov_deg=fov_deg,
                         water=seg["water"], depth_relative=relative)
