"""Cloud detection + multi-date cloud-free compositing.

Satellite photos are frequently obstructed by clouds. The fix the user asked
for, made systematic: detect the clouds (and their shadows) in each scene,
then walk OTHER acquisition dates of the same area and fill the missing
parts — a process that may take several photos.

- `cloud_mask` — classical detector, works on any RGB/gray ortho (no ML, no
  extra bands): clouds are BRIGHT, DESATURATED and locally SMOOTH; their
  shadows are dark desaturated blobs. Returns the INVALID mask
  (cloud ∪ shadow), dilated for safety fringes.
- `composite_scenes` — incremental compositing: start from the best scene,
  radiometrically align each further scene to the composite on the overlap
  of mutually valid pixels (median/MAD + per-channel gain — same math as
  gis.radiometry, applied cross-scene), and fill only the still-missing
  texels. Stops when hole coverage is negligible or scenes run out.

The composite has `utc=None` on purpose: it mixes dates, so shadow-based
height inference must not run on it (shadows move between acquisitions).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dronecv.gis.providers.imagery import OrthoImage
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.clouds")

CLOUD_LUM = 0.60  # min luminance of a cloud candidate
CLOUD_SAT = 0.16  # max channel spread (clouds are white/gray)
SHADOW_DROP = 0.62  # shadow candidate: lum < drop * local median
HOLE_DONE_FRACTION = 0.005  # stop compositing when <0.5% is still missing


def cloud_mask(ortho: OrthoImage) -> np.ndarray:
    """Boolean INVALID mask (clouds + their shadows) on the ortho grid.

    The brightness/darkness references are SCENE-GLOBAL (median): a local
    reference saturates inside any cloud larger than the window and misses
    its interior entirely."""
    import cv2

    g = ortho.gray.astype(np.float32)
    scene_med = float(np.median(g))

    if ortho.rgb is not None:
        spread = ortho.rgb.max(-1) - ortho.rgb.min(-1)
    else:
        spread = np.zeros_like(g)
    smooth = cv2.GaussianBlur(g, (0, 0), max(1.0, 2.0 / ortho.res_m))
    edges = cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
    texture = cv2.boxFilter((edges > 0.10).astype(np.float32), -1, (9, 9))

    # Texture only vetoes MODERATELY bright candidates (white buildings):
    # real cumulus have plenty of internal texture at 10 m and would slip
    # through a hard smoothness gate. Very bright + desaturated = cloud,
    # textured or not (a white industrial roof caught this way just gets
    # filled from another date — harmless).
    # Adaptive term CAPPED at 0.72: on a scene that is mostly/entirely cloud
    # the median IS cloud, and an uncapped med+0.15 would declare the whole
    # scene "clear" — the exact degenerate case of full overcast.
    bright = g > max(CLOUD_LUM, min(scene_med + 0.15, 0.72))
    cloud = bright & (spread < CLOUD_SAT) & ((texture < 0.35) | (g > 0.72))
    shadow = (g < scene_med * SHADOW_DROP) & (spread < CLOUD_SAT + 0.06)

    # Dilation must at least span the texture-gate band (9 px box) that
    # excludes cloud BORDERS, plus a 30 m safety fringe.
    k = max(11, int(30.0 / ortho.res_m) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cloud = cv2.morphologyEx(cloud.astype(np.uint8), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cloud = cv2.dilate(cloud, kernel).astype(bool)
    shadow = cv2.morphologyEx(shadow.astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # A dark blob only counts as CLOUD shadow if a cloud sits within a
    # plausible displacement (~1.2 km — low sun × cloud height): without
    # this gate water and asphalt would be thrown away wholesale.
    reach = max(3, int(1200.0 / ortho.res_m) | 1)
    near_cloud = cv2.dilate(cloud.astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach)))
    shadow = shadow.astype(bool) & near_cloud.astype(bool)
    shadow = cv2.dilate(shadow.astype(np.uint8), kernel).astype(bool)
    return cloud | shadow


@dataclass
class CompositeStats:
    n_scenes_used: int = 0
    first_scene_cloud_fraction: float = 0.0
    final_hole_fraction: float = 1.0
    fills: list[dict] = field(default_factory=list)  # per scene: date, filled px


def _align_to(reference: OrthoImage, ref_valid: np.ndarray,
              scene: OrthoImage, scene_valid: np.ndarray) -> OrthoImage:
    """Radiometrically align `scene` to `reference` on the overlap of valid
    pixels (median shift+scale of luminance, per-channel gain)."""
    overlap = ref_valid & scene_valid
    if overlap.sum() < 500:
        return scene
    r_med = float(np.median(reference.gray[overlap]))
    s_med = float(np.median(scene.gray[overlap]))
    r_mad = float(np.median(np.abs(reference.gray[overlap] - r_med))) or 1e-3
    s_mad = float(np.median(np.abs(scene.gray[overlap] - s_med))) or 1e-3
    scale = float(np.clip(r_mad / s_mad, 0.5, 2.0))
    gray = np.clip((scene.gray - s_med) * scale + r_med, 0.0, 1.0).astype(np.float32)
    rgb = scene.rgb
    if rgb is not None and reference.rgb is not None:
        gain = np.clip(
            np.median(reference.rgb[overlap].reshape(-1, 3), axis=0)
            / np.maximum(np.median(rgb[overlap].reshape(-1, 3), axis=0), 1e-3),
            0.5, 2.0,
        )
        rgb = np.clip(rgb * gain[None, None, :], 0.0, 1.0).astype(np.float32)
    return OrthoImage(gray=gray, res_m=scene.res_m, e0=scene.e0, n0=scene.n0,
                      utc=scene.utc, rgb=rgb)


def composite_scenes(
    scenes: list[tuple[OrthoImage, str]],
    max_scenes: int | None = None,
) -> tuple[OrthoImage, CompositeStats]:
    """Fill cloud holes of the first scene with later ones (order = caller's
    preference, usually most recent first). Returns the composite and stats.

    All scenes must share the same grid (res/e0/n0/shape)."""
    stats = CompositeStats()
    if not scenes:
        raise ValueError("no scenes to composite")
    if max_scenes:
        scenes = scenes[:max_scenes]

    base, base_date = scenes[0]
    invalid = cloud_mask(base)
    stats.first_scene_cloud_fraction = float(invalid.mean())
    gray = base.gray.copy()
    rgb = None if base.rgb is None else base.rgb.copy()
    valid = ~invalid
    stats.n_scenes_used = 1
    stats.fills.append({"date": base_date, "filled_px": int(valid.sum())})

    for scene, date in scenes[1:]:
        holes = ~valid
        if holes.mean() <= HOLE_DONE_FRACTION:
            break
        if scene.gray.shape != gray.shape:
            raise ValueError("composite scenes must share the same grid")
        s_valid = ~cloud_mask(scene)
        ref = OrthoImage(gray=gray, res_m=base.res_m, e0=base.e0, n0=base.n0, rgb=rgb)
        aligned = _align_to(ref, valid, scene, s_valid)
        fill = holes & s_valid
        if not fill.any():
            continue
        gray[fill] = aligned.gray[fill]
        if rgb is not None and aligned.rgb is not None:
            rgb[fill] = aligned.rgb[fill]
        valid |= fill
        stats.n_scenes_used += 1
        stats.fills.append({"date": date, "filled_px": int(fill.sum())})

    stats.final_hole_fraction = float((~valid).mean())
    # Remaining holes: inpaint from the nearest valid texels so downstream
    # consumers never see garbage (they are also excluded via `valid`).
    if (~valid).any():
        import cv2

        gray = cv2.inpaint((np.clip(gray, 0, 1) * 255).astype(np.uint8),
                           (~valid).astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
        if rgb is not None:
            rgb = cv2.inpaint((np.clip(rgb, 0, 1) * 255).astype(np.uint8),
                              (~valid).astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    log.info(
        f"cloud-free composite: {stats.n_scenes_used} scene(s), first-scene clouds "
        f"{stats.first_scene_cloud_fraction:.1%}, residual holes {stats.final_hole_fraction:.2%}"
    )
    out = OrthoImage(gray=gray.astype(np.float32), res_m=base.res_m, e0=base.e0,
                     n0=base.n0, utc=None, rgb=None if rgb is None else rgb.astype(np.float32))
    return out, stats
