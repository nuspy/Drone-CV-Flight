"""Drape an orthophoto onto the ground albedo of a GIS store.

Turns the fetched imagery (Sentinel-2 composite, EOX mosaic, an XYZ tile drape
or a user GeoTIFF) into the visible ground colour layer (`store.albedo`,
uint8 HxWx3) that `GisWorld.albedo_at` and the renderers consume. Where the
imagery does not cover a texel — outside its footprint, or masked as
cloud/invalid — the synthetic per-class colour is used, so the whole ground
stays coloured. Without imagery the store keeps no albedo and rendering falls
back to class colours (the shape-first default).
"""

from __future__ import annotations

import numpy as np

from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.drape")


def _rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb)
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    m = float(a.max()) if a.size else 1.0
    if m <= 1.0:            # floats in [0, 1]
        a = a * 255.0
    elif m > 255.0:         # e.g. uint16 reflectance
        a = a * (255.0 / m)
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def drape_albedo(store, ortho) -> float:
    """Write `store.albedo` from `ortho.rgb`, class-colour where uncovered.
    Returns the fraction of ground texels that got real imagery colour."""
    from dronecv.gis.world import CLASS_COLORS

    meta = store.meta
    h, w = meta.height, meta.width

    # base = synthetic per-class colours (so the whole ground is coloured)
    lut = np.zeros((256, 3), np.float32)
    for cid, rgb in CLASS_COLORS.items():
        lut[cid] = rgb
    cls = np.asarray(store.class_id)
    albedo = np.clip(np.round(lut[cls] * 255.0), 0, 255).astype(np.uint8)

    covered = 0.0
    if ortho is not None and getattr(ortho, "rgb", None) is not None:
        # store texel centres in ENU -> ortho pixel indices
        ee = meta.e0 + (np.arange(w) + 0.5) * meta.res_m
        nn = meta.n0 + (np.arange(h) + 0.5) * meta.res_m
        oc = np.floor((ee - ortho.e0) / ortho.res_m).astype(int)   # (w,)
        orr = np.floor((nn - ortho.n0) / ortho.res_m).astype(int)  # (h,)
        oh, ow = ortho.rgb.shape[:2]
        in_c = (oc >= 0) & (oc < ow)
        in_r = (orr >= 0) & (orr < oh)
        rgb = _rgb_to_uint8(ortho.rgb)
        sampled = rgb[np.ix_(np.clip(orr, 0, oh - 1), np.clip(oc, 0, ow - 1))]  # (h,w,3)
        mask = in_r[:, None] & in_c[None, :]                                    # (h,w)
        if getattr(ortho, "valid", None) is not None:
            mask &= ortho.valid[np.ix_(np.clip(orr, 0, oh - 1), np.clip(oc, 0, ow - 1))]
        albedo[mask] = sampled[mask]
        covered = float(mask.mean())

    dst = store.create_albedo()
    dst[:] = albedo
    return covered
