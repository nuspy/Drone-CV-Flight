"""Vegetation layer: forests as visible 3D canopy in the heightfield.

Forest polygons (with typology from OSM `leaf_type`) become a noisy canopy
height raster: base height per type, per-texel variation and gap holes from a
deterministic hash — so training renders show believable bumpy tree cover and
the Unity/Blender exports can instance individual trees from the same
density/type data.

Bridges get a raised deck (interpolated between end-point terrain + clearance)
and railways a small embankment ridge — both become part of the geometry.
"""

from __future__ import annotations

import numpy as np

from dronecv.geo.anchor import GeoAnchor
from dronecv.gis.geometry import ring_to_enu
from dronecv.gis.providers.landcover import GROUND_CLASSES, LandcoverFeature
from dronecv.gis.store import GisStore
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.vegetation")

CANOPY_BASE_M = {GROUND_CLASSES["forest_broadleaf"]: 16.0, GROUND_CLASSES["forest_conifer"]: 22.0}
GAP_FRACTION = 0.18  # fraction of forest texels left open (paths, clearings)


def _texel_hash(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    h = (rows.astype(np.int64) * 73856093) ^ (cols.astype(np.int64) * 19349663)
    return (h & 0xFFFF).astype(np.float32) / 65535.0


def build_vegetation(store: GisStore) -> dict:
    """Fill veg_h from the forest classes already rasterized in class_id."""
    win = 2048
    n_forest = 0
    for r0 in range(0, store.meta.height, win):
        r1 = min(r0 + win, store.meta.height)
        for c0 in range(0, store.meta.width, win):
            c1 = min(c0 + win, store.meta.width)
            cls = np.asarray(store.class_id[r0:r1, c0:c1])
            veg = np.zeros(cls.shape, dtype=np.float32)
            ys, xs = np.mgrid[r0:r1, c0:c1]
            noise = _texel_hash(ys, xs)
            for cid, base in CANOPY_BASE_M.items():
                m = cls == cid
                if not m.any():
                    continue
                canopy = base * (0.7 + 0.5 * noise)
                canopy[noise < GAP_FRACTION] = 0.0  # clearings
                veg[m] = canopy[m]
                n_forest += int(m.sum())
            # No trees on top of buildings.
            veg[np.asarray(store.build_h[r0:r1, c0:c1]) > 0] = 0.0
            store.veg_h[r0:r1, c0:c1] = veg
    log.info(f"vegetation layer: {n_forest} forest texels "
             f"({n_forest * store.meta.res_m**2 / 1e4:.1f} ha)")
    return {"forest_texels": n_forest}


def stamp_bridges(store: GisStore, anchor: GeoAnchor, feats: list[LandcoverFeature], clearance_m: float = 5.0) -> int:
    """Bridge decks: a strip raised above the terrain, spanning between the
    ground heights at the two ends."""
    from rasterio import features as rio_features
    from rasterio.transform import Affine
    from shapely.geometry import LineString

    n = 0
    res = store.meta.res_m
    transform = Affine(res, 0.0, store.meta.e0, 0.0, res, store.meta.n0)
    for f in feats:
        if f.kind != "bridge" or not f.line_lonlat or len(f.line_lonlat) < 2:
            continue
        line = ring_to_enu(anchor, f.line_lonlat)
        # Ground at endpoints (relative heights already).
        def _g(pt):
            r = int(np.clip((pt[1] - store.meta.n0) / res, 0, store.meta.height - 1))
            c = int(np.clip((pt[0] - store.meta.e0) / res, 0, store.meta.width - 1))
            return float(store.ground[r, c])

        deck = max(_g(line[0]), _g(line[-1])) + clearance_m
        strip = LineString(line).buffer(max(f.width_m, 2.0) / 2.0)
        burned = rio_features.rasterize(
            [(strip, 1)], out_shape=(store.meta.height, store.meta.width),
            transform=transform, dtype="uint8",
        ).astype(bool)
        if not burned.any():
            continue
        # Deck as building-layer height above LOCAL ground.
        rows, cols = np.nonzero(burned)
        rel = deck - np.asarray(store.ground[rows, cols])
        store.build_h[rows, cols] = np.maximum(store.build_h[rows, cols], np.clip(rel, 0.5, 80.0))
        n += 1
    if n:
        log.info(f"stamped {n} bridge decks")
    return n


def stamp_rail_embankment(store: GisStore, ridge_m: float = 0.7) -> int:
    """Small embankment under rail texels (visible geometry cue)."""
    m = np.asarray(store.class_id) == GROUND_CLASSES["rail"]
    count = int(m.sum())
    if count:
        rows, cols = np.nonzero(m)
        store.ground[rows, cols] = np.asarray(store.ground[rows, cols]) + ridge_m
    return count
