"""Rasterization of GIS features into the store mosaics."""

from __future__ import annotations

import numpy as np
from rasterio import features as rio_features
from rasterio.transform import Affine
from shapely.geometry import LineString, Polygon

from dronecv.geo.anchor import GeoAnchor
from dronecv.gis.geometry import ring_to_enu
from dronecv.gis.providers.buildings import Building
from dronecv.gis.providers.landcover import GROUND_CLASSES, LandcoverFeature
from dronecv.gis.store import BUILDING_CLASS_IDS, GisStore
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.raster")

DEFAULT_CLASS_HEIGHTS = {
    "generic": 6.0,
    "residential": 7.0,
    "industrial": 9.0,
    "commercial": 12.0,
    "landmark": 25.0,
}


def _transform(store: GisStore) -> Affine:
    """Affine mapping (col, row) -> ENU (E, N). Row 0 is the SOUTH edge, so
    the N step is positive (unusual for images, fine for rasterio)."""
    m = store.meta
    return Affine(m.res_m, 0.0, m.e0, 0.0, m.res_m, m.n0)


def building_polygon(anchor: GeoAnchor, b: Building) -> Polygon | None:
    outer = ring_to_enu(anchor, b.footprint_lonlat)
    if len(outer) < 4:
        return None
    holes = [ring_to_enu(anchor, h) for h in b.holes_lonlat if len(h) >= 4]
    poly = Polygon(outer, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return None if poly.is_empty else poly


def rasterize_buildings(store: GisStore, anchor: GeoAnchor, buildings: list[Building]) -> dict:
    """Burn building heights + classes into the store. Heights still missing
    after the tag/shadow chain fall back to per-class defaults."""
    height_shapes = []
    class_shapes = []
    n_default = 0
    for b in buildings:
        poly = building_polygon(anchor, b)
        if poly is None:
            continue
        h = b.height_m
        if h is None:
            h = DEFAULT_CLASS_HEIGHTS.get(b.building_class, 6.0)
            b.height_m = h
            b.height_source = "class_default"
            n_default += 1
        height_shapes.append((poly, float(min(h, 400.0))))
        class_shapes.append((poly, BUILDING_CLASS_IDS.get(b.building_class, 10)))
    if not height_shapes:
        return {"n_rasterized": 0, "n_class_default": 0}

    shape = (store.meta.height, store.meta.width)
    transform = _transform(store)
    heights = rio_features.rasterize(
        height_shapes, out_shape=shape, transform=transform, fill=0.0, dtype="float32",
        all_touched=False,
    )
    classes = rio_features.rasterize(
        class_shapes, out_shape=shape, transform=transform, fill=0, dtype="uint8",
    )
    store.build_h[:] = np.maximum(store.build_h, heights)
    mask = classes > 0
    store.class_id[mask] = classes[mask]
    log.info(f"rasterized {len(height_shapes)} buildings ({n_default} class-default heights)")
    return {"n_rasterized": len(height_shapes), "n_class_default": n_default}


def rasterize_landcover(store: GisStore, anchor: GeoAnchor, feats: list[LandcoverFeature]) -> int:
    """Burn ground classes (under buildings: buildings win, burned later)."""
    shapes = []
    for f in feats:
        cid = GROUND_CLASSES.get(f.kind)
        if cid is None or cid == 0:
            continue
        if f.ring_lonlat is not None:
            ring = ring_to_enu(anchor, f.ring_lonlat)
            if len(ring) < 4:
                continue
            poly = Polygon(ring)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty:
                shapes.append((poly, cid))
        elif f.line_lonlat is not None and len(f.line_lonlat) >= 2:
            line = LineString(ring_to_enu(anchor, f.line_lonlat))
            shapes.append((line.buffer(max(f.width_m, 1.0) / 2.0), cid))
    if not shapes:
        return 0
    burned = rio_features.rasterize(
        shapes,
        out_shape=(store.meta.height, store.meta.width),
        transform=_transform(store),
        fill=0,
        dtype="uint8",
    )
    mask = burned > 0
    store.class_id[mask] = burned[mask]
    log.info(f"rasterized {len(shapes)} landcover features")
    return len(shapes)
