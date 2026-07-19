"""Building height inference from satellite/aerial shadows.

For a building with no tagged height, the length of its shadow at a known
UTC time pins its height:

    height = shadow_length * tan(sun_elevation)

with the sun azimuth/elevation from the same NOAA ephemeris the rest of the
system uses (`dronecv.geo.celestial`) at the orthophoto's acquisition time
and the AOI's location — the "artificio" of using shadow length at a given
hour/season/latitude, made systematic.

Method per building:
1. shadow direction = sun azimuth + 180 (ENU);
2. pick boundary points on the down-sun side of the footprint;
3. from each, march along the shadow direction sampling orthophoto luminance;
   the shadow ends where luminance recovers above a locally estimated ground
   reference; samples that run into another footprint are discarded
   (occlusion);
4. robust length = median of surviving rays; gates on sample count, contrast
   and plausible range.

Accuracy scales with the orthophoto: at 0.2-1 m/px this resolves ordinary
houses; at Sentinel-2's 10 m/px only structures taller than ~15-20 m.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep

from dronecv.geo import celestial
from dronecv.geo.anchor import GeoAnchor
from dronecv.gis.providers.buildings import Building
from dronecv.gis.providers.imagery import OrthoImage
from dronecv.gis.raster import building_polygon
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.shadows")

MIN_SUN_ELEVATION_DEG = 12.0
MAX_SHADOW_M = 150.0
SHADOW_LUM_FACTOR = 0.72  # pixel counts as shadow if below this x ground ref
MIN_EDGE_SAMPLES = 3


def estimate_heights_from_shadows(
    buildings: list[Building],
    anchor: GeoAnchor,
    ortho: OrthoImage,
    max_height_m: float = 300.0,
) -> int:
    """Fill `height_m` (source="shadow") for buildings lacking real heights.
    Returns the number of buildings estimated."""
    if ortho.utc is None:
        log.warning("orthophoto has no acquisition time — shadow inference skipped")
        return 0
    sun = celestial.sun_position(ortho.utc, anchor.lat0, anchor.lon0)
    if sun.elevation_deg < MIN_SUN_ELEVATION_DEG:
        log.warning(f"sun too low at ortho time ({sun.elevation_deg:.1f} deg) — skipped")
        return 0
    tan_el = math.tan(math.radians(sun.elevation_deg))
    az_shadow = math.radians((sun.azimuth_deg + 180.0) % 360.0)
    d_e, d_n = math.sin(az_shadow), math.cos(az_shadow)

    polys = []
    for b in buildings:
        poly = building_polygon(anchor, b)
        polys.append(poly)
    prepared = [prep(p) if p is not None else None for p in polys]

    step = max(ortho.res_m, 0.5)
    n_estimated = 0
    for bi, b in enumerate(buildings):
        if b.height_source in ("tag_height", "tag_levels") or polys[bi] is None:
            continue
        poly = polys[bi]
        cx, cy = poly.centroid.x, poly.centroid.y
        if not ortho.contains(cx, cy):
            continue

        # Ground luminance reference: ring around the building, up-sun side
        # (never in this building's own shadow).
        ring_pts = _ring_points(poly, dist=6.0 * step, n=24)
        up_sun = [
            (e, n) for e, n in ring_pts
            if (e - cx) * d_e + (n - cy) * d_n < 0 and not _in_any(prepared, e, n, skip=bi)
        ]
        if len(up_sun) < 4:
            continue
        ground_ref = float(
            np.median(ortho.sample(np.array([p[0] for p in up_sun]), np.array([p[1] for p in up_sun])))
        )
        if ground_ref < 0.15:  # scene too dark to discriminate
            continue
        thresh = ground_ref * SHADOW_LUM_FACTOR

        # Down-sun boundary points.
        edge_pts = [
            (e, n) for e, n in _ring_points(poly, dist=0.6 * step, n=36)
            if (e - cx) * d_e + (n - cy) * d_n > 0
        ]
        lengths = []
        for e0, n0 in edge_pts:
            length = _march_shadow(
                ortho, prepared, bi, e0, n0, d_e, d_n, step, thresh
            )
            if length is not None:
                lengths.append(length)
        if len(lengths) < MIN_EDGE_SAMPLES:
            continue
        shadow_len = float(np.median(lengths))
        if not (step <= shadow_len <= MAX_SHADOW_M):
            continue
        height = shadow_len * tan_el
        if 2.0 <= height <= max_height_m:
            b.height_m = height
            b.height_source = "shadow"
            n_estimated += 1

    log.info(f"shadow inference: estimated heights for {n_estimated} buildings "
             f"(sun el {sun.elevation_deg:.1f} deg, az {sun.azimuth_deg:.1f})")
    return n_estimated


def _ring_points(poly, dist: float, n: int) -> list[tuple[float, float]]:
    outer = poly.exterior if hasattr(poly, "exterior") else None
    if outer is None:
        return []
    buffered = poly.buffer(dist).exterior
    return [
        (buffered.interpolate(i / n, normalized=True).x, buffered.interpolate(i / n, normalized=True).y)
        for i in range(n)
    ]


def _in_any(prepared, e: float, n: float, skip: int) -> bool:
    pt = Point(e, n)
    for i, p in enumerate(prepared):
        if i != skip and p is not None and p.contains(pt):
            return True
    return False


def _march_shadow(
    ortho: OrthoImage, prepared, skip: int, e0: float, n0: float,
    d_e: float, d_n: float, step: float, thresh: float,
) -> float | None:
    """Length of continuous shadow from (e0, n0) along the shadow direction.
    None if the ray is invalid (occluded by another building, out of image,
    or no shadow at all)."""
    lit_run = 0
    length = 0.0
    saw_shadow = False
    for i in range(1, int(MAX_SHADOW_M / step) + 1):
        e, n = e0 + d_e * step * i, n0 + d_n * step * i
        if not ortho.contains(e, n):
            return None
        if _in_any(prepared, e, n, skip):
            return None  # another building interrupts: unreliable ray
        lum = float(ortho.sample(np.array(e), np.array(n)))
        if lum < thresh:
            saw_shadow = True
            lit_run = 0
            length = step * i
        else:
            lit_run += 1
            if lit_run >= 2:  # shadow ended
                break
    return length if saw_shadow else None
