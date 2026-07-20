"""Landmark archetype shaping: famous/POI buildings get recognizable 3D
profiles instead of flat LoD1 boxes.

Because the internal renderer is a heightfield raycaster, archetypes are
implemented as HEIGHT STAMPS on the building extrusion raster — the training
images see the same spire/dome/crenellation silhouette that the Unity/Blender
exports reproduce as meshes. `building:part` data (real per-part heights)
stamps as-is and wins over archetypes.

Archetypes: spire_tower (conical top +40%), cap_tower (rounded cap),
dome (central hemisphere), crenellated (perimeter teeth), crenellated_tower.
"""

from __future__ import annotations

import numpy as np
from rasterio import features as rio_features
from rasterio.transform import Affine

from dronecv.geo.anchor import GeoAnchor
from dronecv.gis.providers.buildings import Building
from dronecv.gis.providers.poi import BuildingPart, Poi
from dronecv.gis.raster import building_polygon
from dronecv.gis.store import GisStore
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.landmarks")


def _transform(store: GisStore) -> Affine:
    m = store.meta
    return Affine(m.res_m, 0.0, m.e0, 0.0, m.res_m, m.n0)


def match_pois_to_buildings(
    anchor: GeoAnchor, pois: list[Poi], buildings: list[Building], max_dist_m: float = 40.0
) -> list[tuple[Poi, Building]]:
    """Pair each shaped POI with the nearest building footprint."""
    from dronecv.gis.geometry import geodetic_to_enu_vec

    pairs = []
    if not buildings:
        return pairs
    centroids = []
    for b in buildings:
        poly = building_polygon(anchor, b)
        centroids.append(None if poly is None else (poly.centroid.x, poly.centroid.y))
    for poi in pois:
        if not poi.archetype:
            continue
        e, n = geodetic_to_enu_vec(
            np.array([poi.lonlat[1]]), np.array([poi.lonlat[0]]), anchor
        )
        best, best_d = None, max_dist_m**2
        for b, c in zip(buildings, centroids, strict=True):
            if c is None:
                continue
            d = (c[0] - float(e[0])) ** 2 + (c[1] - float(n[0])) ** 2
            if d < best_d:
                best, best_d = b, d
        if best is not None:
            pairs.append((poi, best))
    return pairs


def stamp_archetypes(
    store: GisStore, anchor: GeoAnchor, pairs: list[tuple[Poi, Building]]
) -> int:
    """Reshape matched buildings' height raster according to their archetype."""
    n_stamped = 0
    res = store.meta.res_m
    for poi, b in pairs:
        poly = building_polygon(anchor, b)
        if poly is None or b.height_m is None:
            continue
        minx, miny, maxx, maxy = poly.bounds
        r0 = max(0, int((miny - store.meta.n0) / res) - 2)
        r1 = min(store.meta.height, int((maxy - store.meta.n0) / res) + 3)
        c0 = max(0, int((minx - store.meta.e0) / res) - 2)
        c1 = min(store.meta.width, int((maxx - store.meta.e0) / res) + 3)
        if r1 <= r0 or c1 <= c0:
            continue
        window = (slice(r0, r1), slice(c0, c1))
        mask = rio_features.rasterize(
            [(poly, 1)], out_shape=(r1 - r0, c1 - c0),
            transform=_transform(store) * Affine.translation(c0, r0), dtype="uint8",
        ).astype(bool)
        if not mask.any():
            continue
        h = float(b.height_m)
        ys, xs = np.mgrid[r0:r1, c0:c1]
        ce = (poly.centroid.x - store.meta.e0) / res
        cn = (poly.centroid.y - store.meta.n0) / res
        rad = np.sqrt((xs - ce) ** 2 + (ys - cn) ** 2) * res
        max_rad = max(float(rad[mask].max()), res)
        profile = np.zeros_like(rad, dtype=np.float32)

        if poi.archetype in ("spire_tower",):
            profile = h * (1.0 + 0.45 * np.clip(1.0 - rad / max_rad, 0, 1))
        elif poi.archetype == "cap_tower":
            profile = h * (1.0 + 0.2 * np.sqrt(np.clip(1.0 - (rad / max_rad) ** 2, 0, 1)))
        elif poi.archetype == "dome":
            dome_r = max_rad * 0.6
            dome = 0.5 * h * np.sqrt(np.clip(1.0 - (rad / dome_r) ** 2, 0, 1))
            profile = h + dome
        elif poi.archetype in ("crenellated", "crenellated_tower"):
            # Perimeter teeth: every other ~2 m arc raised 15%.
            angle = np.arctan2(ys - cn, xs - ce)
            arc = (angle * max_rad / 2.0)
            teeth = ((np.floor(arc / max(2.0, 2 * res)) % 2) == 0) & (rad > max_rad * 0.72)
            profile = np.where(teeth, h * 1.15, h).astype(np.float32)
        else:
            continue

        region = store.build_h[window]
        store.build_h[window] = np.where(mask, np.maximum(region, profile), region)
        n_stamped += 1
        log.info(f"stamped archetype {poi.archetype} on '{poi.name or b.osm_id}' (h={h:.0f} m)")
    return n_stamped


def stamp_building_parts(store: GisStore, anchor: GeoAnchor, parts: list[BuildingPart]) -> int:
    """Real LoD detail: each building:part sets its own height where mapped."""
    shapes = []
    for p in parts:
        from shapely.geometry import Polygon

        from dronecv.gis.geometry import ring_to_enu

        if len(p.footprint_lonlat) < 4:
            continue
        poly = Polygon(ring_to_enu(anchor, p.footprint_lonlat))
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty:
            shapes.append((poly, float(min(p.height_m, 400.0))))
    if not shapes:
        return 0
    heights = rio_features.rasterize(
        shapes, out_shape=(store.meta.height, store.meta.width),
        transform=_transform(store), fill=0.0, dtype="float32",
    )
    mask = heights > 0
    store.build_h[mask] = np.maximum(store.build_h[mask], heights[mask])
    return len(shapes)
