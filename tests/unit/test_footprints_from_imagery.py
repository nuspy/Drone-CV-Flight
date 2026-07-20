"""Footprint reconstruction from imagery: synthetic orthophoto with known
buildings -> extraction recall/precision, shadow validation, GIS merge."""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon

from dronecv.geo.anchor import GeoAnchor
from dronecv.gis.providers.footprints_from_imagery import (
    extract_footprints,
    merge_footprints,
)
from dronecv.gis.providers.imagery import OrthoImage
from dronecv.gis.raster import building_polygon

ANCHOR = GeoAnchor(lat0=43.32, lon0=11.33, alt0=0.0, true_north_offset_deg=0.0,
                   source="test")
RES = 0.5  # m/px — "good regional orthophoto"
SUN_AZ = 135.0  # sun from SE -> shadows point NW

# Ground-truth buildings in ENU meters (e0, n0, width_e, width_n).
TRUTH = [(-60.0, -50.0, 22.0, 14.0), (10.0, -40.0, 30.0, 18.0),
         (-40.0, 20.0, 16.0, 16.0), (35.0, 45.0, 26.0, 12.0)]


def synthetic_ortho(with_shadows: bool = True) -> OrthoImage:
    """200x200 m: mid-gray textured ground, bright roofs, NW shadows."""
    rng = np.random.default_rng(7)
    size = int(200 / RES)
    gray = 0.45 + rng.normal(0.0, 0.02, (size, size)).astype(np.float32)
    e0 = n0 = -100.0

    def rc(e, n):
        return int((n - n0) / RES), int((e - e0) / RES)

    az = math.radians(SUN_AZ + 180.0)
    dx, dy = math.sin(az), math.cos(az)  # shadow direction (E, N) -> (col, row)
    if with_shadows:  # cast first so roofs overwrite the overlap
        shadow = np.zeros_like(gray, dtype=bool)
        L = 18.0  # ~9 m tall building at ~26 deg sun elevation
        for be, bn, we, wn in TRUTH:
            r0, c0 = rc(be, bn)
            r1, c1 = rc(be + we, bn + wn)
            for step in np.arange(RES, L, RES):
                ro, co = int(round(dy * step / RES)), int(round(dx * step / RES))
                rr0, rr1 = np.clip([r0 + ro, r1 + ro], 0, gray.shape[0])
                cc0, cc1 = np.clip([c0 + co, c1 + co], 0, gray.shape[1])
                shadow[rr0:rr1, cc0:cc1] = True
        gray[shadow] = 0.26 + rng.normal(0.0, 0.01, int(shadow.sum())).astype(np.float32)
    for be, bn, we, wn in TRUTH:
        r0, c0 = rc(be, bn)
        r1, c1 = rc(be + we, bn + wn)
        gray[r0:r1, c0:c1] = 0.82 + rng.normal(0.0, 0.015, (r1 - r0, c1 - c0))
    return OrthoImage(gray=np.clip(gray, 0, 1), res_m=RES, e0=e0, n0=n0, utc=None)


def truth_polygons() -> list[Polygon]:
    return [Polygon([(e, n), (e + w, n), (e + w, n + h), (e, n + h)])
            for e, n, w, h in TRUTH]


def _enu_poly(b) -> Polygon:
    return building_polygon(ANCHOR, b)


class TestExtraction:
    def test_recovers_known_footprints(self):
        found = extract_footprints(synthetic_ortho(), ANCHOR, sun_azimuth_deg=SUN_AZ)
        assert len(found) >= 3, f"only {len(found)} footprints extracted"
        polys = [_enu_poly(b) for b in found]
        matched = 0
        for t in truth_polygons():
            best = max((p.intersection(t).area / p.union(t).area for p in polys), default=0.0)
            if best > 0.5:
                matched += 1
        assert matched >= 3, f"only {matched}/4 truth footprints matched with IoU>0.5"

    def test_precision_no_ghost_buildings(self):
        found = extract_footprints(synthetic_ortho(), ANCHOR, sun_azimuth_deg=SUN_AZ)
        truths = truth_polygons()
        ghosts = [
            b for b in found
            if max((_enu_poly(b).intersection(t).area / _enu_poly(b).area for t in truths),
                   default=0.0) < 0.3
        ]
        assert len(ghosts) <= 1, f"{len(ghosts)} ghost footprints (of {len(found)})"

    def test_flat_ground_yields_nothing(self):
        rng = np.random.default_rng(3)
        gray = (0.45 + rng.normal(0, 0.02, (400, 400))).astype(np.float32)
        ortho = OrthoImage(gray=np.clip(gray, 0, 1), res_m=RES, e0=-100, n0=-100, utc=None)
        found = extract_footprints(ortho, ANCHOR, sun_azimuth_deg=SUN_AZ)
        assert len(found) <= 1

    def test_heights_left_unresolved(self):
        found = extract_footprints(synthetic_ortho(), ANCHOR, sun_azimuth_deg=SUN_AZ)
        assert all(b.height_m is None for b in found)


class TestMerge:
    def test_merge_respects_existing_gis(self):
        found = extract_footprints(synthetic_ortho(), ANCHOR, sun_azimuth_deg=SUN_AZ)
        n_found = len(found)
        assert n_found >= 3
        # GIS already has (roughly) the first truth building -> that extraction
        # must be dropped, the rest added.
        import pymap3d

        e, n, w, h = TRUTH[0]
        ring_enu = [(e, n), (e + w, n), (e + w, n + h), (e, n + h), (e, n)]
        lat, lon, _ = pymap3d.enu2geodetic(
            np.array([p[0] for p in ring_enu]), np.array([p[1] for p in ring_enu]),
            np.zeros(5), ANCHOR.lat0, ANCHOR.lon0, ANCHOR.alt0,
        )
        from dronecv.gis.providers.buildings import Building

        gis = [Building(footprint_lonlat=list(zip(lon.tolist(), lat.tolist(), strict=True)),
                        height_m=9.0, height_source="tag_height")]
        added = merge_footprints(gis, found, ANCHOR)
        assert added < n_found  # the overlapping one was dropped
        assert len(gis) == 1 + added

    def test_merge_into_empty_gis_adds_all(self):
        found = extract_footprints(synthetic_ortho(), ANCHOR, sun_azimuth_deg=SUN_AZ)
        gis: list = []
        added = merge_footprints(gis, found, ANCHOR)
        assert added == len(found) == len(gis)
