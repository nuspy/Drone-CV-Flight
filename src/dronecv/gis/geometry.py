"""Geodetic helpers for the GIS pipeline.

All GIS worlds are built ENU-aligned: sim x = East, sim z = North, sim y = Up,
`true_north_offset_deg = 0`, anchor at the AOI center. That makes the GPS
output of the localizer directly real-world.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymap3d

from dronecv.config import AnchorConfig
from dronecv.geo.anchor import GeoAnchor


@dataclass(frozen=True)
class BBox:
    """Geographic bounding box (degrees)."""

    south: float
    west: float
    north: float
    east: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.south + self.north) / 2.0, (self.west + self.east) / 2.0

    def margin(self, deg: float) -> BBox:
        return BBox(self.south - deg, self.west - deg, self.north + deg, self.east + deg)


def parse_bbox(text: str) -> BBox:
    """Parse "lat1,lon1,lat2,lon2" in any corner order."""
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be lat1,lon1,lat2,lon2")
    lats, lons = sorted((parts[0], parts[2])), sorted((parts[1], parts[3]))
    return BBox(lats[0], lons[0], lats[1], lons[1])


def anchor_for(bbox: BBox, ground_alt0: float = 0.0) -> GeoAnchor:
    lat0, lon0 = bbox.center
    return GeoAnchor.resolve(
        AnchorConfig(lat0=lat0, lon0=lon0, alt0=ground_alt0, true_north_offset_deg=0.0), None
    )


def geodetic_to_enu_vec(
    lats: np.ndarray, lons: np.ndarray, anchor: GeoAnchor
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized geodetic -> local ENU (E, N) at ground level."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    e, n, _u = pymap3d.geodetic2enu(
        lats, lons, np.zeros_like(lats), anchor.lat0, anchor.lon0, anchor.alt0
    )
    return np.asarray(e), np.asarray(n)


def enu_bounds(anchor: GeoAnchor, bbox: BBox) -> tuple[float, float, float, float]:
    """(e_min, n_min, e_max, n_max) of the bbox corners in local ENU."""
    lats = np.array([bbox.south, bbox.south, bbox.north, bbox.north])
    lons = np.array([bbox.west, bbox.east, bbox.west, bbox.east])
    e, n = geodetic_to_enu_vec(lats, lons, anchor)
    return float(e.min()), float(n.min()), float(e.max()), float(n.max())


def ring_to_enu(anchor: GeoAnchor, lonlat_ring: list[tuple[float, float]]) -> np.ndarray:
    """GeoJSON-style (lon, lat) ring -> (N, 2) array of ENU (E, N)."""
    lons = np.array([p[0] for p in lonlat_ring], dtype=float)
    lats = np.array([p[1] for p in lonlat_ring], dtype=float)
    e, n = geodetic_to_enu_vec(lats, lons, anchor)
    return np.stack([e, n], axis=1)
