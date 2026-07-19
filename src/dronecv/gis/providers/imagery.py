"""Orthophoto handling (optional layer).

An orthophoto serves two purposes:
- shadow-based height inference for buildings with no tagged height
  (requires the acquisition UTC datetime — shadows move with the sun);
- optional albedo draping over the ground (shape stays the primary signal).

Sources: a user-supplied GeoTIFF (best: regional orthophotos <1 m/px with a
known acquisition time), or downloads from Sentinel-2 (Copernicus Data Space,
free account; 10 m/px — only useful for tall structures) / EOX Sentinel-2
cloudless (WMS, non-commercial, no timestamp so NO shadow inference).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from dronecv.geo.anchor import GeoAnchor
from dronecv.gis.geometry import geodetic_to_enu_vec


@dataclass
class OrthoImage:
    """Grayscale orthophoto resampled onto the local ENU grid.

    `gray[r, c]`: row 0 = south edge (same convention as the GIS store),
    luminance in [0, 1]. `utc` is the acquisition time (needed for shadows).
    """

    gray: np.ndarray
    res_m: float
    e0: float
    n0: float
    utc: datetime | None = None

    def sample(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        r = np.clip((np.asarray(n) - self.n0) / self.res_m, 0, self.gray.shape[0] - 1)
        c = np.clip((np.asarray(e) - self.e0) / self.res_m, 0, self.gray.shape[1] - 1)
        return self.gray[np.round(r).astype(int), np.round(c).astype(int)]

    def contains(self, e: float, n: float) -> bool:
        return (
            self.e0 <= e < self.e0 + self.gray.shape[1] * self.res_m
            and self.n0 <= n < self.n0 + self.gray.shape[0] * self.res_m
        )


def load_geotiff_ortho(
    path: Path, anchor: GeoAnchor, utc: datetime | None, target_res_m: float = 1.0
) -> OrthoImage:
    """Load a GeoTIFF orthophoto (any CRS rasterio can read; commonly WGS84 or
    UTM) and resample it onto the local ENU grid around the anchor."""
    import rasterio
    from rasterio.warp import transform as rio_transform

    with rasterio.open(path) as src:
        data = src.read()
        gray = data.mean(axis=0).astype(np.float32)
        gray /= max(float(gray.max()), 1e-6)
        # Corner coordinates -> ENU extent.
        rows = [0, src.height]
        cols = [0, src.width]
        xs, ys = [], []
        for r in rows:
            for c in cols:
                x, y = src.transform * (c, r)
                xs.append(x)
                ys.append(y)
        lon, lat = rio_transform(src.crs, "EPSG:4326", xs, ys)
        e, n = geodetic_to_enu_vec(np.array(lat), np.array(lon), anchor)
        e0, e1 = float(e.min()), float(e.max())
        n0, n1 = float(n.min()), float(n.max())

        out_w = max(2, int((e1 - e0) / target_res_m))
        out_h = max(2, int((n1 - n0) / target_res_m))
        # Sample the source at each target texel center (nearest): build
        # lat/lon of targets via inverse ENU (small-area approximation is NOT
        # used — pymap3d exact inverse via anchor).
        ee = e0 + (np.arange(out_w) + 0.5) * (e1 - e0) / out_w
        nn = n0 + (np.arange(out_h) + 0.5) * (n1 - n0) / out_h
        ge, gn = np.meshgrid(ee, nn)
        import pymap3d

        lat_t, lon_t, _ = pymap3d.enu2geodetic(
            ge.ravel(), gn.ravel(), np.zeros(ge.size), anchor.lat0, anchor.lon0, anchor.alt0
        )
        x_t, y_t = rio_transform("EPSG:4326", src.crs, list(lon_t), list(lat_t))
        inv = ~src.transform
        cc, rr = inv * (np.array(x_t), np.array(y_t))
        rr = np.clip(np.round(rr).astype(int), 0, src.height - 1)
        cc = np.clip(np.round(cc).astype(int), 0, src.width - 1)
        resampled = gray[rr, cc].reshape(out_h, out_w)

    return OrthoImage(
        gray=resampled,
        res_m=(e1 - e0) / out_w,
        e0=e0,
        n0=n0,
        utc=utc,
    )
