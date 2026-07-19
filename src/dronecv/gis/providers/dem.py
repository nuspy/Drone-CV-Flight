"""Copernicus GLO-30 DEM provider.

Public Cloud-Optimized GeoTIFFs on AWS S3 (no authentication):
    https://copernicus-dem-30m.s3.amazonaws.com/
    Copernicus_DSM_COG_10_<N|S>YY_00_<E|W>XXX_00_DEM/..._DEM.tif
(one tile per 1x1 degree; "10" is the GLO-30 product code). Elevations are
meters above the EGM2008 geoid — i.e. ~MSL, which is exactly what the
system's alt0/MSL convention expects.

Tiles are cached on disk; `sample_grid` mosaics whatever tiles the bbox
touches and bilinearly samples them onto the requested local ENU grid.
Offline tests inject `tile_dir` fixtures instead of downloading.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from dronecv.gis.geometry import BBox
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.dem")

S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_name(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def tiles_for_bbox(bbox: BBox) -> list[tuple[int, int]]:
    lats = range(math.floor(bbox.south), math.floor(bbox.north) + 1)
    lons = range(math.floor(bbox.west), math.floor(bbox.east) + 1)
    return [(la, lo) for la in lats for lo in lons]


class CopernicusDem:
    def __init__(self, cache_dir: Path | None = None, tile_dir: Path | None = None):
        """`tile_dir`: local directory with pre-downloaded/fixture tiles named
        <tile_name>.tif — when set, no network is touched."""
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "dronecv" / "gis" / "dem")
        self.tile_dir = tile_dir

    def _tile_path(self, lat: int, lon: int) -> Path:
        name = tile_name(lat, lon)
        if self.tile_dir is not None:
            path = self.tile_dir / f"{name}.tif"
            if not path.exists():
                raise FileNotFoundError(f"DEM fixture tile missing: {path}")
            return path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{name}.tif"
        if not path.exists():
            self._download(name, path)
        return path

    def _download(self, name: str, dest: Path) -> None:
        import httpx

        url = f"{S3_BASE}/{name}/{name}.tif"
        log.info(f"downloading DEM tile {name}")
        with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as resp:
            if resp.status_code == 404:
                # Ocean / not-yet-released tile: treat as sea level.
                log.warning(f"DEM tile {name} not available (404) — assuming 0 m")
                dest.with_suffix(".missing").touch()
                return
            resp.raise_for_status()
            tmp = dest.with_suffix(".part")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
            tmp.rename(dest)

    def sample_grid(
        self, bbox: BBox, lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        """Bilinear elevations (m MSL) for point arrays lats/lons (same shape)."""
        import rasterio

        out = np.zeros(lats.shape, dtype=np.float32)
        filled = np.zeros(lats.shape, dtype=bool)
        for la, lo in tiles_for_bbox(bbox):
            sel = (lats >= la) & (lats < la + 1) & (lons >= lo) & (lons < lo + 1)
            if not sel.any():
                continue
            try:
                path = self._tile_path(la, lo)
            except FileNotFoundError:
                raise
            if path.with_suffix(".missing").exists() and not path.exists():
                filled |= sel  # sea level 0
                continue
            with rasterio.open(path) as src:
                data = src.read(1)
                h, w = data.shape
                # Geographic transform of a 1x1 degree tile.
                inv = ~src.transform
                cols, rows = inv * (lons[sel], lats[sel])
                out[sel] = _bilinear(data, rows, cols)
                filled |= sel
        if not filled.all():
            log.warning(f"{(~filled).sum()} DEM samples outside available tiles — set to 0")
        return out


def _bilinear(data: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    h, w = data.shape
    r = np.clip(rows, 0, h - 1.001)
    c = np.clip(cols, 0, w - 1.001)
    r0 = r.astype(int)
    c0 = c.astype(int)
    fr, fc = r - r0, c - c0
    return (
        data[r0, c0] * (1 - fr) * (1 - fc)
        + data[r0, c0 + 1] * (1 - fr) * fc
        + data[r0 + 1, c0] * fr * (1 - fc)
        + data[r0 + 1, c0 + 1] * fr * fc
    ).astype(np.float32)
