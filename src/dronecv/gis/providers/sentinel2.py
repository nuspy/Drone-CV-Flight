"""Sentinel-2 L2A multi-date imagery from the public AWS COG bucket.

The `sentinel-cogs` bucket (no keys, plain HTTPS) holds every Sentinel-2
L2A scene as cloud-optimized GeoTIFFs, organized by MGRS tile and date:

    sentinel-s2-l2a-cogs/{zone}/{band}/{square}/{year}/{month}/{SCENE}/TCI.tif

Different DATES of the same area are therefore directly navigable — exactly
what cloud-hole filling needs: list recent scenes, read only the AOI window
of each (HTTP range requests), and let `gis.clouds.composite_scenes` build
the cloud-free composite. Resolution is 10 m/px (true color).

Data: Copernicus Sentinel-2, ESA — free including commercial use with
attribution ("contains modified Copernicus Sentinel data").
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import numpy as np

from dronecv.geo.anchor import GeoAnchor
from dronecv.gis.geometry import BBox, enu_bounds
from dronecv.gis.providers.imagery import OrthoImage
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.sentinel2")

BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

BAND_LETTERS = "CDEFGHJKLMNPQRSTUVWX"  # 8-deg latitude bands, -80..+72
COL_SETS = ["ABCDEFGH", "JKLMNPQR", "STUVWXYZ"]
ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"


def mgrs_tile(lat: float, lon: float) -> tuple[int, str, str]:
    """(utm_zone, band_letter, 100km_square) for a point — e.g. Budapest
    (47.5, 19.04) -> (34, 'T', 'CT')."""
    import pyproj

    zone = int((lon + 180.0) // 6.0) + 1
    band = BAND_LETTERS[min(len(BAND_LETTERS) - 1, max(0, int((lat + 80.0) // 8.0)))]
    utm = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{32600 + zone}", always_xy=True)
    e, n = utm.transform(lon, lat)
    if lat < 0:
        n += 10_000_000.0
    col = COL_SETS[(zone - 1) % 3][int(e // 100_000) - 1]
    row_idx = int(n // 100_000) % 20
    if zone % 2 == 0:
        row_idx = (row_idx + 5) % 20
    return zone, band, col + ROW_LETTERS[row_idx]


def list_scenes(
    zone: int, band: str, square: str,
    until: datetime | None = None,
    months_back: int = 8,
) -> list[tuple[str, str]]:
    """[(scene_prefix, yyyymmdd)] newest first, walking months backwards."""
    import httpx

    until = until or datetime.now(tz=UTC)
    scenes: list[tuple[str, str]] = []
    year, month = until.year, until.month
    with httpx.Client(timeout=30.0) as client:
        for _ in range(months_back):
            prefix = f"sentinel-s2-l2a-cogs/{zone}/{band}/{square}/{year}/{month}/"
            resp = client.get(f"{BUCKET}/?list-type=2&prefix={prefix}&delimiter=/")
            resp.raise_for_status()
            for el in ET.fromstring(resp.text).iter(f"{NS}Prefix"):
                p = el.text or ""
                if p == prefix:
                    continue
                name = p.rstrip("/").rsplit("/", 1)[-1]  # S2A_34TCT_20260619_0_L2A
                parts = name.split("_")
                if len(parts) >= 3 and len(parts[2]) == 8:
                    scenes.append((p.rstrip("/"), parts[2]))
            month -= 1
            if month == 0:
                year, month = year - 1, 12
    scenes.sort(key=lambda s: s[1], reverse=True)
    return scenes


def fetch_scene_ortho(
    scene_prefix: str, bbox: BBox, anchor: GeoAnchor, res_m: float = 10.0
) -> OrthoImage:
    """Read only the AOI window of a scene's TCI (true color, 10 m) and
    resample it onto the local ENU grid."""
    import pymap3d
    import rasterio
    from rasterio.warp import transform as rio_transform
    from rasterio.windows import from_bounds

    e_min, n_min, e_max, n_max = enu_bounds(anchor, bbox)
    w = max(2, int((e_max - e_min) / res_m))
    h = max(2, int((n_max - n_min) / res_m))
    ee = e_min + (np.arange(w) + 0.5) * res_m
    nn = n_min + (np.arange(h) + 0.5) * res_m
    ge, gn = np.meshgrid(ee, nn)
    lat, lon, _ = pymap3d.enu2geodetic(
        ge.ravel(), gn.ravel(), np.zeros(ge.size), anchor.lat0, anchor.lon0, anchor.alt0
    )
    url = f"{BUCKET}/{scene_prefix}/TCI.tif"
    with rasterio.open(url) as src:
        xs, ys = rio_transform("EPSG:4326", src.crs, list(lon), list(lat))
        xs, ys = np.asarray(xs), np.asarray(ys)
        pad = 12 * float(src.res[0])
        win = from_bounds(xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad,
                          src.transform)
        data = src.read(window=win, boundless=True, fill_value=0)
        inv = ~src.window_transform(win)
        cc, rr = inv * (xs, ys)
        rr = np.clip(np.round(rr).astype(int), 0, data.shape[1] - 1)
        cc = np.clip(np.round(cc).astype(int), 0, data.shape[2] - 1)
        rgb = np.moveaxis(data[:3, rr, cc].reshape(3, h, w), 0, -1).astype(np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    name = scene_prefix.rsplit("/", 1)[-1]
    date = name.split("_")[2]
    utc = datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), 10, 0, tzinfo=UTC)
    return OrthoImage(gray=gray, res_m=res_m, e0=e_min, n0=n_min, utc=utc, rgb=rgb)


def fetch_cloudfree_ortho(
    bbox: BBox,
    anchor: GeoAnchor,
    res_m: float = 10.0,
    max_scenes: int = 8,
    until: datetime | None = None,
):
    """Cloud-free composite of the AOI from recent Sentinel-2 dates.
    Returns (OrthoImage with utc=None, CompositeStats)."""
    from dronecv.gis.clouds import HOLE_DONE_FRACTION, cloud_mask, composite_scenes

    center_lat = (bbox.south + bbox.north) / 2.0
    center_lon = (bbox.west + bbox.east) / 2.0
    zone, band, square = mgrs_tile(center_lat, center_lon)
    scenes = list_scenes(zone, band, square, until=until)
    if not scenes:
        raise RuntimeError(f"no Sentinel-2 scenes found for tile {zone}{band}{square}")
    log.info(f"sentinel-2 tile {zone}{band}{square}: {len(scenes)} scenes available")

    # Fetch lazily: stop as soon as the composite is (almost) hole-free.
    fetched: list[tuple[OrthoImage, str]] = []
    composite = stats = None
    for prefix, date in scenes[:max_scenes]:
        try:
            ortho = fetch_scene_ortho(prefix, bbox, anchor, res_m)
        except Exception as e:  # noqa: BLE001 — a broken scene must not kill the run
            log.warning(f"scene {date} unreadable: {e}")
            continue
        fetched.append((ortho, date))
        composite, stats = composite_scenes(fetched)
        if stats.final_hole_fraction <= HOLE_DONE_FRACTION:
            break
        # A first scene that is mostly cloud makes a poor base: try to find a
        # clearer base among what we fetched so far.
        if len(fetched) > 1 and stats.first_scene_cloud_fraction > 0.5:
            fetched.sort(key=lambda s: float(cloud_mask(s[0]).mean()))
            composite, stats = composite_scenes(fetched)
    if composite is None:
        raise RuntimeError("no readable Sentinel-2 scene")
    return composite, stats
