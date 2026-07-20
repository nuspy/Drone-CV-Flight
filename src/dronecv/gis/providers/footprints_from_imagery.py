"""Building footprint RECONSTRUCTION from satellite/aerial imagery.

GIS vector data (OSM/Overture) can have gaps — unmapped districts, new
construction, rural areas. This module extracts additional footprints from an
orthophoto and merges them into the GIS set, so coverage is bounded by what
the imagery shows rather than by what mappers have traced. Heights for the
reconstructed footprints come from the existing shadow chain
(`gis.shadow_heights`, needs the acquisition UTC) or the neighbor-median /
class defaults in `gis.raster`.

Pipeline: orthophoto -> building mask -> polygonization -> Building objects
(height unresolved) -> `merge_footprints` keeps only those not already
covered by GIS footprints.

Segmentation honesty: the built-in extractor is a classical CV chain
(contrast normalization, brightness/edge evidence, morphology, shape and
optional shadow validation). On sharp sub-meter imagery it recovers block
and house footprints; on 10 m Sentinel-2 only large structures survive the
area gate. A learned model can be plugged in via `mask_fn` (any callable
image->probability mask, e.g. an HF segmentation model) without changing the
rest of the chain.

Imagery sources here (both return `OrthoImage`, so shadow inference and
draping work unchanged):
- `fetch_wms_ortho`  — any WMS 1.1.1 GetMap endpoint (default: EOX Sentinel-2
  cloudless, free for non-commercial use, ~10 m/px, composite -> NO
  acquisition time, so no shadow heights from it);
- `fetch_xyz_ortho`  — any XYZ tile template (e.g. Esri World Imagery at
  z=18-19, ~0.3-0.6 m/px). OPT-IN: the caller must pass the URL template
  explicitly and is responsible for the provider's terms of service.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from dronecv.geo.anchor import GeoAnchor
from dronecv.gis.geometry import BBox
from dronecv.gis.providers.buildings import Building
from dronecv.gis.providers.imagery import OrthoImage
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.footprints")

EOX_WMS = "https://tiles.maps.eox.at/wms"
EOX_LAYER = "s2cloudless-2024_3857"

MIN_AREA_M2 = 40.0
MAX_AREA_M2 = 60_000.0
MIN_RECT_FILL = 0.40  # blob area / min-rotated-rect area: rejects vegetation blobs


# --------------------------------------------------------------- imagery I/O


def _enu_grid_lonlat(anchor: GeoAnchor, e0, n0, res_m, w, h):
    import pymap3d

    ee = e0 + (np.arange(w) + 0.5) * res_m
    nn = n0 + (np.arange(h) + 0.5) * res_m
    ge, gn = np.meshgrid(ee, nn)
    lat, lon, _ = pymap3d.enu2geodetic(
        ge.ravel(), gn.ravel(), np.zeros(ge.size), anchor.lat0, anchor.lon0, anchor.alt0
    )
    return np.asarray(lat), np.asarray(lon), ge.shape


def fetch_wms_ortho(
    bbox: BBox,
    anchor: GeoAnchor,
    res_m: float = 10.0,
    wms_url: str = EOX_WMS,
    layer: str = EOX_LAYER,
    timeout_s: float = 120.0,
) -> OrthoImage:
    """GetMap a WMS layer over the bbox, resampled to the local ENU grid.
    Default endpoint is EOX Sentinel-2 cloudless (attribution required,
    non-commercial). The returned image has NO acquisition time."""
    import cv2
    import httpx

    from dronecv.gis.geometry import enu_bounds

    e_min, n_min, e_max, n_max = enu_bounds(anchor, bbox)
    w = max(2, int((e_max - e_min) / res_m))
    h = max(2, int((n_max - n_min) / res_m))
    params = {
        "service": "WMS", "version": "1.1.1", "request": "GetMap",
        "layers": layer, "styles": "", "format": "image/jpeg",
        "srs": "EPSG:4326", "width": min(w, 4096), "height": min(h, 4096),
        "bbox": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}",
    }
    resp = httpx.get(wms_url, params=params, timeout=timeout_s)
    resp.raise_for_status()
    img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"WMS response is not an image ({resp.headers.get('content-type')})")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # WMS row 0 = north; the store convention is row 0 = south.
    rgb = rgb[::-1].copy()
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    log.info(f"WMS ortho {w}x{h} @ {res_m} m/px from {wms_url}")
    return OrthoImage(gray=gray, res_m=res_m, e0=e_min, n0=n_min, utc=None, rgb=rgb)


def fetch_xyz_ortho(
    bbox: BBox,
    anchor: GeoAnchor,
    url_template: str,
    zoom: int = 18,
    timeout_s: float = 30.0,
) -> OrthoImage:
    """Stitch XYZ tiles ({z}/{y}/{x} or {z}/{x}/{y} template) over the bbox.

    OPT-IN high-resolution source: the caller supplies the template and owns
    compliance with the tile provider's terms of use."""
    import cv2
    import httpx

    from dronecv.gis.geometry import enu_bounds

    def tile_of(lat, lon, z):
        n = 2**z
        x = int((lon + 180.0) / 360.0 * n)
        lat_r = math.radians(lat)
        y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
        return x, y

    x0, y1 = tile_of(bbox.south, bbox.west, zoom)
    x1, y0 = tile_of(bbox.north, bbox.east, zoom)
    xs = range(min(x0, x1), max(x0, x1) + 1)
    ys = range(min(y0, y1), max(y0, y1) + 1)
    n_tiles = len(xs) * len(ys)
    if n_tiles > 400:
        raise ValueError(f"{n_tiles} tiles at z{zoom} — lower the zoom or shrink the bbox")
    rows = []
    with httpx.Client(timeout=timeout_s) as client:
        for ty in ys:
            row = []
            for tx in xs:
                url = url_template.format(z=zoom, x=tx, y=ty)
                r = client.get(url)
                r.raise_for_status()
                tile = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
                if tile is None:
                    raise RuntimeError(f"tile {url} is not an image")
                row.append(tile.astype(np.float32) / 255.0)
            rows.append(np.hstack(row))
    mosaic = np.vstack(rows)  # row 0 = north edge of tile y0

    # Web-mercator tile bounds of the mosaic.
    def merc_bounds(tx, ty, z):
        n = 2**z
        lon_w = tx / n * 360.0 - 180.0
        lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
        return lon_w, lat_n

    lon_w, lat_n = merc_bounds(min(xs), min(ys), zoom)
    lon_e, lat_s = merc_bounds(max(xs) + 1, max(ys) + 1, zoom)

    e_min, n_min, e_max, n_max = enu_bounds(anchor, bbox)
    # Ground resolution of one pixel at the center latitude (256 px tiles).
    res_m = 156543.03 * math.cos(math.radians((bbox.south + bbox.north) / 2)) / (2**zoom) / 256
    w = max(2, int((e_max - e_min) / res_m))
    h = max(2, int((n_max - n_min) / res_m))
    lat_t, lon_t, shape = _enu_grid_lonlat(anchor, e_min, n_min, res_m, w, h)
    # lat/lon -> mosaic pixel (web-mercator inverse, linear in lon / mercator y)
    n_tiles_axis = 2**zoom
    fx = (lon_t + 180.0) / 360.0 * n_tiles_axis
    fy = (1.0 - np.arcsinh(np.tan(np.radians(lat_t))) / math.pi) / 2.0 * n_tiles_axis
    px = (fx - min(xs)) * 256.0
    py = (fy - min(ys)) * 256.0
    px = np.clip(px, 0, mosaic.shape[1] - 1).astype(np.int32)
    py = np.clip(py, 0, mosaic.shape[0] - 1).astype(np.int32)
    gray = mosaic[py, px].reshape(shape)
    log.info(f"XYZ ortho {w}x{h} @ {res_m:.2f} m/px, {n_tiles} tiles z{zoom}")
    return OrthoImage(gray=gray, res_m=res_m, e0=e_min, n0=n_min, utc=None)


# ------------------------------------------------------------- segmentation


def building_mask(
    ortho: OrthoImage,
    sun_azimuth_deg: float | None = None,
    mask_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Boolean building mask on the ortho grid.

    `mask_fn` (image float [0,1] -> probability [0,1]) plugs in a learned
    model; otherwise the classical chain runs: CLAHE -> brightness deviation
    from the large-scale local mean + edge support -> morphology."""
    import cv2

    g = np.clip(ortho.gray, 0.0, 1.0).astype(np.float32)
    if mask_fn is not None:
        prob = np.asarray(mask_fn(g), dtype=np.float32)
        return prob > 0.5

    # Denoise BEFORE thresholding (per-pixel sensor noise must not become
    # candidates), then compare against a large-scale local ground reference:
    # roofs read brighter than their surroundings, shadows darker (excluded —
    # they get used as *validation*, not as buildings). Classical chain =
    # bright-roof detector; dark/complex roofs need the `mask_fn` model hook.
    blur = cv2.GaussianBlur(g, (0, 0), max(1.0, 1.0 / ortho.res_m))
    win = max(15, int(150.0 / ortho.res_m) | 1)
    ref = cv2.boxFilter(blur, -1, (win, win))
    # Threshold in LOCAL-dispersion units, not absolute: an exposure change
    # (composite strips, haze) scales the contrast too, and a fixed +0.10
    # would drop every roof in the darker strip. MAD ~ boxFilter of |dev|.
    mad = cv2.boxFilter(np.abs(blur - ref), -1, (win, win))
    cand = blur > ref + np.maximum(0.06, 2.5 * mad)

    k = max(3, int(3.0 / ortho.res_m) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    m = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    return m.astype(bool)


def _shadow_supported(
    ortho: OrthoImage, cx: float, cy: float, radius_px: float, sun_azimuth_deg: float
) -> bool:
    """True if a darker region sits on the anti-solar side of the blob."""
    az = math.radians(sun_azimuth_deg + 180.0)  # shadow direction, ENU
    dx, dy = math.sin(az), math.cos(az)  # ENU east / north -> col / row
    probes, refs = [], []
    for d in (radius_px + 2, radius_px + 4, radius_px + 7):
        px, py = int(cx + dx * d), int(cy + dy * d)
        rx, ry = int(cx - dx * d), int(cy - dy * d)
        if 0 <= py < ortho.gray.shape[0] and 0 <= px < ortho.gray.shape[1]:
            probes.append(float(ortho.gray[py, px]))
        if 0 <= ry < ortho.gray.shape[0] and 0 <= rx < ortho.gray.shape[1]:
            refs.append(float(ortho.gray[ry, rx]))
    if not probes or not refs:
        return True  # blob at the image edge: don't reject on missing evidence
    return float(np.median(probes)) < 0.85 * float(np.median(refs))


def extract_footprints(
    ortho: OrthoImage,
    anchor: GeoAnchor,
    sun_azimuth_deg: float | None = None,
    min_area_m2: float = MIN_AREA_M2,
    max_area_m2: float = MAX_AREA_M2,
    mask_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[Building]:
    """Extract candidate building footprints from an orthophoto.

    Returns Buildings with unresolved heights (the shadow / neighbor-median /
    default chain fills them later) and class \"generic\"."""
    import cv2
    import pymap3d
    from shapely.geometry import Polygon

    mask = building_mask(ortho, sun_azimuth_deg, mask_fn)
    res = ortho.res_m
    min_px = max(4, int(min_area_m2 / res**2))
    max_px = int(max_area_m2 / res**2)

    n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=4
    )
    out: list[Building] = []
    n_rejected_shape = n_rejected_shadow = 0
    for lbl in range(1, n_lbl):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if not (min_px <= area <= max_px):
            continue
        blob = (labels == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        rect_area = max(rect[1][0] * rect[1][1], 1.0)
        if cv2.contourArea(cnt) / rect_area < MIN_RECT_FILL:
            n_rejected_shape += 1
            continue
        if sun_azimuth_deg is not None and not _shadow_supported(
            ortho, centroids[lbl][0], centroids[lbl][1],
            0.5 * max(rect[1][0], rect[1][1]), sun_azimuth_deg,
        ):
            n_rejected_shadow += 1
            continue
        eps = max(1.5, 1.5 / res)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float64)
        if len(approx) < 3:
            continue
        # Pixel -> ENU (row 0 = south) -> lon/lat ring.
        e = ortho.e0 + (approx[:, 0] + 0.5) * res
        n = ortho.n0 + (approx[:, 1] + 0.5) * res
        if not Polygon(np.stack([e, n], axis=1)).is_valid:
            continue
        lat, lon, _ = pymap3d.enu2geodetic(
            e, n, np.zeros(len(e)), anchor.lat0, anchor.lon0, anchor.alt0
        )
        ring = [(float(lo), float(la)) for lo, la in zip(lon, lat, strict=True)]
        ring.append(ring[0])
        out.append(Building(footprint_lonlat=ring, building_class="generic",
                            height_source="none"))
    log.info(
        f"imagery footprints: {len(out)} extracted @ {res} m/px "
        f"({n_rejected_shape} shape-rejected, {n_rejected_shadow} shadow-rejected)"
    )
    return out


# -------------------------------------------------------------------- merge


def merge_footprints(
    gis_buildings: list[Building],
    extracted: list[Building],
    anchor: GeoAnchor,
    clearance_m: float = 2.0,
) -> int:
    """Append extracted footprints that do NOT overlap existing GIS ones
    (GIS vectors are authoritative where present). Returns how many were
    added; mutates `gis_buildings`."""
    from shapely.strtree import STRtree

    from dronecv.gis.raster import building_polygon

    existing = [p for b in gis_buildings if (p := building_polygon(anchor, b)) is not None]
    tree = STRtree(existing) if existing else None
    added = 0
    for b in extracted:
        poly = building_polygon(anchor, b)
        if poly is None:
            continue
        if tree is not None:
            hit = tree.query(poly.buffer(clearance_m))
            if any(existing[i].intersects(poly.buffer(clearance_m)) for i in np.atleast_1d(hit)):
                continue
        b.height_source = "none"
        gis_buildings.append(b)
        added += 1
    log.info(f"footprint merge: +{added} reconstructed (of {len(extracted)} extracted)")
    return added
