"""Overture Maps provider (GeoParquet on public S3, no keys).

Alternative to Overpass when it is unreachable (e.g. filtered networks) or
when better height coverage is wanted. Reads only what the AOI needs:
S3 listing -> parquet footer scan (HTTP range requests) -> row groups whose
bbox statistics intersect the query -> WKB decode.

Data: Overture Maps Foundation (ODbL/CDLA-Permissive per theme).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dronecv.gis.geometry import BBox
from dronecv.gis.providers.buildings import CLASS_OF_TAG, Building
from dronecv.gis.providers.landcover import LandcoverFeature
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.overture")

BUCKET = "https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

ROAD_WIDTHS = {"motorway": 16.0, "trunk": 14.0, "primary": 10.0, "secondary": 8.0,
               "tertiary": 7.0, "residential": 6.0, "unknown": 5.0}


def latest_release() -> str:
    import httpx

    resp = httpx.get(f"{BUCKET}/?list-type=2&prefix=release/&delimiter=/", timeout=30.0)
    resp.raise_for_status()
    prefixes = [
        el.text for el in ET.fromstring(resp.text).iter(f"{NS}Prefix") if el.text != "release/"
    ]
    return sorted(prefixes)[-1].split("/")[1]


def list_theme_files(release: str, theme: str, type_: str) -> list[str]:
    import httpx

    prefix = f"release/{release}/theme={theme}/type={type_}/"
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}"
        if token:
            import urllib.parse

            url += "&continuation-token=" + urllib.parse.quote(token, safe="")
        resp = httpx.get(url, timeout=60.0)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        keys += [el.text for el in root.iter(f"{NS}Key") if el.text.endswith(".parquet")]
        token_el = root.find(f"{NS}NextContinuationToken")
        if token_el is None:
            break
        token = token_el.text
    return keys


class _FooterIndex:
    """Parquet footer scan with exactly TWO HTTP range requests per file
    (tail-8 for the footer length, then the footer itself) on a shared
    keep-alive connection pool — generic fsspec+pyarrow access makes dozens
    of small range reads per footer, which crawls behind CONNECT proxies."""

    def __init__(self):
        import threading

        import fsspec
        import httpx

        self.fs = fsspec.filesystem("https")  # used only for row-group data reads
        self._local = threading.local()
        self._httpx = httpx

    def _client(self):
        if not hasattr(self._local, "client"):
            self._local.client = self._httpx.Client(timeout=60.0)
        return self._local.client

    def _thread_fs(self):
        """fsspec HTTPFileSystem per thread: the sync wrapper owns an asyncio
        loop, and sharing one instance across worker threads is fragile."""
        import fsspec

        if not hasattr(self._local, "fs"):
            self._local.fs = fsspec.filesystem("https", skip_instance_cache=True)
        return self._local.fs

    def matching_row_groups(self, url: str, bbox: BBox) -> list[int]:
        import pyarrow.parquet as pq

        # NOTE: keep this on the fsspec + ParquetFile path. A "faster" variant
        # that fetched the raw footer bytes and called pq.read_metadata on a
        # tail-only / sparse file SEGFAULTED pyarrow 25 on Overture footers.
        try:
            with self._thread_fs().open(url, "rb", block_size=256 * 1024) as fh:
                pf = pq.ParquetFile(fh)
                meta_all = pf.metadata
                rg0 = meta_all.row_group(0)
            idx = {rg0.column(j).path_in_schema: j for j in range(rg0.num_columns)}
            want = []
            for rg in range(meta_all.num_row_groups):
                meta = meta_all.row_group(rg)
                xmin = meta.column(idx["bbox.xmin"]).statistics
                ymin = meta.column(idx["bbox.ymin"]).statistics
                xmax = meta.column(idx["bbox.xmax"]).statistics
                ymax = meta.column(idx["bbox.ymax"]).statistics
                # `is None` checks only: `None in (...)` invokes ==, which hits
                # pyarrow Statistics.__eq__(None) and SEGFAULTS pyarrow 25.
                if xmin is None or ymin is None or xmax is None or ymax is None:
                    want.append(rg)
                    continue
                if (xmax.max >= bbox.west and xmin.min <= bbox.east
                        and ymax.max >= bbox.south and ymin.min <= bbox.north):
                    want.append(rg)
            return want
        except Exception as e:  # noqa: BLE001
            log.warning(f"footer scan failed for {url.rsplit('/', 1)[-1]}: {e}")
            return []

    def read_rows(self, url: str, row_groups: list[int], columns: list[str], bbox: BBox):
        import pyarrow.parquet as pq

        with self.fs.open(url, "rb", block_size=32 * 1024 * 1024, cache_type="readahead") as fh:
            pf = pq.ParquetFile(fh)
            table = pf.read_row_groups(row_groups, columns=columns + ["bbox"])
        b = table.column("bbox").flatten()
        names = table.column("bbox").type
        cols = {names.field(i).name: b[i] for i in range(names.num_fields)}
        keep = (
            (np.asarray(cols["xmax"]) >= bbox.west) & (np.asarray(cols["xmin"]) <= bbox.east)
            & (np.asarray(cols["ymax"]) >= bbox.south) & (np.asarray(cols["ymin"]) <= bbox.north)
        )
        return table.filter(keep)


def _wkb_rings(geom_wkb: bytes):
    from shapely import wkb as swkb

    geom = swkb.loads(geom_wkb)
    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
    else:
        return []
    return polys


def _scan_chunk(args: tuple[list[str], tuple[float, float, float, float]]):
    """Process-pool worker: scan a chunk of files with its OWN fsspec/pyarrow
    state. Runs in a separate process because fsspec(https)+pyarrow footer
    reads are not reliable across threads (native crashes observed); a crashed
    worker only loses its chunk, which the parent retries sequentially."""
    keys, (s, w, n, e) = args
    bbox = BBox(s, w, n, e)
    index = _FooterIndex()
    out = []
    for key in keys:
        rgs = index.matching_row_groups(f"{BUCKET}/{key}", bbox)
        if rgs:
            out.append((key, rgs))
    return out


def scan_files_parallel(index: _FooterIndex, keys: list[str], bbox: BBox, workers: int = 6):
    """Footer-scan many parquet files in crash-isolated worker processes."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    chunk = 8
    chunks = [keys[i : i + chunk] for i in range(0, len(keys), chunk)]
    bbox_t = (bbox.south, bbox.west, bbox.north, bbox.east)
    matches: list = []
    failed: list[list[str]] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(_scan_chunk, (c, bbox_t)): c for c in chunks}
        from concurrent.futures import as_completed

        done = 0
        for fut in as_completed(futures):
            try:
                matches.extend(fut.result())
            except Exception:  # noqa: BLE001 (incl. BrokenProcessPool)
                failed.append(futures[fut])
            done += 1
            if done % 16 == 0:
                log.info(f"footer scan {done}/{len(chunks)} chunks")
    if failed:  # crashed chunks: retry file-by-file, still crash-isolated
        retry_keys = [k for c in failed for k in c]
        log.warning(f"retrying {len(retry_keys)} files from crashed workers")
        with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
            futures = {pool.submit(_scan_chunk, ([k], bbox_t)): k for k in retry_keys}
            for fut in as_completed(futures):
                try:
                    matches.extend(fut.result())
                except Exception:  # noqa: BLE001
                    log.warning(f"skipping unreadable file {futures[fut]}")
    log.info(f"{len(matches)} file(s) intersect the bbox")
    return matches


class OvertureBuildingsProvider:
    """`.fetch(bbox) -> list[Building]` — drop-in for OverpassBuildings."""

    def __init__(self, release: str | None = None, cache_dir: Path | None = None):
        self.release = release
        self._files: list[str] | None = None

    def fetch(self, bbox: BBox) -> list[Building]:
        release = self.release or latest_release()
        files = list_theme_files(release, "buildings", "building")
        log.info(f"overture {release}: scanning {len(files)} building files for {bbox}")
        index = _FooterIndex()
        matches = scan_files_parallel(index, files, bbox)
        buildings: list[Building] = []
        for key, rgs in matches:
            url = f"{BUCKET}/{key}"
            log.info(f"reading {len(rgs)} row groups from {key.rsplit('/', 1)[-1]}")
            table = index.read_rows(url, rgs, ["geometry", "height", "num_floors", "subtype", "class"], bbox)
            heights = table.column("height").to_pylist()
            floors = table.column("num_floors").to_pylist()
            subtypes = table.column("subtype").to_pylist()
            classes = table.column("class").to_pylist()
            for wkb_val, h, fl, st, cl in zip(
                table.column("geometry").to_pylist(), heights, floors, subtypes, classes, strict=True
            ):
                for poly in _wkb_rings(wkb_val):
                    height, source = None, "none"
                    if h is not None and h > 0:
                        height, source = float(h), "tag_height"
                    elif fl:
                        height, source = float(fl) * 3.0, "tag_levels"
                    cls = CLASS_OF_TAG.get(str(cl or st or "").lower(), None)
                    if cls is None:
                        cls = {"residential": "residential", "industrial": "industrial",
                               "commercial": "commercial", "religious": "landmark",
                               "civic": "commercial", "outbuilding": "generic"}.get(str(st or "").lower(), "generic")
                    buildings.append(Building(
                        footprint_lonlat=list(poly.exterior.coords),
                        holes_lonlat=[list(r.coords) for r in poly.interiors],
                        height_m=height,
                        height_source=source,
                        building_class=cls,
                    ))
        with_h = sum(1 for b in buildings if b.height_m is not None)
        log.info(f"overture buildings: {len(buildings)} in bbox ({with_h} with heights)")
        return buildings


# Overture land / land_use subtype/class -> our ground classes.
GREEN_OF_OVERTURE = {
    "forest": "forest_broadleaf", "wood": "forest_broadleaf", "tree": "forest_broadleaf",
    "shrub": "green", "shrubbery": "green", "scrub": "green",
    "grass": "green", "grassland": "green", "meadow": "green",
    "park": "green", "garden": "green", "cemetery": "green",
    "recreation_ground": "green", "golf_course": "green", "vineyard": "green",
    "orchard": "green", "allotments": "green", "village_green": "green",
}


class OvertureLandcoverProvider:
    """Water polygons + road segments + green/forest areas from Overture."""

    def __init__(self, release: str | None = None):
        self.release = release

    def fetch(self, bbox: BBox) -> list[LandcoverFeature]:
        release = self.release or latest_release()
        index = _FooterIndex()
        feats: list[LandcoverFeature] = []

        # Green + forest polygons (vegetation layer needs the forest classes).
        for theme, type_ in (("base", "land"), ("base", "land_use")):
            for key, rgs in scan_files_parallel(
                index, list_theme_files(release, theme, type_), bbox
            ):
                url = f"{BUCKET}/{key}"
                table = index.read_rows(url, rgs, ["geometry", "subtype", "class"], bbox)
                for wkb_val, st, cl in zip(
                    table.column("geometry").to_pylist(),
                    table.column("subtype").to_pylist(),
                    table.column("class").to_pylist(),
                    strict=True,
                ):
                    kind = GREEN_OF_OVERTURE.get(str(cl or "").lower()) or GREEN_OF_OVERTURE.get(
                        str(st or "").lower()
                    )
                    if kind is None:
                        continue
                    for poly in _wkb_rings(wkb_val):
                        feats.append(
                            LandcoverFeature(kind, ring_lonlat=list(poly.exterior.coords))
                        )

        for key, rgs in scan_files_parallel(index, list_theme_files(release, "base", "water"), bbox):
            url = f"{BUCKET}/{key}"
            table = index.read_rows(url, rgs, ["geometry"], bbox)
            for wkb_val in table.column("geometry").to_pylist():
                for poly in _wkb_rings(wkb_val):
                    feats.append(LandcoverFeature("water", ring_lonlat=list(poly.exterior.coords)))

        for key, rgs in scan_files_parallel(
            index, list_theme_files(release, "transportation", "segment"), bbox
        ):
            url = f"{BUCKET}/{key}"
            table = index.read_rows(url, rgs, ["geometry", "subtype", "class"], bbox)
            from shapely import wkb as swkb

            for wkb_val, st, cl in zip(
                table.column("geometry").to_pylist(),
                table.column("subtype").to_pylist(),
                table.column("class").to_pylist(),
                strict=True,
            ):
                if str(st) != "road":
                    continue
                geom = swkb.loads(wkb_val)
                lines = [geom] if geom.geom_type == "LineString" else (
                    list(geom.geoms) if geom.geom_type == "MultiLineString" else []
                )
                width = ROAD_WIDTHS.get(str(cl or "unknown").lower(), 5.0)
                for line in lines:
                    feats.append(LandcoverFeature("road", line_lonlat=list(line.coords), width_m=width))
        log.info(f"overture landcover: {len(feats)} features")
        return feats


# Overture places category fragments -> landmark archetypes (see gis/poi.py).
ARCHETYPE_OF_CATEGORY = [
    ("castle", "crenellated"), ("fortress", "crenellated"), ("fort", "crenellated"),
    ("church", "dome"), ("cathedral", "dome"), ("basilica", "dome"),
    ("mosque", "dome"), ("synagogue", "dome"),
    ("tower", "spire_tower"), ("lighthouse", "spire_tower"),
    ("monument", "spire_tower"), ("landmark", None), ("tourist_attraction", None),
    ("historic", None), ("government_building", None), ("city_hall", None),
]


class OverturePoiProvider:
    """POIs from the Overture `places` theme -> (Poi, []) — same contract as
    OverpassPoi, so archetype stamping, building matching and Wikimedia photo
    fetch work unchanged. Overture has no building:part geometries."""

    def __init__(self, release: str | None = None):
        self.release = release

    def fetch(self, bbox: BBox):
        from shapely import wkb as swkb

        from dronecv.gis.providers.poi import Poi

        release = self.release or latest_release()
        index = _FooterIndex()
        pois: list[Poi] = []
        for key, rgs in scan_files_parallel(
            index, list_theme_files(release, "places", "place"), bbox
        ):
            url = f"{BUCKET}/{key}"
            table = index.read_rows(url, rgs, ["geometry", "names", "categories"], bbox)
            for wkb_val, names, cats in zip(
                table.column("geometry").to_pylist(),
                table.column("names").to_pylist(),
                table.column("categories").to_pylist(),
                strict=True,
            ):
                cat_primary = str((cats or {}).get("primary") or "").lower()
                alternates = [str(c).lower() for c in ((cats or {}).get("alternate") or [])]
                arch, relevant = None, False
                for frag, a in ARCHETYPE_OF_CATEGORY:
                    if frag in cat_primary or any(frag in c for c in alternates):
                        relevant = True
                        arch = arch or a
                if not relevant:
                    continue
                geom = swkb.loads(wkb_val)
                pt = geom.centroid
                name = (names or {}).get("primary")
                pois.append(Poi(
                    name=name, archetype=arch, lonlat=(float(pt.x), float(pt.y)),
                    tags={"overture_category": cat_primary},
                ))
        log.info(f"overture places: {len(pois)} landmark POIs in bbox")
        return pois, []
