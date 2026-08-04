"""Building footprints + heights.

Primary: OpenStreetMap via the Overpass API (no keys). Heights resolved by
the chain: `height` tag (meters) -> `building:levels` x 3 m -> None (later
filled by shadow inference or per-class defaults — see
dronecv.gis.shadow_heights and raster.DEFAULT_CLASS_HEIGHTS).

Optional: Overture Maps (GeoParquet on S3/Azure; better height coverage,
extra dependency `duckdb`) exposing the same Building dataclass.

Offline tests inject a saved Overpass JSON response via `fixture_path`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dronecv.gis.geometry import BBox
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.buildings")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Coarse class taxonomy shared by raster/albedo/saliency.
CLASS_OF_TAG = {
    "residential": "residential",
    "house": "residential",
    "apartments": "residential",
    "detached": "residential",
    "terrace": "residential",
    "industrial": "industrial",
    "warehouse": "industrial",
    "factory": "industrial",
    "commercial": "commercial",
    "retail": "commercial",
    "office": "commercial",
    "church": "landmark",
    "cathedral": "landmark",
    "tower": "landmark",
    "castle": "landmark",
    "campanile": "landmark",
}


@dataclass
class Building:
    footprint_lonlat: list[tuple[float, float]]  # outer ring (lon, lat)
    holes_lonlat: list[list[tuple[float, float]]] = field(default_factory=list)
    height_m: float | None = None
    height_source: str = "none"  # tag_height | tag_levels | shadow | class_default
    building_class: str = "generic"
    osm_id: int | None = None
    roof_shape: str | None = None  # flat | gabled | hipped | pyramidal | skillion | dome
    roof_height_m: float | None = None
    # Real per-building colors where the source maps them (Overture
    # roof_color/facade_color) — the exports tint the meshes per building
    # instead of only per class. RGB floats in [0, 1].
    roof_color: tuple[float, float, float] | None = None
    facade_color: tuple[float, float, float] | None = None


def _parse_height(tags: dict) -> tuple[float | None, str]:
    raw = tags.get("height") or tags.get("building:height")
    if raw:
        m = re.match(r"\s*([\d.]+)", str(raw))
        if m:
            return float(m.group(1)), "tag_height"
    levels = tags.get("building:levels")
    if levels:
        m = re.match(r"\s*([\d.]+)", str(levels))
        if m:
            return float(m.group(1)) * 3.0, "tag_levels"
    return None, "none"


def _building_class(tags: dict) -> str:
    value = str(tags.get("building", "yes")).lower()
    if value in CLASS_OF_TAG:
        return CLASS_OF_TAG[value]
    if tags.get("man_made") in ("tower", "chimney", "water_tower"):
        return "landmark"
    return "generic"


class OverpassBuildings:
    def __init__(self, fixture_path: Path | None = None, url: str = OVERPASS_URL):
        self.fixture_path = fixture_path
        self.url = url

    def fetch(self, bbox: BBox) -> list[Building]:
        data = self._raw(bbox)
        return parse_overpass(data)

    def _raw(self, bbox: BBox) -> dict:
        if self.fixture_path is not None:
            return json.loads(Path(self.fixture_path).read_text())

        query = (
            f"[out:json][timeout:180];"
            f'(way["building"]({bbox.south},{bbox.west},{bbox.north},{bbox.east});'
            f'relation["building"]({bbox.south},{bbox.west},{bbox.north},{bbox.east}););'
            f"out body geom;"
        )
        log.info(f"querying Overpass for buildings in {bbox}")
        from dronecv.gis.providers.overpass_http import overpass_query

        return overpass_query(query, url=self.url)


def parse_overpass(data: dict) -> list[Building]:
    """Overpass `out body geom` JSON -> Buildings (ways + multipolygon relations)."""
    buildings: list[Building] = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if "building" not in tags and el.get("type") != "relation":
            continue
        height, source = _parse_height(tags)
        cls = _building_class(tags)
        roof_shape = tags.get("roof:shape")
        roof_h = None
        raw_rh = tags.get("roof:height") or tags.get("roof:levels")
        if raw_rh:
            try:
                v = float(str(raw_rh).split()[0])
                roof_h = v if "roof:height" in tags else v * 2.5
            except ValueError:
                pass

        if el.get("type") == "way" and "geometry" in el:
            ring = [(g["lon"], g["lat"]) for g in el["geometry"]]
            if len(ring) >= 4:
                buildings.append(
                    Building(ring, [], height, source, cls, el.get("id"),
                             roof_shape=roof_shape, roof_height_m=roof_h)
                )
        elif el.get("type") == "relation" and "members" in el:
            outers, inners = [], []
            for member in el["members"]:
                if "geometry" not in member:
                    continue
                ring = [(g["lon"], g["lat"]) for g in member["geometry"]]
                if len(ring) < 4:
                    continue
                (outers if member.get("role") == "outer" else inners).append(ring)
            for outer in outers:
                buildings.append(
                    Building(outer, inners if len(outers) == 1 else [], height, source, cls, el.get("id"))
                )
    with_height = sum(1 for b in buildings if b.height_m is not None)
    log.info(f"parsed {len(buildings)} buildings ({with_height} with tagged heights)")
    return buildings


def height_stats(buildings: list[Building]) -> dict:
    n = len(buildings)
    tagged = sum(1 for b in buildings if b.height_source in ("tag_height", "tag_levels"))
    heights = np.array([b.height_m for b in buildings if b.height_m is not None])
    return {
        "n_buildings": n,
        "n_with_height": int(tagged),
        "height_coverage": tagged / n if n else 0.0,
        "median_height_m": float(np.median(heights)) if heights.size else None,
    }
