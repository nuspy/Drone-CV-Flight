"""Ground classes (roads, water, green, rail) from OSM via Overpass.

Only used for the shape-first albedo (class color bands) and as weak texture
for the localizer — geometry stays the authoritative signal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from dronecv.gis.geometry import BBox
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.landcover")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# class name -> (class id, priority when overlapping). Building ids live in
# the same class raster starting at BUILDING_CLASS_BASE (see raster.py).
GROUND_CLASSES = {
    "ground": 0,
    "green": 1,
    "water": 2,
    "road": 3,
    "rail": 4,
    "parking": 5,
    "forest_broadleaf": 6,
    "forest_conifer": 7,
}


@dataclass
class LandcoverFeature:
    kind: str  # one of GROUND_CLASSES
    ring_lonlat: list[tuple[float, float]] | None = None  # polygons
    line_lonlat: list[tuple[float, float]] | None = None  # ways to buffer
    width_m: float = 0.0
    holes_lonlat: list[list[tuple[float, float]]] = field(default_factory=list)


def _road_width(tags: dict) -> float:
    highway = tags.get("highway", "")
    return {
        "motorway": 16.0, "trunk": 14.0, "primary": 10.0, "secondary": 8.0,
        "tertiary": 7.0, "residential": 6.0, "unclassified": 5.0, "service": 4.0,
    }.get(highway, 4.0)


class OverpassLandcover:
    def __init__(self, fixture_path: Path | None = None, url: str = OVERPASS_URL):
        self.fixture_path = fixture_path
        self.url = url

    def fetch(self, bbox: BBox) -> list[LandcoverFeature]:
        return parse_overpass_landcover(self._raw(bbox))

    def _raw(self, bbox: BBox) -> dict:
        if self.fixture_path is not None:
            return json.loads(Path(self.fixture_path).read_text())

        b = f"({bbox.south},{bbox.west},{bbox.north},{bbox.east})"
        query = (
            f"[out:json][timeout:180];("
            f'way["highway"]{b};way["railway"~"^(rail|tram)$"]{b};'
            f'way["natural"="water"]{b};way["waterway"="riverbank"]{b};'
            f'way["landuse"~"^(grass|forest|meadow|farmland|orchard|vineyard)$"]{b};'
            f'way["leisure"~"^(park|garden|pitch)$"]{b};way["natural"~"^(wood|scrub)$"]{b};'
            f'way["amenity"="parking"]{b};'
            f");out body geom;"
        )
        log.info(f"querying Overpass for landcover in {bbox}")
        from dronecv.gis.providers.overpass_http import overpass_query

        return overpass_query(query, url=self.url)


def parse_overpass_landcover(data: dict) -> list[LandcoverFeature]:
    feats: list[LandcoverFeature] = []
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        pts = [(g["lon"], g["lat"]) for g in el["geometry"]]
        closed = len(pts) >= 4 and pts[0] == pts[-1]
        bridge = tags.get("bridge") in ("yes", "viaduct") or tags.get("man_made") == "bridge"
        if "highway" in tags:
            kind = "road"
            feats.append(LandcoverFeature(kind, line_lonlat=pts, width_m=_road_width(tags)))
            if bridge:
                feats.append(LandcoverFeature("bridge", line_lonlat=pts, width_m=_road_width(tags) + 2.0))
        elif "railway" in tags:
            feats.append(LandcoverFeature("rail", line_lonlat=pts, width_m=5.0))
            if bridge:
                feats.append(LandcoverFeature("bridge", line_lonlat=pts, width_m=7.0))
        elif tags.get("natural") == "water" or tags.get("waterway") == "riverbank":
            if closed:
                feats.append(LandcoverFeature("water", ring_lonlat=pts))
        elif tags.get("amenity") == "parking":
            if closed:
                feats.append(LandcoverFeature("parking", ring_lonlat=pts))
        elif closed and (
            tags.get("landuse") == "forest" or tags.get("natural") in ("wood", "scrub")
        ):
            # Forest with typology: leaf_type drives the vegetation layer.
            leaf = str(tags.get("leaf_type", "broadleaved"))
            kind = "forest_conifer" if "needle" in leaf else "forest_broadleaf"
            feats.append(LandcoverFeature(kind, ring_lonlat=pts))
        elif closed:
            feats.append(LandcoverFeature("green", ring_lonlat=pts))
    log.info(f"parsed {len(feats)} landcover features")
    return feats
