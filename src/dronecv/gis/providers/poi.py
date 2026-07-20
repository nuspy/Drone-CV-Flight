"""Points of interest and detailed building parts from OSM.

POIs (historic sites, towers, churches, castles, attractions) drive:
- landmark archetype modeling (a `historic=castle` gets crenellations, a
  `man_made=tower` a spire, a church a dome/campanile — stamped into the
  height mosaic so even the TRAINING renders see the shape);
- online photo retrieval via the `wikidata` tag (Wikimedia Commons, free
  licenses with attribution). Google Maps/Places photos are deliberately NOT
  supported: their ToS forbid offline storage and texture use.

`building:part` elements provide real LoD detail (per-part height/min_height)
where mappers added it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from dronecv.gis.geometry import BBox
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.poi")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# tag -> archetype used by landmark modeling.
ARCHETYPE_OF = {
    ("man_made", "tower"): "spire_tower",
    ("man_made", "water_tower"): "cap_tower",
    ("man_made", "chimney"): "spire_tower",
    ("historic", "castle"): "crenellated",
    ("historic", "fort"): "crenellated",
    ("historic", "tower"): "crenellated_tower",
    ("building", "church"): "dome",
    ("building", "cathedral"): "dome",
    ("building", "mosque"): "dome",
    ("building", "tower"): "spire_tower",
    ("tourism", "attraction"): None,  # POI only, shape from other tags
}


@dataclass
class Poi:
    name: str | None
    archetype: str | None
    lonlat: tuple[float, float]
    wikidata: str | None = None
    wikipedia: str | None = None
    tags: dict = field(default_factory=dict)


@dataclass
class BuildingPart:
    footprint_lonlat: list[tuple[float, float]]
    height_m: float
    min_height_m: float = 0.0


class OverpassPoi:
    def __init__(self, fixture_path: Path | None = None, url: str = OVERPASS_URL):
        self.fixture_path = fixture_path
        self.url = url

    def fetch(self, bbox: BBox) -> tuple[list[Poi], list[BuildingPart]]:
        return parse_poi(self._raw(bbox))

    def _raw(self, bbox: BBox) -> dict:
        if self.fixture_path is not None:
            return json.loads(Path(self.fixture_path).read_text())

        b = f"({bbox.south},{bbox.west},{bbox.north},{bbox.east})"
        query = (
            f"[out:json][timeout:180];("
            f'nwr["historic"]{b};nwr["tourism"~"^(attraction|viewpoint)$"]{b};'
            f'nwr["man_made"~"^(tower|water_tower|chimney|lighthouse)$"]{b};'
            f'way["building:part"]{b};'
            f");out body geom center;"
        )
        log.info(f"querying Overpass for POI/parts in {bbox}")
        from dronecv.gis.providers.overpass_http import overpass_query

        return overpass_query(query, url=self.url)


def _archetype(tags: dict) -> str | None:
    for (k, v), arch in ARCHETYPE_OF.items():
        if tags.get(k) == v and arch:
            return arch
    return None


def parse_poi(data: dict) -> tuple[list[Poi], list[BuildingPart]]:
    pois: list[Poi] = []
    parts: list[BuildingPart] = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if "building:part" in tags and el.get("type") == "way" and "geometry" in el:
            from dronecv.gis.providers.buildings import _parse_height

            h, _src = _parse_height(tags)
            if h is None:
                h = 6.0
            min_h = 0.0
            raw_min = tags.get("min_height") or tags.get("building:min_level")
            if raw_min:
                try:
                    min_h = float(str(raw_min).split()[0])
                    if "building:min_level" in tags and "min_height" not in tags:
                        min_h *= 3.0
                except ValueError:
                    min_h = 0.0
            parts.append(BuildingPart(
                [(g["lon"], g["lat"]) for g in el["geometry"]], height_m=h, min_height_m=min_h,
            ))
            continue

        if not tags:
            continue
        lonlat = None
        if el.get("type") == "node":
            lonlat = (el.get("lon"), el.get("lat"))
        elif "center" in el:
            lonlat = (el["center"]["lon"], el["center"]["lat"])
        elif "geometry" in el and el["geometry"]:
            gs = el["geometry"]
            lonlat = (
                sum(g["lon"] for g in gs) / len(gs),
                sum(g["lat"] for g in gs) / len(gs),
            )
        if lonlat is None or lonlat[0] is None:
            continue
        if any(k in tags for k in ("historic", "tourism")) or _archetype(tags):
            pois.append(Poi(
                name=tags.get("name"),
                archetype=_archetype(tags),
                lonlat=lonlat,
                wikidata=tags.get("wikidata"),
                wikipedia=tags.get("wikipedia"),
                tags=tags,
            ))
    log.info(f"parsed {len(pois)} POIs, {len(parts)} building parts")
    return pois, parts


def fetch_commons_photos(pois: list[Poi], out_dir: Path, max_photos: int = 10) -> list[dict]:
    """Download the primary Wikimedia Commons image (wikidata P18) for POIs
    that carry a wikidata tag. Free licenses; attribution recorded alongside.
    Never raises on network trouble — photos are an optional enrichment."""
    import httpx

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for poi in pois:
        if len(records) >= max_photos or not poi.wikidata:
            continue
        try:
            wd = httpx.get(
                f"https://www.wikidata.org/wiki/Special:EntityData/{poi.wikidata}.json",
                timeout=30.0, follow_redirects=True,
                headers={"User-Agent": "dronecv-gis/0.1"},
            )
            wd.raise_for_status()
            claims = wd.json()["entities"][poi.wikidata]["claims"]
            image_name = claims["P18"][0]["mainsnak"]["datavalue"]["value"]
            url = (
                "https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/"
                + image_name.replace(" ", "_") + "&width=1024"
            )
            img = httpx.get(url, timeout=60.0, follow_redirects=True,
                            headers={"User-Agent": "dronecv-gis/0.1"})
            img.raise_for_status()
            fname = f"{poi.wikidata}.jpg"
            (out_dir / fname).write_bytes(img.content)
            records.append({
                "wikidata": poi.wikidata, "name": poi.name, "file": fname,
                "source": f"Wikimedia Commons: {image_name}",
                "attribution": "see commons.wikimedia.org file page for license/author",
            })
        except Exception as e:  # noqa: BLE001
            log.warning(f"photo fetch failed for {poi.wikidata}: {e}")
    if records:
        (out_dir / "photos.json").write_text(json.dumps(records, indent=1))
    return records
