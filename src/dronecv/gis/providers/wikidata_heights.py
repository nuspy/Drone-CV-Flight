"""Real landmark heights from Wikidata (P2048).

POIs carry `wikidata` tags (OSM) — for the matched buildings whose height came
from a weak source (neighbor median / class default / none), the true height
of the landmark is one API call away: the Parliament is 96 m on Wikidata even
when the map data has no tag. Enrichment is best-effort: offline or blocked
networks simply skip it (a warning, never a failure).
"""

from __future__ import annotations

from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.wikidata")

_WEAK_SOURCES = {"none", "neighbor_median", "class_default"}


def fetch_height_m(wikidata_id: str, timeout_s: float = 10.0) -> float | None:
    """Height (P2048, meters) of a Wikidata entity, or None."""
    import httpx

    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
    resp = httpx.get(url, timeout=timeout_s, follow_redirects=True)
    resp.raise_for_status()
    claims = resp.json()["entities"][wikidata_id]["claims"].get("P2048", [])
    for c in claims:
        try:
            val = c["mainsnak"]["datavalue"]["value"]
            amount = float(val["amount"])
            unit = str(val.get("unit", ""))
            if unit.endswith("Q11573") or unit in ("1", ""):  # meters
                return amount
        except (KeyError, TypeError, ValueError):
            continue
    return None


def enrich_landmark_heights(pois, pairs) -> int:
    """For each (poi, building) match where the poi has a wikidata id and the
    building's height came from a weak source, set the REAL height. Returns
    how many buildings were enriched. Never raises on network trouble."""
    n = 0
    for poi, building in pairs:
        wd = getattr(poi, "wikidata", None) or (poi.tags or {}).get("wikidata")
        if not wd or building.height_source not in _WEAK_SOURCES:
            continue
        try:
            h = fetch_height_m(str(wd))
        except Exception as e:  # noqa: BLE001 — offline/blocked: skip quietly
            log.warning(f"wikidata height lookup failed for {wd}: {e}")
            return n
        if h and h > (building.height_m or 0.0):
            building.height_m = float(h)
            building.height_source = "wikidata"
            n += 1
    if n:
        log.info(f"wikidata heights: {n} landmark buildings enriched")
    return n
