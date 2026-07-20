"""Shared Overpass HTTP layer: identifying User-Agent + mirror fallback.

The public Overpass instances enforce a usage policy: requests without an
identifying User-Agent get rejected (observed as 406 Not Acceptable on
overpass-api.de) and heavy users get 429. All dronecv queries therefore go
through this helper, which sends a proper UA and falls back across public
mirrors on policy/availability errors.
"""

from __future__ import annotations

from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.overpass")

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
HEADERS = {"User-Agent": "dronecv-gis/0.1 (+https://github.com/nuspy/Drone-CV-Flight)"}
RETRYABLE = {403, 406, 429, 502, 503, 504}


def overpass_query(query: str, url: str | None = None, timeout_s: float = 240.0) -> dict:
    """POST an Overpass QL query, walking the mirror list on failure.
    A caller-supplied non-default `url` is honored exclusively (tests,
    private instances)."""
    import httpx

    urls = [url] if url and url not in MIRRORS else MIRRORS
    last: str = "no endpoint tried"
    for u in urls:
        try:
            resp = httpx.post(u, data={"data": query}, headers=HEADERS, timeout=timeout_s)
            if resp.status_code in RETRYABLE:
                last = f"{u} -> HTTP {resp.status_code}"
                log.warning(f"overpass endpoint rejected the request: {last}")
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            last = f"{u} -> {e}"
            log.warning(f"overpass endpoint failed: {last}")
    raise RuntimeError(f"all Overpass endpoints failed (last: {last})")
