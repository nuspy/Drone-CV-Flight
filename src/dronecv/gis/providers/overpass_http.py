"""Shared Overpass HTTP layer: identifying User-Agent + HEDGED mirrors.

The public Overpass instances enforce a usage policy (a plain, identifying
User-Agent is required — else 406). More importantly, under load a mirror
often *hangs*: it accepts the connection but returns nothing for minutes,
while another mirror answers a tiny query in under a second — and which
mirror is fast changes from minute to minute. A fixed order therefore wastes
whole timeouts on whichever mirror is currently stuck.

So requests are HEDGED: the first mirror fires immediately, the next only if
the previous hasn't answered within `stagger_s` (keeping load modest), and
the FIRST mirror to return a good response wins — the others are abandoned.
A short read timeout means a hung mirror is given up on quickly. For the tiny
per-cell queries the cell grid produces, this turns "150 s + 150 s + fast"
into just "fast".
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor

from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.overpass")

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
HEADERS = {"User-Agent": "dronecv-gis/0.1 (+https://github.com/nuspy/Drone-CV-Flight)"}
RETRYABLE = {403, 406, 429, 500, 502, 503, 504}


def overpass_query(
    query: str,
    url: str | None = None,
    timeout_s: float = 40.0,
    stagger_s: float = 7.0,
) -> dict:
    """POST an Overpass QL query, hedged across the public mirrors; returns
    the first good JSON response. A caller-supplied non-default `url` is used
    exclusively (tests, private instances)."""
    import httpx

    urls = [url] if (url and url not in MIRRORS) else MIRRORS
    timeout = httpx.Timeout(timeout_s, connect=10.0)

    if len(urls) == 1:
        resp = httpx.post(urls[0], data={"data": query}, headers=HEADERS, timeout=timeout)
        if resp.status_code in RETRYABLE:
            raise RuntimeError(f"all Overpass endpoints failed (last: {urls[0]} -> "
                               f"HTTP {resp.status_code})")
        resp.raise_for_status()
        return resp.json()

    won = threading.Event()
    results: queue.Queue = queue.Queue()

    def attempt(u: str, delay: float) -> None:
        # Staggered start: hold off until `delay` unless a winner appears first.
        if delay and won.wait(delay):
            results.put(("skip", u, ""))
            return
        if won.is_set():
            results.put(("skip", u, ""))
            return
        try:
            resp = httpx.post(u, data={"data": query}, headers=HEADERS, timeout=timeout)
            if resp.status_code in RETRYABLE:
                log.warning(f"overpass endpoint rejected the request: {u} -> HTTP {resp.status_code}")
                results.put(("retry", u, f"HTTP {resp.status_code}"))
                return
            resp.raise_for_status()
            won.set()
            results.put(("ok", u, resp.json()))
        except httpx.HTTPError as e:
            log.warning(f"overpass endpoint failed: {u} -> {e}")
            results.put(("err", u, str(e)))

    ex = ThreadPoolExecutor(max_workers=len(urls))
    for i, u in enumerate(urls):
        ex.submit(attempt, u, i * stagger_s)

    last = "no endpoint tried"
    for _ in range(len(urls)):
        kind, u, payload = results.get()
        if kind == "ok":
            ex.shutdown(wait=False)  # abandon the stragglers
            return payload
        if kind != "skip":
            last = f"{u} -> {payload}"
    ex.shutdown(wait=False)
    raise RuntimeError(f"all Overpass endpoints failed (last: {last})")
