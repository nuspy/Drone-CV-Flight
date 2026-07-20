"""Shared Overpass HTTP layer: identifying User-Agent + SMART mirror selection.

The public Overpass instances enforce a usage policy (an identifying
User-Agent is required — else 406) and, under load, a mirror often *hangs* or
rate-limits while another answers in a second — and which mirror is fast
changes minute to minute.

So mirror choice is DATA-DRIVEN, not fixed:

- a persistent `MirrorPool` remembers, across sessions, which mirrors have been
  succeeding (success rate, latency, recent success) and orders attempts
  best-first — "start from the ones that already worked";
- a query tries the best free mirror, and on a retryable failure/timeout falls
  through to the next best (smart sequence), returning the first good answer;
- rate-limit/blocking signals (HTTP 429/403 or an Overpass `rate_limited`
  body) put that mirror on a cooldown so the pool automatically stops using it;
- callers (the cell fetcher) read `pool.any_blocked()` to DROP parallelism in
  real time when a provider starts blocking.

Health lives in `~/.cache/dronecv/gis/mirror_health.json` (atomic writes).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.overpass")

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
HEADERS = {"User-Agent": "dronecv-gis/0.1 (+https://github.com/nuspy/Drone-CV-Flight)"}
RETRYABLE = {403, 406, 429, 500, 502, 503, 504}
BLOCK_STATUS = {403, 429}  # provider is refusing us -> back off this mirror
_BLOCK_COOLDOWN_S = 120.0
_MAX_COOLDOWN_S = 900.0


def _health_path() -> Path:
    from dronecv.gis.parcel_cache import default_cache_root

    return default_cache_root().parent / "mirror_health.json"


class MirrorPool:
    """Remembers per-mirror health and hands out the best available mirror.

    Thread-safe: `fetch_cells` calls into it from several worker threads."""

    def __init__(self, mirrors: list[str], path: Path | None = None):
        self.mirrors = list(mirrors)
        self.path = path or _health_path()
        self._lock = threading.Lock()
        self._inflight = {m: 0 for m in self.mirrors}
        self._stats = self._load()

    # ---- persistence ----
    def _blank(self) -> dict:
        return {"ok": 0, "fail": 0, "ewma": 0.0, "last_ok": 0.0,
                "blocked_until": 0.0, "block_streak": 0}

    def _load(self) -> dict:
        stats = {m: self._blank() for m in self.mirrors}
        try:
            data = json.loads(self.path.read_text())
            for m in self.mirrors:
                if m in data:
                    stats[m].update({k: data[m].get(k, v) for k, v in stats[m].items()})
        except (OSError, ValueError):
            pass
        return stats

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._stats, indent=1))
            tmp.replace(self.path)
        except OSError:
            pass  # health is best-effort; never fail a build over it

    # ---- ordering / selection ----
    def _score(self, m: str) -> tuple:
        s = self._stats[m]
        tot = s["ok"] + s["fail"]
        rate = (s["ok"] + 1.0) / (tot + 2.0)          # Laplace-smoothed success rate
        lat = s["ewma"] or 5.0
        # fewer in-flight first (spread load), then higher success, then faster
        return (self._inflight[m], -rate, lat)

    def order(self, exclude=()) -> list[str]:
        now = time.time()
        avail = [m for m in self.mirrors
                 if m not in exclude and self._stats[m]["blocked_until"] <= now]
        return sorted(avail, key=self._score)

    def any_blocked(self) -> bool:
        now = time.time()
        return any(s["blocked_until"] > now for s in self._stats.values())

    def acquire(self, exclude=()) -> str | None:
        with self._lock:
            order = self.order(exclude)
            if not order:
                return None
            m = order[0]
            self._inflight[m] += 1
            return m

    def release(self, m: str, ok: bool, latency: float, blocked: bool = False) -> None:
        now = time.time()
        with self._lock:
            s = self._stats[m]
            self._inflight[m] = max(0, self._inflight[m] - 1)
            if ok:
                s["ok"] += 1
                s["last_ok"] = now
                s["block_streak"] = 0
                s["ewma"] = latency if not s["ewma"] else 0.7 * s["ewma"] + 0.3 * latency
            else:
                s["fail"] += 1
            if blocked:
                s["block_streak"] += 1
                cd = min(_MAX_COOLDOWN_S, _BLOCK_COOLDOWN_S * (2 ** (s["block_streak"] - 1)))
                s["blocked_until"] = now + cd
                log.warning(f"overpass mirror rate-limited/blocked: {m} -> cooling down {cd:.0f}s")
            self._save()


_pool: MirrorPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> MirrorPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = MirrorPool(MIRRORS)
        return _pool


def _reset_pool_for_tests() -> None:
    global _pool
    with _pool_lock:
        _pool = None


def _is_rate_limited(resp) -> bool:
    if resp.status_code in BLOCK_STATUS:
        return True
    if resp.status_code in RETRYABLE:
        try:
            body = resp.text[:2000].lower()
        except Exception:  # noqa: BLE001
            return False
        return "rate_limited" in body or "too many requests" in body
    return False


def overpass_query(
    query: str,
    url: str | None = None,
    timeout_s: float = 30.0,
    max_attempts: int | None = None,
) -> dict:
    """POST an Overpass QL query. A caller-supplied non-mirror `url` is used
    exclusively (tests/private instances); otherwise the MirrorPool picks the
    best mirror, falling through to the next best on a retryable failure and
    returning the first good JSON response."""
    import httpx

    timeout = httpx.Timeout(timeout_s, connect=10.0)

    if url and url not in MIRRORS:  # explicit private/custom endpoint
        resp = httpx.post(url, data={"data": query}, headers=HEADERS, timeout=timeout)
        if resp.status_code in RETRYABLE:
            raise RuntimeError(f"all Overpass endpoints failed (last: {url} -> "
                               f"HTTP {resp.status_code})")
        resp.raise_for_status()
        return resp.json()

    pool = get_pool()
    tried: list[str] = []
    last = "no endpoint available"
    attempts = max_attempts or len(MIRRORS)
    for _ in range(attempts):
        m = pool.acquire(exclude=tried)
        if m is None:
            break
        t0 = time.monotonic()
        try:
            resp = httpx.post(m, data={"data": query}, headers=HEADERS, timeout=timeout)
        except httpx.HTTPError as e:
            pool.release(m, ok=False, latency=time.monotonic() - t0, blocked=False)
            tried.append(m)
            last = f"{m} -> {e}"
            log.warning(f"overpass endpoint failed: {last}")
            continue
        blocked = _is_rate_limited(resp)
        if resp.status_code in RETRYABLE:
            pool.release(m, ok=False, latency=time.monotonic() - t0, blocked=blocked)
            tried.append(m)
            last = f"{m} -> HTTP {resp.status_code}"
            log.warning(f"overpass endpoint rejected the request: {last}")
            continue
        try:
            resp.raise_for_status()
        except httpx.HTTPError as e:
            pool.release(m, ok=False, latency=time.monotonic() - t0, blocked=False)
            tried.append(m)
            last = f"{m} -> {e}"
            continue
        pool.release(m, ok=True, latency=time.monotonic() - t0, blocked=False)
        return resp.json()

    raise RuntimeError(f"all Overpass endpoints failed (last: {last})")


def download_concurrency() -> int:
    """Default number of cells to download in parallel (env-overridable)."""
    env = os.environ.get("DRONECV_DOWNLOAD_CONCURRENCY")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return min(4, len(MIRRORS) + 1)
