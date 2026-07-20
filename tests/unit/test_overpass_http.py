"""Overpass HTTP layer: identifying UA, smart mirror pool, blocking backoff."""

from __future__ import annotations

import httpx
import pytest

from dronecv.gis.providers.overpass_http import (
    MIRRORS,
    MirrorPool,
    _reset_pool_for_tests,
    overpass_query,
)


@pytest.fixture(autouse=True)
def _fresh_pool():
    # The pool is a module singleton persisted to disk; reset it (its health
    # file lives under the tmp DRONECV_CACHE_DIR from conftest) so tests don't
    # contaminate each other's mirror ordering.
    _reset_pool_for_tests()
    yield
    _reset_pool_for_tests()


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {"elements": []}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


def test_first_good_mirror_wins(monkeypatch):
    seen_headers = []

    def fake_post(url, data=None, headers=None, timeout=None):
        seen_headers.append(headers)
        return FakeResponse(200, {"elements": [1]}) if "private.coffee" in url else FakeResponse(504)

    monkeypatch.setattr(httpx, "post", fake_post)
    out = overpass_query("[out:json];")
    assert out == {"elements": [1]}
    assert all("dronecv" in h["User-Agent"] for h in seen_headers)  # UA always sent


def test_all_mirrors_reject_raises(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(504))
    with pytest.raises(RuntimeError, match="all Overpass endpoints failed"):
        overpass_query("[out:json];")


def test_all_mirrors_time_out_raises(monkeypatch):
    def boom(*a, **k):
        raise httpx.ReadTimeout("read timed out")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(RuntimeError, match="all Overpass endpoints failed"):
        overpass_query("[out:json];")


def test_custom_url_used_exclusively(monkeypatch):
    calls = []

    def fake_post(url, data=None, headers=None, timeout=None):
        calls.append(url)
        return FakeResponse(200, {"elements": [7]})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = overpass_query("[out:json];", url="https://my.private/api")
    assert out == {"elements": [7]}
    assert calls == ["https://my.private/api"]  # never touched the public mirrors


def test_mirror_in_list_uses_the_pool(monkeypatch):
    calls = set()

    def fake_post(url, data=None, headers=None, timeout=None):
        calls.add(url)
        return FakeResponse(200 if "kumi" in url else 504)

    monkeypatch.setattr(httpx, "post", fake_post)
    out = overpass_query("[out:json];", url=MIRRORS[0])
    assert out == {"elements": []}
    assert len(calls) >= 1  # tried mirrors via the pool, not just the passed one


class TestMirrorPool:
    def test_order_prefers_better_history(self, tmp_path):
        pool = MirrorPool(MIRRORS, path=tmp_path / "h.json")
        good = MIRRORS[2]
        pool.release(good, ok=True, latency=0.2)         # good has a success
        pool.release(MIRRORS[0], ok=False, latency=5.0)  # first mirror failed
        assert pool.order()[0] == good  # best-first

    def test_blocked_mirror_excluded_until_cooldown(self, tmp_path):
        pool = MirrorPool(MIRRORS, path=tmp_path / "h.json")
        blk = MIRRORS[1]
        pool.release(blk, ok=False, latency=1.0, blocked=True)
        assert blk not in pool.order()
        assert pool.any_blocked()

    def test_health_persists_round_trip(self, tmp_path):
        p = tmp_path / "h.json"
        a = MirrorPool(MIRRORS, path=p)
        a.release(MIRRORS[0], ok=True, latency=0.5)
        b = MirrorPool(MIRRORS, path=p)              # reload from disk
        assert b._stats[MIRRORS[0]]["ok"] == 1


def test_block_detected_from_429(monkeypatch, tmp_path):
    # A 429 marks the mirror blocked so the pool stops using it.
    from dronecv.gis.providers import overpass_http

    pool = MirrorPool(MIRRORS, path=tmp_path / "h.json")
    monkeypatch.setattr(overpass_http, "_pool", pool)

    # The first-ordered mirror (overpass-api.de) rate-limits; the query falls
    # through to a good one, and the 429 mirror is put on a cooldown.
    def fake_post(url, data=None, headers=None, timeout=None):
        return FakeResponse(429) if "de/api" in url else FakeResponse(200, {"elements": [9]})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = overpass_query("[out:json];")
    assert out == {"elements": [9]}
    blocked = [m for m in MIRRORS if pool._stats[m]["blocked_until"] > 0]
    assert blocked  # the rate-limited mirror got a cooldown
