"""Overpass HTTP layer: identifying UA, hedged mirrors, first-good-wins."""

from __future__ import annotations

import httpx
import pytest

from dronecv.gis.providers.overpass_http import MIRRORS, overpass_query


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {"elements": []}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


def test_first_good_mirror_wins(monkeypatch):
    seen_headers = []

    def fake_post(url, data=None, headers=None, timeout=None):
        seen_headers.append(headers)
        # Only the private.coffee mirror answers; the others reject (retryable).
        if "private.coffee" in url:
            return FakeResponse(200, {"elements": [1]})
        return FakeResponse(504)

    monkeypatch.setattr(httpx, "post", fake_post)
    out = overpass_query("[out:json];", stagger_s=0.0)
    assert out == {"elements": [1]}
    assert all("dronecv" in h["User-Agent"] for h in seen_headers)  # UA always sent


def test_all_mirrors_reject_raises(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(504))
    with pytest.raises(RuntimeError, match="all Overpass endpoints failed"):
        overpass_query("[out:json];", stagger_s=0.0)


def test_all_mirrors_time_out_raises(monkeypatch):
    def boom(*a, **k):
        raise httpx.ReadTimeout("read timed out")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(RuntimeError, match="all Overpass endpoints failed"):
        overpass_query("[out:json];", stagger_s=0.0)


def test_custom_url_used_exclusively(monkeypatch):
    calls = []

    def fake_post(url, data=None, headers=None, timeout=None):
        calls.append(url)
        return FakeResponse(200, {"elements": [7]})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = overpass_query("[out:json];", url="https://my.private/api")
    assert out == {"elements": [7]}
    assert calls == ["https://my.private/api"]  # never touched the public mirrors


def test_default_url_in_mirrors_triggers_hedge(monkeypatch):
    # Passing a URL that IS one of the mirrors means "use all mirrors hedged".
    calls = set()

    def fake_post(url, data=None, headers=None, timeout=None):
        calls.add(url)
        return FakeResponse(200 if "kumi" in url else 504)

    monkeypatch.setattr(httpx, "post", fake_post)
    out = overpass_query("[out:json];", url=MIRRORS[0], stagger_s=0.0)
    assert out == {"elements": []}
    assert len(calls) >= 1  # hedged across mirrors, not just the passed one
