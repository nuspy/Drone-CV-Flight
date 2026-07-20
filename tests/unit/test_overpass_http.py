"""Overpass HTTP layer: UA header, mirror fallback, custom URL honored."""

from __future__ import annotations

import httpx
import pytest

from dronecv.gis.providers import overpass_http
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


def test_falls_back_across_mirrors_on_406(monkeypatch):
    calls = []

    def fake_post(url, data=None, headers=None, timeout=None):
        calls.append((url, headers))
        # First mirror rejects like overpass-api.de does without a UA policy
        # match; the second succeeds.
        return FakeResponse(406) if len(calls) == 1 else FakeResponse(200, {"elements": [1]})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = overpass_query("[out:json];")
    assert out == {"elements": [1]}
    assert calls[0][0] == MIRRORS[0] and calls[1][0] == MIRRORS[1]
    assert "dronecv" in calls[0][1]["User-Agent"]  # identifying UA always sent


def test_custom_url_used_exclusively(monkeypatch):
    calls = []

    def fake_post(url, data=None, headers=None, timeout=None):
        calls.append(url)
        return FakeResponse(406)

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(RuntimeError, match="all Overpass endpoints failed"):
        overpass_query("[out:json];", url="https://my.private/api")
    assert calls == ["https://my.private/api"]


def test_all_mirrors_down_raises(monkeypatch):
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("boom")),
    )
    with pytest.raises(RuntimeError, match="all Overpass endpoints failed"):
        overpass_query("[out:json];")
    assert overpass_http.MIRRORS  # sanity: mirror list non-empty
