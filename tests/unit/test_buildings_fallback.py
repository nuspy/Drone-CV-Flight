"""Overpass -> Overture automatic fallback in the build pipeline."""

from __future__ import annotations

import pytest

from dronecv.gis.geometry import BBox
from dronecv.gis.pipeline import BuildSources, _fetch_buildings_resilient
from dronecv.gis.providers.buildings import Building, OverpassBuildings

BBOX = BBox(47.49, 19.02, 47.52, 19.07)


class _Meta:
    def __init__(self):
        self.stats: dict = {}


class _FailingOverpass(OverpassBuildings):
    def fetch(self, bbox):
        raise RuntimeError("all Overpass endpoints failed (last: 504)")


def _sources(buildings):
    return BuildSources(dem=None, buildings=buildings, landcover=None, poi=None)


def test_overpass_failure_raises_by_default():
    # DEFAULT: no silent source switch — the user stays in control.
    sources = _sources(_FailingOverpass())
    meta = _Meta()
    with pytest.raises(RuntimeError, match="Overpass is unavailable"):
        _fetch_buildings_resilient(sources, BBOX, meta)
    assert "buildings_fallback" not in meta.stats
    # sources were NOT mutated to Overture.
    assert isinstance(sources.buildings, _FailingOverpass)


def test_overpass_failure_falls_back_when_opted_in(monkeypatch):
    import dronecv.gis.providers.overture as ov

    sentinel = [Building(footprint_lonlat=[(19.0, 47.5)] * 4, height_m=12.0)]

    class FakeBuildings:
        def fetch(self, bbox):
            return sentinel

    class FakeLandcover:
        pass

    class FakePoi:
        pass

    monkeypatch.setattr(ov, "OvertureBuildingsProvider", FakeBuildings)
    monkeypatch.setattr(ov, "OvertureLandcoverProvider", FakeLandcover)
    monkeypatch.setattr(ov, "OverturePoiProvider", FakePoi)

    sources = _sources(_FailingOverpass())
    meta = _Meta()
    out = _fetch_buildings_resilient(sources, BBOX, meta, allow_overture_fallback=True)

    assert out is sentinel
    assert meta.stats["buildings_fallback"] == "overture"
    # landcover + POI were swapped to the fallback provider too.
    assert isinstance(sources.buildings, FakeBuildings)
    assert isinstance(sources.landcover, FakeLandcover)
    assert isinstance(sources.poi, FakePoi)


def test_overture_source_failure_is_not_masked():
    # If the user explicitly chose Overture and it fails, surface the error
    # (no silent secondary fallback).
    class FailingOverture:
        def fetch(self, bbox):
            raise RuntimeError("overture scan failed")

    with pytest.raises(RuntimeError, match="overture scan failed"):
        _fetch_buildings_resilient(_sources(FailingOverture()), BBOX, _Meta())


def test_success_path_no_fallback():
    hit = [Building(footprint_lonlat=[(19.0, 47.5)] * 4)]

    class OkOverpass(OverpassBuildings):
        def fetch(self, bbox):
            return hit

    meta = _Meta()
    assert _fetch_buildings_resilient(_sources(OkOverpass()), BBOX, meta) is hit
    assert "buildings_fallback" not in meta.stats
