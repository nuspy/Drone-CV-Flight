"""3D-model detail sources: Overture colors/parts, Wikidata heights,
relative-depth alignment."""

from __future__ import annotations

import numpy as np

from dronecv.gis.providers.overture import _parse_hex_color
from dronecv.gis.providers.wikidata_heights import enrich_landmark_heights
from dronecv.localization.geofusion.finepose import view_alignment
from dronecv.localization.geofusion.invariant import InvariantView


class TestHexColor:
    def test_parses_and_rejects(self):
        assert _parse_hex_color("#8b4513") == (139 / 255, 69 / 255, 19 / 255)
        assert _parse_hex_color("8B4513") == (139 / 255, 69 / 255, 19 / 255)
        assert _parse_hex_color(None) is None
        assert _parse_hex_color("red") is None
        assert _parse_hex_color("#zzzzzz") is None


class TestWikidataHeights:
    class _Poi:
        def __init__(self, wd):
            self.tags = {"wikidata": wd} if wd else {}
            self.wikidata = wd

    class _Bld:
        def __init__(self, h, src):
            self.height_m = h
            self.height_source = src

    def test_enriches_weak_sources_only(self, monkeypatch):
        import dronecv.gis.providers.wikidata_heights as wh

        monkeypatch.setattr(wh, "fetch_height_m", lambda wd: 96.0)
        weak = self._Bld(12.0, "neighbor_median")
        tagged = self._Bld(30.0, "tag_height")
        n = enrich_landmark_heights(
            [self._Poi("Q1"), self._Poi("Q2")],
            [(self._Poi("Q1"), weak), (self._Poi("Q2"), tagged)])
        assert n == 1
        assert weak.height_m == 96.0 and weak.height_source == "wikidata"
        assert tagged.height_m == 30.0  # real tags are never overwritten

    def test_network_failure_is_silent(self, monkeypatch):
        import dronecv.gis.providers.wikidata_heights as wh

        def boom(wd):
            raise OSError("blocked")

        monkeypatch.setattr(wh, "fetch_height_m", boom)
        b = self._Bld(10.0, "none")
        assert enrich_landmark_heights([], [(self._Poi("Q1"), b)]) == 0
        assert b.height_m == 10.0


def _view(depth, building, relative=False):
    h, w = depth.shape
    return InvariantView(depth=depth.astype(np.float32), building=building.astype(bool),
                         vegetation=np.zeros_like(building, bool),
                         points=np.zeros((h, w, 3)), pos=np.zeros(3), yaw_deg=0.0,
                         pitch_down_deg=35.0, fov_deg=70.0, depth_relative=relative)


class TestRelativeDepthAlignment:
    def test_scale_cancels_for_relative_depth(self):
        rng = np.random.default_rng(0)
        metric = rng.uniform(20, 400, (48, 64))
        bld = rng.random((48, 64)) > 0.6
        render = _view(metric, bld)
        photo_rel = _view(metric * 0.037, bld, relative=True)  # unknown scale
        assert view_alignment(photo_rel, render) > 0.9

    def test_wrong_structure_scores_lower(self):
        rng = np.random.default_rng(1)
        bld = rng.random((48, 64)) > 0.6
        a = _view(rng.uniform(20, 400, (48, 64)) * 0.05, bld, relative=True)
        b_same = _view(a.depth / 0.05, bld)
        b_wrong = _view(rng.uniform(20, 400, (48, 64)), bld)
        assert view_alignment(a, b_same) > view_alignment(a, b_wrong)


def test_depth_plugin_absent_returns_none():
    import dronecv.localization.geofusion.depth_plugin as dp

    dp._cached = "unset"
    assert dp.get_depth_fn() is None  # no transformers in this environment
