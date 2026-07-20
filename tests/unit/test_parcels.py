"""Fixed 500 m cell grid + per-cell cache + fetch orchestrator/decisions."""

from __future__ import annotations

import pytest

from dronecv.gis.geometry import BBox
from dronecv.gis.parcel_cache import CellCache
from dronecv.gis.parcel_fetch import (
    BuildAborted,
    Decision,
    fetch_cells,
    merge_dedup,
)
from dronecv.gis.parcels import (
    Cell,
    cell_of,
    cells_for_bbox,
    cells_for_selection,
)
from dronecv.gis.providers.buildings import Building


class TestGrid:
    def test_deterministic_same_cell(self):
        # Two points in the same 500 m square map to the same cell + id.
        a = cell_of(47.5001, 19.0401)
        b = cell_of(47.5003, 19.0403)
        assert a == b and a.id == b.id
        # id is stable and encodes the fixed grid indices.
        assert a.id == f"c500_{a.ilat}_{a.ilon}"

    def test_cell_size_about_500m_NS(self):
        c = cell_of(47.5, 19.04)
        b = c.bbox
        assert 499.0 < (b.north - b.south) * 111_320.0 < 501.0

    def test_cells_cover_bbox_and_are_contiguous(self):
        bbox = BBox(47.500, 19.040, 47.510, 19.055)
        cells = cells_for_bbox(bbox)
        assert len(cells) > 1
        # every cell touches the bbox; union spans it
        assert min(c.bbox.south for c in cells) <= bbox.south
        assert max(c.bbox.north for c in cells) >= bbox.north
        # adjacency: consecutive ilat/ilon indices with no gaps
        ilats = sorted({c.ilat for c in cells})
        assert ilats == list(range(ilats[0], ilats[-1] + 1))

    def test_grid_is_global_fixed_independent_of_aoi(self):
        # The same ground square is the same cell whether requested alone or
        # as part of a bigger AOI.
        small = set(c.id for c in cells_for_bbox(BBox(47.5005, 19.0405, 47.5006, 19.0406)))
        big = set(c.id for c in cells_for_bbox(BBox(47.49, 19.03, 47.51, 19.05)))
        assert small.issubset(big)

    def test_selection_union_dedups_overlaps(self):
        a = BBox(47.500, 19.040, 47.506, 19.050)
        b = BBox(47.503, 19.045, 47.509, 19.055)  # overlaps a
        union = cells_for_selection([a, b])
        ids = [c.id for c in union]
        assert len(ids) == len(set(ids))  # no cell processed twice
        # union equals the set-union of the two individually
        expect = {c.id for c in cells_for_bbox(a)} | {c.id for c in cells_for_bbox(b)}
        assert set(ids) == expect

    def test_selection_accepts_geojson(self):
        gj = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates":
              [[[19.040, 47.500], [19.050, 47.500], [19.050, 47.506], [19.040, 47.506],
                [19.040, 47.500]]]}}
        cells = cells_for_selection([gj])
        assert cells and all(isinstance(c, Cell) for c in cells)


class TestCache:
    def test_roundtrip_and_skip(self, tmp_path):
        cache = CellCache("osm", "v1", root=tmp_path)
        assert not cache.has("c500_1_1")
        cache.put("c500_1_1", [{"a": 1}])
        assert cache.has("c500_1_1") and cache.get("c500_1_1") == [{"a": 1}]
        assert not cache.is_skipped("c500_1_1")
        cache.mark_skip("c500_2_2", "504")
        assert cache.is_skipped("c500_2_2")
        cache.clear_skip("c500_2_2")
        assert not cache.is_skipped("c500_2_2")

    def test_source_version_isolation(self, tmp_path):
        a = CellCache("overture", "2026-06-17.0", root=tmp_path)
        b = CellCache("overture", "2026-07-01.0", root=tmp_path)
        a.put("c500_1_1", [{"r": "a"}])
        assert b.get("c500_1_1") is None  # different release -> different cache


def _bld(osm_id):
    return Building(footprint_lonlat=[(19.0, 47.5), (19.001, 47.5), (19.001, 47.501),
                                      (19.0, 47.5)], osm_id=osm_id)


class _Provider:
    """Fake buildings provider: fails on cells in `fail`, else returns 1 bldg."""

    def __init__(self, fail: set[str] | None = None, error=None):
        self.fail = fail or set()
        self.error = error or RuntimeError("all Overpass endpoints failed (504)")
        self.calls: list[str] = []

    def fetch(self, bbox):
        from dronecv.gis.parcels import cell_of

        c = cell_of((bbox.south + bbox.north) / 2, (bbox.west + bbox.east) / 2)
        self.calls.append(c.id)
        if c.id in self.fail:
            raise self.error
        return [_bld(hash(c.id) % 100000)]


class TestOrchestrator:
    def _cells(self):
        return cells_for_bbox(BBox(47.500, 19.040, 47.505, 19.045))

    def test_all_ok_caches_each(self, tmp_path):
        cells = self._cells()
        cache = CellCache("osm", "v", root=tmp_path)
        prov = _Provider()
        per_cell, rep = fetch_cells("buildings", cells, prov, "v", cache)
        assert len(rep.fetched) == len(cells) and not rep.failed
        # second run hits cache, no new fetches
        prov2 = _Provider()
        _, rep2 = fetch_cells("buildings", cells, prov2, "v", cache)
        assert len(rep2.cached) == len(cells) and prov2.calls == []

    def test_defer_default_collects_failures(self, tmp_path):
        cells = self._cells()
        fail = {cells[0].id, cells[2].id}
        cache = CellCache("osm", "v", root=tmp_path)
        per_cell, rep = fetch_cells("buildings", cells, _Provider(fail), "v", cache)
        assert set(rep.deferred) == fail
        assert len(rep.fetched) == len(cells) - 2

    def test_skip_marks_and_persists(self, tmp_path):
        cells = self._cells()
        cache = CellCache("osm", "v", root=tmp_path)
        fail = {cells[0].id}
        fetch_cells("buildings", cells, _Provider(fail), "v", cache,
                    decide=lambda c, e, s: Decision.SKIP)
        assert cache.is_skipped(cells[0].id)
        # re-run: skipped cell is not retried
        prov = _Provider(fail)
        _, rep = fetch_cells("buildings", cells, prov, "v", cache)
        assert cells[0].id in rep.skipped and cells[0].id not in prov.calls

    def test_abort_stops(self, tmp_path):
        cells = self._cells()
        cache = CellCache("osm", "v", root=tmp_path)
        with pytest.raises(BuildAborted):
            fetch_cells("buildings", cells, _Provider({cells[0].id}), "v", cache,
                        decide=lambda c, e, s: Decision.ABORT)

    def test_fallback_uses_other_source(self, tmp_path):
        cells = self._cells()
        cache = CellCache("osm", "v", root=tmp_path)
        fb_cache = CellCache("overture", "r", root=tmp_path)
        fb = _Provider()  # fallback always succeeds
        per_cell, rep = fetch_cells(
            "buildings", cells, _Provider({cells[0].id}), "v", cache,
            decide=lambda c, e, s: Decision.FALLBACK,
            fallback_provider=fb, fallback_cache=fb_cache,
        )
        assert cells[0].id in rep.fallback
        assert fb_cache.has(cells[0].id)

    def test_apply_to_all_same_error(self, tmp_path):
        cells = self._cells()
        cache = CellCache("osm", "v", root=tmp_path)
        seen_sigs = []
        remembered = {}

        def decide(cell, err, sig):
            seen_sigs.append(sig)
            if sig in remembered:
                return remembered[sig]
            remembered[sig] = Decision.SKIP  # "apply to all with this error"
            return Decision.SKIP

        fail = {cells[0].id, cells[1].id}
        fetch_cells("buildings", cells, _Provider(fail), "v", cache, decide=decide)
        assert all(s == "504" for s in seen_sigs)  # same signature grouped


class TestPartialCellRetry:
    """A cell that got landcover but NOT buildings (buildings deferred) must
    refetch buildings on reuse — each data kind is cached independently, and a
    failed fetch is never cached. Only an explicit SKIP marks a cell final."""

    def test_deferred_kind_is_not_cached_and_refetches(self, tmp_path):
        cells = cells_for_bbox(BBox(47.500, 19.040, 47.503, 19.043))
        # buildings cache: one cell failed (deferred -> no cache written)
        bcache = CellCache("osm-buildings", "osm", root=tmp_path)
        fail = {cells[0].id}
        prov = _Provider(fail)
        fetch_cells("buildings", cells, prov, "osm", bcache)
        assert not bcache.has(cells[0].id)  # deferred cell NOT cached
        assert bcache.has(cells[1].id)      # the others are

        # Reuse the same cells: the deferred one is retried (still no cache),
        # the rest are served from cache (not refetched).
        prov2 = _Provider(fail)
        _, rep = fetch_cells("buildings", cells, prov2, "osm", bcache)
        assert cells[0].id in prov2.calls               # retried
        assert cells[1].id not in prov2.calls           # cached, skipped
        assert cells[0].id in rep.deferred

    def test_cache_write_failure_does_not_defer_a_good_fetch(self, tmp_path):
        # A SUCCESSFUL fetch whose cache write fails must still be USED and
        # reported as fetched (not deferred) — otherwise good data is discarded
        # and re-downloaded every run (the reported resume bug).
        cells = cells_for_bbox(BBox(47.500, 19.040, 47.503, 19.043))
        cache = CellCache("osm-buildings", "osm", root=tmp_path)

        def boom(cell_id, obj):
            raise OSError("disk full / permission denied")

        cache.put = boom  # every cache write fails
        deferred = []
        per_cell, rep = fetch_cells(
            "buildings", cells, _Provider(), "osm", cache,
            decide=lambda c, e, s: deferred.append(c.id) or Decision.DEFER,
        )
        assert rep.fetched == [c.id for c in cells]  # all fetched, used
        assert rep.deferred == [] and deferred == []  # none deferred by a cache error
        assert all(len(r) == 1 for r in per_cell)     # the fetched data is returned

    def test_skip_is_final(self, tmp_path):
        cells = cells_for_bbox(BBox(47.500, 19.040, 47.503, 19.043))
        cache = CellCache("osm-buildings", "osm", root=tmp_path)
        fetch_cells("buildings", cells, _Provider({cells[0].id}), "osm", cache,
                    decide=lambda c, e, s: Decision.SKIP)
        prov2 = _Provider({cells[0].id})
        fetch_cells("buildings", cells, prov2, "osm", cache)
        assert cells[0].id not in prov2.calls  # skipped cell is never retried


class TestMerge:
    def test_dedup_boundary_buildings_by_osm_id(self):
        # same building returned by two adjacent cells
        shared = _bld(42)
        merged = merge_dedup("buildings", [[shared, _bld(1)], [shared, _bld(2)]])
        ids = sorted(b.osm_id for b in merged)
        assert ids == [1, 2, 42]  # 42 kept once
