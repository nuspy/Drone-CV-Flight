"""Build pipeline through the cell orchestrator: defer failures, manifest,
resume from cache."""

import json
from pathlib import Path

import pytest

from dronecv.gis.geometry import BBox
from dronecv.gis.pipeline import BuildSources, build_environment
from dronecv.gis.providers.buildings import OverpassBuildings
from dronecv.gis.providers.dem import CopernicusDem
from dronecv.gis.providers.landcover import OverpassLandcover

ROOT = Path(__file__).parent.parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "gis"
BBOX = BBox(south=43.3145, west=11.3225, north=43.3255, east=11.3375)

pytestmark = pytest.mark.integration


class _FlakyBuildings(OverpassBuildings):
    """Fixture buildings that raise on the first fetch of each cell until
    `heal` is set — to exercise defer + resume."""

    def __init__(self, fixture_path, fail_ids):
        super().__init__(fixture_path=fixture_path)
        self.fail_ids = set(fail_ids)
        self.heal = False

    def fetch(self, bbox):
        from dronecv.gis.parcels import cell_of

        c = cell_of(*BBox(bbox.south, bbox.west, bbox.north, bbox.east).center)
        if c.id in self.fail_ids and not self.heal:
            raise RuntimeError("all Overpass endpoints failed (504)")
        return super().fetch(bbox)


def _sources(buildings):
    return BuildSources(
        dem=CopernicusDem(tile_dir=FIXTURES),
        buildings=buildings,
        landcover=OverpassLandcover(fixture_path=FIXTURES / "overpass_landcover.json"),
    )


def test_defer_writes_manifest_and_resumes(tmp_path):
    from dronecv.gis.parcels import cells_for_bbox

    cache = tmp_path / "cache"
    fail = {cells_for_bbox(BBOX)[0].id}
    flaky = _FlakyBuildings(FIXTURES / "overpass_buildings.json", fail)

    gis_dir = build_environment(
        BBOX, "cells_env", out_root=tmp_path / "art", configs_root=tmp_path,
        sources=_sources(flaky), res_m=2.0, cache_root=cache, on_cell_fail="defer",
    )
    manifest = json.loads((gis_dir / "download_manifest.json").read_text())
    assert manifest["cell_m"] == 500.0
    assert set(manifest["deferred"]) == fail  # the failing cell was deferred

    # Resume: heal the provider and rebuild — cached cells are skipped, only
    # the deferred one is refetched, and it now succeeds (no deferred left).
    flaky.heal = True
    gis_dir2 = build_environment(
        BBOX, "cells_env2", out_root=tmp_path / "art2", configs_root=tmp_path,
        sources=_sources(flaky), res_m=2.0, cache_root=cache, on_cell_fail="defer",
    )
    manifest2 = json.loads((gis_dir2 / "download_manifest.json").read_text())
    assert manifest2["deferred"] == []


def test_skip_marks_cell_and_second_build_is_cached(tmp_path):
    cache = tmp_path / "cache"
    flaky = _FlakyBuildings(FIXTURES / "overpass_buildings.json", set())
    build_environment(
        BBOX, "e1", out_root=tmp_path / "a", configs_root=tmp_path,
        sources=_sources(flaky), res_m=2.0, cache_root=cache,
    )
    # Second build of the SAME area: buildings come from cache (no fetch).
    calls = {"n": 0}
    orig = OverpassBuildings.fetch

    def counting_fetch(self, bbox):
        calls["n"] += 1
        return orig(self, bbox)

    fresh = OverpassBuildings(fixture_path=FIXTURES / "overpass_buildings.json")
    fresh.fetch = counting_fetch.__get__(fresh, OverpassBuildings)
    build_environment(
        BBOX, "e2", out_root=tmp_path / "b", configs_root=tmp_path,
        sources=_sources(fresh), res_m=2.0, cache_root=cache,
    )
    assert calls["n"] == 0  # every cell served from the shared cache
