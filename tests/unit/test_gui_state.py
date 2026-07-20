"""Display-independent GUI logic (GuiState) — no Qt required."""

import json

import pytest

from dronecv.gui.app import GuiState


def _rect_geojson(s, w, n, e):
    return json.dumps({
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
        },
    })


def test_mask_to_bbox():
    st = GuiState()
    bbox = st.set_mask(_rect_geojson(43.31, 11.32, 43.33, 11.35))
    assert bbox.south == pytest.approx(43.31)
    assert bbox.east == pytest.approx(11.35)
    # Irregular polygon: bbox is the hull.
    st.set_mask(json.dumps({
        "type": "Polygon",
        "coordinates": [[[11.30, 43.30], [11.36, 43.31], [11.33, 43.34], [11.30, 43.30]]],
    }))
    assert st.bbox.north == pytest.approx(43.34)


def test_coverage_preselection_and_gating():
    st = GuiState()
    ok, why = st.can_build()
    assert not ok and "draw" in why

    st.set_mask(_rect_geojson(43.31, 11.32, 43.33, 11.35))
    ok, why = st.can_build()
    assert not ok and "name" in why

    st.env_name = "siena centro"  # invalid: space
    assert not st.can_build()[0]
    st.env_name = "siena_centro"

    selected = st.apply_coverage({
        "dem": {"coverage": 1.0, "available": 1, "tiles": [{}]},
        "buildings": {"n_buildings": 300, "height_coverage": 0.7},
        "recommended": {"dem": "copernicus_glo30", "buildings": "osm_overpass", "heights": "tags"},
    })
    assert selected["heights"] == "tags"
    assert st.can_build() == (True, "ready")

    # No DEM coverage blocks the build.
    st.apply_coverage({
        "dem": {"coverage": 0.0, "available": 0, "tiles": [{}]},
        "buildings": {"n_buildings": 0, "height_coverage": 0.0},
        "recommended": {"dem": None, "buildings": None, "heights": "shadow+defaults"},
    })
    ok, why = st.can_build()
    assert not ok and "DEM" in why


def test_build_kwargs():
    st = GuiState()
    st.set_mask(_rect_geojson(43.31, 11.32, 43.33, 11.35))
    st.env_name = "test-env"
    st.res_m = 2.0
    st.ortho_utc = "2026-06-21T10:00:00+00:00"
    kw = st.build_kwargs()
    assert kw["env_name"] == "test-env"
    assert kw["res_m"] == 2.0
    assert kw["ortho_utc"].year == 2026
    assert kw["ortho_path"] is None


def test_source_selection_and_build_kwargs():
    st = GuiState()
    st.set_mask(_rect_geojson(43.31, 11.32, 43.33, 11.35))
    st.env_name = "x"
    # Overture wins when it has more buildings; s2 auto-picked when available.
    st.apply_coverage({
        "dem": {"coverage": 1.0},
        "buildings": {"available": True, "n_buildings": 100, "height_coverage": 0.2},
        "buildings_overture": {"available": True, "n_buildings": 900, "height_coverage": 0.6},
        "imagery_s2": {"available": True, "n_recent_scenes": 12},
        "recommended": {"dem": "copernicus_glo30", "buildings": "overture",
                        "heights": "tags", "imagery": "s2"},
    })
    assert st.selected_sources["buildings"] == "overture"
    assert st.imagery == "s2"
    kw = st.build_kwargs()
    assert kw["imagery"] == "s2"
    from dronecv.gis.providers.overture import OvertureBuildingsProvider

    assert isinstance(kw["sources"].buildings, OvertureBuildingsProvider)
    assert kw["sources"].poi is not None


def test_build_kwargs_defaults_osm_no_sources():
    st = GuiState()
    st.set_mask(_rect_geojson(43.31, 11.32, 43.33, 11.35))
    st.env_name = "x"
    kw = st.build_kwargs()
    assert "sources" not in kw          # OSM path uses the pipeline defaults
    assert kw["imagery"] is None        # "none" maps to no imagery
    assert kw["reconstruct_buildings"] is False


def test_coverage_recommendation_prefers_reachable_source():
    from dronecv.gis.coverage import coverage_report  # noqa: F401 (import check)
    # Pure-logic check of the preference rule mirrored in apply_coverage:
    st = GuiState()
    st.set_mask(_rect_geojson(43.31, 11.32, 43.33, 11.35))
    st.apply_coverage({
        "dem": {"coverage": 1.0},
        "buildings": {"available": False, "n_buildings": 0, "height_coverage": 0.0},
        "recommended": {"dem": "copernicus_glo30", "buildings": "overture",
                        "heights": "shadow+defaults"},
    })
    assert st.selected_sources["buildings"] == "overture"


def test_cell_decision_controller_apply_to_all():
    from dronecv.gis.parcel_fetch import Decision
    from dronecv.gui.app import CellDecisionController

    ctrl = CellDecisionController()
    asked = []

    def ask(cell, err, sig):
        asked.append(sig)
        return Decision.SKIP, True  # apply to all with this signature

    # First 504 asks the user; subsequent 504s reuse the memo without asking.
    d1 = ctrl.decide("c1", RuntimeError("504"), "504", ask)
    d2 = ctrl.decide("c2", RuntimeError("504"), "504", ask)
    assert d1 == d2 == Decision.SKIP
    assert asked == ["504"]  # asked once, applied to all

    # A different signature asks again.
    ctrl.decide("c3", RuntimeError("timeout"), "timeout", ask)
    assert asked == ["504", "timeout"]


def test_cell_decision_no_apply_asks_each_time():
    from dronecv.gis.parcel_fetch import Decision
    from dronecv.gui.app import CellDecisionController

    ctrl = CellDecisionController()
    calls = {"n": 0}

    def ask(cell, err, sig):
        calls["n"] += 1
        return Decision.DEFER, False  # do NOT apply to all

    ctrl.decide("c1", RuntimeError("504"), "504", ask)
    ctrl.decide("c2", RuntimeError("504"), "504", ask)
    assert calls["n"] == 2  # asked every time


def test_cells_json_for_selection():
    from dronecv.gui.app import GuiState

    st = GuiState()
    st.set_mask(_rect_geojson(47.500, 19.040, 47.506, 19.050))
    import json
    cells = json.loads(st.cells_json())
    assert cells and all({"id", "s", "w", "n", "e"} <= set(c) for c in cells)
