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
