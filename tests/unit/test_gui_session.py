"""GUI session persistence (map view + selection + params)."""

from dronecv.gui.session import load_session, save_session


def test_roundtrip(tmp_path):
    p = tmp_path / "gui_state.json"
    data = {"map": {"lat": 47.5, "lon": 19.04, "zoom": 14},
            "selection": {"type": "Polygon", "coordinates": [[[19, 47], [19.1, 47]]]},
            "env_name": "budapest", "res_m": 1.0}
    save_session(data, p)
    assert load_session(p) == data


def test_missing_file_is_empty(tmp_path):
    assert load_session(tmp_path / "nope.json") == {}


def test_corrupt_file_is_empty(tmp_path):
    p = tmp_path / "gui_state.json"
    p.write_text("{ not json")
    assert load_session(p) == {}


def test_env_override(monkeypatch, tmp_path):
    from dronecv.gui.session import session_path

    monkeypatch.setenv("DRONECV_CONFIG_DIR", str(tmp_path))
    assert session_path() == tmp_path / "gui_state.json"
