"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_gis_cell_cache(tmp_path_factory, monkeypatch):
    """Point the per-cell cache AND the GUI config dir at throwaway dirs so
    tests never read or write the developer's real ~/.cache or ~/.config and
    stay hermetic."""
    monkeypatch.setenv("DRONECV_CACHE_DIR", str(tmp_path_factory.mktemp("cellcache")))
    monkeypatch.setenv("DRONECV_CONFIG_DIR", str(tmp_path_factory.mktemp("guiconfig")))
