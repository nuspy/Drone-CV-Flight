"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_gis_cell_cache(tmp_path_factory, monkeypatch):
    """Point the per-cell download cache at a throwaway dir so tests never
    read or write the developer's real ~/.cache/dronecv and stay hermetic."""
    d = tmp_path_factory.mktemp("cellcache")
    monkeypatch.setenv("DRONECV_CACHE_DIR", str(d))
