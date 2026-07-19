"""Saliency prior: repetitive fabric gets denser POVs than distinctive areas,
and the planner honors the density function."""

import numpy as np

from dronecv.capture import planner
from dronecv.config import CaptureConfig
from dronecv.gis.saliency import compute_saliency, orbit_points_from_store
from dronecv.gis.store import GisMeta, GisStore

BMIN = np.array([-500.0, 0.0, -500.0])
BMAX = np.array([500.0, 150.0, 500.0])


def _store(tmp_path, build_fn):
    size = 512
    meta = GisMeta(
        res_m=2.0, e0=-size, n0=-size, width=size, height=size,
        anchor={"lat0": 43, "lon0": 11, "alt0": 0, "true_north_offset_deg": 0, "source": "env_config"},
        ground_alt0=0.0, max_height=60.0,
    )
    store = GisStore.create(tmp_path, meta)
    build_fn(store)
    store.flush()
    return store


def _repetitive(store):
    # Identical 10 m blocks on a regular grid everywhere.
    for r in range(20, 500, 60):
        for c in range(20, 500, 60):
            store.build_h[r : r + 12, c : c + 12] = 10.0


def _distinctive(store):
    # One dominant tower + strong relief, empty elsewhere.
    store.build_h[250:262, 250:262] = 55.0
    ys, xs = np.meshgrid(np.arange(512), np.arange(512), indexing="ij")
    store.ground[:] = 40.0 * np.exp(-(((ys - 150) / 90.0) ** 2 + ((xs - 350) / 90.0) ** 2))


def test_repetitive_needs_more_pov_than_distinctive(tmp_path):
    rep = compute_saliency(_store(tmp_path / "rep", _repetitive), cell_m=256.0)
    dis = compute_saliency(_store(tmp_path / "dis", _distinctive), cell_m=256.0)
    assert rep["mean_density_multiplier"] > dis["mean_density_multiplier"]


def test_orbit_points_pick_tallest_structures(tmp_path):
    store = _store(tmp_path / "d", _distinctive)
    pts = orbit_points_from_store(store, top_n=5)
    assert pts, "must find at least the tower"
    e, n = pts[0]
    # Tower is at rows/cols 250-262 -> ENU ~ (-512 + 256*2) = 0.
    assert abs(e) < 30 and abs(n) < 30


def test_grid_plan_density_modulation():
    cfg = CaptureConfig(altitudes_agl_m=[50.0], yaw_bins=2, grid_spacing_m=60.0)

    def density(x, z):
        return 2.2 if x < 0 else 0.5  # west: dense, east: sparse

    poses = planner.grid_plan(BMIN, BMAX, cfg, density=density)
    west = sum(1 for p in poses if p.x < 0)
    east = sum(1 for p in poses if p.x >= 0)
    assert west > east * 2.5  # ~4.4x expected
    # Determinism.
    poses2 = planner.grid_plan(BMIN, BMAX, cfg, density=density)
    assert poses == poses2
    # Without density: uniform baseline unchanged in behavior.
    uniform = planner.grid_plan(BMIN, BMAX, cfg)
    xs = {p.x for p in uniform}
    assert len(xs) > 5
