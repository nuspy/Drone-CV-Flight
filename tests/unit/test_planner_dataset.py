import numpy as np

from dronecv.capture import planner
from dronecv.capture.dataset import CaptureDataset, DatasetWriter
from dronecv.config import CaptureConfig

BMIN = np.array([-250.0, 0.0, -250.0])
BMAX = np.array([250.0, 150.0, 250.0])


def _cfg():
    return CaptureConfig(altitudes_agl_m=[40.0, 80.0], yaw_bins=4, grid_spacing_m=100.0)


def test_grid_plan_covers_bounds():
    poses = planner.grid_plan(BMIN, BMAX, _cfg())
    assert len(poses) > 0
    xs = {p.x for p in poses}
    assert len(xs) >= 4  # multiple stations
    for p in poses:
        assert -250 <= p.x <= 250 and -250 <= p.z <= 250
        assert p.agl_m in (40.0, 80.0)
    yaws = {p.yaw_deg for p in poses}
    assert yaws == {0.0, 90.0, 180.0, 270.0}


def test_orbit_plan_faces_center():
    cfg = _cfg()
    poses = planner.orbit_plan([(0.0, 0.0)], cfg)
    for p in poses[: cfg.orbit_points]:
        # Camera at (x, z) with yaw pointing at origin: bearing of -pos.
        expected = np.degrees(np.arctan2(-p.x, -p.z)) % 360.0
        assert abs((p.yaw_deg - expected + 180) % 360 - 180) < 1e-6


def test_targeted_plan_weights_and_bounds():
    cells = [(0.0, 0.0, 10.0), (100.0, 100.0, 0.1)]
    poses = planner.targeted_plan(cells, 50.0, 100, _cfg(), seed=1)
    assert len(poses) == 100
    near_first = sum(1 for p in poses if abs(p.x) < 30 and abs(p.z) < 30)
    assert near_first > 60  # weight 10 vs 0.1


def test_random_plan_and_shuffle_deterministic():
    p1 = planner.random_plan(BMIN, BMAX, 20, _cfg(), seed=3)
    p2 = planner.random_plan(BMIN, BMAX, 20, _cfg(), seed=3)
    assert p1 == p2
    s1 = planner.shuffle_and_cap(p1, 10, seed=4)
    s2 = planner.shuffle_and_cap(p1, 10, seed=4)
    assert s1 == s2 and len(s1) == 10


def _write_fake_dataset(tmp_path, n=40, cell_pattern=50.0):
    rng = np.random.default_rng(0)
    writer = DatasetWriter(tmp_path, info={"env": "t", "anchor": {
        "lat0": 45.0, "lon0": 9.0, "alt0": 100.0, "true_north_offset_deg": 0.0, "source": "env_config"}})
    for i in range(n):
        img = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
        pos = [float(rng.uniform(-200, 200)), float(rng.uniform(-200, 200)), float(rng.uniform(40, 90))]
        writer.add(img, dict(
            pos_sim=[pos[0], pos[2], pos[1]],
            pos_enu=pos,
            yaw_sim_deg=float(i * 9 % 360),
            heading_deg=float(i * 9 % 360),
            pitch_deg=35.0,
            agl_m=50.0,
            ground_y_sim=10.0,
            utc="2026-06-21T10:00:00+00:00",
            sun_azimuth_deg=150.0,
            sun_elevation_deg=50.0,
        ))
    writer.close()


def test_dataset_round_trip_and_split(tmp_path):
    _write_fake_dataset(tmp_path, n=40)
    ds = CaptureDataset(tmp_path)
    assert len(ds) == 40
    img = ds.image(7)
    assert img.shape == (16, 16, 3)

    train, val = ds.spatial_split(cell_m=50.0, val_fraction=0.3, seed=0)
    assert len(train) + len(val) == 40
    assert len(val) > 0
    # Whole cells are held out: no cell appears on both sides.
    train_cells = {ds.cell_of(i, 50.0) for i in train}
    val_cells = {ds.cell_of(i, 50.0) for i in val}
    assert not (train_cells & val_cells)
    # Deterministic.
    train2, val2 = ds.spatial_split(cell_m=50.0, val_fraction=0.3, seed=0)
    assert train == train2 and val == val2


def test_dataset_append_mode(tmp_path):
    _write_fake_dataset(tmp_path, n=10)
    _write_fake_dataset(tmp_path, n=10)  # second writer appends
    ds = CaptureDataset(tmp_path)
    assert len(ds) == 20
    assert ds.image(15).shape == (16, 16, 3)
