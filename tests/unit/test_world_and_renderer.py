import numpy as np
import pytest

from dronecv.config import WorldConfig
from dronecv.sim.headless import rasterizer
from dronecv.sim.headless.world import World


@pytest.fixture(scope="module")
def world():
    return World(WorldConfig(size_m=500.0, height_scale_m=40.0, n_landmarks=12, grid=96), seed=0)


def test_world_is_seeded_deterministic():
    cfg = WorldConfig(size_m=500.0, n_landmarks=8, grid=64)
    w1, w2 = World(cfg, seed=5), World(cfg, seed=5)
    np.testing.assert_array_equal(w1.heightmap, w2.heightmap)
    assert [(lm.x, lm.z) for lm in w1.landmarks] == [(lm.x, lm.z) for lm in w2.landmarks]
    w3 = World(cfg, seed=6)
    assert not np.array_equal(w1.heightmap, w3.heightmap)


def test_height_bounds_and_landmarks(world):
    assert world.heightmap.min() >= 0.0
    assert world.heightmap.max() <= 40.0 * 1.01
    assert len(world.landmarks) == 12
    for lm in world.landmarks:
        assert abs(lm.x) < world.half and abs(lm.z) < world.half


def test_height_query_vectorized(world):
    xs = np.linspace(-200, 200, 40)
    zs = np.zeros_like(xs)
    h = world.height_at(xs, zs)
    assert h.shape == (40,)
    assert (h >= 0).all()


def test_render_shapes_and_content(world):
    rgb, depth = rasterizer.render(
        world, np.array([0.0, 90.0, 0.0]), yaw_deg=30.0, pitch_down_deg=35.0,
        width=64, height=64, fov_deg=70.0,
        sun_azimuth_sim_deg=150.0, sun_elevation_deg=50.0,
    )
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8
    assert depth.shape == (64, 64) and depth.dtype == np.float32
    # Down-tilted from 90 m: bottom rows terrain, top rows likely sky.
    assert (depth[48:] > 0).mean() > 0.9
    assert rgb.std() > 10  # not a flat image


def test_depth_matches_geometry(world):
    # Straight down ray: depth center ~= altitude - terrain height.
    pos = np.array([50.0, 100.0, 50.0])
    _, depth = rasterizer.render(
        world, pos, yaw_deg=0.0, pitch_down_deg=90.0, width=33, height=33,
        fov_deg=60.0, sun_azimuth_sim_deg=180.0, sun_elevation_deg=45.0,
    )
    center = float(depth[16, 16])
    expected = 100.0 - float(world.height_at(np.array(50.0), np.array(50.0)))
    assert center == pytest.approx(expected, abs=1.5)


def test_sun_disc_appears_in_sky(world):
    # Look straight at the sun: a bright warm disc must be visible.
    rgb, _ = rasterizer.render(
        world, np.array([0.0, 80.0, 0.0]), yaw_deg=90.0, pitch_down_deg=-40.0,
        width=64, height=64, fov_deg=70.0,
        sun_azimuth_sim_deg=90.0, sun_elevation_deg=40.0,
    )
    bright = (rgb[..., 0] > 240) & (rgb[..., 1] > 230) & (rgb[..., 2] < 240)
    assert bright.sum() >= 4
    # Turn 180 degrees away: no sun disc.
    rgb2, _ = rasterizer.render(
        world, np.array([0.0, 80.0, 0.0]), yaw_deg=270.0, pitch_down_deg=-40.0,
        width=64, height=64, fov_deg=70.0,
        sun_azimuth_sim_deg=90.0, sun_elevation_deg=40.0,
    )
    bright2 = (rgb2[..., 0] > 240) & (rgb2[..., 1] > 230) & (rgb2[..., 2] < 240)
    assert bright2.sum() == 0


def test_landmark_visible_when_looked_at(world):
    lm = world.landmarks[0]
    # Camera 150 m south of the landmark at its mid height, looking at it.
    cam = np.array([lm.x, lm.base_y + lm.height * 0.5 + 5, lm.z - 150.0])
    rgb_at, depth_at = rasterizer.render(
        world, cam, yaw_deg=0.0, pitch_down_deg=0.0, width=64, height=64,
        fov_deg=70.0, sun_azimuth_sim_deg=180.0, sun_elevation_deg=50.0,
    )
    rgb_away, _ = rasterizer.render(
        world, cam, yaw_deg=180.0, pitch_down_deg=0.0, width=64, height=64,
        fov_deg=70.0, sun_azimuth_sim_deg=180.0, sun_elevation_deg=50.0,
    )
    # Images should differ a lot; landmark colors are saturated.
    assert np.abs(rgb_at.astype(int) - rgb_away.astype(int)).mean() > 5


def test_drone_body_dynamics():
    from dronecv.config import DroneConfig
    from dronecv.sim.headless.drone import DroneBody

    body = DroneBody(DroneConfig())
    body.reset(np.array([0.0, 50.0, 0.0]))
    body.set_command(np.array([100.0, 0.0, 0.0]), None)  # clamped to max speed
    for _ in range(100):
        body.step(0.1, lambda x, z: 0.0)
    v = body.state.vel_sim
    assert v[0] == pytest.approx(12.0, abs=0.5)  # max_speed_ms
    assert body.state.pos_sim[0] > 50

    body.set_command(np.array([0.0, -100.0, 0.0]), None)
    for _ in range(400):
        body.step(0.1, lambda x, z: 0.0)
    assert body.state.collided
    assert body.state.pos_sim[1] == pytest.approx(0.5)
