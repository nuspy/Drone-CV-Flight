"""VO on rendered frame pairs with known ground-truth motion."""

import numpy as np
import pytest

from dronecv.config import WorldConfig
from dronecv.models.vo import VisualOdometry, body_to_world_disp
from dronecv.sim.headless import rasterizer
from dronecv.sim.headless.world import World

W = H = 128
FOV = 70.0
TILT = 35.0


@pytest.fixture(scope="module")
def world():
    return World(WorldConfig(size_m=500.0, height_scale_m=40.0, n_landmarks=12, grid=96), seed=0)


def _render_gray(world, pos, yaw):
    rgb, _ = rasterizer.render(
        world, np.asarray(pos, float), yaw, TILT, W, H, FOV,
        sun_azimuth_sim_deg=150.0, sun_elevation_deg=50.0,
    )
    import cv2

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _agl(world, x, z, y):
    return y - float(world.height_at(np.array(x), np.array(z)))


def test_vo_detects_yaw_rotation(world):
    pos = [20.0, 80.0, -30.0]
    g1 = _render_gray(world, pos, 40.0)
    g2 = _render_gray(world, pos, 48.0)  # +8 deg yaw
    vo = VisualOdometry(W, H, FOV, TILT)
    res = vo.compute(g1, g2, _agl(world, 20.0, -30.0, 80.0))
    assert res is not None and res.inliers >= 10
    assert res.dyaw_deg == pytest.approx(8.0, abs=2.0)
    if res.disp_body_m is not None:
        assert np.linalg.norm(res.disp_body_m) < 3.0  # no translation


def test_vo_translation_ground_flow(world):
    # Move 6 m along +x, constant yaw: dyaw ~ 0, displacement ~ (6, 0, 0).
    y = 80.0
    agl = _agl(world, 0.0, 0.0, y)
    g1 = _render_gray(world, [0.0, y, 0.0], 0.0)
    g2 = _render_gray(world, [6.0, y, 0.0], 0.0)
    vo = VisualOdometry(W, H, FOV, TILT)
    res = vo.compute(g1, g2, agl)
    assert res is not None and res.inliers >= 10
    assert abs(res.dyaw_deg) < 2.0
    assert res.disp_body_m is not None
    disp_world = body_to_world_disp(res.disp_body_m, 0.0)
    assert disp_world[0] == pytest.approx(6.0, abs=2.0)
    assert abs(disp_world[2]) < 2.0


def test_vo_combined_motion(world):
    # Forward 5 m along the body axis while yawing +5 deg, yaw 90 (facing +x).
    y = 75.0
    agl = _agl(world, -40.0, 60.0, y)
    g1 = _render_gray(world, [-40.0, y, 60.0], 90.0)
    g2 = _render_gray(world, [-35.0, y, 60.0], 95.0)  # moved +5 x (= body fwd at yaw 90)
    vo = VisualOdometry(W, H, FOV, TILT)
    res = vo.compute(g1, g2, agl)
    assert res is not None and res.inliers >= 10
    assert res.dyaw_deg == pytest.approx(5.0, abs=2.5)
    disp_world = body_to_world_disp(res.disp_body_m, 90.0)
    assert disp_world[0] == pytest.approx(5.0, abs=2.5)
    assert abs(disp_world[2]) < 2.5


def test_vo_yaw_only_without_agl(world):
    pos = [20.0, 80.0, -30.0]
    g1 = _render_gray(world, pos, 40.0)
    g2 = _render_gray(world, pos, 47.0)
    vo = VisualOdometry(W, H, FOV, TILT)
    res = vo.compute(g1, g2, None)
    assert res is not None
    assert res.disp_body_m is None
    assert res.dyaw_deg == pytest.approx(7.0, abs=2.0)


def test_vo_rejects_featureless():
    vo = VisualOdometry(W, H, FOV, TILT)
    flat = np.full((H, W), 128, dtype=np.uint8)
    assert vo.compute(flat, flat, 50.0) is None
