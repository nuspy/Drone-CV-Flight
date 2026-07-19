import math

import numpy as np
import pytest

from dronecv.config import LocalizationConfig
from dronecv.localization.fusion import EkfFuser


def make_fuser() -> EkfFuser:
    f = EkfFuser(LocalizationConfig())
    f.initialize(np.array([0.0, 0.0, 50.0]), np.eye(3) * 25.0, yaw_deg=0.0)
    return f


def test_uninitialized_is_inert():
    f = EkfFuser(LocalizationConfig())
    assert not f.initialized
    assert not f.update_position(np.zeros(3), np.eye(3), "apr")
    assert f.confidence(10.0) == 0.0


def test_converges_to_static_truth():
    f = make_fuser()
    rng = np.random.default_rng(0)
    truth = np.array([120.0, -40.0, 70.0])
    for _ in range(60):
        f.predict(0.2)
        z = truth + rng.normal(0, 5.0, 3)
        f.update_position(z, np.eye(3) * 25.0, "apr")
    assert np.linalg.norm(f.pos - truth) < 5.0
    assert f.pos_sigma_h < 4.0


def test_tracks_moving_target_velocity():
    f = make_fuser()
    rng = np.random.default_rng(1)
    vel = np.array([6.0, 3.0, 0.0])
    dt = 0.2
    pos = np.array([0.0, 0.0, 50.0])
    for _ in range(100):
        pos = pos + vel * dt
        f.predict(dt)
        f.update_position(pos + rng.normal(0, 4.0, 3), np.eye(3) * 16.0, "apr")
    assert np.linalg.norm(f.vel[:2] - vel[:2]) < 2.0
    assert np.linalg.norm(f.pos[:2] - pos[:2]) < 8.0


def test_gate_rejects_outliers_and_lost_mode():
    cfg = LocalizationConfig()
    f = EkfFuser(cfg)
    f.initialize(np.array([0.0, 0.0, 50.0]), np.eye(3) * 4.0, yaw_deg=0.0)
    for _ in range(5):
        f.predict(0.2)
        f.update_position(np.array([0.0, 0.0, 50.0]), np.eye(3) * 4.0, "apr")
    # Wild outlier gets gated out.
    assert not f.update_position(np.array([500.0, 500.0, 50.0]), np.eye(3) * 4.0, "apr")
    assert np.linalg.norm(f.pos[:2]) < 2.0
    # A streak of rejections flips lost mode.
    for _ in range(cfg.lost_reject_streak):
        f.predict(0.2)
        f.update_position(np.array([500.0, 500.0, 50.0]), np.eye(3) * 4.0, "apr")
    assert f.lost
    # Re-initialization clears it.
    f.initialize(np.array([500.0, 500.0, 50.0]), np.eye(3) * 25.0, yaw_deg=10.0)
    assert not f.lost


def test_yaw_update_wraps_correctly():
    f = make_fuser()
    f.x[6] = math.radians(350.0)
    for _ in range(20):
        f.predict(0.2)
        f.update_yaw(10.0, 5.0, "sun")  # 20 deg away through the wrap
    err = (f.yaw_deg - 10.0 + 180.0) % 360.0 - 180.0
    assert abs(err) < 3.0


def test_velocity_update_direct():
    f = make_fuser()
    for _ in range(30):
        f.predict(0.2)
        f.update_velocity(np.array([5.0, 0.0, 0.0]), np.eye(3) * 1.0)
    assert f.vel[0] == pytest.approx(5.0, abs=0.6)


def test_altitude_update():
    f = make_fuser()
    for _ in range(20):
        f.predict(0.2)
        f.update_altitude(80.0, sigma=2.0)
    assert f.pos[2] == pytest.approx(80.0, abs=2.0)


def test_confidence_monotone_in_covariance():
    f = make_fuser()
    c_wide = f.confidence(15.0)
    for _ in range(30):
        f.predict(0.2)
        f.update_position(np.array([0.0, 0.0, 50.0]), np.eye(3) * 4.0, "apr")
    c_tight = f.confidence(15.0)
    assert c_tight > c_wide
    assert 0.0 <= c_wide <= 1.0 and 0.0 <= c_tight <= 1.0
    # Health scaling caps it.
    assert f.confidence(15.0, cue_health=0.5) < c_tight


def test_covariance_stays_symmetric_positive():
    f = make_fuser()
    rng = np.random.default_rng(2)
    for i in range(200):
        f.predict(0.2)
        if i % 3 == 0:
            f.update_position(rng.normal(0, 5, 3) + [0, 0, 50], np.eye(3) * 25.0, "apr")
        if i % 5 == 0:
            f.update_yaw(rng.normal(0, 5), 6.0, "sun")
    np.testing.assert_allclose(f.P, f.P.T, atol=1e-9)
    assert (np.linalg.eigvalsh(f.P) > -1e-9).all()
