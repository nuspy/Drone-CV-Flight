from datetime import UTC, datetime

import numpy as np
import pytest

from dronecv.config import AnchorConfig, WorldConfig
from dronecv.geo import celestial
from dronecv.geo.anchor import GeoAnchor
from dronecv.localization import measurements as meas
from dronecv.localization.terrain import TerrainPrior
from dronecv.models.vo import camera_matrix
from dronecv.sim.headless import rasterizer
from dronecv.sim.headless.world import World

ANCHOR = GeoAnchor.resolve(
    AnchorConfig(lat0=45.4642, lon0=9.19, alt0=120.0, true_north_offset_deg=0.0), None
)


def test_terrain_prior_build_query_save(tmp_path):
    rng = np.random.default_rng(0)
    pos = rng.uniform(-100, 100, (200, 3))
    ground = 10.0 + pos[:, 0] * 0.05  # gentle slope
    prior = TerrainPrior.from_records(pos, ground, cell_m=25.0)
    assert prior.elevation(50.0, 0.0) == pytest.approx(10.0 + 50 * 0.05, abs=2.0)
    # Far outside data: falls back near global median, never crashes.
    assert abs(prior.elevation(5000.0, 5000.0) - prior.global_median) < 1e-6
    prior.save(tmp_path / "t.npz")
    loaded = TerrainPrior.load(tmp_path / "t.npz")
    assert loaded.elevation(50.0, 0.0) == pytest.approx(prior.elevation(50.0, 0.0), abs=1e-3)


def test_sun_detection_and_heading_round_trip():
    """Render a sun disc at a known ephemeris position and recover the camera
    heading from it — the full celestial cue in one loop."""
    world = World(WorldConfig(size_m=300.0, height_scale_m=30.0, n_landmarks=8, grid=64), seed=0)
    utc = datetime(2026, 6, 21, 10, 0, tzinfo=UTC)
    eph = celestial.sun_position(utc, ANCHOR.lat0, ANCHOR.lon0)
    # Sun at ~135 deg az / ~62 deg el for this time+place: face it and look up
    # so the disc lands mid-frame (FOV 70 -> vertical half-angle ~35 deg).
    true_heading = 132.0
    width = height = 96
    tilt = -55.0
    rgb, _ = rasterizer.render(
        world, np.array([0.0, 80.0, 0.0]), true_heading, tilt, width, height, 70.0,
        sun_azimuth_sim_deg=eph.azimuth_deg, sun_elevation_deg=eph.elevation_deg,
    )
    k_inv = np.linalg.inv(camera_matrix(width, height, 70.0))
    det = meas.detect_sun_disc(rgb, k_inv, tilt)
    assert det is not None, "sun disc must be detected in this geometry"
    got = meas.sun_heading_measurement(det, utc, ANCHOR)
    assert got is not None
    heading, sigma = got
    err = (heading - true_heading + 180.0) % 360.0 - 180.0
    assert abs(err) < 6.0
    assert sigma > 0


def test_sun_measurement_rejected_at_night():
    det = meas.SunDetection(azimuth_body_deg=0.0, elevation_deg=30.0)
    got = meas.sun_heading_measurement(det, datetime(2026, 6, 21, 0, 0, tzinfo=UTC), ANCHOR)
    assert got is None


def test_sun_measurement_rejects_elevation_mismatch():
    utc = datetime(2026, 6, 21, 10, 0, tzinfo=UTC)
    det = meas.SunDetection(azimuth_body_deg=0.0, elevation_deg=5.0)  # ephemeris says ~55
    assert meas.sun_heading_measurement(det, utc, ANCHOR) is None


def test_apr_measurement_scaling():
    pred = {"pos_enu": np.array([10.0, 20.0, 30.0]), "sigma_pos_m": np.float32(2.0)}
    pos, cov = meas.apr_position_measurement(pred, calibration_scale=3.0, sigma_floor_m=1.5)
    np.testing.assert_array_equal(pos, [10, 20, 30])
    assert cov[0, 0] == pytest.approx(36.0)  # (2*3)^2
    assert cov[2, 2] == pytest.approx(72.0)  # altitude inflated


def test_vo_velocity_measurement():
    vel, cov = meas.vo_velocity_measurement(np.array([2.0, 0.0, 0.0]), dt=0.5, inliers=100)
    np.testing.assert_allclose(vel, [4.0, 0, 0])
    assert cov[0, 0] < 1.0  # many inliers -> tight
    _, cov_few = meas.vo_velocity_measurement(np.array([2.0, 0.0, 0.0]), dt=0.5, inliers=4)
    assert cov_few[0, 0] > cov[0, 0]
