
import numpy as np
import pytest

from dronecv.config import AnchorConfig, GuidanceConfig
from dronecv.geo.anchor import GeoAnchor
from dronecv.guidance.controller import FlightController
from dronecv.guidance.guidance import compute_guidance
from dronecv.localization.localizer import Estimate

CFG = GuidanceConfig()  # standoff 10 m, arrival 3 m, alt tol 2 m
ANCHOR = GeoAnchor.resolve(AnchorConfig(lat0=45.0, lon0=9.0, alt0=100.0, true_north_offset_deg=0.0), None)


def _est(pos, heading=0.0, conf=0.9, lost=False, initialized=True):
    return Estimate(
        initialized=initialized, lat=0, lon=0, alt_msl=0, alt_agl=None,
        pos_enu=np.asarray(pos, float), vel_enu=np.zeros(3), speed_ms=0.0,
        heading_deg=heading, confidence=conf, lost=lost,
    )


def test_goal_is_10m_short_of_target_at_target_altitude():
    drone = np.array([0.0, 0.0, 50.0])
    target = np.array([100.0, 0.0, 80.0])
    g = compute_guidance(drone, target, CFG)
    np.testing.assert_allclose(g.goal_enu, [90.0, 0.0, 80.0], atol=1e-9)
    assert g.bearing_deg_true == pytest.approx(90.0)  # due east
    assert g.horizontal_dist_m == pytest.approx(90.0)
    assert g.altitude_delta_m == pytest.approx(30.0)
    assert not g.at_goal and not g.standoff_violation


def test_bearing_true_north_convention():
    g = compute_guidance(np.zeros(3), np.array([0.0, 50.0, 0.0]), CFG)
    assert g.bearing_deg_true == pytest.approx(0.0)  # north
    g = compute_guidance(np.zeros(3), np.array([-50.0, 0.0, 0.0]), CFG)
    assert g.bearing_deg_true == pytest.approx(270.0)  # west


def test_at_goal_and_violation_flags():
    target = np.array([0.0, 0.0, 60.0])
    # Exactly at the goal point (10 m south of target, at its altitude).
    g = compute_guidance(np.array([0.0, -10.0, 60.0]), target, CFG)
    assert g.at_goal
    assert not g.standoff_violation
    # Inside the standoff sphere.
    g = compute_guidance(np.array([0.0, -4.0, 60.0]), target, CFG)
    assert g.standoff_violation
    # Drone inside standoff: goal pushes it back out (bearing points away).
    assert g.horizontal_dist_m == pytest.approx(6.0)
    assert g.bearing_deg_true == pytest.approx(180.0)


def test_target_directly_above_uses_fallback_direction():
    drone = np.array([0.0, 0.0, 50.0])
    target = np.array([0.0, 0.0, 90.0])
    g = compute_guidance(drone, target, CFG, fallback_bearing_deg=45.0)
    assert np.isfinite(g.bearing_deg_true)
    assert g.horizontal_dist_m == pytest.approx(10.0)  # goal on the standoff ring
    assert g.altitude_delta_m == pytest.approx(40.0)


def test_controller_holds_on_low_confidence_and_lost():
    ctrl = FlightController(CFG, ANCHOR)
    g = compute_guidance(np.zeros(3), np.array([100.0, 0.0, 50.0]), CFG)
    for est in (_est([0, 0, 0], conf=0.05), _est([0, 0, 0], lost=True), _est([0, 0, 0], initialized=False)):
        cmd = ctrl.command(est, g)
        np.testing.assert_array_equal(cmd.vel_sim, np.zeros(3))
        assert cmd.yaw_rate_dps == 0.0


def test_controller_flies_toward_goal():
    ctrl = FlightController(CFG, ANCHOR)
    est = _est([0.0, 0.0, 50.0], heading=0.0, conf=1.0)
    g = compute_guidance(est.pos_enu, np.array([200.0, 0.0, 50.0]), CFG)
    cmd = ctrl.command(est, g)
    # East in ENU = +x in sim (zero offset): sim vel x > 0, z ~ 0.
    assert cmd.vel_sim[0] > 3.0
    assert abs(cmd.vel_sim[2]) < 1e-6
    assert cmd.yaw_rate_dps > 10.0  # turning toward bearing 90


def test_controller_climbs_toward_target_altitude():
    ctrl = FlightController(CFG, ANCHOR)
    est = _est([0.0, 0.0, 40.0], conf=1.0)
    g = compute_guidance(est.pos_enu, np.array([0.0, 100.0, 70.0]), CFG)
    cmd = ctrl.command(est, g)
    assert cmd.vel_sim[1] == pytest.approx(3.0)  # climb clipped at 3


def test_controller_speed_scales_with_confidence():
    ctrl = FlightController(CFG, ANCHOR)
    g = compute_guidance(np.zeros(3), np.array([300.0, 0.0, 0.0]), CFG)
    v_high = np.linalg.norm(ctrl.command(_est([0, 0, 0], conf=1.0), g).vel_sim[[0, 2]])
    v_low = np.linalg.norm(ctrl.command(_est([0, 0, 0], conf=0.35), g).vel_sim[[0, 2]])
    assert v_low < v_high


def test_guidance_with_north_offset_anchor():
    # With a 90 deg north offset, an eastward ENU command maps to sim -z.
    anchor = GeoAnchor.resolve(AnchorConfig(lat0=45.0, lon0=9.0, alt0=100.0, true_north_offset_deg=90.0), None)
    ctrl = FlightController(CFG, anchor)
    est = _est([0.0, 0.0, 50.0], heading=90.0, conf=1.0)
    g = compute_guidance(est.pos_enu, np.array([200.0, 0.0, 50.0]), CFG)
    cmd = ctrl.command(est, g)
    # ENU east -> sim: enu_to_sim([1,0,0]) with offset 90 = z * -1? verify numerically:
    expected_dir = anchor.enu_to_sim(np.array([1.0, 0.0, 0.0]))
    v = cmd.vel_sim / np.linalg.norm(cmd.vel_sim)
    np.testing.assert_allclose(v, expected_dir, atol=1e-6)
