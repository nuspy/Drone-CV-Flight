import json
import math
from pathlib import Path

import numpy as np
import pytest

from dronecv.geo import frames

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_axis_map_zero_offset():
    # sim x -> East, sim z -> North, sim y -> Up
    np.testing.assert_allclose(frames.sim_to_enu([1, 0, 0]), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(frames.sim_to_enu([0, 1, 0]), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(frames.sim_to_enu([0, 0, 1]), [0, 1, 0], atol=1e-12)


def test_north_offset_rotates_forward_axis():
    # If sim +Z has compass bearing 90 deg, it must map onto East.
    v = frames.sim_to_enu([0, 0, 1], true_north_offset_deg=90.0)
    np.testing.assert_allclose(v, [1, 0, 0], atol=1e-12)
    # And bearing 45: halfway between north and east.
    v = frames.sim_to_enu([0, 0, 1], true_north_offset_deg=45.0)
    np.testing.assert_allclose(v, [math.sqrt(0.5), math.sqrt(0.5), 0], atol=1e-12)


@pytest.mark.parametrize("offset", [0.0, 33.3, 90.0, -120.0])
def test_sim_enu_round_trip(offset):
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.normal(size=3) * 100
        back = frames.enu_to_sim(frames.sim_to_enu(p, offset), offset)
        np.testing.assert_allclose(back, p, atol=1e-9)


def test_rotation_conjugation_is_proper():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        r_sim = frames.quat_to_matrix(q)
        r_enu = frames.rot_sim_to_enu(r_sim, true_north_offset_deg=17.0)
        assert np.linalg.det(r_enu) == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(r_enu @ r_enu.T, np.eye(3), atol=1e-9)


def test_identity_attitude_heading_matches_offset():
    # Unity body aligned with world (identity quaternion) faces sim +Z,
    # whose compass bearing is exactly the north offset.
    for offset in [0.0, 30.0, 90.0, 200.0]:
        r = frames.sim_quat_to_enu_matrix([0, 0, 0, 1], offset)
        assert frames.heading_deg_from_rotation(r) == pytest.approx(offset % 360.0, abs=1e-6)


def test_unity_yaw_rotation_heading():
    # Unity yaw of +90 deg about Y (left-handed) turns +Z toward +X (east at
    # zero offset). Unity quaternion for 90 deg about Y: (0, sin45, 0, cos45).
    s, c = math.sin(math.pi / 4), math.cos(math.pi / 4)
    r = frames.sim_quat_to_enu_matrix([0, s, 0, c], 0.0)
    assert frames.heading_deg_from_rotation(r) == pytest.approx(90.0, abs=1e-6)


def test_quat_matrix_round_trip():
    rng = np.random.default_rng(2)
    for _ in range(50):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        if q[3] < 0:
            q = -q
        r = frames.quat_to_matrix(q)
        q2 = frames.matrix_to_quat(r)
        np.testing.assert_allclose(q2, q, atol=1e-9)


def test_heading_from_vector():
    assert frames.heading_deg_from_enu_vector([0, 1, 0]) == pytest.approx(0.0)
    assert frames.heading_deg_from_enu_vector([1, 0, 0]) == pytest.approx(90.0)
    assert frames.heading_deg_from_enu_vector([0, -1, 0]) == pytest.approx(180.0)
    assert frames.heading_deg_from_enu_vector([-1, 0, 0]) == pytest.approx(270.0)


def test_enu_geodetic_round_trip():
    lat0, lon0, alt0 = 45.4642, 9.19, 120.0
    rng = np.random.default_rng(3)
    for _ in range(50):
        p = rng.uniform(-2000, 2000, size=3)
        lat, lon, alt = frames.enu_to_geodetic(p, lat0, lon0, alt0)
        back = frames.geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0)
        np.testing.assert_allclose(back, p, atol=1e-6)


def test_geodetic_scale_sanity():
    # 1 degree of latitude is ~111 km.
    lat0, lon0, alt0 = 45.0, 9.0, 0.0
    p = frames.geodetic_to_enu(46.0, 9.0, 0.0, lat0, lon0, alt0)
    assert p[1] == pytest.approx(111_000, rel=0.01)


def test_wrap_deg():
    assert frames.wrap_deg(190.0) == pytest.approx(-170.0)
    assert frames.wrap_deg(-190.0) == pytest.approx(170.0)
    assert frames.wrap_deg(180.0) == pytest.approx(180.0)
    assert frames.wrap_deg(0.0) == pytest.approx(0.0)


def test_shared_fixture_cases():
    """Cases consumed by BOTH pytest and the Unity EditMode tests."""
    cases = json.loads((FIXTURES / "frames_cases.json").read_text())
    for case in cases:
        got = frames.sim_to_enu(case["pos_sim"], case["true_north_offset_deg"])
        np.testing.assert_allclose(got, case["expected_enu"], atol=1e-6, err_msg=case["name"])
        if "quat_sim" in case:
            r = frames.sim_quat_to_enu_matrix(case["quat_sim"], case["true_north_offset_deg"])
            assert frames.heading_deg_from_rotation(r) == pytest.approx(
                case["expected_heading_deg"], abs=1e-4
            ), case["name"]
