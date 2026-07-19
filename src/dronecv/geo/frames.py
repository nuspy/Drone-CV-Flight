"""Coordinate frame conversions.

Frames:
- **sim**: the simulator's local frame. Unity is left-handed, Y-up, Z-forward;
  the headless sim uses the same convention so the Python side is identical
  for both. `true_north_offset_deg` is the compass bearing (degrees clockwise
  from true north) that the sim's +Z axis points to in the real world.
- **ENU**: right-handed local East/North/Up in meters, origin at the sim origin.
- **WGS84**: geodetic lat/lon (degrees) and MSL altitude (meters).

All internal navigation state (filter, guidance) lives in ENU; WGS84 appears
only at the API boundary. Attitude conversions go through rotation matrices to
avoid quaternion handedness mistakes; the Unity C# UnityEnuConverter mirrors
this file and both are tested against tests/fixtures/frames_cases.json.
"""

from __future__ import annotations

import math

import numpy as np
import pymap3d

# Axis map sim -> ENU before the north-offset yaw: E = x, N = z, U = y.
# Improper orthogonal (det = -1); conjugating a sim rotation matrix with it
# yields a proper ENU rotation matrix.
_M_SIM_TO_ENU0 = np.array(
    [
        [1.0, 0.0, 0.0],  # E <- x
        [0.0, 0.0, 1.0],  # N <- z
        [0.0, 1.0, 0.0],  # U <- y
    ]
)


def _yaw_matrix_enu(offset_deg: float) -> np.ndarray:
    """Rotation applied to naive ENU vectors when the sim +Z axis has compass
    bearing `offset_deg`. Derived so that offset=0 keeps N=z, and offset=90
    maps the sim +Z axis onto East."""
    o = math.radians(offset_deg)
    return np.array(
        [
            [math.cos(o), math.sin(o), 0.0],
            [-math.sin(o), math.cos(o), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def sim_to_enu(p_sim: np.ndarray, true_north_offset_deg: float = 0.0) -> np.ndarray:
    """Position/vector from sim (x, y, z) to ENU (E, N, U)."""
    p_sim = np.asarray(p_sim, dtype=float)
    p0 = _M_SIM_TO_ENU0 @ p_sim
    return _yaw_matrix_enu(true_north_offset_deg) @ p0


def enu_to_sim(p_enu: np.ndarray, true_north_offset_deg: float = 0.0) -> np.ndarray:
    p_enu = np.asarray(p_enu, dtype=float)
    p0 = _yaw_matrix_enu(true_north_offset_deg).T @ p_enu
    return _M_SIM_TO_ENU0.T @ p0


def rot_sim_to_enu(r_sim: np.ndarray, true_north_offset_deg: float = 0.0) -> np.ndarray:
    """Rotation matrix (body->sim world) to rotation matrix (body RFU -> ENU).

    The sim frame is left-handed; conjugation by the improper axis map plus
    the yaw offset produces a proper right-handed rotation. The resulting
    matrix maps body RFU coordinates (X right, Y forward, Z up) to ENU.
    """
    yaw = _yaw_matrix_enu(true_north_offset_deg)
    m = yaw @ _M_SIM_TO_ENU0
    return m @ np.asarray(r_sim, dtype=float) @ _M_SIM_TO_ENU0.T


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (x, y, z, w) -> rotation matrix. Convention-agnostic:
    interprets the quaternion in whatever frame it was expressed."""
    x, y, z, w = np.asarray(q, dtype=float)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (x, y, z, w), w >= 0."""
    r = np.asarray(r, dtype=float)
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    if q[3] < 0:
        q = -q
    return q / np.linalg.norm(q)


def sim_quat_to_enu_matrix(q_sim: np.ndarray, true_north_offset_deg: float = 0.0) -> np.ndarray:
    """Unity/sim body quaternion -> body->ENU rotation matrix."""
    return rot_sim_to_enu(quat_to_matrix(q_sim), true_north_offset_deg)


def heading_deg_from_enu_vector(v_enu: np.ndarray) -> float:
    """Compass heading (degrees clockwise from true north) of an ENU vector."""
    e, n = float(v_enu[0]), float(v_enu[1])
    return (math.degrees(math.atan2(e, n)) + 360.0) % 360.0


def heading_deg_from_rotation(r_body_to_enu: np.ndarray) -> float:
    """Heading of the body forward axis.

    Body convention here is RFU (X right, Y forward, Z up, right-handed):
    it is what `rot_sim_to_enu` produces, because the same axis map that takes
    the sim world frame to ENU takes the Unity body frame (X right, Z forward,
    Y up) to RFU. Forward is therefore body +Y.
    """
    fwd = np.asarray(r_body_to_enu, dtype=float) @ np.array([0.0, 1.0, 0.0])
    return heading_deg_from_enu_vector(fwd)


def enu_to_geodetic(p_enu: np.ndarray, lat0: float, lon0: float, alt0: float) -> tuple[float, float, float]:
    """ENU (m) -> (lat deg, lon deg, alt m MSL)."""
    lat, lon, alt = pymap3d.enu2geodetic(
        float(p_enu[0]), float(p_enu[1]), float(p_enu[2]), lat0, lon0, alt0
    )
    return float(lat), float(lon), float(alt)


def geodetic_to_enu(lat: float, lon: float, alt: float, lat0: float, lon0: float, alt0: float) -> np.ndarray:
    e, n, u = pymap3d.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
    return np.array([e, n, u], dtype=float)


def wrap_deg(angle: float) -> float:
    """Wrap an angle difference to (-180, 180]."""
    a = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a
