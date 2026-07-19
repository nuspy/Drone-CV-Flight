"""Structural/physical tests for the sun/moon position algorithms.

Rather than pinning table values, these assert well-known physical facts
(solar noon geometry, equinox declination, east/west symmetry) that hold to
much better than the algorithm's accuracy, plus shared fixture vectors that
the Unity C# port must also reproduce.
"""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dronecv.geo import celestial

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_julian_day_epoch():
    # J2000.0 epoch: 2000-01-01 12:00 UTC = JD 2451545.0
    assert celestial.julian_day(datetime(2000, 1, 1, 12, 0, tzinfo=UTC)) == pytest.approx(2451545.0)


def test_sun_due_south_at_solar_noon_northern_hemisphere():
    # Milan, ~solar noon (12:00 UTC+1 civil ~ 11:00 UTC; scan for max elevation)
    lat, lon = 45.4642, 9.19
    day = datetime(2026, 6, 21, tzinfo=UTC)
    best = max(
        (celestial.sun_position(day + timedelta(minutes=m), lat, lon) for m in range(9 * 60, 15 * 60, 2)),
        key=lambda p: p.elevation_deg,
    )
    assert abs(best.azimuth_deg - 180.0) < 3.0
    # Max elevation ~= 90 - lat + declination (23.44 at June solstice)
    assert best.elevation_deg == pytest.approx(90 - lat + 23.44, abs=1.0)


def test_sun_rises_east_sets_west():
    lat, lon = 45.4642, 9.19
    day = datetime(2026, 3, 20, tzinfo=UTC)  # near equinox: sunrise ~ due east
    morning = celestial.sun_position(day + timedelta(hours=5, minutes=30), lat, lon)
    evening = celestial.sun_position(day + timedelta(hours=17, minutes=30), lat, lon)
    assert 60 < morning.azimuth_deg < 120
    assert 240 < evening.azimuth_deg < 300


def test_sun_below_horizon_at_night():
    p = celestial.sun_position(datetime(2026, 6, 21, 0, 30, tzinfo=UTC), 45.4642, 9.19)
    assert p.elevation_deg < -10


def test_equinox_declination_small():
    jd = celestial.julian_day(datetime(2026, 3, 20, 12, 0, tzinfo=UTC))
    jc = (jd - 2451545.0) / 36525.0
    decl, _ = celestial._sun_declination_eqtime(jc)
    assert abs(decl) < 1.0


def test_equator_equinox_sun_overhead():
    # On the equator near the equinox, max elevation approaches 90.
    day = datetime(2026, 3, 20, tzinfo=UTC)
    best = max(
        (celestial.sun_position(day + timedelta(minutes=m), 0.0, 0.0) for m in range(10 * 60, 14 * 60, 2)),
        key=lambda p: p.elevation_deg,
    )
    assert best.elevation_deg > 88.0


def test_moon_position_is_sane():
    p = celestial.moon_position(datetime(2026, 1, 3, 22, 0, tzinfo=UTC), 45.4642, 9.19)
    assert 0.0 <= p.azimuth_deg < 360.0
    assert -90.0 <= p.elevation_deg <= 90.0


def test_moon_moves_relative_to_stars():
    # The moon moves ~13 deg/day in ecliptic longitude.
    t0 = datetime(2026, 1, 3, tzinfo=UTC)
    jc = lambda t: (celestial.julian_day(t) - 2451545.0) / 36525.0  # noqa: E731
    lon0, _ = celestial._moon_ecliptic(jc(t0))
    lon1, _ = celestial._moon_ecliptic(jc(t0 + timedelta(days=1)))
    delta = (lon1 - lon0) % 360.0
    assert 11.0 < delta < 15.0


def test_sun_direction_enu_consistency():
    t = datetime(2026, 6, 21, 10, 0, tzinfo=UTC)
    lat, lon = 45.4642, 9.19
    p = celestial.sun_position(t, lat, lon)
    d = celestial.sun_direction_enu(t, lat, lon)
    assert math.hypot(*d) == pytest.approx(1.0, abs=1e-9)
    heading = (math.degrees(math.atan2(d[0], d[1])) + 360) % 360
    assert heading == pytest.approx(p.azimuth_deg, abs=1e-6)


def test_shared_fixture_cases():
    """Vectors consumed by BOTH pytest and Unity EditMode tests (SunController port)."""
    cases = json.loads((FIXTURES / "celestial_cases.json").read_text())
    for case in cases:
        t = datetime.fromisoformat(case["utc"])
        p = celestial.sun_position(t, case["lat"], case["lon"])
        assert p.azimuth_deg == pytest.approx(case["sun_azimuth_deg"], abs=0.5), case["utc"]
        assert p.elevation_deg == pytest.approx(case["sun_elevation_deg"], abs=0.5), case["utc"]
