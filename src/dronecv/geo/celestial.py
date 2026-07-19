"""Sun and moon apparent positions — dependency-free.

Sun: NOAA "General Solar Position Calculations" algorithm (the one behind the
NOAA solar calculator spreadsheet), accuracy well under 0.1 deg for the
1900-2100 range. Moon: Astronomical Almanac low-precision series (~0.3 deg in
ecliptic coordinates), plenty for a heading cue with sigma 2-5 deg.

The same formulas are ported line-by-line to C# in the Unity package
(SunController); shared test vectors keep the two implementations aligned, so
the light direction rendered by either simulator matches what the localizer
predicts when it turns a detected sun/moon disc into an absolute heading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class CelestialPosition:
    azimuth_deg: float  # compass, clockwise from true north
    elevation_deg: float  # above horizon


def _to_utc(t: datetime) -> datetime:
    if t.tzinfo is None:
        return t.replace(tzinfo=UTC)
    return t.astimezone(UTC)


def julian_day(t: datetime) -> float:
    t = _to_utc(t)
    year, month = t.year, t.month
    day = (
        t.day
        + t.hour / 24.0
        + t.minute / 1440.0
        + (t.second + t.microsecond / 1e6) / 86400.0
    )
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day + b - 1524.5


def _sun_declination_eqtime(jc: float) -> tuple[float, float]:
    """Returns (declination deg, equation of time minutes) at julian century jc."""
    l0 = (280.46646 + jc * (36000.76983 + 0.0003032 * jc)) % 360.0
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mrad = math.radians(m)
    c = (
        math.sin(mrad) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + math.sin(2 * mrad) * (0.019993 - 0.000101 * jc)
        + math.sin(3 * mrad) * 0.000289
    )
    true_long = l0 + c
    omega = math.radians(125.04 - 1934.136 * jc)
    app_long = true_long - 0.00569 - 0.00478 * math.sin(omega)
    mean_obliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    obliq = mean_obliq + 0.00256 * math.cos(omega)
    decl = math.degrees(
        math.asin(math.sin(math.radians(obliq)) * math.sin(math.radians(app_long)))
    )
    var_y = math.tan(math.radians(obliq / 2.0)) ** 2
    l0r, mr = math.radians(l0), mrad
    eq_time = 4.0 * math.degrees(
        var_y * math.sin(2 * l0r)
        - 2 * ecc * math.sin(mr)
        + 4 * ecc * var_y * math.sin(mr) * math.cos(2 * l0r)
        - 0.5 * var_y * var_y * math.sin(4 * l0r)
        - 1.25 * ecc * ecc * math.sin(2 * mr)
    )
    return decl, eq_time


def sun_position(t: datetime, lat_deg: float, lon_deg: float, refraction: bool = True) -> CelestialPosition:
    t = _to_utc(t)
    jd = julian_day(t)
    jc = (jd - 2451545.0) / 36525.0
    decl, eq_time = _sun_declination_eqtime(jc)

    utc_minutes = t.hour * 60.0 + t.minute + (t.second + t.microsecond / 1e6) / 60.0
    true_solar_min = (utc_minutes + eq_time + 4.0 * lon_deg) % 1440.0
    ha = true_solar_min / 4.0 + 180.0 if true_solar_min / 4.0 < 0 else true_solar_min / 4.0 - 180.0

    lat, dec, h = math.radians(lat_deg), math.radians(decl), math.radians(ha)
    cos_zen = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(h)
    cos_zen = max(-1.0, min(1.0, cos_zen))
    zenith = math.degrees(math.acos(cos_zen))
    elevation = 90.0 - zenith

    if refraction and elevation < 85.0:
        te = math.tan(math.radians(max(elevation, -0.99)))
        if elevation > 5.0:
            corr = 58.1 / te - 0.07 / te**3 + 0.000086 / te**5
        elif elevation > -0.575:
            corr = 1735.0 + elevation * (-518.2 + elevation * (103.4 + elevation * (-12.79 + elevation * 0.711)))
        else:
            corr = -20.772 / te
        elevation += corr / 3600.0

    sin_zen = math.sin(math.radians(zenith))
    if abs(sin_zen) < 1e-9:
        azimuth = 0.0
    else:
        cos_az = (math.sin(lat) * cos_zen - math.sin(dec)) / (math.cos(lat) * sin_zen)
        cos_az = max(-1.0, min(1.0, cos_az))
        az = math.degrees(math.acos(cos_az))
        azimuth = (az + 180.0) % 360.0 if ha > 0 else (540.0 - az) % 360.0

    return CelestialPosition(azimuth_deg=azimuth, elevation_deg=elevation)


def _moon_ecliptic(jc: float) -> tuple[float, float]:
    """Low-precision geocentric ecliptic (longitude, latitude) of the moon, degrees."""
    t = jc
    s = math.sin
    r = math.radians
    lon = (
        218.32
        + 481267.881 * t
        + 6.29 * s(r(135.0 + 477198.87 * t))
        - 1.27 * s(r(259.3 - 413335.36 * t))
        + 0.66 * s(r(235.7 + 890534.22 * t))
        + 0.21 * s(r(269.9 + 954397.74 * t))
        - 0.19 * s(r(357.5 + 35999.05 * t))
        - 0.11 * s(r(186.5 + 966404.03 * t))
    ) % 360.0
    lat = (
        5.13 * s(r(93.3 + 483202.02 * t))
        + 0.28 * s(r(228.2 + 960400.89 * t))
        - 0.28 * s(r(318.3 + 6003.15 * t))
        - 0.17 * s(r(217.6 - 407332.21 * t))
    )
    return lon, lat


def moon_position(t: datetime, lat_deg: float, lon_deg: float) -> CelestialPosition:
    t = _to_utc(t)
    jd = julian_day(t)
    jc = (jd - 2451545.0) / 36525.0

    ecl_lon, ecl_lat = _moon_ecliptic(jc)
    obliq = math.radians(23.4393 - 0.01300 * jc)
    lam, beta = math.radians(ecl_lon), math.radians(ecl_lat)

    sin_dec = math.sin(beta) * math.cos(obliq) + math.cos(beta) * math.sin(obliq) * math.sin(lam)
    dec = math.asin(max(-1.0, min(1.0, sin_dec)))
    ra = math.atan2(
        math.sin(lam) * math.cos(obliq) - math.tan(beta) * math.sin(obliq), math.cos(lam)
    )

    # Local sidereal time (degrees)
    lst = (280.46061837 + 360.98564736629 * (jd - 2451545.0) + lon_deg) % 360.0
    h = math.radians((lst - math.degrees(ra)) % 360.0)

    lat = math.radians(lat_deg)
    sin_el = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(h)
    el = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))
    az = math.degrees(
        math.atan2(
            -math.cos(dec) * math.sin(h),
            math.sin(dec) * math.cos(lat) - math.cos(dec) * math.cos(h) * math.sin(lat),
        )
    ) % 360.0
    return CelestialPosition(azimuth_deg=az, elevation_deg=el)


def sun_direction_enu(t: datetime, lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """Unit vector pointing from the observer toward the sun, in ENU."""
    p = sun_position(t, lat_deg, lon_deg)
    az, el = math.radians(p.azimuth_deg), math.radians(p.elevation_deg)
    return (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))
