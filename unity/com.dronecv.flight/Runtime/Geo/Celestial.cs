// Sun and moon apparent positions — C# port of src/dronecv/geo/celestial.py
// (NOAA solar algorithm + Astronomical Almanac low-precision moon).
// Pinned against tests/fixtures/celestial_cases.json so Unity's rendered sun
// matches the ephemeris the Python localizer predicts.

using System;

namespace DroneCV.Flight.Geo
{
    public struct CelestialPosition
    {
        public double AzimuthDeg;    // compass, clockwise from true north
        public double ElevationDeg;  // above horizon
    }

    public static class Celestial
    {
        private static double Rad(double d) => d * Math.PI / 180.0;
        private static double Deg(double r) => r * 180.0 / Math.PI;

        public static double JulianDay(DateTime utc)
        {
            utc = utc.ToUniversalTime();
            int year = utc.Year, month = utc.Month;
            double day = utc.Day
                + utc.Hour / 24.0
                + utc.Minute / 1440.0
                + (utc.Second + utc.Millisecond / 1000.0) / 86400.0;
            if (month <= 2) { year -= 1; month += 12; }
            int a = year / 100;
            int b = 2 - a + a / 4;
            return Math.Floor(365.25 * (year + 4716)) + Math.Floor(30.6001 * (month + 1)) + day + b - 1524.5;
        }

        private static void SunDeclinationEqTime(double jc, out double declDeg, out double eqTimeMin)
        {
            double l0 = (280.46646 + jc * (36000.76983 + 0.0003032 * jc)) % 360.0;
            if (l0 < 0) l0 += 360.0;
            double m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc);
            double ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc);
            double mrad = Rad(m);
            double c = Math.Sin(mrad) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
                     + Math.Sin(2 * mrad) * (0.019993 - 0.000101 * jc)
                     + Math.Sin(3 * mrad) * 0.000289;
            double trueLong = l0 + c;
            double omega = Rad(125.04 - 1934.136 * jc);
            double appLong = trueLong - 0.00569 - 0.00478 * Math.Sin(omega);
            double meanObliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0;
            double obliq = meanObliq + 0.00256 * Math.Cos(omega);
            declDeg = Deg(Math.Asin(Math.Sin(Rad(obliq)) * Math.Sin(Rad(appLong))));
            double varY = Math.Tan(Rad(obliq / 2.0)); varY *= varY;
            double l0r = Rad(l0), mr = mrad;
            eqTimeMin = 4.0 * Deg(
                varY * Math.Sin(2 * l0r)
                - 2 * ecc * Math.Sin(mr)
                + 4 * ecc * varY * Math.Sin(mr) * Math.Cos(2 * l0r)
                - 0.5 * varY * varY * Math.Sin(4 * l0r)
                - 1.25 * ecc * ecc * Math.Sin(2 * mr));
        }

        public static CelestialPosition SunPosition(DateTime utc, double latDeg, double lonDeg, bool refraction = true)
        {
            utc = utc.ToUniversalTime();
            double jd = JulianDay(utc);
            double jc = (jd - 2451545.0) / 36525.0;
            SunDeclinationEqTime(jc, out var decl, out var eqTime);

            double utcMinutes = utc.Hour * 60.0 + utc.Minute + (utc.Second + utc.Millisecond / 1000.0) / 60.0;
            double trueSolarMin = (utcMinutes + eqTime + 4.0 * lonDeg) % 1440.0;
            if (trueSolarMin < 0) trueSolarMin += 1440.0;
            double ha = trueSolarMin / 4.0 < 0 ? trueSolarMin / 4.0 + 180.0 : trueSolarMin / 4.0 - 180.0;

            double lat = Rad(latDeg), dec = Rad(decl), h = Rad(ha);
            double cosZen = Math.Sin(lat) * Math.Sin(dec) + Math.Cos(lat) * Math.Cos(dec) * Math.Cos(h);
            cosZen = Math.Max(-1.0, Math.Min(1.0, cosZen));
            double zenith = Deg(Math.Acos(cosZen));
            double elevation = 90.0 - zenith;

            if (refraction && elevation < 85.0)
            {
                double te = Math.Tan(Rad(Math.Max(elevation, -0.99)));
                double corr;
                if (elevation > 5.0)
                    corr = 58.1 / te - 0.07 / Math.Pow(te, 3) + 0.000086 / Math.Pow(te, 5);
                else if (elevation > -0.575)
                    corr = 1735.0 + elevation * (-518.2 + elevation * (103.4 + elevation * (-12.79 + elevation * 0.711)));
                else
                    corr = -20.772 / te;
                elevation += corr / 3600.0;
            }

            double sinZen = Math.Sin(Rad(zenith));
            double azimuth;
            if (Math.Abs(sinZen) < 1e-9)
            {
                azimuth = 0.0;
            }
            else
            {
                double cosAz = (Math.Sin(lat) * cosZen - Math.Sin(dec)) / (Math.Cos(lat) * sinZen);
                cosAz = Math.Max(-1.0, Math.Min(1.0, cosAz));
                double az = Deg(Math.Acos(cosAz));
                azimuth = ha > 0 ? (az + 180.0) % 360.0 : (540.0 - az) % 360.0;
            }
            return new CelestialPosition { AzimuthDeg = azimuth, ElevationDeg = elevation };
        }

        private static void MoonEcliptic(double t, out double lonDeg, out double latDeg)
        {
            double S(double d) => Math.Sin(Rad(d));
            lonDeg = (218.32 + 481267.881 * t
                + 6.29 * S(135.0 + 477198.87 * t)
                - 1.27 * S(259.3 - 413335.36 * t)
                + 0.66 * S(235.7 + 890534.22 * t)
                + 0.21 * S(269.9 + 954397.74 * t)
                - 0.19 * S(357.5 + 35999.05 * t)
                - 0.11 * S(186.5 + 966404.03 * t)) % 360.0;
            if (lonDeg < 0) lonDeg += 360.0;
            latDeg = 5.13 * S(93.3 + 483202.02 * t)
                + 0.28 * S(228.2 + 960400.89 * t)
                - 0.28 * S(318.3 + 6003.15 * t)
                - 0.17 * S(217.6 - 407332.21 * t);
        }

        public static CelestialPosition MoonPosition(DateTime utc, double latDeg, double lonDeg)
        {
            utc = utc.ToUniversalTime();
            double jd = JulianDay(utc);
            double jc = (jd - 2451545.0) / 36525.0;
            MoonEcliptic(jc, out var eclLon, out var eclLat);
            double obliq = Rad(23.4393 - 0.01300 * jc);
            double lam = Rad(eclLon), beta = Rad(eclLat);

            double sinDec = Math.Sin(beta) * Math.Cos(obliq) + Math.Cos(beta) * Math.Sin(obliq) * Math.Sin(lam);
            double dec = Math.Asin(Math.Max(-1.0, Math.Min(1.0, sinDec)));
            double ra = Math.Atan2(Math.Sin(lam) * Math.Cos(obliq) - Math.Tan(beta) * Math.Sin(obliq), Math.Cos(lam));

            double lst = (280.46061837 + 360.98564736629 * (jd - 2451545.0) + lonDeg) % 360.0;
            if (lst < 0) lst += 360.0;
            double h = Rad(((lst - Deg(ra)) % 360.0 + 360.0) % 360.0);

            double lat = Rad(latDeg);
            double sinEl = Math.Sin(lat) * Math.Sin(dec) + Math.Cos(lat) * Math.Cos(dec) * Math.Cos(h);
            double el = Deg(Math.Asin(Math.Max(-1.0, Math.Min(1.0, sinEl))));
            double az = Deg(Math.Atan2(
                -Math.Cos(dec) * Math.Sin(h),
                Math.Sin(dec) * Math.Cos(lat) - Math.Cos(dec) * Math.Cos(h) * Math.Sin(lat)));
            az = (az % 360.0 + 360.0) % 360.0;
            return new CelestialPosition { AzimuthDeg = az, ElevationDeg = el };
        }
    }
}
