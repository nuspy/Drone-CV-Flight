// EditMode tests: the C# celestial port must agree with the Python NOAA
// implementation within 0.05 deg on the shared vectors.

using System;
using System.IO;
using DroneCV.Flight.Geo;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace DroneCV.Flight.Tests
{
    public class CelestialTests
    {
        [Test]
        public void JulianDayEpoch()
        {
            var t = new DateTime(2000, 1, 1, 12, 0, 0, DateTimeKind.Utc);
            Assert.AreEqual(2451545.0, Celestial.JulianDay(t), 1e-6);
        }

        [Test]
        public void SunMatchesPythonFixtures()
        {
            var cases = JArray.Parse(File.ReadAllText(FixturePaths.File("celestial_cases.json")));
            foreach (var c in cases)
            {
                var utc = DateTime.Parse((string)c["utc"], null,
                    System.Globalization.DateTimeStyles.AdjustToUniversal);
                var p = Celestial.SunPosition(utc, (double)c["lat"], (double)c["lon"]);
                Assert.AreEqual((double)c["sun_azimuth_deg"], p.AzimuthDeg, 0.05, $"azimuth {c["utc"]}");
                Assert.AreEqual((double)c["sun_elevation_deg"], p.ElevationDeg, 0.05, $"elevation {c["utc"]}");
            }
        }

        [Test]
        public void MoonIsSane()
        {
            var p = Celestial.MoonPosition(new DateTime(2026, 1, 3, 22, 0, 0, DateTimeKind.Utc), 45.4642, 9.19);
            Assert.That(p.AzimuthDeg, Is.InRange(0.0, 360.0));
            Assert.That(p.ElevationDeg, Is.InRange(-90.0, 90.0));
        }
    }
}
