// EditMode tests: the C# frame conversions must reproduce the SAME numbers
// as the Python implementation, pinned by the shared fixture file.

using System.IO;
using DroneCV.Flight.Geo;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace DroneCV.Flight.Tests
{
    public class FramesConversionTests
    {
        [Test]
        public void SharedFixtureCases()
        {
            var cases = JArray.Parse(File.ReadAllText(FixturePaths.File("frames_cases.json")));
            foreach (var c in cases)
            {
                var name = (string)c["name"];
                var posSim = c["pos_sim"].ToObject<double[]>();
                var offset = (double)c["true_north_offset_deg"];
                var expected = c["expected_enu"].ToObject<double[]>();
                var got = UnityEnuConverter.SimToEnu(posSim, offset);
                for (int i = 0; i < 3; i++)
                    Assert.AreEqual(expected[i], got[i], 1e-6, $"{name} axis {i}");

                // Round trip.
                var back = UnityEnuConverter.EnuToSim(got, offset);
                for (int i = 0; i < 3; i++)
                    Assert.AreEqual(posSim[i], back[i], 1e-9, $"{name} round-trip axis {i}");
            }
        }

        [Test]
        public void HeadingFromUnityYawMatchesFixtures()
        {
            var cases = JArray.Parse(File.ReadAllText(FixturePaths.File("frames_cases.json")));
            foreach (var c in cases)
            {
                if (c["quat_sim"] == null || c["expected_heading_deg"] == null) continue;
                var q = c["quat_sim"].ToObject<double[]>();
                // Fixture quaternions are pure yaw rotations about +Y.
                var yawDeg = 2.0 * System.Math.Atan2(q[1], q[3]) * 180.0 / System.Math.PI;
                var offset = (double)c["true_north_offset_deg"];
                var expected = (double)c["expected_heading_deg"];
                var got = UnityEnuConverter.HeadingDegFromUnityYaw(yawDeg, offset);
                Assert.AreEqual(expected, got, 1e-4, (string)c["name"]);
            }
        }

        [Test]
        public void WrapDeg()
        {
            Assert.AreEqual(-170.0, UnityEnuConverter.WrapDeg(190.0), 1e-9);
            Assert.AreEqual(170.0, UnityEnuConverter.WrapDeg(-190.0), 1e-9);
            Assert.AreEqual(180.0, UnityEnuConverter.WrapDeg(180.0), 1e-9);
        }
    }
}
