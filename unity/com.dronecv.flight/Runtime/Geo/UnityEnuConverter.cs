// Unity local frame <-> ENU conversions.
// Line-by-line mirror of src/dronecv/geo/frames.py; both implementations are
// pinned by tests/fixtures/frames_cases.json (pytest + EditMode tests).
//
// Convention: trueNorthOffsetDeg is the compass bearing (deg clockwise from
// true north) of the Unity world +Z axis. Axis map before the yaw offset:
// E = x, N = z, U = y.

using System;

namespace DroneCV.Flight.Geo
{
    public static class UnityEnuConverter
    {
        public static double[] SimToEnu(double[] pSim, double trueNorthOffsetDeg)
        {
            double e0 = pSim[0], n0 = pSim[2], u0 = pSim[1];
            double o = trueNorthOffsetDeg * Math.PI / 180.0;
            double co = Math.Cos(o), so = Math.Sin(o);
            return new[]
            {
                co * e0 + so * n0,
                -so * e0 + co * n0,
                u0,
            };
        }

        public static double[] EnuToSim(double[] pEnu, double trueNorthOffsetDeg)
        {
            double o = trueNorthOffsetDeg * Math.PI / 180.0;
            double co = Math.Cos(o), so = Math.Sin(o);
            double e0 = co * pEnu[0] - so * pEnu[1];
            double n0 = so * pEnu[0] + co * pEnu[1];
            double u0 = pEnu[2];
            return new[] { e0, u0, n0 };
        }

        /// Compass heading of a Unity body given its yaw about +Y
        /// (Unity yaw 0 = facing +Z): heading = yaw + north offset.
        public static double HeadingDegFromUnityYaw(double yawDeg, double trueNorthOffsetDeg)
        {
            return ((yawDeg + trueNorthOffsetDeg) % 360.0 + 360.0) % 360.0;
        }

        public static double WrapDeg(double angle)
        {
            var a = ((angle + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
            return a == -180.0 ? 180.0 : a;
        }
    }
}
