// Drives the scene's directional light from the real solar ephemeris
// (C# port of the same NOAA algorithm the Python side uses), given the sim
// clock and the geo anchor. This is what makes the celestial heading cue
// consistent between Unity, the headless sim and the localizer.

using System;
using DroneCV.Flight.Geo;
using UnityEngine;

namespace DroneCV.Flight.Sim
{
    public class SunController : MonoBehaviour
    {
        [Tooltip("Directional light representing the sun")]
        public Light SunLight;

        [Tooltip("Simulation clock start, ISO-8601 UTC (e.g. 2026-06-21T10:00:00Z)")]
        public string StartUtc = "2026-06-21T10:00:00Z";

        public GeoAnchorProvider AnchorProvider;

        private DateTime _startUtc;
        private double _trueNorthOffsetDeg;
        private double _lat, _lon;

        private void Awake() => Configure();

        public void Configure()
        {
            _startUtc = DateTime.Parse(
                string.IsNullOrEmpty(StartUtc) ? "2026-06-21T10:00:00Z" : StartUtc,
                null, System.Globalization.DateTimeStyles.AdjustToUniversal
                    | System.Globalization.DateTimeStyles.AssumeUniversal);
            if (AnchorProvider != null)
            {
                _trueNorthOffsetDeg = AnchorProvider.TrueNorthOffsetDeg();
                (_lat, _lon) = AnchorProvider.LatLon();
            }
        }

        public void OverrideStartUtc(string iso)
        {
            StartUtc = iso;
            Configure();
        }

        public DateTime UtcAt(double simTime) => _startUtc.AddSeconds(simTime);

        public CelestialPosition Sun(double simTime) => Celestial.SunPosition(UtcAt(simTime), _lat, _lon);

        public CelestialPosition Moon(double simTime) => Celestial.MoonPosition(UtcAt(simTime), _lat, _lon);

        /// Update the directional light to match the ephemeris at simTime.
        public void Apply(double simTime)
        {
            if (SunLight == null) return;
            var sun = Sun(simTime);
            // Unity-frame azimuth: compass azimuth minus the north offset.
            var azSim = (float)(sun.AzimuthDeg - _trueNorthOffsetDeg);
            // Light rotation: elevation tips it down, azimuth spins about Y;
            // a light pointing "from the sun" has direction opposite the
            // sun's sky position.
            SunLight.transform.rotation = Quaternion.Euler((float)sun.ElevationDeg, azSim + 180f, 0f);
            SunLight.enabled = sun.ElevationDeg > 0.0;
        }
    }
}
