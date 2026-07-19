// Fallback geo anchor as a project asset (Create > DroneCV > Geo Anchor).
// If the imported environment ships a dronecv_geo.json sidecar, that wins —
// see GeoAnchorProvider.

using UnityEngine;

namespace DroneCV.Flight.Geo
{
    [CreateAssetMenu(fileName = "GeoAnchor", menuName = "DroneCV/Geo Anchor")]
    public class GeoAnchorAsset : ScriptableObject
    {
        [Tooltip("Latitude of the Unity world origin (degrees)")]
        public double Lat0 = 45.4642;

        [Tooltip("Longitude of the Unity world origin (degrees)")]
        public double Lon0 = 9.19;

        [Tooltip("MSL altitude of the Unity world origin (meters)")]
        public double Alt0 = 120.0;

        [Tooltip("Compass bearing of the Unity +Z axis (degrees clockwise from true north)")]
        public double TrueNorthOffsetDeg = 0.0;
    }
}
