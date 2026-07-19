// Resolves the geo anchor for the loaded environment (dual mode):
//   1. a `dronecv_geo.json` sidecar next to the imported environment
//      (or any path configured in the inspector) — "embedded metadata";
//   2. else the GeoAnchorAsset assigned in the inspector;
//   3. else nothing: hello_ack.geo_meta stays null and the Python side falls
//      back to its per-environment YAML anchor.
//
// Sidecar format:
//   { "lat0": 45.4642, "lon0": 9.19, "alt0": 120.0, "true_north_offset_deg": 0.0 }

using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

namespace DroneCV.Flight.Geo
{
    public class GeoAnchorProvider : MonoBehaviour
    {
        [Tooltip("Optional path to a dronecv_geo.json sidecar; relative paths resolve against Application.streamingAssetsPath, then the project root.")]
        public string SidecarPath = "dronecv_geo.json";

        [Tooltip("Fallback anchor asset used when no sidecar is found.")]
        public GeoAnchorAsset FallbackAsset;

        public Dictionary<string, double> ResolveGeoMeta()
        {
            var sidecar = FindSidecar();
            if (sidecar != null)
            {
                try
                {
                    var meta = JsonConvert.DeserializeObject<Dictionary<string, double>>(File.ReadAllText(sidecar));
                    if (meta != null && meta.ContainsKey("lat0") && meta.ContainsKey("lon0"))
                    {
                        Debug.Log($"[DroneCV] geo anchor from sidecar {sidecar}");
                        if (!meta.ContainsKey("alt0")) meta["alt0"] = 0.0;
                        if (!meta.ContainsKey("true_north_offset_deg")) meta["true_north_offset_deg"] = 0.0;
                        return meta;
                    }
                }
                catch (System.Exception e)
                {
                    Debug.LogWarning($"[DroneCV] failed to parse geo sidecar {sidecar}: {e.Message}");
                }
            }
            if (FallbackAsset != null)
            {
                Debug.Log("[DroneCV] geo anchor from GeoAnchorAsset");
                return new Dictionary<string, double>
                {
                    { "lat0", FallbackAsset.Lat0 },
                    { "lon0", FallbackAsset.Lon0 },
                    { "alt0", FallbackAsset.Alt0 },
                    { "true_north_offset_deg", FallbackAsset.TrueNorthOffsetDeg },
                };
            }
            Debug.Log("[DroneCV] no geo anchor in Unity — Python will use its env-config fallback");
            return null;
        }

        public double TrueNorthOffsetDeg()
        {
            var meta = ResolveGeoMeta();
            return meta != null && meta.TryGetValue("true_north_offset_deg", out var v) ? v : 0.0;
        }

        public (double lat, double lon) LatLon()
        {
            var meta = ResolveGeoMeta();
            if (meta == null) return (45.4642, 9.19);
            return (meta["lat0"], meta["lon0"]);
        }

        private string FindSidecar()
        {
            if (string.IsNullOrEmpty(SidecarPath)) return null;
            if (Path.IsPathRooted(SidecarPath) && File.Exists(SidecarPath)) return SidecarPath;
            var candidates = new[]
            {
                Path.Combine(Application.streamingAssetsPath ?? "", SidecarPath),
                Path.Combine(Directory.GetCurrentDirectory(), SidecarPath),
            };
            foreach (var c in candidates)
                if (File.Exists(c)) return c;
            return null;
        }
    }
}
