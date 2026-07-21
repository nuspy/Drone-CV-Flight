// Native building importer: builds the buildings directly in C# from the
// exported buildings.json + palette.json, so materials are ALWAYS created and
// assigned in any render pipeline WITHOUT needing a glTF importer package.
// LoD1 (footprint extrusion + flat roof); the glTFast path stays the richer
// option (LoD2 + facade textures) when that package is installed.
//
// Coordinate mapping: buildings.json rings are Z-up ENU (e = East, n = North);
// the Unity terrain uses x = East, z = North, so a ring point (e, n) maps to
// world (e, y, n). Base height is sampled from the imported Terrain.

using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace DroneCV.Flight.Editor.Gis
{
    public static class BuildingBuilder
    {
        public static bool TryBuild(string dir)
        {
            var path = Path.Combine(dir, "buildings.json");
            if (!File.Exists(path)) return false;
            var buildings = JObject.Parse(File.ReadAllText(path))["buildings"] as JArray;
            if (buildings == null || buildings.Count == 0) return false;

            var palette = LoadPalette(dir);
            var terrain = Terrain.activeTerrain ?? Object.FindFirstObjectByType<Terrain>();
            float baseOffset = terrain != null ? terrain.GetPosition().y : 0f;

            // Walls share one material; roofs get a per-class colour (mirrors the
            // glb export). Accumulate one mesh per group.
            var groups = new Dictionary<string, Mesh_>();
            Mesh_ Group(string key)
            {
                if (!groups.TryGetValue(key, out var g)) groups[key] = g = new Mesh_();
                return g;
            }

            foreach (var b in buildings)
            {
                var ring = b["ring_enu"] as JArray;
                if (ring == null || ring.Count < 3) continue;
                float height = b["height_m"] != null && b["height_m"].Type != JTokenType.Null
                    ? (float)b["height_m"] : 6.0f;
                string cls = (string)(b["class"] ?? "generic");

                var pts = new Vector2[ring.Count];
                float cx = 0f, cz = 0f;
                for (int i = 0; i < ring.Count; i++)
                {
                    pts[i] = new Vector2((float)ring[i][0], (float)ring[i][1]);  // (e, n)
                    cx += pts[i].x; cz += pts[i].y;
                }
                cx /= ring.Count; cz /= ring.Count;
                float baseY = (terrain != null
                    ? terrain.SampleHeight(new Vector3(cx, 0f, cz)) : 0f) + baseOffset;
                float top = baseY + Mathf.Max(2f, height);

                var walls = Group("walls");
                for (int i = 0; i < ring.Count; i++)
                {
                    var a = pts[i];
                    var c = pts[(i + 1) % ring.Count];
                    var p0 = new Vector3(a.x, baseY, a.y);
                    var p1 = new Vector3(c.x, baseY, c.y);
                    var p2 = new Vector3(c.x, top, c.y);
                    var p3 = new Vector3(a.x, top, a.y);
                    walls.Quad(p0, p1, p2, p3);
                }
                // flat roof: fan from the centroid (double-sided material, so
                // winding does not matter; LoD1 fallback).
                var roof = Group("roof_" + cls);
                var centre = new Vector3(cx, top, cz);
                for (int i = 0; i < ring.Count; i++)
                {
                    var a = pts[i];
                    var c = pts[(i + 1) % ring.Count];
                    roof.Tri(centre, new Vector3(a.x, top, a.y), new Vector3(c.x, top, c.y));
                }
            }

            var parent = new GameObject("DroneCV Buildings");
            foreach (var kv in groups)
            {
                Color col = kv.Key == "walls"
                    ? palette.Wall : palette.Roof(kv.Key.Substring("roof_".Length));
                var go = new GameObject(kv.Key);
                go.transform.SetParent(parent.transform, false);
                var mf = go.AddComponent<MeshFilter>();
                var mr = go.AddComponent<MeshRenderer>();
                mf.sharedMesh = kv.Value.ToMesh(kv.Key);
                mr.sharedMaterial = PipelineMaterials.Lit(col, doubleSided: true);
                go.AddComponent<MeshCollider>().sharedMesh = mf.sharedMesh;
                GameObjectUtility.SetStaticEditorFlags(go,
                    StaticEditorFlags.BatchingStatic | StaticEditorFlags.OccluderStatic |
                    StaticEditorFlags.OccludeeStatic);
            }
            Debug.Log($"[DroneCV] built {buildings.Count} buildings natively " +
                      $"({groups.Count} material groups).");
            return true;
        }

        // ---- palette ----
        class Pal
        {
            public Color Wall = new Color(0.80f, 0.76f, 0.68f);
            public Color[] Roofs = { new Color(0.62f, 0.34f, 0.28f) };
            public Color Roof(string cls)
            {
                switch (cls)
                {
                    case "commercial": return Roofs[Mathf.Min(1, Roofs.Length - 1)];
                    case "industrial": return new Color(0.50f, 0.50f, 0.52f);
                    case "landmark": return new Color(0.82f, 0.78f, 0.70f);
                    default: return Roofs[0];
                }
            }
        }

        static Pal LoadPalette(string dir)
        {
            var p = new Pal();
            var path = Path.Combine(dir, "palette.json");
            if (!File.Exists(path)) return p;
            try
            {
                var j = JObject.Parse(File.ReadAllText(path));
                var wall = j["wall"] as JArray;
                if (wall != null && wall.Count > 0) p.Wall = ToColor(wall[0]);
                var roof = j["roof"] as JArray;
                if (roof != null && roof.Count > 0)
                {
                    var list = new List<Color>();
                    foreach (var c in roof) list.Add(ToColor(c));
                    p.Roofs = list.ToArray();
                }
            }
            catch (System.Exception e) { Debug.LogWarning("[DroneCV] palette.json unreadable: " + e.Message); }
            return p;
        }

        static Color ToColor(JToken t) =>
            new Color((float)t[0], (float)t[1], (float)t[2]);

        // ---- mesh accumulator ----
        class Mesh_
        {
            readonly List<Vector3> v = new();
            readonly List<int> t = new();

            public void Tri(Vector3 a, Vector3 b, Vector3 c)
            {
                int i = v.Count; v.Add(a); v.Add(b); v.Add(c);
                t.Add(i); t.Add(i + 1); t.Add(i + 2);
            }

            public void Quad(Vector3 a, Vector3 b, Vector3 c, Vector3 d)
            {
                Tri(a, b, c); Tri(a, c, d);
            }

            public Mesh ToMesh(string name)
            {
                var m = new Mesh { name = name };
                if (v.Count > 65000) m.indexFormat = IndexFormat.UInt32;
                m.SetVertices(v);
                m.SetTriangles(t, 0);
                m.RecalculateNormals();
                m.RecalculateBounds();
                return m;
            }
        }
    }
}
