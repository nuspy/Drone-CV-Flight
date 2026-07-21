// District merge: reduce the imported building objects to ONE object + ONE
// material per "district", following the standard batching best practice for
// repeatable/tiling materials.
//
// The GIS export already groups buildings by class into a handful of meshes
// (walls_<class> + roof_<class>), all sharing a single tiling window texture on
// the walls and per-class solid roof colors. This utility combines a chosen
// GROUP into one mesh and bakes each source material's base color into VERTEX
// COLORS, so a single material (a tiling window texture multiplied by the vertex
// color) reproduces every building: walls keep their per-facade tiling UVs;
// roofs are remapped to the texture's plain plaster corner so they read as their
// solid color. That is the textbook way to keep a repeatable texture working
// "across the whole object" while collapsing to one draw material.
//
// Grouping units: WholeScene (1 object), PerClass (1 per building class),
// Cell500 (1 per 500 m world cell — a real neighbourhood block). Terrain and
// trees are left untouched.

using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace DroneCV.Flight.Editor.Gis
{
    public enum MergeUnit { WholeScene, PerClass, Cell500 }

    public static class DistrictMerger
    {
        const float CellM = 500f;
        // Plain plaster corner of the procedural window atlas (white area) — roofs
        // sample here so texture ≈ white and the roof reads as its vertex color.
        static readonly Vector2 PlasterUv = new Vector2(0.03f, 0.03f);

        public struct Result { public int mergedObjects; public int sourceObjects; }

        /// Merge the building meshes under `root` into one object + one material
        /// per group. Returns counts. Registers Undo. `root` is typically the
        /// imported "DroneCV GIS Scene (glTF)" / "DroneCV Buildings" object.
        public static Result Merge(GameObject root, MergeUnit unit, float cellSize = CellM)
        {
            var sources = CollectBuildingRenderers(root);
            if (sources.Count == 0)
            {
                Debug.LogWarning("[DroneCV] no building meshes found to merge under " + root.name);
                return new Result { mergedObjects = 0, sourceObjects = 0 };
            }

            Texture sharedTex = FindWindowTexture(sources);
            var mat = BuildSharedMaterial(sharedTex);

            // World bounds (for the cell grid origin).
            var min = new Vector3(float.MaxValue, 0, float.MaxValue);
            foreach (var s in sources)
            {
                var b = s.renderer.bounds;
                min.x = Mathf.Min(min.x, b.min.x); min.z = Mathf.Min(min.z, b.min.z);
            }

            // Accumulate triangles per group key.
            var groups = new Dictionary<string, MeshBuf>();
            foreach (var s in sources)
            {
                var mesh = s.filter.sharedMesh;
                if (mesh == null) continue;
                var mtx = s.renderer.transform.localToWorldMatrix;
                var verts = mesh.vertices;
                var normals = mesh.normals.Length == verts.Length ? mesh.normals : null;
                var uvs = mesh.uv.Length == verts.Length ? mesh.uv : null;
                Color32 col = s.color;

                for (int sm = 0; sm < mesh.subMeshCount; sm++)
                {
                    var tris = mesh.GetTriangles(sm);
                    for (int i = 0; i < tris.Length; i += 3)
                    {
                        int a = tris[i], b = tris[i + 1], c = tris[i + 2];
                        Vector3 wa = mtx.MultiplyPoint3x4(verts[a]);
                        Vector3 wb = mtx.MultiplyPoint3x4(verts[b]);
                        Vector3 wc = mtx.MultiplyPoint3x4(verts[c]);
                        string key = GroupKey(unit, s, (wa + wb + wc) / 3f, min, cellSize);
                        if (!groups.TryGetValue(key, out var buf)) groups[key] = buf = new MeshBuf();

                        AddVert(buf, wa, Norm(normals, a, mtx), Uv(uvs, a, s.isWall), col);
                        AddVert(buf, wb, Norm(normals, b, mtx), Uv(uvs, b, s.isWall), col);
                        AddVert(buf, wc, Norm(normals, c, mtx), Uv(uvs, c, s.isWall), col);
                    }
                }
            }

            var parent = new GameObject(root.name + " (merged)");
            Undo.RegisterCreatedObjectUndo(parent, "Merge district");
            int made = 0;
            foreach (var kv in groups)
            {
                var go = new GameObject(kv.Key);
                go.transform.SetParent(parent.transform, false);
                var mf = go.AddComponent<MeshFilter>();
                var mr = go.AddComponent<MeshRenderer>();
                mf.sharedMesh = kv.Value.ToMesh(kv.Key);
                mr.sharedMaterial = mat;
                go.AddComponent<MeshCollider>().sharedMesh = mf.sharedMesh;
                GameObjectUtility.SetStaticEditorFlags(go,
                    StaticEditorFlags.BatchingStatic | StaticEditorFlags.OccluderStatic |
                    StaticEditorFlags.OccludeeStatic);
                made++;
            }

            // Hide the originals (keep them for Undo rather than destroy).
            Undo.RegisterFullObjectHierarchyUndo(root, "Merge district");
            root.SetActive(false);
            Debug.Log($"[DroneCV] merged {sources.Count} building meshes into {made} object(s), 1 material.");
            return new Result { mergedObjects = made, sourceObjects = sources.Count };
        }

        // ---------------------------------------------------------------- helpers

        struct Src
        {
            public MeshRenderer renderer;
            public MeshFilter filter;
            public Color32 color;
            public bool isWall;
        }

        static List<Src> CollectBuildingRenderers(GameObject root)
        {
            var list = new List<Src>();
            foreach (var mr in root.GetComponentsInChildren<MeshRenderer>(true))
            {
                var mf = mr.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;
                string n = mr.gameObject.name.ToLowerInvariant();
                if (n.Contains("terrain")) continue;                 // leave the ground
                bool isWall = n.StartsWith("walls") || HasMainTex(mr.sharedMaterial);
                list.Add(new Src { renderer = mr, filter = mf, color = MatColor(mr.sharedMaterial), isWall = isWall });
            }
            return list;
        }

        static bool HasMainTex(Material m)
        {
            if (m == null) return false;
            if (m.HasProperty("_BaseMap") && m.GetTexture("_BaseMap") != null) return true;
            if (m.HasProperty("_MainTex") && m.GetTexture("_MainTex") != null) return true;
            return false;
        }

        static Color32 MatColor(Material m)
        {
            if (m == null) return new Color32(200, 200, 200, 255);
            if (m.HasProperty("_BaseColor")) return m.GetColor("_BaseColor");
            if (m.HasProperty("_Color")) return m.GetColor("_Color");
            return new Color32(200, 200, 200, 255);
        }

        static Texture FindWindowTexture(List<Src> sources)
        {
            foreach (var s in sources)
            {
                if (!s.isWall) continue;
                var m = s.renderer.sharedMaterial;
                if (m == null) continue;
                if (m.HasProperty("_BaseMap") && m.GetTexture("_BaseMap") != null) return m.GetTexture("_BaseMap");
                if (m.HasProperty("_MainTex") && m.GetTexture("_MainTex") != null) return m.GetTexture("_MainTex");
            }
            return Texture2D.whiteTexture;
        }

        static Material BuildSharedMaterial(Texture tex)
        {
            // The custom vertex-tint shader is Built-in-only; on URP/HDRP use the
            // pipeline's Lit shader (one material, no per-vertex tint).
            var shader = GraphicsSettings.currentRenderPipeline == null
                ? Shader.Find("DroneCV/DistrictMerged") : null;
            Material mat;
            if (shader != null) mat = new Material(shader);
            else
            {
                // No custom shader available: fall back to the active pipeline's
                // lit shader (vertex colours won't multiply, but geometry merges).
                mat = new Material(PipelineMaterials.LitShader());
                Debug.LogWarning("[DroneCV] DroneCV/DistrictMerged shader not found — " +
                                 "merged object will use a single plain material without per-vertex tint.");
            }
            if (tex != null)
            {
                if (mat.HasProperty("_MainTex")) mat.SetTexture("_MainTex", tex);
                if (mat.HasProperty("_BaseMap")) mat.SetTexture("_BaseMap", tex);
                tex.wrapMode = TextureWrapMode.Repeat;
            }
            mat.name = "District Merged";
            // Kept in-memory: it serializes into the scene when saved (no stray
            // .mat asset, and unit-testable). The window may persist it if needed.
            return mat;
        }

        static string GroupKey(MergeUnit unit, Src s, Vector3 centroid, Vector3 min, float cell)
        {
            switch (unit)
            {
                case MergeUnit.PerClass:
                    return "buildings_" + ClassOf(s.renderer.gameObject.name);
                case MergeUnit.Cell500:
                    int ix = Mathf.FloorToInt((centroid.x - min.x) / cell);
                    int iz = Mathf.FloorToInt((centroid.z - min.z) / cell);
                    return $"cell_{ix}_{iz}";
                default:
                    return "district";
            }
        }

        static string ClassOf(string name)
        {
            int us = name.IndexOf('_');
            return us >= 0 && us + 1 < name.Length ? name.Substring(us + 1) : name;
        }

        static Vector3 Norm(Vector3[] normals, int i, Matrix4x4 mtx) =>
            normals != null ? mtx.MultiplyVector(normals[i]).normalized : Vector3.up;

        static Vector2 Uv(Vector2[] uvs, int i, bool isWall) =>
            isWall && uvs != null ? uvs[i] : PlasterUv;

        static void AddVert(MeshBuf b, Vector3 p, Vector3 n, Vector2 uv, Color32 c)
        {
            b.pos.Add(p); b.nrm.Add(n); b.uv.Add(uv); b.col.Add(c); b.idx.Add(b.pos.Count - 1);
        }

        class MeshBuf
        {
            public readonly List<Vector3> pos = new();
            public readonly List<Vector3> nrm = new();
            public readonly List<Vector2> uv = new();
            public readonly List<Color32> col = new();
            public readonly List<int> idx = new();

            public Mesh ToMesh(string name)
            {
                var m = new Mesh { name = name };
                if (pos.Count > 65000) m.indexFormat = IndexFormat.UInt32;
                m.SetVertices(pos);
                m.SetNormals(nrm);
                m.SetUVs(0, uv);
                m.SetColors(col);
                m.SetTriangles(idx, 0);
                m.RecalculateBounds();
                return m;
            }
        }
    }
}
