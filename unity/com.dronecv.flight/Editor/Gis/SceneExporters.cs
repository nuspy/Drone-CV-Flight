// Export the imported / merged environment to other formats from inside Unity.
//   OBJ    — native C# writer (no extra package), .obj + .mtl.
//   glTF   — reuses the pipeline's scene.glb (the richest artifact); if the
//            glTFast exporter is present it can export the live (merged) scene.
//   Prefab — saves the imported root as a reusable prefab + the TerrainData.
//   FBX    — only if com.unity.formats.fbx is installed (reflection, so the
//            package stays an optional dependency).

using System;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DroneCV.Flight.Editor.Gis
{
    public static class SceneExporters
    {
        /// Write `root` to `objPath` as Wavefront OBJ (+ sibling .mtl). Walks all
        /// MeshFilters, in world space, one OBJ group + one material per renderer.
        public static void ExportObj(GameObject root, string objPath)
        {
            var mtlPath = Path.ChangeExtension(objPath, ".mtl");
            var obj = new StringBuilder();
            var mtl = new StringBuilder();
            var ci = CultureInfo.InvariantCulture;
            obj.AppendLine("# DroneCV GIS export");
            obj.AppendLine("mtllib " + Path.GetFileName(mtlPath));

            int vBase = 1, nBase = 1, tBase = 1, matId = 0;
            foreach (var mf in root.GetComponentsInChildren<MeshFilter>(true))
            {
                var mesh = mf.sharedMesh;
                if (mesh == null) continue;
                var tf = mf.transform;
                var mr = mf.GetComponent<MeshRenderer>();
                string matName = $"mat_{matId++}";
                WriteMtl(mtl, matName, mr != null ? mr.sharedMaterial : null, ci);

                obj.AppendLine("o " + SafeName(mf.gameObject.name));
                var verts = mesh.vertices;
                foreach (var v in verts)
                {
                    var w = tf.TransformPoint(v);
                    obj.AppendLine($"v {w.x.ToString(ci)} {w.y.ToString(ci)} {w.z.ToString(ci)}");
                }
                var normals = mesh.normals;
                bool hasN = normals.Length == verts.Length;
                if (hasN)
                    foreach (var n in normals)
                    {
                        var w = tf.TransformDirection(n).normalized;
                        obj.AppendLine($"vn {w.x.ToString(ci)} {w.y.ToString(ci)} {w.z.ToString(ci)}");
                    }
                var uvs = mesh.uv;
                bool hasT = uvs.Length == verts.Length;
                if (hasT)
                    foreach (var t in uvs)
                        obj.AppendLine($"vt {t.x.ToString(ci)} {t.y.ToString(ci)}");

                obj.AppendLine("usemtl " + matName);
                for (int sm = 0; sm < mesh.subMeshCount; sm++)
                {
                    var tris = mesh.GetTriangles(sm);
                    for (int i = 0; i < tris.Length; i += 3)
                    {
                        int a = tris[i] + vBase, b = tris[i + 1] + vBase, c = tris[i + 2] + vBase;
                        int at = tris[i] + tBase, bt = tris[i + 1] + tBase, ct = tris[i + 2] + tBase;
                        int an = tris[i] + nBase, bn = tris[i + 1] + nBase, cn = tris[i + 2] + nBase;
                        obj.AppendLine("f " + Face(a, at, an, hasT, hasN) + " " +
                                              Face(b, bt, bn, hasT, hasN) + " " +
                                              Face(c, ct, cn, hasT, hasN));
                    }
                }
                vBase += verts.Length;
                if (hasN) nBase += verts.Length;
                if (hasT) tBase += verts.Length;
            }
            Directory.CreateDirectory(Path.GetDirectoryName(objPath));
            File.WriteAllText(objPath, obj.ToString());
            File.WriteAllText(mtlPath, mtl.ToString());
        }

        static string Face(int v, int t, int n, bool hasT, bool hasN)
        {
            if (hasT && hasN) return $"{v}/{t}/{n}";
            if (hasN) return $"{v}//{n}";
            if (hasT) return $"{v}/{t}";
            return v.ToString();
        }

        static void WriteMtl(StringBuilder mtl, string name, Material m, CultureInfo ci)
        {
            Color c = Color.gray;
            if (m != null)
            {
                if (m.HasProperty("_BaseColor")) c = m.GetColor("_BaseColor");
                else if (m.HasProperty("_Color")) c = m.GetColor("_Color");
            }
            mtl.AppendLine("newmtl " + name);
            mtl.AppendLine($"Kd {c.r.ToString(ci)} {c.g.ToString(ci)} {c.b.ToString(ci)}");
            mtl.AppendLine("Ka 0 0 0");
            mtl.AppendLine("d 1");
            mtl.AppendLine("illum 1");
        }

        static string SafeName(string s) => s.Replace(' ', '_');

        /// Copy the pipeline's scene.glb (Y-up, PBR, facade texture) to `dst`.
        public static bool ExportGlb(string sceneExportDir, string dst)
        {
            var src = Path.Combine(sceneExportDir, "scene.glb");
            if (!File.Exists(src)) return false;
            Directory.CreateDirectory(Path.GetDirectoryName(dst));
            File.Copy(src, dst, true);
            return true;
        }

        /// Save `root` (and any TerrainData under it) as reusable assets.
        public static string ExportPrefab(GameObject root, string assetDir)
        {
            Directory.CreateDirectory(assetDir);
            var terrain = root.GetComponentInChildren<Terrain>();
            if (terrain != null && terrain.terrainData != null &&
                string.IsNullOrEmpty(AssetDatabase.GetAssetPath(terrain.terrainData)))
            {
                AssetDatabase.CreateAsset(terrain.terrainData, $"{assetDir}/{SafeName(root.name)}_Terrain.asset");
            }
            var path = AssetDatabase.GenerateUniqueAssetPath($"{assetDir}/{SafeName(root.name)}.prefab");
            PrefabUtility.SaveAsPrefabAsset(root, path);
            AssetDatabase.SaveAssets();
            return path;
        }

        /// True only if the FBX Exporter package (com.unity.formats.fbx) is
        /// present; discovered by reflection so it isn't a hard dependency.
        public static bool FbxAvailable() => FbxExportMethod() != null;

        /// Export via the FBX package if available. Returns the path or null.
        public static string ExportFbx(GameObject root, string fbxPath)
        {
            var mi = FbxExportMethod();
            if (mi == null) return null;
            Directory.CreateDirectory(Path.GetDirectoryName(fbxPath));
            // ModelExporter.ExportObject(string filePath, UnityEngine.Object singleObject)
            var res = mi.Invoke(null, new object[] { fbxPath, root });
            return res as string ?? fbxPath;
        }

        static System.Reflection.MethodInfo FbxExportMethod()
        {
            var t = Type.GetType("UnityEditor.Formats.Fbx.Exporter.ModelExporter, Unity.Formats.Fbx.Editor");
            if (t == null) return null;
            return t.GetMethod("ExportObject", new[] { typeof(string), typeof(UnityEngine.Object) });
        }
    }
}
