// Imports a dronecv GIS scene export (dronecv gis export-scene) into Unity:
//   Terrain with real orography (terrain.raw) + class splatmap + tree
//   instances (density and typology from trees.json), buildings.obj placed
//   as meshes, GeoAnchorAsset created from scene_meta.json.
//
// Menu: DroneCV > Import GIS Scene…  (pick the export FOLDER)
//
// Coordinate mapping: export is Z-up ENU (x=E, y=N, z=Up); Unity terrain is
// Y-up with x=E, z=N — heights map directly, building meshes are rotated
// -90° around X on import by Unity's OBJ importer convention handling below.

using System.IO;
using DroneCV.Flight.Editor.Gis;
using DroneCV.Flight.Geo;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DroneCV.Flight.Editor
{
    public static class GisSceneImporter
    {
        [MenuItem("DroneCV/Import GIS Scene…")]
        public static void Import()
        {
            var dir = EditorUtility.OpenFolderPanel("Select dronecv GIS export folder", "", "");
            if (string.IsNullOrEmpty(dir)) return;
            ImportFolder(dir, intoNewScene: false);
        }

        /// Import a dronecv GIS export folder into Unity. When `intoNewScene` is
        /// true a fresh empty scene is created first; otherwise the objects are
        /// added to the currently open scene (the product default). Reused by
        /// the GIS Environment Builder window — no logic is duplicated.
        public static void ImportFolder(string dir, bool intoNewScene)
        {
            if (intoNewScene)
                EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var meta = JObject.Parse(File.ReadAllText(Path.Combine(dir, "scene_meta.json")));

            int res = (int)meta["terrain_resolution"];
            float extentE = (float)meta["extent_e_m"];
            float extentN = (float)meta["extent_n_m"];
            float hMin = (float)meta["height_min_m"];
            float hScale = (float)meta["height_scale_m"];
            float e0 = (float)meta["e0"];
            float n0 = (float)meta["n0"];

            // ---- TerrainData: orography ----
            var data = new TerrainData
            {
                heightmapResolution = res,
                size = new Vector3(extentE, hScale, extentN),
            };
            var raw = File.ReadAllBytes(Path.Combine(dir, "terrain.raw"));
            var heights = new float[res, res];
            for (int r = 0; r < res; r++)
                for (int c = 0; c < res; c++)
                {
                    int i = (r * res + c) * 2;
                    heights[r, c] = (raw[i] | (raw[i + 1] << 8)) / 65535f;
                }
            data.SetHeights(0, 0, heights);

            // ---- splat layers (solid-color procedural textures) ----
            data.terrainLayers = new[]
            {
                MakeLayer("ground", new Color(0.42f, 0.40f, 0.33f)),
                MakeLayer("vegetation", new Color(0.24f, 0.38f, 0.20f)),
                MakeLayer("hard", new Color(0.36f, 0.36f, 0.38f)),
                MakeLayer("water", new Color(0.16f, 0.30f, 0.50f)),
            };
            var splatTex = new Texture2D(2, 2);
            splatTex.LoadImage(File.ReadAllBytes(Path.Combine(dir, "splatmap.png")));
            int ar = data.alphamapResolution;
            var alpha = new float[ar, ar, 4];
            for (int r = 0; r < ar; r++)
                for (int c = 0; c < ar; c++)
                {
                    var px = splatTex.GetPixelBilinear(c / (float)(ar - 1), r / (float)(ar - 1));
                    float sum = Mathf.Max(px.r + px.g + px.b + px.a, 1e-3f);
                    alpha[r, c, 0] = px.r / sum;
                    alpha[r, c, 1] = px.g / sum;
                    alpha[r, c, 2] = px.b / sum;
                    alpha[r, c, 3] = px.a / sum;
                }
            data.SetAlphamaps(0, 0, alpha);

            // ---- trees: density + typology ----
            data.treePrototypes = new[]
            {
                new TreePrototype { prefab = TreePrefab("broadleaf", new Color(0.20f, 0.35f, 0.16f)) },
                new TreePrototype { prefab = TreePrefab("conifer", new Color(0.14f, 0.28f, 0.18f)) },
            };
            var trees = JObject.Parse(File.ReadAllText(Path.Combine(dir, "trees.json")))["trees"];
            var instances = new System.Collections.Generic.List<TreeInstance>();
            foreach (var t in trees)
            {
                float e = (float)t["e"];
                float n = (float)t["n"];
                float h = (float)t["height"];
                instances.Add(new TreeInstance
                {
                    position = new Vector3((e - e0) / extentE, 0f, (n - n0) / extentN),
                    prototypeIndex = (string)t["type"] == "conifer" ? 1 : 0,
                    widthScale = h / 15f,
                    heightScale = h / 15f,
                    color = Color.white,
                    lightmapColor = Color.white,
                });
            }
            data.SetTreeInstances(instances.ToArray(), true);

            AssetDatabase.CreateAsset(data, "Assets/DroneCV_TerrainData.asset");

            // ---- scene objects ----
            var terrainGo = Terrain.CreateTerrainGameObject(data);
            terrainGo.name = "DroneCV Terrain";
            // Terrain origin at its SW corner: place so ENU origin = anchor.
            terrainGo.transform.position = new Vector3(e0, hMin, n0);
            // URP/HDRP: a freshly created Terrain has no valid material and
            // renders magenta — assign the pipeline's terrain material.
            var terrMat = PipelineMaterials.TerrainMaterial();
            if (terrMat != null) terrainGo.GetComponent<Terrain>().materialTemplate = terrMat;

            ImportBuildings(dir);
            CreateAnchorAsset(meta);

            Debug.Log($"[DroneCV] GIS scene imported: {instances.Count} trees, " +
                      $"attribution: {meta["attribution"]}");
        }

        private static void ImportBuildings(string dir)
        {
            // 1) scene.glb via a glTF importer (best: LoD2 roofs + PBR palette
            //    materials + facade texture), if one is installed.
            if (TryImportGlb(dir)) return;
            // 2) native builder from buildings.json + palette.json: works in any
            //    render pipeline with per-class materials ALWAYS assigned, no
            //    external package needed (LoD1).
            if (BuildingBuilder.TryBuild(dir)) return;
            // 3) last resort: the plain OBJ (now carries a .mtl, but no pipeline
            //    material upgrade).
            ImportObj(dir);
        }

        private static bool TryImportGlb(string dir)
        {
            var glb = Path.Combine(dir, "scene.glb");
            if (!File.Exists(glb)) return false;
            Directory.CreateDirectory("Assets/DroneCVImported");
            var glbDst = "Assets/DroneCVImported/scene.glb";
            File.Copy(glb, glbDst, true);
            AssetDatabase.ImportAsset(glbDst);
            var glbPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(glbDst);
            if (glbPrefab == null)
            {
                Debug.Log("[DroneCV] no glTF importer package — building natively " +
                          "from buildings.json (install com.unity.cloud.gltfast for LoD2 + textures).");
                return false;
            }
            var scene = Object.Instantiate(glbPrefab);
            scene.name = "DroneCV GIS Scene (glTF)";
            foreach (var mf in scene.GetComponentsInChildren<MeshFilter>())
                mf.gameObject.AddComponent<MeshCollider>();
            return true;
        }

        private static void ImportObj(string dir)
        {
            var src = Path.Combine(dir, "buildings.obj");
            if (!File.Exists(src)) return;
            Directory.CreateDirectory("Assets/DroneCVImported");
            var dst = "Assets/DroneCVImported/buildings.obj";
            File.Copy(src, dst, true);
            var mtl = Path.Combine(dir, "buildings.mtl");
            if (File.Exists(mtl)) File.Copy(mtl, "Assets/DroneCVImported/buildings.mtl", true);
            AssetDatabase.ImportAsset(dst);
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(dst);
            if (prefab != null)
            {
                var go = Object.Instantiate(prefab);
                go.name = "DroneCV Buildings";
                // Export is Z-up; Unity's obj import applies -90 X — undo the
                // axis mismatch so E,N,Up land on x,z,y.
                go.transform.rotation = Quaternion.Euler(-90f, 0f, 0f);
                go.transform.localScale = new Vector3(-1f, 1f, 1f); // handedness
                foreach (var mf in go.GetComponentsInChildren<MeshFilter>())
                    mf.gameObject.AddComponent<MeshCollider>();
            }
        }

        private static void CreateAnchorAsset(JObject meta)
        {
            var anchor = ScriptableObject.CreateInstance<GeoAnchorAsset>();
            anchor.Lat0 = (double)meta["anchor"]["lat0"];
            anchor.Lon0 = (double)meta["anchor"]["lon0"];
            anchor.Alt0 = (double)meta["anchor"]["alt0"];
            anchor.TrueNorthOffsetDeg = 0.0;
            AssetDatabase.CreateAsset(anchor, "Assets/DroneCV_GeoAnchor.asset");
        }

        private static TerrainLayer MakeLayer(string name, Color color)
        {
            var tex = new Texture2D(4, 4);
            var px = new Color[16];
            for (int i = 0; i < 16; i++) px[i] = color;
            tex.SetPixels(px);
            tex.Apply();
            var layer = new TerrainLayer { diffuseTexture = tex, tileSize = new Vector2(8, 8) };
            AssetDatabase.CreateAsset(layer, $"Assets/DroneCV_Layer_{name}.asset");
            return layer;
        }

        private static GameObject TreePrefab(string kind, Color color)
        {
            var path = $"Assets/DroneCVImported/tree_{kind}.prefab";
            var existing = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (existing != null) return existing;
            Directory.CreateDirectory("Assets/DroneCVImported");
            var go = GameObject.CreatePrimitive(kind == "conifer"
                ? PrimitiveType.Cylinder : PrimitiveType.Sphere);
            go.name = $"tree_{kind}";
            go.transform.localScale = kind == "conifer"
                ? new Vector3(4f, 7.5f, 4f) : new Vector3(8f, 6f, 8f);
            go.transform.position = new Vector3(0, kind == "conifer" ? 7.5f : 9f, 0);
            var mat = PipelineMaterials.Lit(color);  // pipeline-aware (no magenta in URP)
            AssetDatabase.CreateAsset(mat, $"Assets/DroneCVImported/tree_{kind}_mat.asset");
            go.GetComponent<MeshRenderer>().sharedMaterial = mat;
            var prefab = PrefabUtility.SaveAsPrefabAsset(go, path);
            Object.DestroyImmediate(go);
            return prefab;
        }
    }
}
