// DroneCV GIS Environment Builder — Editor window.
//
// Mirrors the desktop PySide6 GUI (minus the training / localization /
// reliability parts, which are not part of the 3D product) and drives the SAME
// Python pipeline under the hood (auto-installed by PythonEnv). It builds a
// real-world area, imports the generated terrain + objects directly into the
// current or a new scene (the product default), exports to other formats and
// can merge a whole district into one object + one material.
//
// It REUSES: GisSceneImporter (import), the Python `dronecv` CLI (generation +
// export), SceneExporters and DistrictMerger. Nothing is duplicated.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using DroneCV.Flight.Editor;   // GisSceneImporter (same assembly)
using UnityEditor;
using UnityEngine;

namespace DroneCV.Flight.Editor.Gis
{
    public class DroneCVGisWindow : EditorWindow
    {
        const string P = "DroneCV.Gis.";   // EditorPrefs key prefix

        // --- build config
        string _env = "my_city";
        string _place = "";
        double _minLat = 43.31, _minLon = 11.32, _maxLat = 43.33, _maxLon = 11.35;
        float _resM = 1.0f;
        int _buildings = 0;                 // 0 osm_overpass, 1 overture
        int _imagery = 0;                   // 0 none, 1 s2, 2 eox
        bool _reconstruct = false;
        bool _fallback = false;
        string _palette = "";
        float _cellM = 500f;

        // --- import / export / merge
        bool _newScene = false;
        int _terrainRes = 513;
        bool _expObj = true, _expGlb = true, _expPrefab = false, _expFbx = false;
        string _exportFolder = "";
        int _mergeUnit = 2;                 // 0 whole, 1 per-class, 2 cell500

        // --- runtime
        readonly StringBuilder _log = new();
        readonly object _logLock = new();
        readonly Queue<Action> _main = new();
        readonly object _mainLock = new();
        Vector2 _logScroll, _scroll;
        bool _busy;
        string _status = "";
        GameObject _lastImported;

        [MenuItem("DroneCV/GIS Environment Builder…")]
        public static void Open()
        {
            var w = GetWindow<DroneCVGisWindow>("GIS Builder");
            w.minSize = new Vector2(420, 640);
            w.LoadPrefs();
        }

        void OnEnable() { EditorApplication.update += Pump; }
        void OnDisable() { EditorApplication.update -= Pump; SavePrefs(); }

        void Pump()
        {
            Action a = null;
            lock (_mainLock) { if (_main.Count > 0) a = _main.Dequeue(); }
            a?.Invoke();
            if (a != null) Repaint();
        }

        void OnGUI()
        {
            _scroll = EditorGUILayout.BeginScrollView(_scroll);
            using (new EditorGUI.DisabledScope(_busy))
            {
                PythonPanel();
                EditorGUILayout.Space();
                BuildPanel();
                EditorGUILayout.Space();
                ImportPanel();
                EditorGUILayout.Space();
                ExportPanel();
                EditorGUILayout.Space();
                MergePanel();
                EditorGUILayout.Space();
                ViewerPanel();
            }
            EditorGUILayout.EndScrollView();
            LogPanel();
        }

        // ------------------------------------------------------------- panels

        void PythonPanel()
        {
            Header("1 · Python environment (auto-installed)");
            EditorGUILayout.LabelField("Status", PythonEnv.IsReady() ? PythonEnv.StatusLine() : "not installed");
            EditorGUILayout.HelpBox(
                "The builder uses a private Python it installs for you under Library/DroneCV/py " +
                "(Python + the dronecv package with the GIS extra). If your system has no suitable " +
                "Python, a standalone one is downloaded automatically. Nothing is installed system-wide.",
                MessageType.Info);
            if (GUILayout.Button(PythonEnv.IsReady() ? "Repair / update environment" : "Install environment"))
                RunAsync(() => PythonEnv.Ensure(Log), "environment ready");
        }

        void BuildPanel()
        {
            Header("2 · Build the area");
            _env = Field("Env name", _env,
                "Short name (letters/digits/_/-). Becomes artifacts/gis/<name> and the --env of every command.");
            _place = Field("Place (optional)", _place,
                "A place name (Nominatim). If set, it geocodes to a bbox. Otherwise use the bbox below.");
            EditorGUILayout.LabelField(new GUIContent("Bounding box (lat/lon)",
                "The rectangle to build, in degrees. Smaller = faster and lighter."));
            EditorGUI.indentLevel++;
            _minLat = EditorGUILayout.DoubleField("min lat", _minLat);
            _minLon = EditorGUILayout.DoubleField("min lon", _minLon);
            _maxLat = EditorGUILayout.DoubleField("max lat", _maxLat);
            _maxLon = EditorGUILayout.DoubleField("max lon", _maxLon);
            EditorGUI.indentLevel--;
            _resM = EditorGUILayout.Slider(new GUIContent("Resolution m/px",
                "Ground sampling of the imagery mosaic. LOWER = finer texture but the mosaic grows " +
                "quadratically and the build is slower; 1 m is a good default, use 2–4 for large areas. " +
                "Building shapes come from vector data, so this mainly affects the draped texture."), _resM, 0.5f, 10f);
            _buildings = EditorGUILayout.Popup(new GUIContent("Buildings",
                "Footprint + height source. osm_overpass: best in Europe; overture: often better heights in the US."),
                _buildings, new[] { "osm_overpass", "overture" });
            _imagery = EditorGUILayout.Popup(new GUIContent("Imagery",
                "Satellite colour to drape. none = synthetic class colours (shape-first). s2 = Sentinel-2 " +
                "cloud-free (~10 m, free). eox = Sentinel-2 mosaic (non-commercial)."),
                _imagery, new[] { "none", "s2", "eox" });
            _reconstruct = EditorGUILayout.Toggle(new GUIContent("Reconstruct buildings",
                "Extract extra footprints from imagery ONLY where GIS has none. Needs an imagery source; " +
                "quality tracks its resolution."), _reconstruct);
            _fallback = EditorGUILayout.Toggle(new GUIContent("Overture fallback",
                "OFF keeps you in control of the source. ON silently retries a failed OSM cell from Overture."), _fallback);
            _palette = FolderField("Palette photos", _palette,
                "Optional folder of area photos; their roof/wall colours tint the synthetic materials.");
            _cellM = EditorGUILayout.FloatField(new GUIContent("Cell size (m)",
                "Fixed download-grid cell size. Smaller cells = smaller, more resumable requests."), _cellM);

            if (GUILayout.Button("Build (download + generate)"))
                DoBuild();
        }

        void ImportPanel()
        {
            Header("3 · Import into Unity (product default)");
            _newScene = EditorGUILayout.Popup(new GUIContent("Target scene",
                "Where to place the generated terrain + objects."),
                _newScene ? 1 : 0, new[] { "Current scene", "New empty scene" }) == 1;
            _terrainRes = EditorGUILayout.IntPopup(new GUIContent("Terrain resolution",
                "Heightmap grid side (2^n+1). HIGHER = finer relief but larger/slower; 513 is a good default, " +
                "1025/2049 for mountains, 257 for flat areas. Only affects the ground, not buildings."),
                _terrainRes, new[] { "129", "257", "513", "1025", "2049" }, new[] { 129, 257, 513, 1025, 2049 });
            if (GUILayout.Button("Export scene + import into Unity"))
                DoExportAndImport();
        }

        void ExportPanel()
        {
            Header("4 · Export to other formats");
            _expObj = EditorGUILayout.Toggle(new GUIContent("OBJ", "Wavefront .obj + .mtl (native writer)."), _expObj);
            _expGlb = EditorGUILayout.Toggle(new GUIContent("glTF / GLB", "The pipeline's scene.glb (PBR, facade texture)."), _expGlb);
            _expPrefab = EditorGUILayout.Toggle(new GUIContent("Unity Prefab + TerrainData", "Reusable prefab in the project."), _expPrefab);
            using (new EditorGUI.DisabledScope(!SceneExporters.FbxAvailable()))
                _expFbx = EditorGUILayout.Toggle(new GUIContent("FBX",
                    SceneExporters.FbxAvailable() ? "Via the FBX Exporter package." :
                    "Install com.unity.formats.fbx to enable."), _expFbx);
            _exportFolder = FolderField("Output folder", _exportFolder, "Where OBJ/GLB/FBX files are written.");
            if (GUILayout.Button("Export selected formats"))
                DoExport();
        }

        void MergePanel()
        {
            Header("5 · Merge district (fewer objects, one material)");
            _mergeUnit = EditorGUILayout.Popup(new GUIContent("Unit",
                "Whole scene = one object+material for all buildings. Per class = one per building class. " +
                "Cell 500 m = one per neighbourhood block (recommended, composable)."),
                _mergeUnit, new[] { "Whole scene", "Per class", "Cell 500 m" });
            EditorGUILayout.HelpBox(
                "Combines the imported building meshes into one mesh + one material per unit, baking each " +
                "building's colour into vertex colours over a single shared tiling facade texture. Select the " +
                "imported root (or the last import is used).", MessageType.None);
            if (GUILayout.Button("Merge selected / last import"))
                DoMerge();
        }

        void ViewerPanel()
        {
            Header("6 · Realtime 3D viewer");
            // `gis view` serves + opens the browser and blocks until stopped, so
            // it runs on a detached thread (it must NOT gate the busy state).
            if (GUILayout.Button("Open 3D viewer (browser)"))
            {
                _status = "starting 3D viewer…";
                new Thread(() => PythonEnv.RunDronecv($"gis view --env {_env}", Log)) { IsBackground = true }.Start();
            }
        }

        void LogPanel()
        {
            EditorGUILayout.LabelField(_busy ? "Working…  " + _status : _status, EditorStyles.miniLabel);
            _logScroll = EditorGUILayout.BeginScrollView(_logScroll, GUILayout.Height(140));
            string text; lock (_logLock) text = _log.ToString();
            EditorGUILayout.TextArea(text, GUILayout.ExpandHeight(true));
            EditorGUILayout.EndScrollView();
        }

        // ------------------------------------------------------------- actions

        void DoBuild()
        {
            SavePrefs();
            var args = new StringBuilder($"gis build --env-name {_env}");
            if (!string.IsNullOrWhiteSpace(_place)) args.Append($" --place \"{_place}\"");
            else args.Append($" --bbox {_minLat},{_minLon},{_maxLat},{_maxLon}");
            args.Append($" --res {_resM.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
            args.Append($" --cell-m {_cellM.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
            if (_imagery == 1) args.Append(" --imagery s2");
            else if (_imagery == 2) args.Append(" --imagery eox");
            if (_reconstruct) args.Append(" --reconstruct-buildings");
            if (_fallback) args.Append(" --allow-overture-fallback");
            if (!string.IsNullOrWhiteSpace(_palette)) args.Append($" --palette-photos \"{_palette}\"");
            args.Append(" --on-cell-fail defer");
            RunAsync(() =>
            {
                if (!PythonEnv.IsReady()) PythonEnv.Ensure(Log);
                int code = PythonEnv.RunDronecv(args.ToString(), Log);
                if (code != 0) throw new Exception("build failed (exit " + code + ")");
                return null;
            }, "build complete");
        }

        void DoExportAndImport()
        {
            SavePrefs();
            string outDir = Path.Combine(PythonEnv.ManagedRoot, "exports", _env);
            RunAsync(() =>
            {
                if (!PythonEnv.IsReady()) PythonEnv.Ensure(Log);
                int code = PythonEnv.RunDronecv(
                    $"gis export-scene --env {_env} --out \"{outDir}\" --terrain-res {_terrainRes}", Log);
                if (code != 0) throw new Exception("export-scene failed (exit " + code + ")");
                return outDir;
            }, "imported into Unity", onMain: dir =>
            {
                // Unity API must run on the main thread.
                GisSceneImporter.ImportFolder((string)dir, _newScene);
                _lastImported = GameObject.Find("DroneCV GIS Scene (glTF)")
                                ?? GameObject.Find("DroneCV Buildings");
            });
        }

        void DoExport()
        {
            if (string.IsNullOrWhiteSpace(_exportFolder)) { _exportFolder = PythonEnv.WorkspaceDir; }
            var root = SelectedOrLast();
            if (root == null) { Log("Select an imported scene object first."); return; }
            string outDir = _exportFolder;
            RunOnMain(() =>
            {
                if (_expObj) { SceneExporters.ExportObj(root, Path.Combine(outDir, _env + ".obj")); Log("OBJ written."); }
                if (_expGlb)
                {
                    var sceneExport = Path.Combine(PythonEnv.ManagedRoot, "exports", _env);
                    if (SceneExporters.ExportGlb(sceneExport, Path.Combine(outDir, _env + ".glb"))) Log("GLB copied.");
                    else Log("scene.glb not found — run Import (section 3) first.");
                }
                if (_expPrefab) Log("Prefab: " + SceneExporters.ExportPrefab(root, "Assets/DroneCVImported"));
                if (_expFbx && SceneExporters.FbxAvailable())
                    Log("FBX: " + SceneExporters.ExportFbx(root, Path.Combine(outDir, _env + ".fbx")));
                AssetDatabase.Refresh();
                _status = "export done → " + outDir;
            });
        }

        void DoMerge()
        {
            var root = SelectedOrLast();
            if (root == null) { Log("Select an imported building root first."); return; }
            RunOnMain(() =>
            {
                var r = DistrictMerger.Merge(root, (MergeUnit)_mergeUnit, _cellM);
                _status = $"merged {r.sourceObjects} meshes → {r.mergedObjects} object(s)";
                Log(_status);
            });
        }

        GameObject SelectedOrLast() => Selection.activeGameObject ?? _lastImported;

        // ------------------------------------------------------------- plumbing

        /// Run `work` on a background thread; marshal `onMain(result)` (if any)
        /// back to the main thread. `work` returns a value passed to `onMain`.
        void RunAsync(Func<object> work, string okStatus, Action<object> onMain = null)
        {
            _busy = true; _status = "working…";
            new Thread(() =>
            {
                object result = null; string err = null;
                try { result = work(); }
                catch (Exception e) { err = e.Message; }
                lock (_mainLock)
                {
                    _main.Enqueue(() =>
                    {
                        _busy = false;
                        if (err != null) { _status = "error: " + err; Log("ERROR: " + err); }
                        else { _status = okStatus; onMain?.Invoke(result); }
                    });
                }
            }) { IsBackground = true }.Start();
        }

        void RunOnMain(Action a) { lock (_mainLock) _main.Enqueue(a); }

        void Log(string line)
        {
            lock (_logLock) { _log.AppendLine(line); if (_log.Length > 60000) _log.Remove(0, 20000); }
        }

        // ------------------------------------------------------------- widgets

        void Header(string s) => EditorGUILayout.LabelField(s, EditorStyles.boldLabel);

        string Field(string label, string val, string tip) =>
            EditorGUILayout.TextField(new GUIContent(label, tip), val);

        string FolderField(string label, string val, string tip)
        {
            EditorGUILayout.BeginHorizontal();
            val = EditorGUILayout.TextField(new GUIContent(label, tip), val);
            if (GUILayout.Button("…", GUILayout.Width(28)))
            {
                var p = EditorUtility.OpenFolderPanel(label, val, "");
                if (!string.IsNullOrEmpty(p)) val = p;
            }
            EditorGUILayout.EndHorizontal();
            return val;
        }

        // ------------------------------------------------------------- prefs

        void SavePrefs()
        {
            EditorPrefs.SetString(P + "env", _env);
            EditorPrefs.SetString(P + "place", _place);
            EditorPrefs.SetFloat(P + "minLat", (float)_minLat);
            EditorPrefs.SetFloat(P + "minLon", (float)_minLon);
            EditorPrefs.SetFloat(P + "maxLat", (float)_maxLat);
            EditorPrefs.SetFloat(P + "maxLon", (float)_maxLon);
            EditorPrefs.SetFloat(P + "resM", _resM);
            EditorPrefs.SetInt(P + "buildings", _buildings);
            EditorPrefs.SetInt(P + "imagery", _imagery);
            EditorPrefs.SetBool(P + "reconstruct", _reconstruct);
            EditorPrefs.SetBool(P + "fallback", _fallback);
            EditorPrefs.SetString(P + "palette", _palette);
            EditorPrefs.SetFloat(P + "cellM", _cellM);
            EditorPrefs.SetInt(P + "terrainRes", _terrainRes);
            EditorPrefs.SetBool(P + "newScene", _newScene);
            EditorPrefs.SetInt(P + "mergeUnit", _mergeUnit);
            EditorPrefs.SetString(P + "exportFolder", _exportFolder);
        }

        void LoadPrefs()
        {
            _env = EditorPrefs.GetString(P + "env", _env);
            _place = EditorPrefs.GetString(P + "place", _place);
            _minLat = EditorPrefs.GetFloat(P + "minLat", (float)_minLat);
            _minLon = EditorPrefs.GetFloat(P + "minLon", (float)_minLon);
            _maxLat = EditorPrefs.GetFloat(P + "maxLat", (float)_maxLat);
            _maxLon = EditorPrefs.GetFloat(P + "maxLon", (float)_maxLon);
            _resM = EditorPrefs.GetFloat(P + "resM", _resM);
            _buildings = EditorPrefs.GetInt(P + "buildings", _buildings);
            _imagery = EditorPrefs.GetInt(P + "imagery", _imagery);
            _reconstruct = EditorPrefs.GetBool(P + "reconstruct", _reconstruct);
            _fallback = EditorPrefs.GetBool(P + "fallback", _fallback);
            _palette = EditorPrefs.GetString(P + "palette", _palette);
            _cellM = EditorPrefs.GetFloat(P + "cellM", _cellM);
            _terrainRes = EditorPrefs.GetInt(P + "terrainRes", _terrainRes);
            _newScene = EditorPrefs.GetBool(P + "newScene", _newScene);
            _mergeUnit = EditorPrefs.GetInt(P + "mergeUnit", _mergeUnit);
            _exportFolder = EditorPrefs.GetString(P + "exportFolder", _exportFolder);
        }
    }
}
