// EditMode tests for the pipeline-aware materials, the native building builder
// and the terrain-in-OBJ export — the fixes for "no terrain / no materials /
// materials not assigned" on import/export.

using System.IO;
using DroneCV.Flight.Editor.Gis;
using NUnit.Framework;
using UnityEngine;

namespace DroneCV.Flight.Tests
{
    public class ImporterTests
    {
        readonly System.Collections.Generic.List<Object> _spawned = new();

        [TearDown]
        public void Cleanup()
        {
            foreach (var o in _spawned) if (o != null) Object.DestroyImmediate(o);
            _spawned.Clear();
        }

        [Test]
        public void PipelineMaterials_Lit_IsNeverNull()
        {
            var m = PipelineMaterials.Lit(Color.red);
            Assert.IsNotNull(m);
            Assert.IsNotNull(m.shader);
            _spawned.Add(m);
        }

        [Test]
        public void BuildingBuilder_BuildsGroupsWithAssignedMaterials()
        {
            var dir = Path.Combine(Path.GetTempPath(), "dronecv_bld_" + System.Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, "buildings.json"),
                "{\"buildings\":[" +
                "{\"ring_enu\":[[0,0],[10,0],[10,10],[0,10]],\"height_m\":9.0,\"class\":\"residential\"}," +
                "{\"ring_enu\":[[20,0],[30,0],[30,10],[20,10]],\"height_m\":12.0,\"class\":\"industrial\"}]}");

            bool built = BuildingBuilder.TryBuild(dir);
            Assert.IsTrue(built);
            var parent = GameObject.Find("DroneCV Buildings");
            Assert.IsNotNull(parent);
            _spawned.Add(parent);

            var renderers = parent.GetComponentsInChildren<MeshRenderer>();
            Assert.GreaterOrEqual(renderers.Length, 2);   // walls + roof group(s)
            foreach (var r in renderers)
            {
                Assert.IsNotNull(r.sharedMaterial, "every group has an assigned material");
                Assert.Greater(r.GetComponent<MeshFilter>().sharedMesh.vertexCount, 0);
            }
            Directory.Delete(dir, true);
        }

        [Test]
        public void ExportObj_IncludesTheTerrain()
        {
            var td = new TerrainData { heightmapResolution = 33, size = new Vector3(100, 20, 100) };
            var terrainGo = Terrain.CreateTerrainGameObject(td);
            terrainGo.name = "T";
            _spawned.Add(terrainGo);
            _spawned.Add(td);

            var path = Path.Combine(Path.GetTempPath(), "dronecv_terr.obj");
            SceneExporters.ExportObj(terrainGo, path);
            var text = File.ReadAllText(path);
            StringAssert.Contains("o DroneCV_Terrain", text);
            Assert.Greater(System.Text.RegularExpressions.Regex.Matches(text, @"(?m)^v ").Count, 100);
            File.Delete(path);
            File.Delete(Path.ChangeExtension(path, ".mtl"));
        }
    }
}
