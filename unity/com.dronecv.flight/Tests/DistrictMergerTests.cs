// EditMode tests for the district merge + OBJ export utilities. They build a
// tiny "district" (one textured wall mesh + one solid roof mesh) and check that
// the merge collapses it to a single object with a single material, and that
// the OBJ writer emits valid geometry.

using System.IO;
using DroneCV.Flight.Editor.Gis;
using NUnit.Framework;
using UnityEngine;

namespace DroneCV.Flight.Tests
{
    public class DistrictMergerTests
    {
        static GameObject Quad(string name, Transform parent, Material mat, float z)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent);
            var mf = go.AddComponent<MeshFilter>();
            var mr = go.AddComponent<MeshRenderer>();
            var m = new Mesh();
            m.vertices = new[]
            {
                new Vector3(0, 0, z), new Vector3(1, 0, z),
                new Vector3(1, 1, z), new Vector3(0, 1, z),
            };
            m.uv = new[] { Vector2.zero, Vector2.right, Vector2.one, Vector2.up };
            m.normals = new[] { Vector3.up, Vector3.up, Vector3.up, Vector3.up };
            m.triangles = new[] { 0, 1, 2, 0, 2, 3 };
            mf.sharedMesh = m;
            mr.sharedMaterial = mat;
            return go;
        }

        GameObject _root, _merged;

        [TearDown]
        public void Cleanup()
        {
            if (_root != null) Object.DestroyImmediate(_root);
            if (_merged != null) Object.DestroyImmediate(_merged);
        }

        [Test]
        public void Merge_WholeScene_CollapsesToOneObjectOneMaterial()
        {
            _root = new GameObject("DroneCV GIS Scene (glTF)");
            var wallMat = new Material(Shader.Find("Unlit/Texture"));
            wallMat.mainTexture = Texture2D.whiteTexture;               // -> treated as wall
            var roofMat = new Material(Shader.Find("Unlit/Color")) { color = Color.red }; // -> roof
            Quad("walls_residential", _root.transform, wallMat, 0f);
            Quad("roof_residential", _root.transform, roofMat, 5f);

            var res = DistrictMerger.Merge(_root, MergeUnit.WholeScene);

            Assert.AreEqual(2, res.sourceObjects);
            Assert.AreEqual(1, res.mergedObjects, "whole-scene merge must yield one object");

            _merged = GameObject.Find("DroneCV GIS Scene (glTF) (merged)");
            Assert.IsNotNull(_merged, "merged parent created");
            var renderers = _merged.GetComponentsInChildren<MeshRenderer>();
            Assert.AreEqual(1, renderers.Length, "one merged renderer");
            Assert.IsNotNull(renderers[0].sharedMaterial, "one shared material");
            // 2 quads * 2 triangles * 3 unshared verts = 12
            Assert.AreEqual(12, renderers[0].GetComponent<MeshFilter>().sharedMesh.vertexCount);
            Assert.IsFalse(_root.activeSelf, "originals hidden after merge");
        }

        [Test]
        public void Merge_PerClass_OneObjectPerClass()
        {
            _root = new GameObject("Buildings");
            var m1 = new Material(Shader.Find("Unlit/Color")) { color = Color.gray };
            Quad("roof_residential", _root.transform, m1, 0f);
            Quad("walls_residential", _root.transform, new Material(Shader.Find("Unlit/Texture")) { mainTexture = Texture2D.whiteTexture }, 1f);
            Quad("roof_industrial", _root.transform, m1, 2f);

            var res = DistrictMerger.Merge(_root, MergeUnit.PerClass);
            _merged = GameObject.Find("Buildings (merged)");
            Assert.AreEqual(2, res.mergedObjects, "residential + industrial => 2 groups");
        }

        [Test]
        public void ExportObj_WritesVerticesAndFaces()
        {
            _root = new GameObject("obj_root");
            Quad("q", _root.transform, new Material(Shader.Find("Unlit/Color")), 0f);
            var path = Path.Combine(Path.GetTempPath(), "dronecv_test.obj");
            SceneExporters.ExportObj(_root, path);

            var text = File.ReadAllText(path);
            StringAssert.Contains("mtllib", text);
            Assert.AreEqual(4, System.Text.RegularExpressions.Regex.Matches(text, @"(?m)^v ").Count);
            Assert.AreEqual(2, System.Text.RegularExpressions.Regex.Matches(text, @"(?m)^f ").Count);
            File.Delete(path);
            File.Delete(Path.ChangeExtension(path, ".mtl"));
        }
    }
}
