// One-click scene setup: GameObject > DroneCV > Create Sim Rig builds the
// full bridge hierarchy with sensible defaults; the user only assigns the
// sun light and (optionally) a GeoAnchorAsset, then presses Play.

using DroneCV.Flight.Capture;
using DroneCV.Flight.Geo;
using DroneCV.Flight.Sim;
using UnityEditor;
using UnityEngine;

namespace DroneCV.Flight.Editor
{
    public static class CreateSimRigMenu
    {
        [MenuItem("GameObject/DroneCV/Create Sim Rig", false, 10)]
        public static void CreateSimRig()
        {
            var root = new GameObject("DroneCV Sim");
            var anchor = root.AddComponent<GeoAnchorProvider>();
            var sun = root.AddComponent<SunController>();
            sun.AnchorProvider = anchor;
            var loop = root.AddComponent<SimLoop>();

            var drone = new GameObject("Drone");
            drone.transform.SetParent(root.transform);
            drone.transform.position = new Vector3(0, 60, 0);
            var body = drone.AddComponent<DroneBody>();
            var lidar = drone.AddComponent<MockLidar>();

            var camGo = new GameObject("CaptureRig");
            camGo.transform.SetParent(drone.transform, false);
            var cam = camGo.AddComponent<Camera>();
            cam.nearClipPlane = 0.3f;
            cam.farClipPlane = 3000f;
            var rig = camGo.AddComponent<CaptureRig>();
            rig.DepthShader = Shader.Find("DroneCV/DepthCapture");

            loop.Drone = body;
            loop.Rig = rig;
            loop.Lidar = lidar;
            loop.Sun = sun;
            loop.Anchor = anchor;

            var light = Object.FindFirstObjectByType<Light>();
            if (light != null && light.type == LightType.Directional)
                sun.SunLight = light;
            else
                Debug.LogWarning("[DroneCV] assign a directional light to SunController.SunLight");

            Selection.activeGameObject = root;
            Undo.RegisterCreatedObjectUndo(root, "Create DroneCV Sim Rig");
            Debug.Log("[DroneCV] Sim rig created. Set SimLoop bounds to the flyable volume, " +
                      "check the environment has colliders, press Play, then run " +
                      "`dronecv protocol verify --host <this machine> --port 7601`.");
        }
    }
}
