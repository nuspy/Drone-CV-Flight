// The DroneCV simulation bridge: one MonoBehaviour that makes a Unity scene
// speak the dronecv wire protocol, indistinguishable from the headless sim.
//
// Scene setup (see Documentation~/setup.md):
//   - an imported 3D environment with colliders (terrain/meshes)
//   - a GameObject with SimLoop + GeoAnchorProvider + SunController
//   - a child DroneBody with a CaptureRig camera and a MockLidar
//   - press Play, then run:  dronecv protocol verify --host <ip> --port 7601
//
// Role enforcement matches the headless server exactly: the truth channel is
// only granted to clients that said role="harness" in their hello, so the
// localizer stays blind here too.

using System;
using System.Collections.Generic;
using DroneCV.Flight.Capture;
using DroneCV.Flight.Geo;
using DroneCV.Flight.Protocol;
using UnityEngine;

namespace DroneCV.Flight.Sim
{
    public class SimLoop : MonoBehaviour
    {
        [Header("Network")]
        public int Port = 7601;

        [Header("Identity")]
        public string EnvId = "unity_example";

        [Header("Camera")]
        public int ImageWidth = 224;
        public int ImageHeight = 224;
        public float FovDeg = 70f;
        public float CameraTiltDeg = 35f;
        public int JpegQuality = 92;

        [Header("Streaming")]
        public float SensorHz = 5f;

        [Header("Flyable volume (Unity local frame)")]
        public Vector3 BoundsMin = new Vector3(-250, 0, -250);
        public Vector3 BoundsMax = new Vector3(250, 150, 250);

        [Header("References")]
        public DroneBody Drone;
        public CaptureRig Rig;
        public MockLidar Lidar;
        public SunController Sun;
        public GeoAnchorProvider Anchor;

        private TcpBridge _bridge;
        private double _simTime;
        private long _frameId;
        private float _tickAccum;
        private System.Random _spawnRng = new System.Random(0);

        private static readonly List<string> Capabilities =
            new List<string> { "capture", "depth", "sun", "truth", "teleport", "reset" };

        private void Start()
        {
            if (Rig != null)
            {
                Rig.Cam.fieldOfView = VerticalFov();
                Rig.Cam.enabled = false; // rendered manually
            }
            _bridge = new TcpBridge();
            _bridge.Start(Port);
            Debug.Log($"[DroneCV] SimLoop listening on port {Port} (env '{EnvId}')");
        }

        private float VerticalFov()
        {
            // Config FOV is horizontal; Unity Camera.fieldOfView is vertical.
            float h = FovDeg * Mathf.Deg2Rad;
            return 2f * Mathf.Atan(Mathf.Tan(h / 2f) * ImageHeight / ImageWidth) * Mathf.Rad2Deg;
        }

        private void OnDestroy() => _bridge?.Dispose();

        // ------------------------------------------------------------- update

        private void Update()
        {
            while (_bridge.Inbound.TryDequeue(out var inbound))
                Dispatch(inbound.Client, inbound.Message);

            bool anySensor = false, anyTruth = false;
            foreach (var c in _bridge.Clients)
            {
                if (c.Channels.Contains("sensors")) anySensor = true;
                if (c.Channels.Contains("truth")) anyTruth = true;
            }
            if (!(anySensor || anyTruth)) return;

            _tickAccum += Time.deltaTime;
            var dt = 1f / SensorHz;
            if (_tickAccum < dt) return;
            _tickAccum -= dt;

            Drone.Step(dt);
            _simTime += dt;
            _frameId += 1;
            Sun?.Apply(_simTime);
            Tick(anySensor);
        }

        // ----------------------------------------------------------- dispatch

        private void Dispatch(BridgeClient client, DecodedMessage msg)
        {
            if (!client.HelloDone)
            {
                if (msg.Type != "hello")
                {
                    _bridge.Send(client, new ErrorMsg { Message = "expected hello" });
                    return;
                }
                var hello = msg.As<Hello>();
                var major = ProtocolConstants.ProtocolVersion.Split('.')[0];
                if (hello.Version.Split('.')[0] != major)
                {
                    _bridge.Send(client, new ErrorMsg
                    {
                        Message = $"protocol major version mismatch: bridge {ProtocolConstants.ProtocolVersion}, client {hello.Version}",
                    });
                    return;
                }
                client.Role = hello.Role;
                client.HelloDone = true;
                _bridge.Send(client, new HelloAck
                {
                    EnvId = EnvId,
                    Capabilities = Capabilities,
                    GeoMeta = Anchor != null ? Anchor.ResolveGeoMeta() : null,
                    Camera = new CameraInfo
                    {
                        Width = ImageWidth, Height = ImageHeight, FovDeg = FovDeg, TiltDeg = CameraTiltDeg,
                    },
                    SimTime = _simTime,
                });
                return;
            }

            switch (msg.Type)
            {
                case "ping":
                    _bridge.Send(client, new Pong { SimTime = _simTime });
                    break;
                case "env_info_request":
                    _bridge.Send(client, new EnvInfo
                    {
                        BoundsMinSim = ToArr(BoundsMin),
                        BoundsMaxSim = ToArr(BoundsMax),
                        GroundAltMinM = 0,
                        GroundAltMaxM = BoundsMax.y * 0.5,
                        SimTime = _simTime,
                    });
                    break;
                case "subscribe":
                {
                    var sub = msg.As<Subscribe>();
                    var granted = new List<string>();
                    foreach (var ch in sub.Channels)
                    {
                        if (ch == "truth" && client.Role != "harness")
                        {
                            Debug.LogWarning($"[DroneCV] denied truth subscription to role={client.Role}");
                            continue;
                        }
                        granted.Add(ch);
                    }
                    client.Channels.Clear();
                    foreach (var g in granted) client.Channels.Add(g);
                    _bridge.Send(client, new SubscribeAck { Channels = granted, SimTime = _simTime });
                    break;
                }
                case "capture_request":
                    HandleCapture(client, msg.As<CaptureRequest>());
                    break;
                case "command":
                {
                    var cmd = msg.As<Command>();
                    Drone.SetCommand(
                        cmd.VelSim != null ? ToVec(cmd.VelSim) : (Vector3?)null,
                        cmd.YawRateDps.HasValue ? (float?)cmd.YawRateDps.Value : null);
                    break;
                }
                case "reset":
                    HandleReset(client, msg.As<Reset>());
                    break;
                case "teleport":
                {
                    if (client.Role != "harness")
                    {
                        _bridge.Send(client, new ErrorMsg { Message = "teleport requires role=harness" });
                        break;
                    }
                    var tp = msg.As<Teleport>();
                    Drone.ResetTo(ToVec(tp.PosSim), (float)tp.YawDeg);
                    break;
                }
                default:
                    _bridge.Send(client, new ErrorMsg { Message = $"unsupported message type '{msg.Type}'" });
                    break;
            }
        }

        // ------------------------------------------------------------ capture

        private void HandleCapture(BridgeClient client, CaptureRequest req)
        {
            var pos = ToVec(req.PosSim);
            Rig.SetPose(pos, (float)req.YawDeg, (float)req.PitchDeg);
            var blobs = new List<Blob>();
            if (req.Want.Contains("rgb"))
            {
                var jpg = Rig.RenderRgb(ImageWidth, ImageHeight, "jpeg", JpegQuality);
                blobs.Add(MessageCodec.EncodedImageBlob("rgb", jpg, "jpeg", ImageHeight, ImageWidth));
            }
            if (req.Want.Contains("depth"))
            {
                var depth = Rig.RenderDepth(ImageWidth, ImageHeight);
                blobs.Add(MessageCodec.RawFloat32Blob("depth", depth, new[] { ImageHeight, ImageWidth }));
            }

            var result = new CaptureResult
            {
                PosSim = ToArr(pos),
                QuatSim = YawQuat((float)req.YawDeg),
                YawDeg = req.YawDeg,
                PitchDeg = req.PitchDeg,
                Utc = Sun != null ? Sun.UtcAt(_simTime).ToString("o") : DateTime.UtcNow.ToString("o"),
                SimTime = _simTime,
            };
            if (req.Want.Contains("sun") && Sun != null)
            {
                var sun = Sun.Sun(_simTime);
                var moon = Sun.Moon(_simTime);
                result.SunAzimuthDeg = sun.AzimuthDeg;
                result.SunElevationDeg = sun.ElevationDeg;
                result.MoonAzimuthDeg = moon.AzimuthDeg;
                result.MoonElevationDeg = moon.ElevationDeg;
            }
            _bridge.Send(client, result, blobs);
        }

        // -------------------------------------------------------------- reset

        private void HandleReset(BridgeClient client, Reset reset)
        {
            _spawnRng = new System.Random(reset.Seed);
            Lidar?.Reseed(reset.Seed);
            if (!string.IsNullOrEmpty(reset.Utc)) Sun?.OverrideStartUtc(reset.Utc);

            Vector3 pos;
            if (reset.StartPosSim != null)
            {
                pos = ToVec(reset.StartPosSim);
            }
            else
            {
                var x = Mathf.Lerp(BoundsMin.x * 0.7f, BoundsMax.x * 0.7f, (float)_spawnRng.NextDouble());
                var z = Mathf.Lerp(BoundsMin.z * 0.7f, BoundsMax.z * 0.7f, (float)_spawnRng.NextDouble());
                pos = new Vector3(x, 0f, z);
                Drone.transform.position = pos + Vector3.up * 500f;
                var ground = Drone.GroundHeightBelow() ?? 0f;
                pos.y = ground + 50f;
            }
            Drone.ResetTo(pos, (float)reset.StartYawDeg);
            _simTime = 0;
            _frameId = 0;
            Sun?.Apply(0);
            _bridge.Send(client, new ResetDone { PosSim = ToArr(pos), SimTime = 0 });
        }

        // --------------------------------------------------------------- tick

        private void Tick(bool renderSensor)
        {
            byte[] jpg = null;
            double? lidarRange = null;
            if (renderSensor)
            {
                Rig.SetPose(Drone.transform.position, Drone.YawDeg, CameraTiltDeg);
                jpg = Rig.RenderRgb(ImageWidth, ImageHeight, "jpeg", JpegQuality);
                lidarRange = Lidar != null ? Lidar.RangeDown() : null;
            }
            foreach (var client in _bridge.Clients)
            {
                if (jpg != null && client.Channels.Contains("sensors"))
                {
                    var frame = new SensorFrame
                    {
                        FrameId = _frameId,
                        Utc = Sun != null ? Sun.UtcAt(_simTime).ToString("o") : DateTime.UtcNow.ToString("o"),
                        LidarRangeM = lidarRange,
                        SimTime = _simTime,
                    };
                    _bridge.Send(client, frame, new List<Blob>
                    {
                        MessageCodec.EncodedImageBlob("rgb", jpg, "jpeg", ImageHeight, ImageWidth),
                    });
                }
                if (client.Channels.Contains("truth"))
                {
                    _bridge.Send(client, new TruthState
                    {
                        FrameId = _frameId,
                        PosSim = ToArr(Drone.transform.position),
                        QuatSim = YawQuat(Drone.YawDeg),
                        VelSim = ToArr(Drone.Velocity),
                        Collided = Drone.Collided,
                        SimTime = _simTime,
                    });
                }
            }
        }

        // ------------------------------------------------------------ helpers

        private static double[] ToArr(Vector3 v) => new double[] { v.x, v.y, v.z };
        private static Vector3 ToVec(double[] a) => new Vector3((float)a[0], (float)a[1], (float)a[2]);

        private static double[] YawQuat(float yawDeg)
        {
            var half = yawDeg * Mathf.Deg2Rad / 2f;
            return new double[] { 0.0, Math.Sin(half), 0.0, Math.Cos(half) };
        }

        private void OnDrawGizmosSelected()
        {
            Gizmos.color = new Color(0.1f, 0.6f, 0.9f, 0.35f);
            var center = (BoundsMin + BoundsMax) / 2f;
            Gizmos.DrawWireCube(center, BoundsMax - BoundsMin);
        }
    }
}
