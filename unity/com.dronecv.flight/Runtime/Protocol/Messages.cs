// dronecv wire protocol v1.0 message headers.
// Mirror of src/dronecv/protocol/messages.py — keep the two in sync and bump
// ProtocolVersion together. Field names must match the JSON exactly
// (snake_case), hence the JsonProperty annotations.

using System.Collections.Generic;
using Newtonsoft.Json;

namespace DroneCV.Flight.Protocol
{
    public static class ProtocolConstants
    {
        public const string ProtocolVersion = "1.0";
        public const int MaxHeaderLen = 1 << 20;
        public const long MaxBlobLen = 1L << 28;
    }

    public class BlobSpec
    {
        [JsonProperty("name")] public string Name;
        [JsonProperty("dtype")] public string Dtype = "bytes";        // uint8 | float32 | bytes
        [JsonProperty("shape")] public int[] Shape;                    // null for encoded streams
        [JsonProperty("encoding")] public string Encoding = "raw";     // raw | jpeg | png
        [JsonProperty("byte_len")] public long ByteLen;
    }

    public class Header
    {
        [JsonProperty("type")] public string Type;
        [JsonProperty("version")] public string Version = ProtocolConstants.ProtocolVersion;
        [JsonProperty("msg_id")] public long MsgId;
        [JsonProperty("sim_time")] public double SimTime;
        [JsonProperty("blobs")] public List<BlobSpec> Blobs = new List<BlobSpec>();
    }

    public class Hello : Header
    {
        public Hello() { Type = "hello"; }
        [JsonProperty("role")] public string Role = "localizer";
    }

    public class CameraInfo
    {
        [JsonProperty("width")] public int Width;
        [JsonProperty("height")] public int Height;
        [JsonProperty("fov_deg")] public float FovDeg;
        [JsonProperty("tilt_deg")] public float TiltDeg;
    }

    public class HelloAck : Header
    {
        public HelloAck() { Type = "hello_ack"; }
        [JsonProperty("sim_kind")] public string SimKind = "unity";
        [JsonProperty("env_id")] public string EnvId;
        [JsonProperty("capabilities")] public List<string> Capabilities = new List<string>();
        [JsonProperty("geo_meta", NullValueHandling = NullValueHandling.Include)]
        public Dictionary<string, double> GeoMeta;
        [JsonProperty("camera")] public CameraInfo Camera;
    }

    public class EnvInfoRequest : Header
    {
        public EnvInfoRequest() { Type = "env_info_request"; }
    }

    public class EnvInfo : Header
    {
        public EnvInfo() { Type = "env_info"; }
        [JsonProperty("bounds_min_sim")] public double[] BoundsMinSim;
        [JsonProperty("bounds_max_sim")] public double[] BoundsMaxSim;
        [JsonProperty("ground_alt_min_m")] public double GroundAltMinM;
        [JsonProperty("ground_alt_max_m")] public double GroundAltMaxM;
    }

    public class Subscribe : Header
    {
        public Subscribe() { Type = "subscribe"; }
        [JsonProperty("channels")] public List<string> Channels = new List<string>();
    }

    public class SubscribeAck : Header
    {
        public SubscribeAck() { Type = "subscribe_ack"; }
        [JsonProperty("channels")] public List<string> Channels = new List<string>();
    }

    public class CaptureRequest : Header
    {
        public CaptureRequest() { Type = "capture_request"; }
        [JsonProperty("pos_sim")] public double[] PosSim;
        [JsonProperty("yaw_deg")] public double YawDeg;
        [JsonProperty("pitch_deg")] public double PitchDeg;
        [JsonProperty("want")] public List<string> Want = new List<string> { "rgb" };
    }

    public class CaptureResult : Header
    {
        public CaptureResult() { Type = "capture_result"; }
        [JsonProperty("pos_sim")] public double[] PosSim;
        [JsonProperty("quat_sim")] public double[] QuatSim;
        [JsonProperty("yaw_deg")] public double YawDeg;
        [JsonProperty("pitch_deg")] public double PitchDeg;
        [JsonProperty("utc")] public string Utc;
        [JsonProperty("sun_azimuth_deg", NullValueHandling = NullValueHandling.Include)]
        public double? SunAzimuthDeg;
        [JsonProperty("sun_elevation_deg", NullValueHandling = NullValueHandling.Include)]
        public double? SunElevationDeg;
        [JsonProperty("moon_azimuth_deg", NullValueHandling = NullValueHandling.Include)]
        public double? MoonAzimuthDeg;
        [JsonProperty("moon_elevation_deg", NullValueHandling = NullValueHandling.Include)]
        public double? MoonElevationDeg;
    }

    public class SensorFrame : Header
    {
        public SensorFrame() { Type = "sensor_frame"; }
        [JsonProperty("frame_id")] public long FrameId;
        [JsonProperty("utc")] public string Utc;
        [JsonProperty("lidar_range_m", NullValueHandling = NullValueHandling.Include)]
        public double? LidarRangeM;
        // Deliberately NO pose fields: the localizer flies blind.
    }

    public class Command : Header
    {
        public Command() { Type = "command"; }
        [JsonProperty("vel_sim", NullValueHandling = NullValueHandling.Include)]
        public double[] VelSim;
        [JsonProperty("yaw_rate_dps", NullValueHandling = NullValueHandling.Include)]
        public double? YawRateDps;
    }

    public class TruthState : Header
    {
        public TruthState() { Type = "truth_state"; }
        [JsonProperty("frame_id")] public long FrameId;
        [JsonProperty("pos_sim")] public double[] PosSim;
        [JsonProperty("quat_sim")] public double[] QuatSim;
        [JsonProperty("vel_sim")] public double[] VelSim;
        [JsonProperty("collided")] public bool Collided;
    }

    public class Reset : Header
    {
        public Reset() { Type = "reset"; }
        [JsonProperty("seed")] public int Seed;
        [JsonProperty("start_pos_sim", NullValueHandling = NullValueHandling.Include)]
        public double[] StartPosSim;
        [JsonProperty("start_yaw_deg")] public double StartYawDeg;
        [JsonProperty("utc", NullValueHandling = NullValueHandling.Include)]
        public string Utc;
    }

    public class ResetDone : Header
    {
        public ResetDone() { Type = "reset_done"; }
        [JsonProperty("pos_sim")] public double[] PosSim;
    }

    public class Teleport : Header
    {
        public Teleport() { Type = "teleport"; }
        [JsonProperty("pos_sim")] public double[] PosSim;
        [JsonProperty("yaw_deg")] public double YawDeg;
    }

    public class ErrorMsg : Header
    {
        public ErrorMsg() { Type = "error"; }
        [JsonProperty("message")] public string Message;
    }

    public class Ping : Header { public Ping() { Type = "ping"; } }
    public class Pong : Header { public Pong() { Type = "pong"; } }
}
