// EditMode tests: the C# codec must decode every golden frame produced by
// the Python encoder (scripts/gen_protocol_fixtures.py), and its own
// encoding must round-trip through its decoder.

using System.Collections.Generic;
using System.IO;
using DroneCV.Flight.Protocol;
using NUnit.Framework;

namespace DroneCV.Flight.Tests
{
    public class CodecGoldenTests
    {
        private static IEnumerable<string> GoldenFiles()
        {
            var dir = Path.Combine(FixturePaths.Root, "protocol");
            Assert.That(Directory.Exists(dir), $"missing {dir} — run scripts/gen_protocol_fixtures.py");
            return Directory.GetFiles(dir, "*.bin");
        }

        [Test]
        public void DecodesEveryPythonGoldenFrame()
        {
            var count = 0;
            foreach (var file in GoldenFiles())
            {
                var decoded = MessageCodec.DecodeBytes(File.ReadAllBytes(file));
                var expectedType = Path.GetFileNameWithoutExtension(file);
                Assert.AreEqual(expectedType, decoded.Type, file);
                foreach (var blob in decoded.Blobs.Values)
                    Assert.AreEqual(blob.Spec.ByteLen, blob.Data.LongLength, file);
                count++;
            }
            Assert.Greater(count, 10, "expected a full set of golden frames");
        }

        [Test]
        public void CaptureResultGoldenHasImageAndDepth()
        {
            var path = Path.Combine(FixturePaths.Root, "protocol", "capture_result.bin");
            var decoded = MessageCodec.DecodeBytes(File.ReadAllBytes(path));
            var result = decoded.As<CaptureResult>();
            Assert.AreEqual("2026-06-21T10:00:00+00:00", result.Utc);
            Assert.NotNull(result.SunAzimuthDeg);
            Assert.That(decoded.Blobs.ContainsKey("rgb"));
            Assert.That(decoded.Blobs.ContainsKey("depth"));
            var depth = decoded.Blobs["depth"];
            Assert.AreEqual("float32", depth.Spec.Dtype);
            Assert.AreEqual(new[] { 4, 6 }, depth.Spec.Shape);
        }

        [Test]
        public void EncodeDecodeRoundTrip()
        {
            var frame = new SensorFrame
            {
                FrameId = 42,
                Utc = "2026-06-21T10:00:05+00:00",
                LidarRangeM = 38.75,
                SimTime = 5.0,
            };
            var depth = MessageCodec.RawFloat32Blob("depth", new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
            var bytes = MessageCodec.Encode(frame, new List<Blob> { depth });
            var decoded = MessageCodec.DecodeBytes(bytes);
            Assert.AreEqual("sensor_frame", decoded.Type);
            var round = decoded.As<SensorFrame>();
            Assert.AreEqual(42, round.FrameId);
            Assert.AreEqual(38.75, round.LidarRangeM.Value, 1e-9);
            Assert.AreEqual(16, decoded.Blobs["depth"].Data.Length);
        }

        [Test]
        public void TruthDeniedMessageRoundTrip()
        {
            var err = new ErrorMsg { Message = "truth channel requires role=harness" };
            var decoded = MessageCodec.DecodeBytes(MessageCodec.Encode(err));
            Assert.AreEqual("error", decoded.Type);
            Assert.AreEqual(err.Message, decoded.As<ErrorMsg>().Message);
        }
    }
}
