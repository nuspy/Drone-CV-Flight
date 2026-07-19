import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from dronecv.protocol import messages as m
from dronecv.protocol.framing import decode_message, encode_message

FIXTURES = Path(__file__).parent.parent / "fixtures" / "protocol"


def _all_sample_messages() -> dict[str, tuple[m.AnyMessage, dict]]:
    """One representative instance of every message type (used for round-trip
    tests AND to generate the golden fixture files the C# codec validates
    against — see scripts/gen_protocol_fixtures.py)."""
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[1, 2] = [255, 10, 0]
    depth = np.linspace(0, 50, 24, dtype=np.float32).reshape(4, 6)
    return {
        "hello": (m.Hello(role="harness"), {}),
        "hello_ack": (
            m.HelloAck(
                sim_kind="headless",
                env_id="headless_ci",
                capabilities=["capture", "truth", "teleport"],
                geo_meta={"lat0": 45.4642, "lon0": 9.19, "alt0": 120.0, "true_north_offset_deg": 12.5},
                camera=m.CameraInfo(width=64, height=64, fov_deg=70.0, tilt_deg=35.0),
            ),
            {},
        ),
        "env_info_request": (m.EnvInfoRequest(), {}),
        "env_info": (
            m.EnvInfo(
                bounds_min_sim=[-250.0, 0.0, -250.0],
                bounds_max_sim=[250.0, 150.0, 250.0],
                ground_alt_min_m=0.0,
                ground_alt_max_m=42.0,
            ),
            {},
        ),
        "subscribe": (m.Subscribe(channels=["sensors", "truth"]), {}),
        "subscribe_ack": (m.SubscribeAck(channels=["sensors"]), {}),
        "capture_request": (
            m.CaptureRequest(pos_sim=[10.0, 40.0, -20.0], yaw_deg=135.0, pitch_deg=35.0, want=["rgb", "depth", "sun"]),
            {},
        ),
        "capture_result": (
            m.CaptureResult(
                pos_sim=[10.0, 40.0, -20.0],
                quat_sim=[0.0, 0.3826834323650898, 0.0, 0.9238795325112867],
                yaw_deg=45.0,
                pitch_deg=35.0,
                utc="2026-06-21T10:00:00+00:00",
                sun_azimuth_deg=123.4,
                sun_elevation_deg=55.1,
            ),
            {"rgb": (rgb, "png"), "depth": (depth, "raw")},
        ),
        "sensor_frame": (
            m.SensorFrame(frame_id=42, utc="2026-06-21T10:00:05+00:00", lidar_range_m=38.75, sim_time=5.0),
            {"rgb": (rgb, "png")},
        ),
        "command": (m.Command(vel_sim=[1.0, 0.0, 3.5], yaw_rate_dps=-15.0), {}),
        "truth_state": (
            m.TruthState(
                frame_id=42,
                pos_sim=[11.0, 40.0, -18.0],
                quat_sim=[0.0, 0.0, 0.0, 1.0],
                vel_sim=[1.0, 0.0, 3.5],
                collided=False,
            ),
            {},
        ),
        "reset": (m.Reset(seed=7, start_pos_sim=[0.0, 50.0, 0.0], start_yaw_deg=90.0, utc="2026-06-21T10:00:00+00:00"), {}),
        "reset_done": (m.ResetDone(pos_sim=[0.0, 50.0, 0.0]), {}),
        "teleport": (m.Teleport(pos_sim=[100.0, 60.0, 100.0], yaw_deg=180.0), {}),
        "error": (m.ErrorMsg(message="truth channel requires role=harness"), {}),
        "ping": (m.Ping(), {}),
        "pong": (m.Pong(), {}),
    }


@pytest.mark.parametrize("name", sorted(_all_sample_messages()))
def test_round_trip_every_message(name):
    msg, blobs = _all_sample_messages()[name]
    data = encode_message(msg, blobs)
    decoded, decoded_blobs = decode_message(data)
    assert decoded.type == msg.type
    # Header fields survive (blobs list is filled by the codec).
    original = msg.model_dump(exclude={"blobs"})
    got = decoded.model_dump(exclude={"blobs"})
    assert got == original
    for blob_name, (value, encoding) in blobs.items():
        out = decoded_blobs[blob_name]
        if encoding in ("raw", "png"):  # lossless
            np.testing.assert_array_equal(out, value)
        else:  # jpeg: lossy, just check shape
            assert out.shape == value.shape


def test_jpeg_blob_lossy_but_close():
    img = (np.random.default_rng(0).uniform(0, 255, (32, 32, 3))).astype(np.uint8)
    smooth = np.zeros_like(img)
    smooth[:, :16] = [200, 30, 30]
    smooth[:, 16:] = [30, 30, 200]
    msg = m.SensorFrame(frame_id=1, utc="2026-01-01T00:00:00+00:00")
    _, blobs = decode_message(encode_message(msg, {"rgb": (smooth, "jpeg")}))
    assert np.abs(blobs["rgb"].astype(int) - smooth.astype(int)).mean() < 8


def test_unknown_message_type_rejected():
    import struct

    header = json.dumps({"type": "evil", "version": "1.0", "msg_id": 0, "sim_time": 0, "blobs": []}).encode()
    with pytest.raises(Exception):
        decode_message(struct.pack(">I", len(header)) + header)


def test_golden_fixtures_stable():
    """The C# MessageCodec is validated against these exact bytes. If this
    test fails, the protocol changed: bump PROTOCOL_VERSION and regenerate
    with scripts/gen_protocol_fixtures.py."""
    if not FIXTURES.exists():
        pytest.skip("fixtures not generated yet")
    samples = _all_sample_messages()
    for name, (msg, blobs) in samples.items():
        path = FIXTURES / f"{name}.bin"
        assert path.exists(), f"missing golden fixture {name}.bin — regenerate"
        golden = path.read_bytes()
        decoded, decoded_blobs = decode_message(golden)
        assert decoded.type == msg.type
        assert decoded.model_dump(exclude={"blobs"}) == msg.model_dump(exclude={"blobs"})


def test_transport_round_trip_over_tcp():
    from dronecv.protocol.transport import Connection, connect

    async def scenario():
        received = []

        async def handle(reader, writer):
            conn = Connection(reader, writer)
            msg, blobs = await conn.recv()
            received.append((msg, blobs))
            await conn.send(m.Pong())
            await conn.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = await connect("127.0.0.1", port)
        depth = np.arange(12, dtype=np.float32).reshape(3, 4)
        await client.send(m.CaptureResult(
            pos_sim=[1, 2, 3], quat_sim=[0, 0, 0, 1], yaw_deg=0, pitch_deg=0,
            utc="2026-01-01T00:00:00+00:00",
        ), {"depth": (depth, "raw")})
        reply, _ = await client.recv()
        assert isinstance(reply, m.Pong)
        await client.close()
        server.close()
        await server.wait_closed()

        msg, blobs = received[0]
        assert isinstance(msg, m.CaptureResult)
        np.testing.assert_array_equal(blobs["depth"], depth)

    asyncio.run(scenario())
