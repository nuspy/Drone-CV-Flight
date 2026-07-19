"""Integration tests: full protocol conversation against a live headless sim."""

import asyncio

import numpy as np
import pytest

from dronecv.config import load_config
from dronecv.protocol import messages as m
from dronecv.protocol.framing import ProtocolError
from dronecv.protocol.sim_client import SimClient
from dronecv.sim.headless.server import HeadlessSimServer

from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


@pytest.fixture()
def cfg():
    return load_config("headless_ci", root=ROOT)


def run(coro):
    return asyncio.run(coro)


async def _with_server(cfg, fn):
    server = HeadlessSimServer(cfg)
    host, port = await server.start(port=0)
    try:
        return await fn(host, port)
    finally:
        await server.stop()


def test_handshake_and_geo_meta(cfg):
    async def scenario(host, port):
        client = await SimClient.connect(host, port, role="capture")
        assert client.ack.sim_kind == "headless"
        assert client.ack.env_id == "headless_ci"
        assert client.ack.camera.width == 64
        # headless_ci embeds geo metadata by default
        assert client.geo_meta is not None
        assert client.geo_meta["lat0"] == pytest.approx(45.4642)
        info = await client.env_info()
        assert info.bounds_max_sim[0] == pytest.approx(cfg.world.size_m / 2)
        await client.close()

    run(_with_server(cfg, scenario))


def test_capture_returns_image_depth_and_sun(cfg):
    async def scenario(host, port):
        client = await SimClient.connect(host, port, role="capture")
        pos = np.array([0.0, 80.0, 0.0])
        result, blobs = await client.capture(pos, yaw_deg=45.0, pitch_deg=35.0)
        assert result.pos_sim == [0.0, 80.0, 0.0]
        assert blobs["rgb"].shape == (64, 64, 3)
        assert blobs["depth"].shape == (64, 64)
        # Camera tilted down from 80 m: most of the image must hit terrain.
        assert (blobs["depth"] > 0).mean() > 0.4
        # Sun angles present and daytime (config starts at 10:00 UTC June 21).
        assert result.sun_elevation_deg > 20
        assert 0 <= result.sun_azimuth_deg < 360
        await client.close()

    run(_with_server(cfg, scenario))


def test_determinism_same_pose_same_pixels(cfg):
    async def scenario(host, port):
        client = await SimClient.connect(host, port, role="capture")
        pos = np.array([30.0, 70.0, -40.0])
        _, blobs1 = await client.capture(pos, 120.0, 35.0)
        _, blobs2 = await client.capture(pos, 120.0, 35.0)
        np.testing.assert_array_equal(blobs1["rgb"], blobs2["rgb"])
        await client.close()

    run(_with_server(cfg, scenario))


def test_different_poses_different_pixels(cfg):
    async def scenario(host, port):
        client = await SimClient.connect(host, port, role="capture")
        _, b1 = await client.capture(np.array([0.0, 70.0, 0.0]), 0.0, 35.0)
        _, b2 = await client.capture(np.array([120.0, 70.0, 100.0]), 180.0, 35.0)
        assert np.abs(b1["rgb"].astype(int) - b2["rgb"].astype(int)).mean() > 3
        await client.close()

    run(_with_server(cfg, scenario))


def test_truth_channel_denied_to_localizer(cfg):
    async def scenario(host, port):
        client = await SimClient.connect(host, port, role="localizer")
        with pytest.raises(ProtocolError, match="truth"):
            await client.subscribe(["sensors", "truth"])
        await client.close()

    run(_with_server(cfg, scenario))


def test_streaming_flight_with_truth_for_harness(cfg):
    async def scenario(host, port):
        harness = await SimClient.connect(host, port, role="harness")
        await harness.reset(seed=3, start_pos_sim=np.array([0.0, 80.0, 0.0]))
        await harness.subscribe(["sensors", "truth"])
        await harness.send_command(vel_sim=np.array([5.0, 0.0, 0.0]))

        frames, truths = [], []
        async for msg, blobs in harness.stream():
            if isinstance(msg, m.SensorFrame):
                frames.append((msg, blobs))
            elif isinstance(msg, m.TruthState):
                truths.append(msg)
            if len(frames) >= 8 and len(truths) >= 8:
                break

        # Sensor frames carry image + lidar but no pose.
        frame, blobs = frames[-1]
        assert blobs["rgb"].shape == (64, 64, 3)
        assert not hasattr(frame, "pos_sim")
        # Truth shows eastward motion (commanded +x).
        assert truths[-1].pos_sim[0] > truths[0].pos_sim[0]
        # Lidar mostly valid at 80 m over low terrain.
        valid = [f.lidar_range_m for f, _ in frames if f.lidar_range_m is not None]
        assert len(valid) >= len(frames) // 2
        await harness.close()

    run(_with_server(cfg, scenario))


def test_reset_is_deterministic(cfg):
    async def scenario(host, port):
        harness = await SimClient.connect(host, port, role="harness")
        done1 = await harness.reset(seed=11)
        done2 = await harness.reset(seed=11)
        assert done1.pos_sim == done2.pos_sim
        done3 = await harness.reset(seed=12)
        assert done1.pos_sim != done3.pos_sim
        await harness.close()

    run(_with_server(cfg, scenario))


def test_teleport_requires_harness(cfg):
    async def scenario(host, port):
        loc = await SimClient.connect(host, port, role="localizer")
        await loc.teleport(np.array([10.0, 90.0, 10.0]))
        msg, _ = await loc.conn.recv()
        assert isinstance(msg, m.ErrorMsg)
        await loc.close()

    run(_with_server(cfg, scenario))
