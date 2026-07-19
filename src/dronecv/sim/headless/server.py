"""Headless simulator server — speaks the full dronecv wire protocol.

One shared sim state (world, drone, clock) per server; any number of clients.
The `truth` channel is only granted to clients that declared role="harness"
in their hello, which is how the test system guarantees the localizer never
receives ground truth.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np

from dronecv.config import Config
from dronecv.geo.anchor import GeoAnchor
from dronecv.protocol import messages as m
from dronecv.protocol.transport import Connection
from dronecv.sim.headless.drone import DroneBody
from dronecv.sim.headless.sensors import Camera, CelestialModel, MockLidar
from dronecv.sim.headless.world import World
from dronecv.util.logging import get_logger
from dronecv.util.seeding import rng as make_rng

log = get_logger("dronecv.sim")

CAPABILITIES = ["capture", "depth", "sun", "truth", "teleport", "reset"]


class _Client:
    def __init__(self, conn: Connection):
        self.conn = conn
        self.role: m.Role = "localizer"
        self.channels: set[str] = set()


class HeadlessSimServer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.anchor = GeoAnchor.resolve(cfg.env.anchor, None)
        self.world = World(cfg.world, cfg.env.seed)
        self.drone = DroneBody(cfg.sim.drone)
        self._start_utc = datetime.fromisoformat(cfg.env.start_utc)
        self.sky = CelestialModel(self.anchor, self._start_utc)
        self.camera = Camera(cfg.sim, self.world, self.sky)
        self._sensor_gen = make_rng(cfg.env.seed * 31 + 7)
        self.lidar = MockLidar(cfg.sim.lidar, self.world, self._sensor_gen)
        self.sim_time = 0.0
        self.frame_id = 0
        self.clients: list[_Client] = []
        self._server: asyncio.Server | None = None
        self._tick_task: asyncio.Task | None = None
        self.drone.reset(self.world.spawn_position(make_rng(cfg.env.seed), agl_m=50.0))

    # ------------------------------------------------------------------ server

    async def start(self, host: str | None = None, port: int | None = None) -> tuple[str, int]:
        host = host or self.cfg.sim.host
        port = self.cfg.sim.port if port is None else port
        self._server = await asyncio.start_server(self._handle_client, host, port)
        self._tick_task = asyncio.create_task(self._tick_loop())
        sock = self._server.sockets[0].getsockname()
        log.info(f"headless sim '{self.cfg.env.name}' listening on {sock[0]}:{sock[1]}")
        return sock[0], sock[1]

    async def serve_forever(self) -> None:
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._tick_task:
            self._tick_task.cancel()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    # ----------------------------------------------------------------- clients

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = Connection(reader, writer)
        client = _Client(conn)
        try:
            hello, _ = await conn.recv()
            if not isinstance(hello, m.Hello):
                await conn.send(m.ErrorMsg(message="expected hello"))
                return
            client.role = hello.role
            self.clients.append(client)
            geo_meta = None
            if self.cfg.world.embed_geo_meta:
                geo_meta = {
                    "lat0": self.anchor.lat0,
                    "lon0": self.anchor.lon0,
                    "alt0": self.anchor.alt0,
                    "true_north_offset_deg": self.anchor.true_north_offset_deg,
                }
            await conn.send(
                m.HelloAck(
                    sim_kind="headless",
                    env_id=self.cfg.env.name or "headless",
                    capabilities=CAPABILITIES,
                    geo_meta=geo_meta,
                    camera=m.CameraInfo(
                        width=self.cfg.sim.image_width,
                        height=self.cfg.sim.image_height,
                        fov_deg=self.cfg.sim.fov_deg,
                        tilt_deg=self.cfg.sim.camera_tilt_deg,
                    ),
                    sim_time=self.sim_time,
                )
            )
            while True:
                msg, _blobs = await conn.recv()
                await self._dispatch(client, msg)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            if client in self.clients:
                self.clients.remove(client)
            await conn.close()

    async def _dispatch(self, client: _Client, msg: m.AnyMessage) -> None:
        conn = client.conn
        if isinstance(msg, m.Ping):
            await conn.send(m.Pong(sim_time=self.sim_time))
        elif isinstance(msg, m.EnvInfoRequest):
            await conn.send(
                m.EnvInfo(
                    bounds_min_sim=[float(v) for v in self.world.bounds_min],
                    bounds_max_sim=[float(v) for v in self.world.bounds_max],
                    ground_alt_min_m=float(self.world.heightmap.min()),
                    ground_alt_max_m=float(self.world.heightmap.max()),
                    sim_time=self.sim_time,
                )
            )
        elif isinstance(msg, m.Subscribe):
            granted: list[str] = []
            for ch in msg.channels:
                if ch == "truth" and client.role != "harness":
                    log.warning(f"{conn.peer} (role={client.role}) denied truth subscription")
                    continue
                granted.append(ch)
            client.channels = set(granted)
            await conn.send(m.SubscribeAck(channels=granted, sim_time=self.sim_time))
        elif isinstance(msg, m.CaptureRequest):
            await self._capture(client, msg)
        elif isinstance(msg, m.Command):
            self.drone.set_command(
                None if msg.vel_sim is None else np.array(msg.vel_sim), msg.yaw_rate_dps
            )
        elif isinstance(msg, m.Reset):
            await self._reset(client, msg)
        elif isinstance(msg, m.Teleport):
            if client.role != "harness":
                await conn.send(m.ErrorMsg(message="teleport requires role=harness"))
                return
            self.drone.state.pos_sim = np.array(msg.pos_sim, dtype=float)
            self.drone.state.yaw_deg = msg.yaw_deg
            self.drone.state.vel_sim = np.zeros(3)
        else:
            await conn.send(m.ErrorMsg(message=f"unsupported message type '{msg.type}'"))

    async def _capture(self, client: _Client, req: m.CaptureRequest) -> None:
        pos = np.array(req.pos_sim, dtype=float)
        rgb, depth, sun, moon = self.camera.render(pos, req.yaw_deg, req.pitch_deg, self.sim_time)
        blobs: dict = {}
        if "rgb" in req.want:
            fmt = self.cfg.capture.image_format
            blobs["rgb"] = (rgb, fmt)
        if "depth" in req.want:
            blobs["depth"] = (depth, "raw")
        # Body->sim quaternion for a yaw-then-pitch camera pose.
        yaw_r = np.radians(req.yaw_deg)
        quat = [0.0, float(np.sin(yaw_r / 2)), 0.0, float(np.cos(yaw_r / 2))]
        result = m.CaptureResult(
            pos_sim=[float(v) for v in pos],
            quat_sim=quat,
            yaw_deg=req.yaw_deg,
            pitch_deg=req.pitch_deg,
            utc=self.sky.utc_at(self.sim_time).isoformat(),
            sun_azimuth_deg=sun.azimuth_deg if "sun" in req.want else None,
            sun_elevation_deg=sun.elevation_deg if "sun" in req.want else None,
            moon_azimuth_deg=moon.azimuth_deg if "sun" in req.want else None,
            moon_elevation_deg=moon.elevation_deg if "sun" in req.want else None,
            sim_time=self.sim_time,
        )
        await client.conn.send(result, blobs, jpeg_quality=self.cfg.capture.jpeg_quality)

    async def _reset(self, client: _Client, msg: m.Reset) -> None:
        gen = make_rng(msg.seed)
        self._sensor_gen = make_rng(msg.seed * 31 + 7)
        self.lidar = MockLidar(self.cfg.sim.lidar, self.world, self._sensor_gen)
        if msg.utc:
            self.sky = CelestialModel(self.anchor, datetime.fromisoformat(msg.utc))
            self.camera = Camera(self.cfg.sim, self.world, self.sky)
        pos = (
            np.array(msg.start_pos_sim, dtype=float)
            if msg.start_pos_sim is not None
            else self.world.spawn_position(gen, agl_m=50.0)
        )
        self.drone.reset(pos, msg.start_yaw_deg)
        self.sim_time = 0.0
        self.frame_id = 0
        await client.conn.send(m.ResetDone(pos_sim=[float(v) for v in pos], sim_time=0.0))

    # -------------------------------------------------------------------- tick

    async def _tick_loop(self) -> None:
        dt = 1.0 / self.cfg.sim.sensor_hz
        substeps = 4
        while True:
            has_sensor = any("sensors" in c.channels for c in self.clients)
            has_truth = any("truth" in c.channels for c in self.clients)
            if not (has_sensor or has_truth):
                await asyncio.sleep(0.02)
                continue

            for _ in range(substeps):
                self.drone.step(dt / substeps, lambda x, z: self.world.height_at(np.array(x), np.array(z)))
            self.sim_time += dt
            self.frame_id += 1
            state = self.drone.state

            frame_msg = None
            frame_blobs = None
            if has_sensor:
                rgb, _depth, _sun, _moon = self.camera.render(
                    state.pos_sim, state.yaw_deg, None, self.sim_time
                )
                rng_range = self.lidar.range_down(state.pos_sim)
                frame_msg = dict(
                    frame_id=self.frame_id,
                    utc=self.sky.utc_at(self.sim_time).isoformat(),
                    lidar_range_m=rng_range,
                    sim_time=self.sim_time,
                )
                frame_blobs = {"rgb": (rgb, "jpeg")}

            for client in list(self.clients):
                try:
                    if frame_msg is not None and "sensors" in client.channels:
                        await client.conn.send(m.SensorFrame(**frame_msg), dict(frame_blobs))
                    if "truth" in client.channels:
                        await client.conn.send(
                            m.TruthState(
                                frame_id=self.frame_id,
                                pos_sim=[float(v) for v in state.pos_sim],
                                quat_sim=_yaw_quat(state.yaw_deg),
                                vel_sim=[float(v) for v in state.vel_sim],
                                collided=state.collided,
                                sim_time=self.sim_time,
                            )
                        )
                except (ConnectionError, asyncio.IncompleteReadError):
                    if client in self.clients:
                        self.clients.remove(client)

            if self.cfg.sim.realtime:
                await asyncio.sleep(dt)
            else:
                await asyncio.sleep(0)


def _yaw_quat(yaw_deg: float) -> list[float]:
    yaw_r = np.radians(yaw_deg)
    return [0.0, float(np.sin(yaw_r / 2)), 0.0, float(np.cos(yaw_r / 2))]


async def run_server(cfg: Config, port: int | None = None) -> None:
    server = HeadlessSimServer(cfg)
    await server.start(port=port)
    await server.serve_forever()
