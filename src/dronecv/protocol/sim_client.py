"""High-level client API over the wire protocol.

Used identically against the headless simulator and the Unity bridge — the
caller cannot tell which one is on the other end (by design).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np

from dronecv.protocol import messages as m
from dronecv.protocol.framing import ProtocolError
from dronecv.protocol.transport import Connection, connect


class SimClient:
    def __init__(self, conn: Connection, ack: m.HelloAck):
        self.conn = conn
        self.ack = ack

    @classmethod
    async def connect(cls, host: str, port: int, role: m.Role, timeout: float = 10.0) -> SimClient:
        conn = await connect(host, port, timeout)
        await conn.send(m.Hello(role=role))
        msg, _ = await conn.recv()
        if isinstance(msg, m.ErrorMsg):
            raise ProtocolError(f"handshake rejected: {msg.message}")
        if not isinstance(msg, m.HelloAck):
            raise ProtocolError(f"expected hello_ack, got {msg.type}")
        major_local = m.PROTOCOL_VERSION.split(".")[0]
        major_remote = msg.version.split(".")[0]
        if major_local != major_remote:
            raise ProtocolError(
                f"protocol major version mismatch: local {m.PROTOCOL_VERSION}, sim {msg.version}"
            )
        return cls(conn, msg)

    @property
    def geo_meta(self) -> dict[str, float] | None:
        return self.ack.geo_meta

    async def _request(self, msg: m.AnyMessage, expect: type) -> tuple[m.AnyMessage, dict]:
        await self.conn.send(msg)
        while True:
            reply, blobs = await self.conn.recv()
            if isinstance(reply, m.ErrorMsg):
                raise ProtocolError(reply.message)
            if isinstance(reply, expect):
                return reply, blobs
            # Ignore unrelated pushed messages (e.g. a late sensor frame).

    async def env_info(self) -> m.EnvInfo:
        reply, _ = await self._request(m.EnvInfoRequest(), m.EnvInfo)
        return reply

    async def capture(
        self,
        pos_sim: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        want: list[str] = ("rgb", "depth", "sun"),
    ) -> tuple[m.CaptureResult, dict[str, np.ndarray | bytes]]:
        req = m.CaptureRequest(
            pos_sim=[float(v) for v in pos_sim],
            yaw_deg=float(yaw_deg),
            pitch_deg=float(pitch_deg),
            want=list(want),
        )
        reply, blobs = await self._request(req, m.CaptureResult)
        return reply, blobs

    async def reset(self, seed: int, start_pos_sim: np.ndarray | None = None, start_yaw_deg: float = 0.0, utc: str | None = None) -> m.ResetDone:
        msg = m.Reset(
            seed=seed,
            start_pos_sim=None if start_pos_sim is None else [float(v) for v in start_pos_sim],
            start_yaw_deg=float(start_yaw_deg),
            utc=utc,
        )
        reply, _ = await self._request(msg, m.ResetDone)
        return reply

    async def subscribe(self, channels: list[m.Channel]) -> None:
        reply, _ = await self._request(m.Subscribe(channels=channels), m.SubscribeAck)
        missing = set(channels) - set(reply.channels)
        if missing:
            raise ProtocolError(f"subscription refused for channels: {sorted(missing)}")

    async def send_command(self, vel_sim: np.ndarray | None = None, yaw_rate_dps: float | None = None) -> None:
        await self.conn.send(
            m.Command(
                vel_sim=None if vel_sim is None else [float(v) for v in vel_sim],
                yaw_rate_dps=yaw_rate_dps,
            )
        )

    async def teleport(self, pos_sim: np.ndarray, yaw_deg: float = 0.0) -> None:
        await self.conn.send(m.Teleport(pos_sim=[float(v) for v in pos_sim], yaw_deg=float(yaw_deg)))

    async def stream(self) -> AsyncIterator[tuple[m.AnyMessage, dict[str, np.ndarray | bytes]]]:
        """Yield pushed messages (sensor_frame / truth_state) forever."""
        while True:
            msg, blobs = await self.conn.recv()
            if isinstance(msg, m.ErrorMsg):
                raise ProtocolError(msg.message)
            yield msg, blobs

    async def close(self) -> None:
        await self.conn.close()
