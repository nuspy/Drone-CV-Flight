"""Streaming localization service: connects to a live sim (headless or Unity
bridge — indistinguishable) with role="localizer" and turns sensor frames
into geo estimates. This is exactly what would run on a companion computer."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from dronecv.config import Config
from dronecv.geo.anchor import GeoAnchor
from dronecv.localization.localizer import Estimate, Localizer
from dronecv.protocol import messages as m
from dronecv.protocol.sim_client import SimClient
from dronecv.training.bundle import ModelBundle
from dronecv.util.logging import get_logger

log = get_logger("dronecv.service")


class LocalizationService:
    def __init__(self, cfg: Config, bundle: ModelBundle):
        self.cfg = cfg
        self.bundle = bundle
        self.client: SimClient | None = None
        self.localizer: Localizer | None = None

    async def connect(self, host: str | None = None, port: int | None = None) -> None:
        self.client = await SimClient.connect(
            host or self.cfg.sim.host, port or self.cfg.sim.port, role="localizer"
        )
        anchor = GeoAnchor.resolve(self.cfg.env.anchor, self.client.geo_meta)
        self.localizer = Localizer(self.cfg, self.bundle, anchor)
        await self.client.subscribe(["sensors"])
        log.info(f"localizing against {self.client.ack.sim_kind} sim '{self.client.ack.env_id}'")

    async def estimates(self) -> AsyncIterator[tuple[int, Estimate]]:
        """Yield (frame_id, estimate) forever.

        Frames that arrive while a step is still being processed are DROPPED
        in favor of the newest one — exactly what a real camera pipeline does
        when inference is slower than the frame rate. Without this, a
        faster-than-realtime sim runs ahead and the estimates (and any
        control decisions built on them) refer to an ever-older state.
        """
        assert self.client and self.localizer
        import asyncio

        queue: asyncio.Queue = asyncio.Queue()

        async def pump() -> None:
            async for item in self.client.stream():
                queue.put_nowait(item)

        pump_task = asyncio.create_task(pump())
        dropped = 0
        try:
            while True:
                msg, blobs = await queue.get()
                # Drain the backlog, keeping only the newest sensor frame.
                while not queue.empty():
                    nxt_msg, nxt_blobs = queue.get_nowait()
                    if isinstance(nxt_msg, m.SensorFrame):
                        if isinstance(msg, m.SensorFrame):
                            dropped += 1
                        msg, blobs = nxt_msg, nxt_blobs
                if not isinstance(msg, m.SensorFrame):
                    continue
                est = self.localizer.step(blobs["rgb"], msg.lidar_range_m, msg.utc, msg.sim_time)
                if dropped and dropped % 200 == 0:
                    log.info(f"localizer lagging: {dropped} stale frames dropped so far")
                yield msg.frame_id, est
                if pump_task.done() and queue.empty():
                    pump_task.result()  # surface connection errors
                    return
        finally:
            pump_task.cancel()

    async def run(self, on_estimate: Callable[[int, Estimate], None]) -> None:
        async for frame_id, est in self.estimates():
            on_estimate(frame_id, est)

    async def close(self) -> None:
        if self.client:
            await self.client.close()
