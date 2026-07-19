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
        """Yield (frame_id, estimate) forever."""
        assert self.client and self.localizer
        async for msg, blobs in self.client.stream():
            if not isinstance(msg, m.SensorFrame):
                continue
            est = self.localizer.step(blobs["rgb"], msg.lidar_range_m, msg.utc, msg.sim_time)
            yield msg.frame_id, est

    async def run(self, on_estimate: Callable[[int, Estimate], None]) -> None:
        async for frame_id, est in self.estimates():
            on_estimate(frame_id, est)

    async def close(self) -> None:
        if self.client:
            await self.client.close()
