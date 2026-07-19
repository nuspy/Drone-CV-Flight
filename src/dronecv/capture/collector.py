"""Drives the sim (headless or Unity — same protocol) to build a dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from dronecv.capture.dataset import DatasetWriter
from dronecv.capture.planner import CapturePose
from dronecv.config import Config
from dronecv.geo.anchor import GeoAnchor
from dronecv.protocol.sim_client import SimClient
from dronecv.util.logging import get_logger

log = get_logger("dronecv.capture")

PROBE_ALTITUDE_MARGIN = 30.0


class Collector:
    def __init__(self, cfg: Config, client: SimClient, anchor: GeoAnchor):
        self.cfg = cfg
        self.client = client
        self.anchor = anchor
        self._ground_cache: dict[tuple[int, int], float] = {}
        self._env_info = None

    async def env_info(self):
        if self._env_info is None:
            self._env_info = await self.client.env_info()
        return self._env_info

    async def ground_y(self, x: float, z: float) -> float:
        """Probe terrain height with a straight-down depth capture (cached
        per ~10 m cell). Works against any simulator: no height-query message
        is needed in the protocol."""
        key = (int(x // 10), int(z // 10))
        if key in self._ground_cache:
            return self._ground_cache[key]
        info = await self.env_info()
        probe_y = info.bounds_max_sim[1] - 1.0
        result, blobs = await self.client.capture(
            np.array([x, probe_y, z]), yaw_deg=0.0, pitch_deg=90.0, want=["depth"]
        )
        depth = blobs["depth"]
        h, w = depth.shape
        center = depth[h // 2 - 1 : h // 2 + 2, w // 2 - 1 : w // 2 + 2]
        valid = center[center > 0]
        ground = probe_y - float(np.median(valid)) if valid.size else 0.0
        self._ground_cache[key] = ground
        return ground

    async def collect(self, poses: list[CapturePose], out_dir: Path, progress_label: str = "capture") -> int:
        writer = DatasetWriter(
            out_dir,
            info={
                "env": self.cfg.env.name,
                "anchor": self.anchor.to_dict(),
                "camera": self.client.ack.camera.model_dump() if self.client.ack.camera else None,
                "sim_kind": self.client.ack.sim_kind,
            },
        )
        tilt = self.cfg.sim.camera_tilt_deg
        n_written = 0
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "{task.completed}/{task.total}",
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(progress_label, total=len(poses))
            for pose in poses:
                ground = await self.ground_y(pose.x, pose.z)
                y = ground + pose.agl_m
                result, blobs = await self.client.capture(
                    np.array([pose.x, y, pose.z]),
                    yaw_deg=pose.yaw_deg,
                    pitch_deg=tilt if pose.pitch_deg == 0.0 else pose.pitch_deg,
                    want=["rgb", "sun"],
                )
                pos_sim = np.array(result.pos_sim)
                pos_enu = self.anchor.sim_to_enu(pos_sim)
                heading = (result.yaw_deg + self.anchor.true_north_offset_deg) % 360.0
                writer.add(
                    blobs["rgb"],
                    meta=dict(
                        pos_sim=[float(v) for v in pos_sim],
                        pos_enu=[float(v) for v in pos_enu],
                        yaw_sim_deg=float(result.yaw_deg),
                        heading_deg=float(heading),
                        pitch_deg=float(result.pitch_deg),
                        agl_m=float(pose.agl_m),
                        ground_y_sim=float(ground),
                        utc=result.utc,
                        sun_azimuth_deg=result.sun_azimuth_deg,
                        sun_elevation_deg=result.sun_elevation_deg,
                    ),
                )
                n_written += 1
                progress.advance(task)
        writer.close()
        log.info(f"collected {n_written} captures into {out_dir}")
        return n_written


async def open_sim_client(cfg: Config, role: str = "capture") -> tuple[SimClient, GeoAnchor]:
    """Connect to the sim and resolve the geo anchor (embedded metadata first,
    env-config fallback — the dual mode the user selected)."""
    client = await SimClient.connect(cfg.sim.host, cfg.sim.port, role=role)
    anchor = GeoAnchor.resolve(cfg.env.anchor, client.geo_meta)
    log.info(
        f"connected to {client.ack.sim_kind} sim '{client.ack.env_id}' — "
        f"anchor {anchor.lat0:.4f},{anchor.lon0:.4f} (source: {anchor.source})"
    )
    return client, anchor
