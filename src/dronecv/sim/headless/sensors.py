"""Sensor models for the headless sim: camera, mock lidar, celestial angles.

The mock lidar is a single downward beam with Gaussian noise, max range and
dropout — the exact model MockLidar.cs replicates with a Physics.Raycast.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from dronecv.config import LidarConfig, SimConfig
from dronecv.geo import celestial
from dronecv.geo.anchor import GeoAnchor
from dronecv.sim.headless import rasterizer
from dronecv.sim.headless.world import World


class CelestialModel:
    """Sun/moon angles for the sim clock, expressed in BOTH true-compass and
    sim frames (sim azimuth = true azimuth - anchor north offset)."""

    def __init__(self, anchor: GeoAnchor, start_utc: datetime):
        self.anchor = anchor
        self.start_utc = start_utc

    def utc_at(self, sim_time_s: float) -> datetime:
        return self.start_utc + timedelta(seconds=sim_time_s)

    def sun(self, sim_time_s: float) -> celestial.CelestialPosition:
        return celestial.sun_position(self.utc_at(sim_time_s), self.anchor.lat0, self.anchor.lon0)

    def moon(self, sim_time_s: float) -> celestial.CelestialPosition:
        return celestial.moon_position(self.utc_at(sim_time_s), self.anchor.lat0, self.anchor.lon0)

    def to_sim_azimuth(self, true_azimuth_deg: float) -> float:
        return (true_azimuth_deg - self.anchor.true_north_offset_deg) % 360.0


class Camera:
    def __init__(self, cfg: SimConfig, world: World, sky: CelestialModel):
        self.cfg = cfg
        self.world = world
        self.sky = sky

    def render(
        self, pos_sim: np.ndarray, yaw_deg: float, pitch_down_deg: float | None, sim_time_s: float,
        width: int | None = None, height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, celestial.CelestialPosition, celestial.CelestialPosition]:
        sun = self.sky.sun(sim_time_s)
        moon = self.sky.moon(sim_time_s)
        rgb, depth = rasterizer.render(
            self.world,
            pos_sim,
            yaw_deg,
            self.cfg.camera_tilt_deg if pitch_down_deg is None else pitch_down_deg,
            width or self.cfg.image_width,
            height or self.cfg.image_height,
            self.cfg.fov_deg,
            self.sky.to_sim_azimuth(sun.azimuth_deg),
            sun.elevation_deg,
            self.sky.to_sim_azimuth(moon.azimuth_deg),
            moon.elevation_deg,
        )
        return rgb, depth, sun, moon


class MockLidar:
    def __init__(self, cfg: LidarConfig, world: World, gen: np.random.Generator):
        self.cfg = cfg
        self.world = world
        self.gen = gen

    def range_down(self, pos_sim: np.ndarray) -> float | None:
        """Downward beam range in meters, or None (dropout / out of range)."""
        if self.gen.random() < self.cfg.dropout_prob:
            return None
        ground = float(self.world.height_at(np.asarray(pos_sim[0]), np.asarray(pos_sim[2])))
        true_range = float(pos_sim[1]) - ground
        if true_range < 0 or true_range > self.cfg.max_range_m:
            return None
        return max(0.0, true_range + float(self.gen.normal(0.0, self.cfg.noise_sigma_m)))
