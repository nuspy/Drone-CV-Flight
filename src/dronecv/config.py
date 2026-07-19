"""Layered configuration: configs/default.yaml <- configs/envs/<env>.yaml <- CLI overrides.

All sections are validated by pydantic models; unknown keys are rejected so a
typo in a YAML file fails loudly instead of silently using a default.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnchorConfig(StrictModel):
    lat0: float
    lon0: float
    alt0: float
    true_north_offset_deg: float = 0.0


class EnvConfig(StrictModel):
    name: str | None = None
    kind: Literal["headless", "unity"] = "headless"
    seed: int = 0
    anchor: AnchorConfig
    start_utc: str = "2026-06-21T10:00:00Z"


class LidarConfig(StrictModel):
    max_range_m: float = 120.0
    noise_sigma_m: float = 0.05
    dropout_prob: float = 0.01


class DroneConfig(StrictModel):
    max_speed_ms: float = 12.0
    max_climb_ms: float = 4.0
    max_yaw_rate_dps: float = 90.0
    response_tau_s: float = 0.6


class SimConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = 7601
    sensor_hz: float = 5.0
    realtime: bool = False  # False: tick as fast as consumers allow (CI)
    image_width: int = 96
    image_height: int = 96
    fov_deg: float = 70.0
    camera_tilt_deg: float = 35.0
    lidar: LidarConfig = Field(default_factory=LidarConfig)
    drone: DroneConfig = Field(default_factory=DroneConfig)


class WorldConfig(StrictModel):
    size_m: float = 1000.0
    height_scale_m: float = 60.0
    n_landmarks: int = 30
    grid: int = 128
    # Whether the headless sim reports geo metadata in hello_ack (exercises the
    # "embedded metadata" anchor path; False exercises the env-config fallback).
    embed_geo_meta: bool = True


class CaptureConfig(StrictModel):
    altitudes_agl_m: list[float] = Field(default_factory=lambda: [40.0, 80.0])
    yaw_bins: int = 8
    grid_spacing_m: float = 75.0
    orbit_radius_m: float = 60.0
    orbit_points: int = 12
    image_format: Literal["jpeg", "png"] = "jpeg"
    jpeg_quality: int = 92


class TrainingConfig(StrictModel):
    batch_size: int = 32
    lr: float = 1e-3
    epochs_per_round: int = 12
    embedding_dim: int = 128
    backbone_width: int = 32
    triplet_margin: float = 0.3
    pos_radius_m: float = 30.0
    pos_yaw_deg: float = 60.0
    val_cell_m: float = 50.0
    val_fraction: float = 0.15
    num_workers: int = 2


class Thresholds(StrictModel):
    p95_pos_err_m: float = 12.0
    p95_alt_err_m: float = 6.0
    max_ece: float = 0.15


class ActiveLoopConfig(StrictModel):
    budget_captures: int = 4000
    max_rounds: int = 6
    initial_fraction: float = 0.35
    growth_fraction: float = 0.5
    probe_captures: int = 150
    thresholds: Thresholds = Field(default_factory=Thresholds)
    plateau_epsilon: float = 0.05
    plateau_rounds: int = 2


class ProcessNoise(StrictModel):
    accel_sigma: float = 1.2
    yaw_rate_sigma_dps: float = 10.0


class LocalizationConfig(StrictModel):
    retrieval_topk: int = 5
    gate_chi2: float = 9.21
    apr_sigma_floor_m: float = 1.5
    vo_min_inliers: int = 25
    sun_min_elevation_deg: float = 5.0
    lost_reject_streak: int = 12
    process: ProcessNoise = Field(default_factory=ProcessNoise)


class GuidanceConfig(StrictModel):
    standoff_m: float = 10.0
    arrival_tol_m: float = 3.0
    alt_tol_m: float = 2.0
    cruise_speed_ms: float = 8.0
    min_confidence: float = 0.25


class PassThresholds(StrictModel):
    p95_pos_err_m: float = 15.0
    reach_success_rate: float = 0.6


class HarnessConfig(StrictModel):
    episodes: int = 5
    time_cap_s: float = 240.0
    visual_target_fraction: float = 0.4
    kidnap_test: bool = True
    # True-distance radius within which a declared arrival counts as success.
    # Scale it with the environment's achievable localization accuracy.
    success_radius_m: float = 15.0
    pass_thresholds: PassThresholds = Field(default_factory=PassThresholds)


class ReportConfig(StrictModel):
    out_dir: str = "artifacts"


class Config(StrictModel):
    env: EnvConfig
    sim: SimConfig = Field(default_factory=SimConfig)
    world: WorldConfig = Field(default_factory=WorldConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    active_loop: ActiveLoopConfig = Field(default_factory=ActiveLoopConfig)
    localization: LocalizationConfig = Field(default_factory=LocalizationConfig)
    guidance: GuidanceConfig = Field(default_factory=GuidanceConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @property
    def artifacts_dir(self) -> Path:
        assert self.env.name, "env.name must be set"
        return Path(self.report.out_dir) / self.env.name

    @property
    def dataset_dir(self) -> Path:
        return self.artifacts_dir / "dataset"

    @property
    def bundle_dir(self) -> Path:
        return self.artifacts_dir / "model_bundle"

    @property
    def report_dir(self) -> Path:
        return self.artifacts_dir / "report"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def find_config_root(start: Path | None = None) -> Path:
    """Walk up from `start` (or cwd) until a `configs/` directory is found."""
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "configs" / "default.yaml").exists():
            return candidate
    raise FileNotFoundError("could not locate configs/default.yaml above the current directory")


def load_config(
    env: str,
    overrides: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Config:
    """Load default.yaml, merge configs/envs/<env>.yaml, merge overrides."""
    root = root or find_config_root()
    default_path = root / "configs" / "default.yaml"
    env_path = root / "configs" / "envs" / f"{env}.yaml"
    if not env_path.exists():
        available = sorted(p.stem for p in (root / "configs" / "envs").glob("*.yaml"))
        raise FileNotFoundError(f"unknown environment '{env}'. Available: {', '.join(available)}")

    merged = yaml.safe_load(default_path.read_text())
    merged = _deep_merge(merged, yaml.safe_load(env_path.read_text()) or {})
    if overrides:
        merged = _deep_merge(merged, overrides)
    cfg = Config.model_validate(merged)
    if cfg.env.name is None:
        cfg.env.name = env
    return cfg


def list_envs(root: Path | None = None) -> list[str]:
    root = root or find_config_root()
    return sorted(p.stem for p in (root / "configs" / "envs").glob("*.yaml"))
