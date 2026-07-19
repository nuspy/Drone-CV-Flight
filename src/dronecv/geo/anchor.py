"""Geo anchor: ties the sim's local origin to the real world.

Resolution order (dual mode, per user decision):
1. geo metadata embedded in the environment and reported by the simulator in
   `hello_ack.geo_meta` (Unity reads a `dronecv_geo.json` sidecar or a
   GeoAnchorAsset; the headless sim echoes its config);
2. fallback: the `env.anchor` section of the environment YAML.

The resolved anchor and its provenance are stamped into the trained model
bundle manifest so a model can never silently be used with the wrong anchor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dronecv.config import AnchorConfig
from dronecv.geo import frames


@dataclass(frozen=True)
class GeoAnchor:
    lat0: float
    lon0: float
    alt0: float
    true_north_offset_deg: float
    source: str  # "sim_metadata" | "env_config"

    @classmethod
    def resolve(cls, config_anchor: AnchorConfig, geo_meta: dict[str, Any] | None) -> GeoAnchor:
        if geo_meta:
            return cls(
                lat0=float(geo_meta["lat0"]),
                lon0=float(geo_meta["lon0"]),
                alt0=float(geo_meta["alt0"]),
                true_north_offset_deg=float(geo_meta.get("true_north_offset_deg", 0.0)),
                source="sim_metadata",
            )
        return cls(
            lat0=config_anchor.lat0,
            lon0=config_anchor.lon0,
            alt0=config_anchor.alt0,
            true_north_offset_deg=config_anchor.true_north_offset_deg,
            source="env_config",
        )

    # --- conversions bound to this anchor ---

    def sim_to_enu(self, p_sim: np.ndarray) -> np.ndarray:
        return frames.sim_to_enu(p_sim, self.true_north_offset_deg)

    def enu_to_sim(self, p_enu: np.ndarray) -> np.ndarray:
        return frames.enu_to_sim(p_enu, self.true_north_offset_deg)

    def sim_quat_to_enu_matrix(self, q_sim: np.ndarray) -> np.ndarray:
        return frames.sim_quat_to_enu_matrix(q_sim, self.true_north_offset_deg)

    def enu_to_geodetic(self, p_enu: np.ndarray) -> tuple[float, float, float]:
        return frames.enu_to_geodetic(p_enu, self.lat0, self.lon0, self.alt0)

    def geodetic_to_enu(self, lat: float, lon: float, alt: float) -> np.ndarray:
        return frames.geodetic_to_enu(lat, lon, alt, self.lat0, self.lon0, self.alt0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GeoAnchor:
        return cls(**d)
