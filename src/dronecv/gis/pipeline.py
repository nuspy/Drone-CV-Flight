"""GIS build pipeline: bbox/mask -> ready-to-fly dronecv environment.

    dronecv gis build --bbox 43.31,11.32,43.34,11.36 --env-name siena

Steps: DEM -> buildings (tag heights -> SHADOW inference -> class defaults)
-> landcover -> raster mosaics -> saliency + orbit anchors -> env YAML.
The result plugs into the standard flow: `dronecv run-all --env siena`
trains, tests and reports on the real area, and the localizer outputs REAL
GPS coordinates (anchor = area center, ENU-aligned).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from dronecv.gis.geometry import BBox, anchor_for, enu_bounds
from dronecv.gis.providers.buildings import OverpassBuildings, height_stats
from dronecv.gis.providers.dem import CopernicusDem
from dronecv.gis.providers.imagery import OrthoImage
from dronecv.gis.providers.landcover import OverpassLandcover
from dronecv.gis.raster import rasterize_buildings, rasterize_landcover
from dronecv.gis.saliency import compute_saliency, orbit_points_from_store
from dronecv.gis.shadow_heights import estimate_heights_from_shadows
from dronecv.gis.store import GisMeta, GisStore
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.build")

ATTRIBUTION = [
    "Building/landcover data © OpenStreetMap contributors (ODbL)",
    "Elevation: Copernicus GLO-30 DEM © European Union, ESA, Airbus",
]


@dataclass
class BuildSources:
    """Injectable providers (tests pass fixture-backed instances)."""

    dem: CopernicusDem
    buildings: OverpassBuildings
    landcover: OverpassLandcover
    ortho: OrthoImage | None = None


def build_environment(
    bbox: BBox,
    env_name: str,
    out_root: Path,
    configs_root: Path,
    sources: BuildSources | None = None,
    res_m: float = 1.0,
    ortho_path: Path | None = None,
    ortho_utc: datetime | None = None,
    max_extent_m: float = 60_000.0,
) -> Path:
    sources = sources or BuildSources(
        dem=CopernicusDem(), buildings=OverpassBuildings(), landcover=OverpassLandcover()
    )
    anchor = anchor_for(bbox)
    e_min, n_min, e_max, n_max = enu_bounds(anchor, bbox)
    extent_e, extent_n = e_max - e_min, n_max - n_min
    if max(extent_e, extent_n) > max_extent_m:
        raise ValueError(f"AOI extent {extent_e:.0f}x{extent_n:.0f} m exceeds {max_extent_m} m")
    # Center the mosaic on the anchor.
    half_e, half_n = extent_e / 2.0, extent_n / 2.0
    width = int(math.ceil(extent_e / res_m))
    height = int(math.ceil(extent_n / res_m))
    log.info(f"building '{env_name}': {extent_e:.0f} x {extent_n:.0f} m at {res_m} m/px "
             f"({width}x{height} px), anchor {anchor.lat0:.5f},{anchor.lon0:.5f}")

    gis_dir = Path(out_root) / env_name
    meta = GisMeta(
        res_m=res_m,
        e0=-half_e,
        n0=-half_n,
        width=width,
        height=height,
        anchor=anchor.to_dict(),
        ground_alt0=0.0,
        max_height=0.0,
        attribution=ATTRIBUTION,
    )
    store = GisStore.create(gis_dir, meta)

    # ---- terrain (windowed: bounded RAM for big AOIs) ----
    import pymap3d

    win = 2048
    ground_alt0 = None
    min_ground, max_ground = np.inf, -np.inf
    for r0 in range(0, height, win):
        rows = np.arange(r0, min(r0 + win, height))
        for c0 in range(0, width, win):
            cols = np.arange(c0, min(c0 + win, width))
            ge, gn = np.meshgrid(
                meta.e0 + (cols + 0.5) * res_m, meta.n0 + (rows + 0.5) * res_m
            )
            lat, lon, _ = pymap3d.enu2geodetic(
                ge.ravel(), gn.ravel(), np.zeros(ge.size), anchor.lat0, anchor.lon0, anchor.alt0
            )
            elev = sources.dem.sample_grid(
                bbox.margin(0.01), np.asarray(lat), np.asarray(lon)
            ).reshape(ge.shape)
            if ground_alt0 is None:
                ground_alt0 = float(np.median(elev))
            store.ground[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1] = elev - ground_alt0
            min_ground = min(min_ground, float(elev.min() - ground_alt0))
            max_ground = max(max_ground, float(elev.max() - ground_alt0))
    meta.ground_alt0 = float(ground_alt0 or 0.0)

    # ---- buildings: fetch, resolve heights (tags -> shadows -> defaults) ----
    buildings = sources.buildings.fetch(bbox)
    n_shadow = 0
    if sources.ortho is not None or ortho_path is not None:
        ortho = sources.ortho
        if ortho is None:
            from dronecv.gis.providers.imagery import load_geotiff_ortho

            ortho = load_geotiff_ortho(ortho_path, anchor, ortho_utc, target_res_m=max(res_m, 0.5))
        n_shadow = estimate_heights_from_shadows(buildings, anchor, ortho)

    landcover = sources.landcover.fetch(bbox)
    rasterize_landcover(store, anchor, landcover)
    b_stats = rasterize_buildings(store, anchor, buildings)

    meta.max_height = float(max_ground + float(np.asarray(store.build_h).max(initial=0.0)))
    meta.stats = {
        "min_ground": min_ground,
        "max_ground": max_ground,
        "n_shadow_heights": n_shadow,
        **height_stats(buildings),
        **b_stats,
        "n_landcover": len(landcover),
    }

    # ---- saliency prior + orbit anchors ----
    saliency = compute_saliency(store)
    meta.stats["saliency"] = {
        "mean_density_multiplier": saliency["mean_density_multiplier"],
        "cell_m": saliency["cell_m"],
    }
    (gis_dir / "saliency.json").write_text(__import__("json").dumps(saliency))
    meta.orbit_points = orbit_points_from_store(store)
    store.flush()

    # ---- env YAML: plugs into the standard pipeline ----
    env_yaml = {
        "env": {
            "name": env_name,
            "kind": "headless",
            "seed": 0,
            "anchor": {
                "lat0": anchor.lat0,
                "lon0": anchor.lon0,
                "alt0": meta.ground_alt0,
                "true_north_offset_deg": 0.0,
            },
        },
        "world": {"kind": "gis", "gis_dir": str(gis_dir)},
        "sim": {"image_width": 128, "image_height": 128},
        "capture": {
            "grid_spacing_m": 60.0,
            "altitudes_agl_m": [50.0, 90.0],
        },
        "training": {"backbone_width": 32},
    }
    env_path = Path(configs_root) / "configs" / "envs" / f"{env_name}.yaml"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(yaml.safe_dump(env_yaml, sort_keys=False))
    log.info(
        f"environment '{env_name}' ready: {meta.stats['n_buildings']} buildings "
        f"({meta.stats['n_with_height']} tagged + {n_shadow} shadow heights), "
        f"config {env_path} — next: dronecv run-all --env {env_name}"
    )
    return gis_dir
