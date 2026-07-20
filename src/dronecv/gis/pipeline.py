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
    poi: object | None = None  # OverpassPoi (optional: POIs/landmarks/parts)


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
    fetch_photos: bool = False,
    reconstruct_buildings: bool = False,
    imagery: str | None = None,
    imagery_res_m: float = 10.0,
    palette_photos_dir: Path | None = None,
    ortho_normalize: bool = True,
) -> Path:
    """`reconstruct_buildings` extracts extra footprints from imagery and
    merges them where GIS vectors have nothing (see
    providers.footprints_from_imagery). `imagery` selects the source when no
    local ortho is given: "eox" (Sentinel-2 cloudless WMS, ~10 m/px) or
    "xyz:<url-template>" (caller owns the provider's terms of service)."""
    if sources is None:
        from dronecv.gis.providers.poi import OverpassPoi

        sources = BuildSources(
            dem=CopernicusDem(), buildings=OverpassBuildings(), landcover=OverpassLandcover(),
            poi=OverpassPoi(),
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
    ortho = sources.ortho
    if ortho is None and ortho_path is not None:
        from dronecv.gis.providers.imagery import load_geotiff_ortho

        ortho = load_geotiff_ortho(ortho_path, anchor, ortho_utc, target_res_m=max(res_m, 0.5))
    if ortho is None and reconstruct_buildings and imagery:
        from dronecv.gis.providers.footprints_from_imagery import (
            fetch_wms_ortho,
            fetch_xyz_ortho,
        )

        if imagery == "eox":
            ortho = fetch_wms_ortho(bbox, anchor, res_m=imagery_res_m)
            meta.attribution.append(
                "Sentinel-2 cloudless by EOX IT Services (CC BY-NC-SA 4.0), non-commercial"
            )
        elif imagery == "s2":
            from dronecv.gis.providers.sentinel2 import fetch_cloudfree_ortho

            ortho, s2_stats = fetch_cloudfree_ortho(bbox, anchor, res_m=max(imagery_res_m, 10.0))
            meta.stats["cloudfree_composite"] = {
                "n_scenes_used": s2_stats.n_scenes_used,
                "first_scene_cloud_fraction": round(s2_stats.first_scene_cloud_fraction, 4),
                "final_hole_fraction": round(s2_stats.final_hole_fraction, 4),
                "fills": s2_stats.fills,
            }
            meta.attribution.append("Contains modified Copernicus Sentinel data")
        elif imagery.startswith("xyz:"):
            zoom = max(12, min(19, int(round(math.log2(156543.03 / 256.0 / max(imagery_res_m, 0.05))))))
            ortho = fetch_xyz_ortho(bbox, anchor, imagery[4:], zoom=zoom)
        else:
            raise ValueError(f"unknown imagery source '{imagery}' (use 'eox' or 'xyz:<template>')")

    # Composite mosaics: detect radiometric strips and align them BEFORE any
    # luminance/color consumer (reconstruction, palette, vegetation, shadows).
    rad_stats = None
    if ortho is not None and ortho_normalize:
        from dronecv.gis.radiometry import normalize_zones

        ortho, rad_stats = normalize_zones(ortho)

    # Single-date imagery may contain clouds: mask them so no consumer
    # mistakes a cloud for a roof or its shadow for water. (The "s2" source
    # arrives already cloud-free by multi-date compositing.)
    if ortho is not None and imagery != "s2":
        from dronecv.gis.clouds import cloud_mask

        inv = cloud_mask(ortho)
        if inv.any():
            ortho.valid = ~inv
            frac = float(inv.mean())
            meta.stats["cloud_fraction"] = round(frac, 4)
            if frac > 0.02:
                log.warning(
                    f"imagery is {frac:.0%} cloud-obstructed — those texels are "
                    "excluded; consider --imagery s2 (multi-date cloud-free composite)"
                )

    n_reconstructed = 0
    if reconstruct_buildings:
        if ortho is None:
            log.warning("reconstruct_buildings requested but no imagery available — skipped")
        else:
            from dronecv.gis.providers.footprints_from_imagery import (
                extract_footprints,
                merge_footprints,
            )

            sun_az = None
            if ortho.utc is not None:
                from dronecv.geo import celestial

                sun = celestial.sun_position(ortho.utc, anchor.lat0, anchor.lon0)
                sun_az = sun.azimuth_deg if sun.elevation_deg > 5.0 else None
            extracted = extract_footprints(ortho, anchor, sun_azimuth_deg=sun_az)
            n_reconstructed = merge_footprints(buildings, extracted, anchor)

    n_shadow = 0
    if ortho is not None:
        n_shadow = estimate_heights_from_shadows(buildings, anchor, ortho)

    landcover = sources.landcover.fetch(bbox)
    rasterize_landcover(store, anchor, landcover)
    # Flatten water to a single level: the 30 m DSM is noisy over rivers and
    # renders as blocky waves that visually drown the scene.
    water = np.asarray(store.class_id) == 2
    if water.any():
        store.ground[water] = float(np.percentile(np.asarray(store.ground)[water], 10.0))
    b_stats = rasterize_buildings(store, anchor, buildings)

    # Persist the VECTOR footprints (heights now resolved): LoD2 scene export
    # builds real prisms + shaped roofs from these instead of re-vectorizing
    # the raster (which loses footprints and roof metadata).
    from dronecv.gis.geometry import ring_to_enu

    vec = []
    for b in buildings:
        ring = ring_to_enu(anchor, b.footprint_lonlat)
        if len(ring) < 4:
            continue
        vec.append({
            "ring_enu": [[round(e, 2), round(n, 2)] for e, n in ring],
            "holes_enu": [[[round(e, 2), round(n, 2)] for e, n in ring_to_enu(anchor, h)]
                          for h in b.holes_lonlat],
            "height_m": b.height_m,
            "class": b.building_class,
            "height_source": b.height_source,
            "roof_shape": b.roof_shape,
            "roof_height_m": b.roof_height_m,
        })
    import json as _json

    (gis_dir / "buildings.json").write_text(_json.dumps({"buildings": vec}))

    # ---- POIs: landmark archetypes, building:part LoD, Commons photos ----
    pois, parts, poi_stats = [], [], {}
    if sources.poi is not None:
        from dronecv.gis.landmarks import (
            match_pois_to_buildings,
            stamp_archetypes,
            stamp_building_parts,
        )

        pois, parts = sources.poi.fetch(bbox)
        n_parts = stamp_building_parts(store, anchor, parts)
        pairs = match_pois_to_buildings(anchor, pois, buildings)
        n_arch = stamp_archetypes(store, anchor, pairs)
        poi_stats = {"n_pois": len(pois), "n_building_parts": n_parts, "n_archetypes": n_arch}
        if fetch_photos:
            from dronecv.gis.providers.poi import fetch_commons_photos

            photos = fetch_commons_photos(pois, gis_dir / "photos")
            poi_stats["n_photos"] = len(photos)

    # ---- 3D territory features: forests, bridges, rail embankments ----
    from dronecv.gis.vegetation import (
        build_vegetation,
        stamp_bridges,
        stamp_rail_embankment,
        vegetation_from_imagery,
    )

    imagery_veg_stats = vegetation_from_imagery(store, ortho)
    veg_stats = build_vegetation(store)
    n_bridges = stamp_bridges(store, anchor, landcover)
    n_rail = stamp_rail_embankment(store)

    above = np.maximum(np.asarray(store.build_h), np.asarray(store.veg_h))
    meta.max_height = float(max_ground + float(above.max(initial=0.0)))
    meta.stats = {
        "min_ground": min_ground,
        "max_ground": max_ground,
        "n_shadow_heights": n_shadow,
        "n_reconstructed": n_reconstructed,
        **({"radiometric_zones": {"n_zones": rad_stats.n_zones,
                                  "corrections": rad_stats.corrections}}
           if rad_stats is not None else {}),
        **height_stats(buildings),
        **b_stats,
        "n_landcover": len(landcover),
        **poi_stats,
        **imagery_veg_stats,
        **veg_stats,
        "n_bridges": n_bridges,
        "n_rail_texels": n_rail,
    }

    # ---- zone palette: roofs/walls from photos of the area + ortho roofs ----
    from dronecv.gis.palette import extract_palette

    photo_dirs = [gis_dir / "photos"]
    if palette_photos_dir is not None:
        photo_dirs.append(Path(palette_photos_dir))
    roof_pixels = None
    if ortho is not None and ortho.rgb is not None:
        rs, cs = np.nonzero(np.asarray(store.build_h) > 0)
        if len(rs):
            step = max(1, len(rs) // 100_000)
            rs, cs = rs[::step], cs[::step]
            ge = meta.e0 + (cs + 0.5) * res_m
            gn = meta.n0 + (rs + 0.5) * res_m
            pr = np.clip(((gn - ortho.n0) / ortho.res_m).astype(int), 0, ortho.rgb.shape[0] - 1)
            pc = np.clip(((ge - ortho.e0) / ortho.res_m).astype(int), 0, ortho.rgb.shape[1] - 1)
            roof_pixels = ortho.rgb[pr, pc]
            if ortho.valid is not None:  # cloud texels are not roof colors
                roof_pixels = roof_pixels[ortho.valid[pr, pc]]
    palette = extract_palette(photo_dirs, roof_pixels=roof_pixels)
    palette.save(gis_dir)
    meta.stats["palette"] = {"n_roof_px": palette.n_roof_px, "n_wall_px": palette.n_wall_px,
                             "sources": palette.sources}

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
