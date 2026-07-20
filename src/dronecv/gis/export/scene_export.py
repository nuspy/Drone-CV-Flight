"""Scene export: a built GIS environment -> Unity Terrain package and/or
Blender-ready assets.

Common assets (written for both targets):
    terrain.raw        16-bit heightmap (orography) + scene_meta.json scales
    splatmap.png       ground classes as RGBA weights (ground/green/road/water)
    trees.json         individual tree instances (pos ENU, type, height) sampled
                       from the vegetation layer (density and typology preserved)
    buildings.obj      extruded footprints grouped by class + landmark meshes
                       (spires/domes for matched POIs)
    lines.json         road / river / rail centerlines (ENU polylines)
    scene_meta.json    anchor, extents, height scale, class legend, attribution

Unity: `unity/com.dronecv.flight/Editor/GisSceneImporter.cs` consumes the
folder (menu *DroneCV > Import GIS Scene*): Terrain with orography + splat +
trees (density/typology), buildings placed as meshes.
Blender: `blender_build_scene.py` (bundled into the export) assembles the
same assets with dupli-vert forests.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from dronecv.gis.export import mesh as m
from dronecv.gis.providers.landcover import GROUND_CLASSES
from dronecv.gis.store import GisStore
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.export")

TREE_MIN_SPACING_TEXELS = 4  # ~ every 4th forest texel becomes an instance


def export_scene(gis_dir: Path, out_dir: Path, terrain_resolution: int = 513) -> Path:
    store = GisStore.open(gis_dir)
    meta = store.meta
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- terrain heightmap (RAW16 for Unity, also generic) ----
    ground = np.asarray(store.ground)
    hmin, hmax = float(ground.min()), float(ground.max())
    scale = max(hmax - hmin, 1.0)
    rows = np.linspace(0, meta.height - 1, terrain_resolution).astype(int)
    cols = np.linspace(0, meta.width - 1, terrain_resolution).astype(int)
    resampled = ground[np.ix_(rows, cols)]
    norm = ((resampled - hmin) / scale * 65535).astype("<u2")
    (out_dir / "terrain.raw").write_bytes(norm.tobytes())

    # ---- splatmap: 4 weight layers ----
    cls = np.asarray(store.class_id)[np.ix_(rows, cols)]
    splat = np.zeros((terrain_resolution, terrain_resolution, 4), dtype=np.uint8)
    green_ids = (GROUND_CLASSES["green"], GROUND_CLASSES["forest_broadleaf"], GROUND_CLASSES["forest_conifer"])
    hard_ids = (GROUND_CLASSES["road"], GROUND_CLASSES["rail"], GROUND_CLASSES["parking"])
    splat[..., 1] = np.isin(cls, green_ids) * 255
    splat[..., 2] = np.isin(cls, hard_ids) * 255
    splat[..., 3] = (cls == GROUND_CLASSES["water"]) * 255
    splat[..., 0] = 255 - splat[..., 1:].max(axis=-1)
    Image.fromarray(splat, "RGBA").save(out_dir / "splatmap.png")

    # ---- tree instances from the vegetation layer ----
    veg = np.asarray(store.veg_h)
    step = TREE_MIN_SPACING_TEXELS
    trees = []
    sub = veg[::step, ::step]
    cls_sub = np.asarray(store.class_id)[::step, ::step]
    rr, cc = np.nonzero(sub > 1.0)
    for r, c in zip(rr, cc, strict=True):
        e = meta.e0 + (c * step + 0.5) * meta.res_m
        n = meta.n0 + (r * step + 0.5) * meta.res_m
        conifer = cls_sub[r, c] == GROUND_CLASSES["forest_conifer"]
        trees.append({
            "e": round(float(e), 2), "n": round(float(n), 2),
            "type": "conifer" if conifer else "broadleaf",
            "height": round(float(sub[r, c]), 1),
        })
    (out_dir / "trees.json").write_text(json.dumps({"trees": trees}))

    # ---- buildings: LoD2 from the persisted vectors (fallback: raster) ----
    from shapely import geometry as sgeom

    objects: dict[str, m.Mesh] = {}
    n_buildings = 0
    vec_path = Path(gis_dir) / "buildings.json"
    if vec_path.exists():
        walls_of: dict[str, m.Mesh] = {}
        roofs_of: dict[str, m.Mesh] = {}
        for b in json.loads(vec_path.read_text())["buildings"]:
            try:
                poly = sgeom.Polygon(b["ring_enu"], b.get("holes_enu") or None)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty or poly.area < 4.0:
                    continue
                r = int(np.clip((poly.centroid.y - meta.n0) / meta.res_m, 0, meta.height - 1))
                c = int(np.clip((poly.centroid.x - meta.e0) / meta.res_m, 0, meta.width - 1))
                base = float(ground[r, c])
                walls, roof = m.building_meshes(
                    poly, base, float(b.get("height_m") or 6.0),
                    b.get("roof_shape"), b.get("roof_height_m"),
                    b.get("class", "generic"),
                )
            except Exception:  # noqa: BLE001 — one bad footprint must not kill the export
                continue
            cls = b.get("class", "generic")
            walls_of.setdefault(cls, m.Mesh()).add(walls.vertices, walls.faces, walls.uvs)
            roofs_of.setdefault(cls, m.Mesh()).add(roof.vertices, roof.faces, roof.uvs)
            n_buildings += 1
        for cls, mesh in walls_of.items():
            objects[f"walls_{cls}"] = mesh
        for cls, mesh in roofs_of.items():
            objects[f"roof_{cls}"] = mesh
    else:  # stores built before vector persistence: raster re-vectorization
        from rasterio import features as rio_features
        from rasterio.transform import Affine

        build = np.asarray(store.build_h)
        transform = Affine(meta.res_m, 0.0, meta.e0, 0.0, meta.res_m, meta.n0)
        quant = np.round(build / 2.0).astype(np.int16)  # 2 m height bins
        bmesh = m.Mesh()
        for geom, value in rio_features.shapes(quant, mask=quant > 0, transform=transform):
            height = float(value) * 2.0
            poly = sgeom.shape(geom).simplify(meta.res_m)
            if poly.is_empty or poly.area < 4.0:
                continue
            r = int(np.clip((poly.centroid.y - meta.n0) / meta.res_m, 0, meta.height - 1))
            c = int(np.clip((poly.centroid.x - meta.e0) / meta.res_m, 0, meta.width - 1))
            base = float(ground[r, c])
            prism = m.extrude_polygon(poly, base, base + height)
            bmesh.add(prism.vertices, prism.faces)
            n_buildings += 1
        objects["buildings"] = bmesh
    m.write_obj(out_dir / "buildings.obj", objects)

    # ---- scene.glb: terrain (class vertex colors) + LoD2 buildings with
    # palette PBR materials and a procedural facade window texture ----
    from dronecv.gis.export.gltf import GlbBuilder, window_atlas_png
    from dronecv.gis.palette import ZonePalette
    from dronecv.gis.world import CLASS_COLORS

    palette = ZonePalette.load(gis_dir) or ZonePalette()
    glb = GlbBuilder(name="dronecv-gis-scene")
    mat_wall = glb.add_material("walls", palette.wall[0], texture_png=window_atlas_png())
    roof_color_of = {
        "generic": palette.roof[0], "residential": palette.roof[0],
        "commercial": palette.roof[min(1, len(palette.roof) - 1)],
        "industrial": CLASS_COLORS[12], "landmark": CLASS_COLORS[14],
    }
    tmesh = m.grid_terrain(ground, meta.res_m, meta.e0, meta.n0,
                           step=max(1, meta.width // 256))
    tpos = np.asarray(tmesh.vertices, dtype=np.float32)
    color_lut = np.zeros((32, 3), np.float32)
    for cid, rgb in CLASS_COLORS.items():
        color_lut[cid] = rgb
    cls_full = np.asarray(store.class_id)
    tr = np.clip(((tpos[:, 1] - meta.n0) / meta.res_m).astype(int), 0, meta.height - 1)
    tc = np.clip(((tpos[:, 0] - meta.e0) / meta.res_m).astype(int), 0, meta.width - 1)
    tcol = color_lut[cls_full[tr, tc]]
    mat_terr = glb.add_material("terrain", (1.0, 1.0, 1.0))
    glb.add_mesh("terrain", tpos, np.asarray(tmesh.faces, np.uint32), mat_terr, colors=tcol)
    for name, mesh in objects.items():
        if not mesh.vertices:
            continue
        cls = name.split("_", 1)[-1]
        if name.startswith("walls"):
            mat = mat_wall
        else:
            mat = glb.add_material(name, roof_color_of.get(cls, palette.roof[0]))
        uv = np.asarray(mesh.uvs, np.float32) if mesh.uvs else None
        glb.add_mesh(name, np.asarray(mesh.vertices, np.float32),
                     np.asarray(mesh.faces, np.uint32), mat, uvs=uv)
    glb.save(out_dir / "scene.glb")

    # ---- centerlines for splines (roads/rivers/rail) ----
    lines = {"note": "ENU meters; classes from splatmap; centerlines not vectorized in v1"}
    (out_dir / "lines.json").write_text(json.dumps(lines))

    # Sun position for scene lighting (same reference time as the renders).
    from datetime import UTC, datetime

    from dronecv.geo import celestial

    sun = celestial.sun_position(
        datetime(2026, 6, 21, 10, 0, tzinfo=UTC),
        float(meta.anchor["lat0"]), float(meta.anchor["lon0"]),
    )

    (out_dir / "scene_meta.json").write_text(json.dumps({
        "anchor": meta.anchor,
        "sun_azimuth_deg": round(sun.azimuth_deg, 2),
        "sun_elevation_deg": round(sun.elevation_deg, 2),
        "extent_e_m": meta.width * meta.res_m,
        "extent_n_m": meta.height * meta.res_m,
        "e0": meta.e0,
        "n0": meta.n0,
        "terrain_resolution": terrain_resolution,
        "height_min_m": hmin,
        "height_scale_m": scale,
        "n_trees": len(trees),
        "n_building_meshes": n_buildings,
        "splat_layers": ["ground", "vegetation", "hard_surface", "water"],
        "attribution": meta.attribution,
        "coordinate_convention": "Z-up ENU meters, origin at anchor",
    }, indent=2))

    # ---- Blender assembly script bundled with the export ----
    shutil.copy(Path(__file__).parent / "blender_build_scene.py", out_dir / "blender_build_scene.py")
    log.info(f"scene exported to {out_dir}: {n_buildings} building meshes, {len(trees)} trees")
    return out_dir
