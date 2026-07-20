"""Assemble the dronecv GIS scene export inside Blender.

Usage (Blender 3.6+ / 4.x):
    blender --background --python blender_build_scene.py -- /path/to/export_dir [out.blend]

Preferred path: imports `scene.glb` (LoD2 buildings with palette PBR
materials, facade window texture, class-colored terrain) and adds instanced
forests (density/typology from trees.json), a sun light at the real solar
position, and a sky. Falls back to terrain.raw + buildings.obj for exports
produced before the glTF pipeline. If an output .blend path is given the
scene is saved there — the native-format deliverable.
Everything in Z-up ENU meters, origin at the AOI anchor.
"""

import json
import struct
import sys
from math import radians
from pathlib import Path

import bpy  # noqa: F401 (available inside Blender)


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :]
    export_dir = Path(args[0])
    blend_out = Path(args[1]) if len(args) > 1 else None
    meta = json.loads((export_dir / "scene_meta.json").read_text())

    # ---- clean scene ----
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    if (export_dir / "scene.glb").exists():
        bpy.ops.import_scene.gltf(filepath=str(export_dir / "scene.glb"))
        verts, res = None, None
    else:  # legacy assets
        verts, res = _legacy_terrain(export_dir, meta)
        if (export_dir / "buildings.obj").exists():
            try:
                bpy.ops.wm.obj_import(filepath=str(export_dir / "buildings.obj"))
            except AttributeError:  # Blender < 4
                bpy.ops.import_scene.obj(filepath=str(export_dir / "buildings.obj"))

    # ---- forests: dupli-vert instancing by type ----
    trees = json.loads((export_dir / "trees.json").read_text())["trees"]
    ground_at = _ground_lookup(export_dir, meta, verts, res)
    for kind, maker in (("broadleaf", _broadleaf_template), ("conifer", _conifer_template)):
        pts = [t for t in trees if t["type"] == kind]
        if not pts:
            continue
        cloud = bpy.data.meshes.new(f"forest_{kind}_points")
        cloud.from_pydata([(t["e"], t["n"], ground_at(t["e"], t["n"])) for t in pts], [], [])
        cloud_obj = bpy.data.objects.new(f"Forest_{kind}", cloud)
        bpy.context.collection.objects.link(cloud_obj)
        template = maker(sum(t["height"] for t in pts) / len(pts))
        template.parent = cloud_obj
        cloud_obj.instance_type = "VERTS"

    # ---- sun at the real solar position + simple sky ----
    az = meta.get("sun_azimuth_deg", 160.0)
    el = meta.get("sun_elevation_deg", 55.0)
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 4.0
    sun = bpy.data.objects.new("Sun", sun_data)
    # Blender sun points along -Z of its own frame; rotate so it comes FROM
    # (azimuth, elevation): tilt from zenith by (90-el), then yaw to azimuth.
    sun.rotation_euler = (radians(90.0 - el), 0.0, radians(180.0 - az))
    bpy.context.collection.objects.link(sun)
    world = bpy.data.worlds.new("Sky") if bpy.context.scene.world is None else bpy.context.scene.world
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.55, 0.70, 0.90, 1.0)
        bg.inputs[1].default_value = 1.0

    print(f"dronecv scene assembled: {len(trees)} trees, attribution: {meta['attribution']}")
    if blend_out is not None:
        blend_out.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_out))
        print(f"saved {blend_out}")


def _legacy_terrain(export_dir: Path, meta: dict):
    res = meta["terrain_resolution"]
    raw = (export_dir / "terrain.raw").read_bytes()
    heights = struct.unpack(f"<{res * res}H", raw)
    ex, en = meta["extent_e_m"], meta["extent_n_m"]
    e0, n0 = meta["e0"], meta["n0"]
    verts = []
    for r in range(res):
        for c in range(res):
            h = meta["height_min_m"] + heights[r * res + c] / 65535.0 * meta["height_scale_m"]
            verts.append((e0 + c * ex / (res - 1), n0 + r * en / (res - 1), h))
    faces = []
    for r in range(res - 1):
        for c in range(res - 1):
            a = r * res + c
            faces.append((a, a + 1, a + res + 1, a + res))
    tmesh = bpy.data.meshes.new("terrain")
    tmesh.from_pydata(verts, [], faces)
    terrain = bpy.data.objects.new("Terrain", tmesh)
    bpy.context.collection.objects.link(terrain)
    return verts, res


def _ground_lookup(export_dir: Path, meta: dict, verts, res):
    """Terrain height at (e, n) from terrain.raw (works for both paths)."""
    if verts is None:
        res = meta["terrain_resolution"]
        raw = (export_dir / "terrain.raw").read_bytes()
        heights = struct.unpack(f"<{res * res}H", raw)
        ex, en = meta["extent_e_m"], meta["extent_n_m"]
        e0, n0 = meta["e0"], meta["n0"]

        def f(e, n):
            c = min(res - 1, max(0, round((e - e0) / ex * (res - 1))))
            r = min(res - 1, max(0, round((n - n0) / en * (res - 1))))
            h = heights[int(r) * res + int(c)]
            return meta["height_min_m"] + h / 65535.0 * meta["height_scale_m"]

        return f

    ex, en = meta["extent_e_m"], meta["extent_n_m"]
    e0, n0 = meta["e0"], meta["n0"]

    def f(e, n):
        c = min(res - 1, max(0, round((e - e0) / ex * (res - 1))))
        r = min(res - 1, max(0, round((n - n0) / en * (res - 1))))
        return verts[int(r) * res + int(c)][2]

    return f


def _material(name: str, rgb):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.9
    return mat


def _conifer_template(height: float):
    bpy.ops.mesh.primitive_cone_add(radius1=height * 0.25, depth=height, location=(0, 0, height / 2))
    obj = bpy.context.object
    obj.name = "tree_conifer"
    obj.data.materials.append(_material("conifer", (0.10, 0.22, 0.12)))
    return obj


def _broadleaf_template(height: float):
    bpy.ops.mesh.primitive_ico_sphere_add(radius=height * 0.35, location=(0, 0, height * 0.7))
    obj = bpy.context.object
    obj.name = "tree_broadleaf"
    obj.data.materials.append(_material("broadleaf", (0.16, 0.32, 0.12)))
    return obj


if __name__ == "__main__":
    main()
