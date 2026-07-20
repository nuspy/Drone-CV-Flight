"""Assemble the dronecv GIS scene export inside Blender.

Usage (Blender 3.6+ / 4.x):
    blender --python blender_build_scene.py -- /path/to/export_dir

Builds: terrain mesh from terrain.raw (orography), buildings from
buildings.obj, forests as dupli-vert instanced trees (density and typology
from trees.json — conifers get cones, broadleaves get icospheres on trunks).
Everything in Z-up ENU meters, origin at the AOI anchor.
"""

import json
import struct
import sys
from pathlib import Path

import bpy  # noqa: F401 (available inside Blender)


def main() -> None:
    export_dir = Path(sys.argv[sys.argv.index("--") + 1])
    meta = json.loads((export_dir / "scene_meta.json").read_text())
    res = meta["terrain_resolution"]

    # ---- clean scene ----
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # ---- terrain from RAW16 ----
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

    # ---- buildings ----
    if (export_dir / "buildings.obj").exists():
        try:
            bpy.ops.wm.obj_import(filepath=str(export_dir / "buildings.obj"))
        except AttributeError:  # Blender < 4
            bpy.ops.import_scene.obj(filepath=str(export_dir / "buildings.obj"))

    # ---- forests: dupli-vert instancing by type ----
    trees = json.loads((export_dir / "trees.json").read_text())["trees"]
    ground_at = _ground_lookup(verts, res, e0, n0, ex, en)
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

    print(f"dronecv scene assembled: {len(trees)} trees, attribution: {meta['attribution']}")


def _ground_lookup(verts, res, e0, n0, ex, en):
    def f(e, n):
        c = min(res - 1, max(0, round((e - e0) / ex * (res - 1))))
        r = min(res - 1, max(0, round((n - n0) / en * (res - 1))))
        return verts[int(r) * res + int(c)][2]

    return f


def _conifer_template(height: float):
    bpy.ops.mesh.primitive_cone_add(radius1=height * 0.25, depth=height, location=(0, 0, height / 2))
    obj = bpy.context.object
    obj.name = "tree_conifer"
    return obj


def _broadleaf_template(height: float):
    bpy.ops.mesh.primitive_ico_sphere_add(radius=height * 0.35, location=(0, 0, height * 0.7))
    obj = bpy.context.object
    obj.name = "tree_broadleaf"
    return obj


if __name__ == "__main__":
    main()
