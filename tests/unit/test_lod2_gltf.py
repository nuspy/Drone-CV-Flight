"""LoD2 roof meshes + GLB writer validity."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from shapely.geometry import Polygon

from dronecv.gis.export.gltf import GlbBuilder, window_atlas_png
from dronecv.gis.export.mesh import building_meshes


def _rect(w=10.0, l=24.0):  # noqa: E741
    return Polygon([(0, 0), (l, 0), (l, w), (0, w)])


def _zmax(mesh):
    return max(v[2] for v in mesh.vertices)


class TestRoofs:
    def test_flat_roof_at_height(self):
        walls, roof = building_meshes(_rect(), 100.0, 12.0, "flat", None)
        assert _zmax(roof) == pytest.approx(112.0)
        assert _zmax(walls) == pytest.approx(112.0)
        assert len(walls.uvs) == len(walls.vertices)  # facades carry UVs

    def test_gabled_ridge_and_walls(self):
        walls, roof = building_meshes(_rect(), 0.0, 12.0, "gabled", 4.0)
        assert _zmax(roof) == pytest.approx(12.0)  # ridge at total height
        # eaves at wall_top
        eaves = sorted({round(v[2], 3) for v in roof.vertices})
        assert eaves[0] == pytest.approx(8.0)
        # gable triangles reach the ridge in the walls mesh
        assert _zmax(walls) == pytest.approx(12.0)

    @pytest.mark.parametrize("shape", ["hipped", "pyramidal", "skillion"])
    def test_shaped_roofs_bounded(self, shape):
        walls, roof = building_meshes(_rect(), 0.0, 10.0, shape, 3.0)
        assert _zmax(roof) == pytest.approx(10.0)
        assert roof.faces and walls.faces

    def test_default_narrow_residential_is_gabled(self):
        _, roof = building_meshes(_rect(w=9.0, l=18.0), 0.0, 9.0, None, None, "residential")
        zs = {round(v[2], 2) for v in roof.vertices}
        assert len(zs) > 1, "expected a sloped (gabled) roof, got flat"

    def test_irregular_footprint_falls_back_flat(self):
        poly = Polygon([(0, 0), (30, 0), (30, 10), (18, 10), (18, 22), (0, 22)])  # L-shape
        _, roof = building_meshes(poly, 0.0, 10.0, "gabled", 3.0, "residential")
        zs = {round(v[2], 2) for v in roof.vertices}
        assert zs == {10.0}, f"L-shape must get a flat roof, got z levels {zs}"

    def test_roof_never_below_ground(self):
        _, roof = building_meshes(_rect(), 0.0, 3.0, "gabled", 10.0)
        assert min(v[2] for v in roof.vertices) >= 0.0


class TestGlb:
    def _parse(self, path):
        raw = path.read_bytes()
        magic, version, total = struct.unpack_from("<III", raw, 0)
        assert magic == 0x46546C67 and version == 2 and total == len(raw)
        jlen, jtype = struct.unpack_from("<II", raw, 12)
        assert jtype == 0x4E4F534A
        gltf = json.loads(raw[20 : 20 + jlen])
        blen, btype = struct.unpack_from("<II", raw, 20 + jlen)
        assert btype == 0x004E4942
        assert gltf["buffers"][0]["byteLength"] <= blen
        return gltf

    def test_glb_roundtrip(self, tmp_path):
        glb = GlbBuilder()
        mat = glb.add_material("wall", (0.8, 0.75, 0.65), texture_png=window_atlas_png())
        tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        glb.add_mesh("t", tri, np.array([[0, 1, 2]]), mat,
                     uvs=np.zeros((3, 2), np.float32))
        mat2 = glb.add_material("roof", (0.6, 0.4, 0.3))
        glb.add_mesh("t2", tri + 5, np.array([[0, 1, 2]]), mat2,
                     colors=np.ones((3, 3), np.float32))
        path = glb.save(tmp_path / "s.glb")
        gltf = self._parse(path)
        assert len(gltf["meshes"]) == 2 and len(gltf["materials"]) == 2
        assert gltf["images"] and gltf["textures"]
        prims = gltf["meshes"][0]["primitives"][0]
        assert "TEXCOORD_0" in prims["attributes"]
        assert "COLOR_0" in gltf["meshes"][1]["primitives"][0]["attributes"]
        # Z-up -> Y-up root rotation present
        assert gltf["nodes"][0]["rotation"][0] == pytest.approx(-0.7071067811865476)
        # position accessor has min/max (required by spec)
        pa = prims["attributes"]["POSITION"]
        assert "min" in gltf["accessors"][pa]

    def test_atlas_is_png(self):
        assert window_atlas_png()[:8] == b"\x89PNG\r\n\x1a\n"
