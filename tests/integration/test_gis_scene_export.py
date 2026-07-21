"""POI/vegetation/bridge features + Unity/Blender scene export, on fixtures."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from dronecv.gis.geometry import BBox
from dronecv.gis.pipeline import BuildSources, build_environment
from dronecv.gis.providers.buildings import OverpassBuildings
from dronecv.gis.providers.dem import CopernicusDem
from dronecv.gis.providers.landcover import GROUND_CLASSES, OverpassLandcover
from dronecv.gis.providers.poi import OverpassPoi
from dronecv.gis.store import GisStore
from dronecv.gis.world import GisWorld

ROOT = Path(__file__).parent.parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "gis"
BBOX = BBox(south=43.3145, west=11.3225, north=43.3255, east=11.3375)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("gisenv2")
    gis_dir = build_environment(
        BBOX, "gis_scene", out_root=root / "artifacts", configs_root=root,
        sources=BuildSources(
            dem=CopernicusDem(tile_dir=FIXTURES),
            buildings=OverpassBuildings(fixture_path=FIXTURES / "overpass_buildings.json"),
            landcover=OverpassLandcover(fixture_path=FIXTURES / "overpass_landcover.json"),
            poi=OverpassPoi(fixture_path=FIXTURES / "overpass_poi.json"),
        ),
        res_m=2.0,
    )
    return gis_dir


def test_poi_archetypes_and_parts(built):
    store = GisStore.open(built)
    stats = store.meta.stats
    assert stats["n_pois"] == 2
    assert stats["n_building_parts"] == 1
    assert stats["n_archetypes"] >= 1  # the tower got its spire
    # Spire: tower raster max exceeds the tagged 55 m (archetype adds ~45%).
    assert float(np.asarray(store.build_h).max()) > 60.0


def test_vegetation_layer(built):
    store = GisStore.open(built)
    veg = np.asarray(store.veg_h)
    cls = np.asarray(store.class_id)
    assert store.meta.stats["forest_texels"] > 500
    assert (cls == GROUND_CLASSES["forest_broadleaf"]).any()
    assert (cls == GROUND_CLASSES["forest_conifer"]).any()
    # Conifers taller than broadleaves on average; gaps exist.
    broad = veg[cls == GROUND_CLASSES["forest_broadleaf"]]
    conif = veg[cls == GROUND_CLASSES["forest_conifer"]]
    assert conif[conif > 0].mean() > broad[broad > 0].mean()
    assert (broad == 0).mean() > 0.05  # clearings
    # Canopy is part of the flyable geometry.
    world = GisWorld.open(built)
    rr, cc = np.nonzero(veg > 5.0)
    m = store.meta
    e = m.e0 + (cc[0] + 0.5) * m.res_m
    n = m.n0 + (rr[0] + 0.5) * m.res_m
    h_forest = float(world.height_at(np.array(e), np.array(n)))
    ground = float(store.ground[rr[0], cc[0]])
    assert h_forest - ground > 5.0


def test_bridge_and_rail(built):
    store = GisStore.open(built)
    assert store.meta.stats["n_bridges"] == 1
    assert store.meta.stats["n_rail_texels"] > 50


def test_scene_export_unity_blender_assets(built, tmp_path):
    from dronecv.gis.export.scene_export import export_scene

    out = export_scene(built, tmp_path / "scene", terrain_resolution=129)
    meta = json.loads((out / "scene_meta.json").read_text())

    raw = (out / "terrain.raw").read_bytes()
    assert len(raw) == 129 * 129 * 2
    heights = struct.unpack(f"<{129 * 129}H", raw)
    assert max(heights) > min(heights)  # orography present

    from PIL import Image

    splat = np.array(Image.open(out / "splatmap.png"))
    assert splat.shape == (129, 129, 4)
    assert (splat[..., 1] > 0).any()  # vegetation layer
    assert (splat[..., 3] > 0).any()  # water

    trees = json.loads((out / "trees.json").read_text())["trees"]
    assert meta["n_trees"] == len(trees) > 50
    types = {t["type"] for t in trees}
    assert types == {"broadleaf", "conifer"}

    obj_text = (out / "buildings.obj").read_text()
    assert meta["n_building_meshes"] > 5
    assert obj_text.count("v ") > 100 and obj_text.count("f ") > 100
    # per-class materials for DCC/Unity: OBJ references a .mtl with usemtl groups
    assert "mtllib buildings.mtl" in obj_text and "usemtl " in obj_text
    assert (out / "buildings.mtl").read_text().count("newmtl ") >= 2

    # data the NATIVE Unity importer needs (no glTF importer package required)
    assert (out / "buildings.json").exists() and (out / "palette.json").exists()

    assert (out / "blender_build_scene.py").exists()
    assert "OpenStreetMap" in " ".join(meta["attribution"])


def test_mesh_primitives():
    from shapely.geometry import Polygon

    from dronecv.gis.export import mesh as m

    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    prism = m.extrude_polygon(square, 0.0, 12.0)
    assert len(prism.vertices) == 8
    assert len(prism.faces) == 4 + 8  # 2 caps x 2 tris + 4 walls x 2
    holed = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)],
                    [[(8, 8), (12, 8), (12, 12), (8, 12)]])
    hp = m.extrude_polygon(holed, 0.0, 9.0)
    assert len(hp.vertices) == 16
    sp = m.spire(0, 0, 10, 3, 8)
    assert len(sp.faces) == 12
    dm = m.dome(0, 0, 10, 5)
    assert len(dm.vertices) > 40
