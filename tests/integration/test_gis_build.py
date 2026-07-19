"""GIS pipeline on offline fixtures: build a real-ish area, fly the raycaster
over it, check heights/classes/shadows/saliency and the env plumbing."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from dronecv.gis.geometry import BBox, anchor_for
from dronecv.gis.pipeline import BuildSources, build_environment
from dronecv.gis.providers.buildings import OverpassBuildings
from dronecv.gis.providers.dem import CopernicusDem
from dronecv.gis.providers.landcover import OverpassLandcover
from dronecv.gis.world import GisWorld

ROOT = Path(__file__).parent.parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "gis"

# Matches scripts/gen_gis_fixtures.py: ~1.2 x 1.1 km around (43.320, 11.330).
BBOX = BBox(south=43.3145, west=11.3225, north=43.3255, east=11.3375)

pytestmark = pytest.mark.integration


def _sources() -> BuildSources:
    return BuildSources(
        dem=CopernicusDem(tile_dir=FIXTURES),
        buildings=OverpassBuildings(fixture_path=FIXTURES / "overpass_buildings.json"),
        landcover=OverpassLandcover(fixture_path=FIXTURES / "overpass_landcover.json"),
    )


@pytest.fixture(scope="module")
def built_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("gisenv")
    gis_dir = build_environment(
        BBOX, "gis_fixture", out_root=root / "artifacts", configs_root=root,
        sources=_sources(), res_m=2.0,
    )
    return root, gis_dir


def test_store_contents(built_env):
    root, gis_dir = built_env
    world = GisWorld.open(gis_dir)
    m = world.store.meta
    assert 500 < m.width * m.res_m < 1500  # ~1.2 km
    # Terrain: fixture has a hill + slope, so relief must exist.
    assert m.stats["max_ground"] - m.stats["min_ground"] > 10.0
    # Buildings rasterized with the full height chain.
    assert m.stats["n_buildings"] >= 15
    assert m.stats["n_rasterized"] >= 15
    assert m.stats["n_with_height"] >= 13  # levels + explicit heights
    assert (np.asarray(world.store.build_h) > 0).any()
    # The 55 m tower is in the mosaic.
    assert float(np.asarray(world.store.build_h).max()) == pytest.approx(55.0, abs=1.0)


def test_world_heights_and_walls(built_env):
    _, gis_dir = built_env
    world = GisWorld.open(gis_dir)
    # Tower position: from fixture geometry it is ~East+North of center.
    build = np.asarray(world.store.build_h)
    r, c = np.unravel_index(np.argmax(build), build.shape)
    m = world.store.meta
    e = m.e0 + (c + 0.5) * m.res_m
    n = m.n0 + (r + 0.5) * m.res_m
    h_on = float(world.height_at(np.array(e), np.array(n)))
    h_off = float(world.height_at(np.array(e + 60.0), np.array(n)))
    assert h_on - h_off > 40.0  # the tower stands out of the terrain
    # Vertical wall: height drops within a couple of meters horizontally.
    profile = [float(world.height_at(np.array(e + dx), np.array(n))) for dx in range(0, 40, 2)]
    assert max(profile) - min(profile) > 40.0


def test_env_yaml_and_server_integration(built_env):
    root, gis_dir = built_env
    env_path = root / "configs" / "envs" / "gis_fixture.yaml"
    assert env_path.exists()
    cfg_yaml = yaml.safe_load(env_path.read_text())
    assert cfg_yaml["world"]["kind"] == "gis"
    assert abs(cfg_yaml["env"]["anchor"]["lat0"] - 43.32) < 0.01

    # The standard sim server + raycaster fly over the real area.
    import asyncio

    # Merge with the repo defaults (root only has the env file).
    import shutil

    from dronecv.config import load_config
    from dronecv.protocol.sim_client import SimClient
    from dronecv.sim.headless.server import HeadlessSimServer

    shutil.copy(ROOT / "configs" / "default.yaml", root / "configs" / "default.yaml")
    cfg = load_config("gis_fixture", root=root)

    async def scenario():
        server = HeadlessSimServer(cfg)
        host, port = await server.start(port=0)
        try:
            client = await SimClient.connect(host, port, role="capture")
            info = await client.env_info()
            assert info.bounds_max_sim[1] > 50.0
            world = server.world
            spawn = world.spawn_position(np.random.default_rng(0), agl_m=80.0)
            result, blobs = await client.capture(spawn, yaw_deg=45.0, pitch_deg=35.0)
            rgb, depth = blobs["rgb"], blobs["depth"]
            assert rgb.std() > 8  # textured scene, not a flat frame
            assert (depth > 0).mean() > 0.3
            await client.close()
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_saliency_prior(built_env):
    import json

    _, gis_dir = built_env
    saliency = json.loads((gis_dir / "saliency.json").read_text())
    dens = np.array(saliency["density_multiplier"])
    assert dens.shape[0] >= 1 and (dens >= 0.5).all() and (dens <= 2.2).all()
    meta = GisWorld.open(gis_dir).store.meta
    assert len(meta.orbit_points) >= 3  # tower + sheds picked as anchors


def test_shadow_height_inference():
    """Full shadow chain on a synthetic orthophoto with a KNOWN geometry:
    building of height H at known sun position -> shadow of length
    H / tan(elev) painted dark -> inference recovers H."""
    from dronecv.geo import celestial
    from dronecv.gis.providers.buildings import parse_overpass
    from dronecv.gis.providers.imagery import OrthoImage
    from dronecv.gis.shadow_heights import estimate_heights_from_shadows

    anchor = anchor_for(BBOX)
    utc = datetime(2026, 6, 21, 10, 0, tzinfo=UTC)
    sun = celestial.sun_position(utc, anchor.lat0, anchor.lon0)
    import math

    true_h = 20.0
    shadow_len = true_h / math.tan(math.radians(sun.elevation_deg))

    # One untagged square building at the ENU origin, ~24 m side.
    d = 0.0003
    lon_c, lat_c = BBOX.center[1], BBOX.center[0]
    fixture = {
        "elements": [
            {
                "type": "way", "id": 1, "tags": {"building": "yes"},
                "geometry": [
                    {"lat": lat_c, "lon": lon_c},
                    {"lat": lat_c, "lon": lon_c + d},
                    {"lat": lat_c + d * 0.75, "lon": lon_c + d},
                    {"lat": lat_c + d * 0.75, "lon": lon_c},
                    {"lat": lat_c, "lon": lon_c},
                ],
            }
        ]
    }
    buildings = parse_overpass(fixture)
    assert buildings[0].height_m is None

    # Paint the orthophoto the way a real shadow works: the footprint
    # silhouette swept along the shadow direction for shadow_len meters.
    res = 0.5
    size = 600
    gray = np.full((size, size), 0.8, dtype=np.float32)
    e0 = n0 = -size * res / 2
    az_sh = math.radians((sun.azimuth_deg + 180.0) % 360.0)
    d_e, d_n = math.sin(az_sh), math.cos(az_sh)
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    from dronecv.gis.raster import building_polygon

    poly = building_polygon(anchor, buildings[0])
    transform = Affine(res, 0.0, e0, 0.0, res, n0)  # row 0 = south, like OrthoImage
    mask = rasterize([(poly, 1)], out_shape=(size, size), transform=transform, dtype="uint8")
    shadow = np.zeros_like(mask)
    for t in np.arange(res / 2, shadow_len, res / 2):
        dr = int(round(d_n * t / res))
        dc = int(round(d_e * t / res))
        shadow |= np.roll(np.roll(mask, dr, axis=0), dc, axis=1)
    gray[shadow > 0] = 0.25
    gray[mask > 0] = 0.7  # sunlit roof

    ortho = OrthoImage(gray=gray, res_m=res, e0=e0, n0=n0, utc=utc)
    n_est = estimate_heights_from_shadows(buildings, anchor, ortho)
    assert n_est == 1
    assert buildings[0].height_source == "shadow"
    assert buildings[0].height_m == pytest.approx(true_h, rel=0.3)
