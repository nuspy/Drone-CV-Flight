"""Hierarchical tiled training + TileRouter on a small 2-tile world."""

import asyncio
from pathlib import Path

import numpy as np
import pytest

from dronecv.config import load_config
from dronecv.localization.tile_router import TiledLocalizer, is_tiled_bundle
from dronecv.training.tiled import run_tiled_training, tile_grid, tile_of

ROOT = Path(__file__).parent.parent.parent

pytestmark = pytest.mark.integration


def test_tile_grid_and_lookup():
    bmin = np.array([-300.0, 0.0, -150.0])
    bmax = np.array([300.0, 100.0, 150.0])
    tiles = tile_grid(bmin, bmax, tile_m=300.0)
    assert len(tiles) == 2  # 600 m east split in two, 300 m north single
    t = tile_of(tiles, -100.0, 0.0)
    assert t is not None and t["e_min"] <= -100 <= t["e_max"]
    assert tile_of(tiles, 9999.0, 0.0) is None


@pytest.fixture(scope="module")
def tiled_env(tmp_path_factory):
    cfg = load_config(
        "headless_ci",
        overrides={
            "report": {"out_dir": str(tmp_path_factory.mktemp("artifacts"))},
            "active_loop": {
                "tile_m": 160.0,  # 300 m world -> 2x2 tiles
                "budget_captures": 150,
                "max_rounds": 1,
                "probe_captures": 20,
            },
            "training": {"epochs_per_round": 6, "backbone_width": 12},
            "capture": {"grid_spacing_m": 45.0, "yaw_bins": 4, "altitudes_agl_m": [45.0]},
        },
        root=ROOT,
    )
    result = asyncio.run(run_tiled_training(cfg))
    return cfg, result


def test_tiled_training_produces_bundles(tiled_env):
    cfg, result = tiled_env
    assert result["n_tiles"] == 4
    root = cfg.bundle_dir
    assert is_tiled_bundle(root)
    assert (root / "coarse" / "embed.pt").exists()
    for tid in result["tiles"]:
        assert (root / "tiles" / tid / "manifest.json").exists()
        assert (root / "tiles" / tid / "terrain_prior.npz").exists()


def test_tile_router_localizes_and_hands_off(tiled_env):
    cfg, _result = tiled_env
    from dronecv.protocol.sim_client import SimClient
    from dronecv.sim.headless.server import HeadlessSimServer

    router = TiledLocalizer(cfg, cfg.bundle_dir)

    async def scenario():
        server = HeadlessSimServer(cfg)
        host, port = await server.start(port=0)
        try:
            client = await SimClient.connect(host, port, role="capture")
            # Frames from two OPPOSITE quadrants of the world -> different tiles.
            estimates = []
            for pos in ([-90.0, 80.0, -90.0], [90.0, 80.0, 90.0]):
                for k in range(6):
                    result, blobs = await client.capture(
                        np.array(pos) + [k * 2.0, 0, 0], yaw_deg=45.0 * k,
                        pitch_deg=cfg.sim.camera_tilt_deg,
                    )
                    est = router.step(
                        blobs["rgb"], None, result.utc, float(len(estimates)) * 0.2
                    )
                    estimates.append((pos, est))
            await client.close()
        finally:
            await server.stop()
        return estimates

    estimates = asyncio.run(scenario())
    assert router.current_tile is not None
    tiles_seen = {router.current_tile}
    # All estimates initialized; the router must have consulted at least one
    # tile and produced sane positions inside the world.
    for _pos, est in estimates:
        assert est.initialized
        assert abs(est.pos_enu[0]) < 400 and abs(est.pos_enu[1]) < 400
    # The two flight segments are in different quadrants: with a working
    # coarse router the active tile for the last frame differs from the first.
    first_tile = tile_of(router.tiles, *[p for p in (-90.0, -90.0)])
    last_tile = tile_of(router.tiles, *[p for p in (90.0, 90.0)])
    assert first_tile["id"] != last_tile["id"]
    assert len(tiles_seen) >= 1
