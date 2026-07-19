"""Hierarchical training for large areas (>~5 km): per-tile fine bundles plus
a coarse global place-recognition model that answers "which tile am I over?".

    artifacts/<env>/model_bundle/
        tiling.json                tile grid geometry
        coarse/                    EmbeddingNet + LandmarkDB over the whole AOI
        tiles/t<i>_<j>/            standard fine bundle per tile (active loop)

Enabled by `active_loop.tile_m > 0` in the env config; `dronecv train` and
`run-all` pick it up automatically, and the localizer routes through
`TiledLocalizer` when it finds tiling.json in the bundle.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from dronecv.capture import planner
from dronecv.capture.collector import Collector, open_sim_client
from dronecv.capture.dataset import CaptureDataset, TorchCaptureView
from dronecv.commands.common import sim_endpoint
from dronecv.config import Config
from dronecv.models.retrieval import (
    EmbeddingNet,
    LandmarkDB,
    PositivePairBatchSampler,
    batch_hard_triplet_loss,
    embed_images,
    positive_mask,
)
from dronecv.training.active_loop import run_active_loop
from dronecv.util.logging import get_logger

log = get_logger("dronecv.tiled")

COARSE_WIDTH = 16
COARSE_DIM = 64


def tile_grid(bmin: np.ndarray, bmax: np.ndarray, tile_m: float) -> list[dict]:
    """Split the flyable volume into tiles (sim/ENU frame)."""
    e0, e1 = float(bmin[0]), float(bmax[0])
    n0, n1 = float(bmin[2]), float(bmax[2])
    n_e = max(1, int(np.ceil((e1 - e0) / tile_m)))
    n_n = max(1, int(np.ceil((n1 - n0) / tile_m)))
    tiles = []
    for i in range(n_e):
        for j in range(n_n):
            tiles.append(
                {
                    "id": f"t{i}_{j}",
                    "e_min": e0 + i * (e1 - e0) / n_e,
                    "e_max": e0 + (i + 1) * (e1 - e0) / n_e,
                    "n_min": n0 + j * (n1 - n0) / n_n,
                    "n_max": n0 + (j + 1) * (n1 - n0) / n_n,
                }
            )
    return tiles


def tile_of(tiles: list[dict], e: float, n: float) -> dict | None:
    for t in tiles:
        if t["e_min"] <= e <= t["e_max"] and t["n_min"] <= n <= t["n_max"]:
            return t
    return None


async def run_tiled_training(cfg: Config) -> dict:
    """Coarse capture+train over the whole AOI, then per-tile active loops."""
    tile_m = cfg.active_loop.tile_m
    assert tile_m > 0
    bundle_root = cfg.bundle_dir
    bundle_root.mkdir(parents=True, exist_ok=True)

    # ---- discover bounds + coarse capture ----
    async with sim_endpoint(cfg) as (host, port):
        cfg.sim.host, cfg.sim.port = host, port
        client, anchor = await open_sim_client(cfg)
        try:
            collector = Collector(cfg, client, anchor)
            info = await collector.env_info()
            bmin, bmax = np.array(info.bounds_min_sim), np.array(info.bounds_max_sim)
            tiles = tile_grid(bmin, bmax, tile_m)

            coarse_cfg = copy.deepcopy(cfg.capture)
            coarse_cfg.grid_spacing_m = max(tile_m / 6.0, cfg.capture.grid_spacing_m * 2)
            coarse_cfg.yaw_bins = 4
            coarse_cfg.altitudes_agl_m = [max(cfg.capture.altitudes_agl_m)]
            poses = planner.grid_plan(bmin, bmax, coarse_cfg)
            poses = planner.shuffle_and_cap(poses, 1500, cfg.env.seed)
            coarse_ds_dir = cfg.artifacts_dir / "coarse_dataset"
            if not (coarse_ds_dir / "meta.jsonl").exists():
                await collector.collect(poses, coarse_ds_dir, "coarse capture")
        finally:
            await client.close()

    # ---- coarse retrieval model (embedding only: it answers WHERE-ish) ----
    coarse = _train_coarse(cfg, CaptureDataset(coarse_ds_dir))
    coarse_dir = bundle_root / "coarse"
    coarse_dir.mkdir(parents=True, exist_ok=True)
    torch.save(coarse["net"].state_dict(), coarse_dir / "embed.pt")
    coarse["db"].save(coarse_dir / "landmark_db.npz")
    (coarse_dir / "manifest.json").write_text(json.dumps({
        "backbone_width": COARSE_WIDTH, "embedding_dim": COARSE_DIM,
        "anchor": json.loads((coarse_ds_dir / "info.json").read_text())["anchor"],
    }))

    # ---- per-tile fine bundles ----
    results = {}
    for tile in tiles:
        t_cfg = copy.deepcopy(cfg)
        t_cfg.env.name = f"{cfg.env.name}/tiles/{tile['id']}"
        t_bmin = np.array([tile["e_min"], bmin[1], tile["n_min"]])
        t_bmax = np.array([tile["e_max"], bmax[1], tile["n_max"]])
        log.info(f"training tile {tile['id']} "
                 f"({tile['e_max']-tile['e_min']:.0f} x {tile['n_max']-tile['n_min']:.0f} m)")
        results[tile["id"]] = await run_active_loop(t_cfg, bounds_sim=(t_bmin, t_bmax))
        # Standard bundle location for the router.
        src = t_cfg.bundle_dir
        dst = bundle_root / "tiles" / tile["id"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            import shutil

            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    (bundle_root / "tiling.json").write_text(json.dumps({"tile_m": tile_m, "tiles": tiles}, indent=1))
    return {
        "tiles": {tid: {"stop_reason": r["stop_reason"], "captures_used": r["captures_used"]}
                  for tid, r in results.items()},
        "n_tiles": len(tiles),
        "stop_reason": "tiled",
        "rounds": max(r["rounds"] for r in results.values()),
        "captures_used": sum(r["captures_used"] for r in results.values()),
        "final_metrics": next(iter(results.values()))["final_metrics"],
    }


def _train_coarse(cfg: Config, ds: CaptureDataset) -> dict:
    net = EmbeddingNet(COARSE_WIDTH, COARSE_DIM)
    idx = list(range(len(ds)))
    view = TorchCaptureView(ds, idx, augment=True, seed=cfg.env.seed)
    sampler = PositivePairBatchSampler(
        ds.positions_enu(), ds.headings_deg(),
        radius_m=cfg.active_loop.tile_m / 4.0, yaw_window_deg=100.0,
        batch_size=32, seed=cfg.env.seed,
    )
    loader = DataLoader(view, batch_sampler=sampler, num_workers=0)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    net.train()
    for _epoch in range(6):
        for img, pos, heading_sc, _i in loader:
            heading = torch.rad2deg(torch.atan2(heading_sc[:, 0], heading_sc[:, 1]))
            emb = net(img)
            mask = positive_mask(pos, heading, cfg.active_loop.tile_m / 4.0, 100.0)
            loss = batch_hard_triplet_loss(emb, mask, 0.3)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    clean = TorchCaptureView(ds, idx, augment=False)
    imgs = torch.stack([clean[i][0] for i in range(len(clean))])
    db = LandmarkDB(embed_images(net, imgs), ds.positions_enu(), ds.headings_deg())
    log.info(f"coarse model trained on {len(ds)} sparse captures")
    return {"net": net, "db": db}
