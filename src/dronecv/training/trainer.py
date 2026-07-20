"""Trains the retrieval embedding and the APR pose network on a dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from rich.progress import Progress
from torch.utils.data import DataLoader

from dronecv.capture.dataset import CaptureDataset, TorchCaptureView
from dronecv.config import Config
from dronecv.geo.anchor import GeoAnchor
from dronecv.models.pose_net import PoseNet
from dronecv.models.retrieval import (
    EmbeddingNet,
    LandmarkDB,
    PositivePairBatchSampler,
    batch_hard_triplet_loss,
    embed_images,
    positive_mask,
)
from dronecv.protocol.messages import PROTOCOL_VERSION
from dronecv.training.bundle import BUNDLE_VERSION, ModelBundle
from dronecv.util.logging import get_logger
from dronecv.util.seeding import seed_everything

log = get_logger("dronecv.train")


def _pos_scale(ds: CaptureDataset) -> float:
    pos = np.abs(ds.positions_enu())
    return float(max(pos.max() * 1.3, 50.0))


def train_models(
    cfg: Config,
    ds: CaptureDataset,
    warm_start: ModelBundle | None = None,
    device: str = "cpu",
) -> tuple[ModelBundle, dict]:
    """One training round over the current dataset. Returns bundle + info."""
    seed_everything(cfg.env.seed)
    t = cfg.training
    train_idx, val_idx = ds.spatial_split(t.val_cell_m, t.val_fraction, seed=cfg.env.seed)
    if len(train_idx) < 8:
        raise RuntimeError(f"dataset too small to train ({len(ds)} captures)")

    pos_scale = warm_start.pose_net.pos_scale_m if warm_start else _pos_scale(ds)
    embed_net = EmbeddingNet(t.backbone_width, t.embedding_dim).to(device)
    pose_net = PoseNet(t.backbone_width, pos_scale).to(device)
    if warm_start is not None:
        embed_net.load_state_dict(warm_start.embed_net.state_dict())
        pose_net.load_state_dict(warm_start.pose_net.state_dict())

    from dronecv.vision.preprocess_filter import FilterSpec

    filter_spec = FilterSpec(
        mode=t.filter_mode, edge_weight=t.filter_edge_weight, clahe=t.filter_clahe
    )
    view = TorchCaptureView(ds, train_idx, augment=True, seed=cfg.env.seed,
                            filter_spec=filter_spec)
    last_stats: dict = {}

    # --- retrieval embedding: anchor+positive pair batches ---
    pair_sampler = PositivePairBatchSampler(
        ds.positions_enu()[train_idx],
        ds.headings_deg()[train_idx],
        t.pos_radius_m,
        t.pos_yaw_deg,
        t.batch_size,
        seed=cfg.env.seed,
    )
    pair_loader = DataLoader(view, batch_sampler=pair_sampler, num_workers=0)
    opt_e = torch.optim.AdamW(embed_net.parameters(), lr=t.lr, weight_decay=1e-4)
    steps_e = max(1, t.epochs_per_round * max(1, len(pair_sampler)))
    sched_e = torch.optim.lr_scheduler.CosineAnnealingLR(opt_e, T_max=steps_e)
    embed_net.train()
    with Progress() as progress:
        bar = progress.add_task("train retrieval", total=steps_e)
        for _epoch in range(t.epochs_per_round):
            for img, pos, heading_sc, _idx in pair_loader:
                img, pos = img.to(device), pos.to(device)
                heading_deg = torch.rad2deg(torch.atan2(heading_sc[:, 0], heading_sc[:, 1])).to(device)
                emb = embed_net(img)
                mask = positive_mask(pos, heading_deg, t.pos_radius_m, t.pos_yaw_deg)
                loss_triplet = batch_hard_triplet_loss(emb, mask, t.triplet_margin)
                opt_e.zero_grad(set_to_none=True)
                loss_triplet.backward()
                torch.nn.utils.clip_grad_norm_(embed_net.parameters(), 5.0)
                opt_e.step()
                sched_e.step()
                last_stats["triplet"] = float(loss_triplet.detach())
                progress.advance(bar)

    # --- APR: plain shuffled batches (pair batches are too correlated) ---
    apr_loader = DataLoader(
        view, batch_size=t.batch_size, shuffle=True, num_workers=0,
        drop_last=len(view) > t.batch_size,
    )
    opt_p = torch.optim.AdamW(pose_net.parameters(), lr=t.lr, weight_decay=1e-4)
    steps_p = max(1, t.epochs_per_round * max(1, len(apr_loader)))
    sched_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=steps_p)
    pose_net.train()
    with Progress() as progress:
        bar = progress.add_task("train APR", total=steps_p)
        for _epoch in range(t.epochs_per_round):
            for img, pos, heading_sc, _idx in apr_loader:
                img, pos, heading_sc = img.to(device), pos.to(device), heading_sc.to(device)
                out = pose_net(img)
                loss_pose, stats = pose_net.loss(out, pos, heading_sc)
                opt_p.zero_grad(set_to_none=True)
                loss_pose.backward()
                torch.nn.utils.clip_grad_norm_(pose_net.parameters(), 5.0)
                opt_p.step()
                sched_p.step()
                last_stats.update(stats)
                progress.advance(bar)

    log.info(f"train done: {last_stats}")

    # Landmark DB from the training captures (clean images, no augmentation).
    embed_net.eval()
    clean_view = TorchCaptureView(ds, train_idx, augment=False, filter_spec=filter_spec)
    imgs = torch.stack([clean_view[i][0] for i in range(len(clean_view))])
    embeddings = embed_images(embed_net, imgs.to(device))
    positions = ds.positions_enu()[train_idx]
    headings = ds.headings_deg()[train_idx]
    db = LandmarkDB(embeddings, positions, headings)

    # Terrain elevation prior from the capture ground probes (all captures,
    # not only train: ground truth terrain is not a learned quantity).
    from dronecv.localization.terrain import TerrainPrior

    ground_up = np.array([r["ground_y_sim"] for r in ds.records], dtype=float)
    terrain = TerrainPrior.from_records(ds.positions_enu(), ground_up, t.val_cell_m)

    anchor = GeoAnchor.from_dict(ds.info["anchor"])
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "env": cfg.env.name,
        "anchor": anchor.to_dict(),
        "camera": ds.info.get("camera"),
        "hyperparams": {
            "backbone_width": t.backbone_width,
            "embedding_dim": t.embedding_dim,
            "pos_scale_m": pos_scale,
        },
        "dataset": {"n_captures": len(ds), "n_train": len(train_idx), "n_val": len(val_idx)},
        "preprocess_filter": filter_spec.to_dict(),
        "calibration": {"apr_sigma_scale": 1.0},
        "metrics": {},
    }
    bundle = ModelBundle(embed_net.cpu(), pose_net.cpu(), db, manifest, terrain)
    return bundle, {"train_idx": train_idx, "val_idx": val_idx, "last_stats": last_stats}


def load_dataset(cfg: Config) -> CaptureDataset:
    root = cfg.dataset_dir
    if not (Path(root) / "meta.jsonl").exists():
        raise FileNotFoundError(f"no dataset at {root} — run: dronecv capture --env {cfg.env.name}")
    return CaptureDataset(root)
