"""Capture -> train -> evaluate on a tiny world (CPU, minutes)."""

import asyncio
from pathlib import Path

import numpy as np
import pytest

from dronecv.capture import planner
from dronecv.capture.collector import Collector, open_sim_client
from dronecv.capture.dataset import CaptureDataset
from dronecv.config import load_config
from dronecv.sim.headless.server import HeadlessSimServer
from dronecv.training.evaluator import evaluate_bundle
from dronecv.training.trainer import train_models

ROOT = Path(__file__).parent.parent.parent

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tiny_cfg(tmp_path_factory):
    cfg = load_config(
        "headless_ci",
        overrides={
            "training": {"epochs_per_round": 25, "backbone_width": 12, "batch_size": 32},
            "capture": {"grid_spacing_m": 45.0, "yaw_bins": 4, "altitudes_agl_m": [45.0]},
            "report": {"out_dir": str(tmp_path_factory.mktemp("artifacts"))},
        },
        root=ROOT,
    )
    return cfg


@pytest.fixture(scope="module")
def trained(tiny_cfg):
    """Capture a small dataset and train one round."""

    async def scenario():
        server = HeadlessSimServer(tiny_cfg)
        host, port = await server.start(port=0)
        tiny_cfg.sim.host, tiny_cfg.sim.port = host, port
        try:
            client, anchor = await open_sim_client(tiny_cfg)
            collector = Collector(tiny_cfg, client, anchor)
            info = await collector.env_info()
            bmin, bmax = np.array(info.bounds_min_sim), np.array(info.bounds_max_sim)
            poses = planner.grid_plan(bmin, bmax, tiny_cfg.capture)
            poses = planner.shuffle_and_cap(poses, 180, seed=0)
            await collector.collect(poses, tiny_cfg.dataset_dir)
            await client.close()
        finally:
            await server.stop()

    asyncio.run(scenario())
    ds = CaptureDataset(tiny_cfg.dataset_dir)
    bundle, split = train_models(tiny_cfg, ds)
    return ds, bundle, split


def test_dataset_contents(trained):
    ds, _, _ = trained
    assert 100 <= len(ds) <= 180
    rec = ds.records[0]
    assert rec["sun_elevation_deg"] > 0
    assert abs(rec["pos_enu"][0] - rec["pos_sim"][0]) < 1e-6  # zero north offset
    img = ds.image(0)
    assert img.shape == (64, 64, 3)


def test_training_learns_capacity(tiny_cfg, trained):
    """On TRAIN cells the models must localize well — proves the training
    machinery learns. Generalization to unseen cells at this tiny dataset
    scale is the active loop's job (tested end-to-end in tests/e2e)."""
    ds, bundle, split = trained
    subset = split["train_idx"][:80]
    metrics = evaluate_bundle(tiny_cfg, bundle, ds, subset)
    assert metrics["retrieval"]["fix_err_h_p50_m"] < 35.0
    assert metrics["retrieval"]["recall_at_k"] > 0.6
    assert metrics["apr"]["pos_err_h_p50_m"] < 60.0


def test_holdout_metrics_machinery(tiny_cfg, trained):
    """Held-out evaluation runs, fits calibration, and produces the error map
    that drives targeted capture."""
    ds, bundle, split = trained
    metrics = evaluate_bundle(tiny_cfg, bundle, ds, split["val_idx"], fit_calibration=True)
    world_size = tiny_cfg.world.size_m
    assert metrics["apr"]["pos_err_h_p95_m"] < world_size  # sane at all
    assert 0.0 <= metrics["calibration"]["ece"] <= 1.0
    assert metrics["calibration"]["apr_sigma_scale"] > 0
    assert len(metrics["cell_p95_m"]) > 0
    assert bundle.manifest["calibration"]["apr_sigma_scale"] == metrics["calibration"]["apr_sigma_scale"]


def test_bundle_save_load_round_trip(tiny_cfg, trained, tmp_path):
    _, bundle, _ = trained
    from dronecv.training.bundle import ModelBundle

    bundle.save(tmp_path / "bundle")
    loaded = ModelBundle.load(tmp_path / "bundle")
    assert loaded.anchor.lat0 == pytest.approx(bundle.anchor.lat0)
    assert len(loaded.landmark_db) == len(bundle.landmark_db)
    import torch

    x = torch.rand(2, 3, 64, 64)
    torch.testing.assert_close(loaded.pose_net.predict(x)["pos_enu"], bundle.pose_net.predict(x)["pos_enu"])
