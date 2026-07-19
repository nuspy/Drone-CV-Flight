"""Single-photo localization, mobile export (ONNX parity) and the HTTP server.

Trains a tiny throwaway bundle once (shared fixture) and exercises the whole
photo path: capture a view at a KNOWN pose -> localize the photo -> compare;
export to ONNX -> assert torch/onnxruntime parity; POST the photo to the
FastAPI server -> same numbers.
"""

import asyncio
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronecv.capture import planner
from dronecv.capture.collector import Collector, open_sim_client
from dronecv.capture.dataset import CaptureDataset
from dronecv.config import load_config
from dronecv.localization.single_shot import SingleShotLocalizer, preprocess
from dronecv.sim.headless.server import HeadlessSimServer
from dronecv.training.trainer import train_models

ROOT = Path(__file__).parent.parent.parent

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def trained_env(tmp_path_factory):
    """Small trained bundle + a probe photo captured at a known pose."""
    cfg = load_config(
        "headless_ci",
        overrides={
            "training": {"epochs_per_round": 20, "backbone_width": 12},
            "capture": {"grid_spacing_m": 45.0, "yaw_bins": 4, "altitudes_agl_m": [45.0]},
            "report": {"out_dir": str(tmp_path_factory.mktemp("artifacts"))},
        },
        root=ROOT,
    )
    probe = {}

    async def scenario():
        server = HeadlessSimServer(cfg)
        host, port = await server.start(port=0)
        cfg.sim.host, cfg.sim.port = host, port
        try:
            client, anchor = await open_sim_client(cfg)
            collector = Collector(cfg, client, anchor)
            info = await collector.env_info()
            bmin, bmax = np.array(info.bounds_min_sim), np.array(info.bounds_max_sim)
            poses = planner.shuffle_and_cap(planner.grid_plan(bmin, bmax, cfg.capture), 170, 0)
            await collector.collect(poses, cfg.dataset_dir)
            # A "user photo": rendered at a station-adjacent pose the model saw
            # (this is the quick-test scenario: photo of a known area).
            pose = poses[3]
            ground = await collector.ground_y(pose.x, pose.z)
            result, blobs = await client.capture(
                np.array([pose.x, ground + pose.agl_m, pose.z]),
                yaw_deg=pose.yaw_deg,
                pitch_deg=cfg.sim.camera_tilt_deg,
            )
            probe["rgb"] = blobs["rgb"]
            probe["pos_enu"] = anchor.sim_to_enu(np.array(result.pos_sim))
            probe["anchor"] = anchor
            await client.close()
        finally:
            await server.stop()

    asyncio.run(scenario())
    ds = CaptureDataset(cfg.dataset_dir)
    bundle, _split = train_models(cfg, ds)
    bundle.save(cfg.bundle_dir)
    return cfg, bundle, probe


def test_photo_localization_close_to_truth(trained_env):
    cfg, bundle, probe = trained_env
    fix = SingleShotLocalizer(bundle).localize(probe["rgb"])
    err_h = np.linalg.norm(fix.pos_enu[:2] - probe["pos_enu"][:2])
    # The photo shows a trained-on view: the fix must land in the right area
    # (tiny-bundle regime: tens of meters, not the world's other side).
    assert err_h < 85.0
    assert abs(fix.pos_enu[2] - probe["pos_enu"][2]) < 30.0
    assert 0.0 <= fix.confidence <= 1.0
    # WGS84 output is consistent with the anchor conversion.
    back = probe["anchor"].geodetic_to_enu(fix.lat, fix.lon, fix.alt_msl)
    np.testing.assert_allclose(back, fix.pos_enu, atol=1e-3)


def test_photo_preprocess_handles_any_size(trained_env):
    cfg, bundle, probe = trained_env
    # Simulate a phone photo: bigger, different aspect ratio.
    big = cv2.resize(probe["rgb"], (1280, 720), interpolation=cv2.INTER_CUBIC)
    fix_big = SingleShotLocalizer(bundle).localize(big)
    fix_ref = SingleShotLocalizer(bundle).localize(probe["rgb"])
    # Same scene: fixes should agree within tens of meters.
    assert np.linalg.norm(fix_big.pos_enu[:2] - fix_ref.pos_enu[:2]) < 80.0
    arr = preprocess(big, 64, 64)
    assert arr.shape == (64, 64, 3) and arr.dtype == np.float32


def test_mobile_export_onnx_parity(trained_env, tmp_path):
    import onnxruntime as ort
    import torch

    from dronecv.export.mobile import export_mobile_bundle, read_landmark_db_bin

    cfg, bundle, probe = trained_env
    out = export_mobile_bundle(bundle, tmp_path / "mobile")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["anchor"]["lat0"] == pytest.approx(bundle.anchor.lat0)
    w, h = manifest["camera"]["width"], manifest["camera"]["height"]

    arr = preprocess(probe["rgb"], w, h)
    x = arr.transpose(2, 0, 1)[None]

    sess_e = ort.InferenceSession(str(out / "embed.onnx"), providers=["CPUExecutionProvider"])
    desc_onnx = sess_e.run(None, {"image": x})[0][0]
    with torch.no_grad():
        desc_torch = bundle.embed_net(torch.from_numpy(x))[0].numpy()
    np.testing.assert_allclose(desc_onnx, desc_torch, atol=1e-4)

    sess_p = ort.InferenceSession(str(out / "posenet.onnx"), providers=["CPUExecutionProvider"])
    pos_scaled, heading_vec, _lv_p, _lv_h = sess_p.run(None, {"image": x})
    with torch.no_grad():
        ref = bundle.pose_net(torch.from_numpy(x))
    np.testing.assert_allclose(pos_scaled[0], ref["pos_scaled"][0].numpy(), atol=1e-4)
    np.testing.assert_allclose(heading_vec[0], ref["heading_vec"][0].numpy(), atol=1e-4)

    emb, pos, heading = read_landmark_db_bin(out / "landmark_db.bin")
    np.testing.assert_allclose(emb, bundle.landmark_db.embeddings, atol=1e-6)
    np.testing.assert_allclose(pos, bundle.landmark_db.pos_enu, atol=1e-3)
    assert heading.shape[0] == emb.shape[0]


def test_http_server_localize_and_bundle(trained_env, tmp_path):
    from fastapi.testclient import TestClient

    from dronecv.server.app import create_app

    cfg, bundle, probe = trained_env
    app = create_app(bundle, mobile_dir=tmp_path / "mobile")
    client = TestClient(app)

    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/info").json()
    assert info["anchor"]["lat0"] == pytest.approx(45.4642)

    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(probe["rgb"], cv2.COLOR_RGB2BGR))
    assert ok
    resp = client.post("/localize", files={"image": ("photo.jpg", jpg.tobytes(), "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    ref = SingleShotLocalizer(bundle).localize(probe["rgb"])
    # JPEG round trip is lossy: allow meters, not micrometers.
    assert abs(body["lat"] - ref.lat) < 0.01
    assert abs(body["lon"] - ref.lon) < 0.01
    assert 0.0 <= body["confidence"] <= 1.0
    assert "diagnostics" in body

    resp = client.post("/localize", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
    assert resp.status_code == 400

    resp = client.get("/mobile-bundle.zip")
    assert resp.status_code == 200
    zf = zipfile.ZipFile(__import__("io").BytesIO(resp.content))
    assert {"embed.onnx", "posenet.onnx", "landmark_db.bin", "manifest.json"} <= set(zf.namelist())
