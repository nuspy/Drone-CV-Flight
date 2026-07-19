import numpy as np
import pytest
import torch

from dronecv.models.backbone import TinyConvNet
from dronecv.models.pose_net import PoseNet
from dronecv.models.retrieval import (
    EmbeddingNet,
    LandmarkDB,
    batch_hard_triplet_loss,
    positive_mask,
)


def test_backbone_shapes():
    net = TinyConvNet(width=8)
    out = net(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 64)


def test_embedding_normalized():
    net = EmbeddingNet(width=8, embedding_dim=32)
    emb = net(torch.rand(4, 3, 64, 64))
    assert emb.shape == (4, 32)
    np.testing.assert_allclose(emb.norm(dim=1).detach().numpy(), 1.0, atol=1e-5)


def test_positive_mask_radius_and_yaw():
    pos = torch.tensor([[0.0, 0, 0], [5.0, 0, 0], [500.0, 0, 0], [0.0, 3, 0]])
    heading = torch.tensor([0.0, 10.0, 0.0, 350.0])
    mask = positive_mask(pos, heading, radius_m=30.0, yaw_window_deg=60.0)
    assert mask[0, 1] and mask[0, 3]  # close, similar yaw (350 wraps to -10)
    assert not mask[0, 2]  # too far
    assert not mask[0, 0]  # diagonal excluded


def test_triplet_loss_orders_correctly():
    # Embeddings where positives are close: loss should be near zero.
    emb_good = torch.tensor([[1.0, 0], [0.99, 0.05], [-1.0, 0], [-0.99, -0.05]])
    emb_good = torch.nn.functional.normalize(emb_good, dim=1)
    mask = torch.tensor(
        [[False, True, False, False],
         [True, False, False, False],
         [False, False, False, True],
         [False, False, True, False]]
    )
    loss_good = batch_hard_triplet_loss(emb_good, mask, margin=0.3)
    # Scrambled: positives far, negatives close -> big loss.
    emb_bad = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [-1.0, 0], [0.99, 0.05], [-0.99, -0.05]]), dim=1)
    loss_bad = batch_hard_triplet_loss(emb_bad, mask, margin=0.3)
    assert float(loss_good) < 0.05
    assert float(loss_bad) > float(loss_good) + 0.5


def test_landmark_db_consensus_and_spread():
    rng = np.random.default_rng(0)
    # Two clusters of descriptors at different places.
    d1 = np.tile([1.0, 0.0, 0.0], (10, 1)) + rng.normal(0, 0.01, (10, 3))
    d2 = np.tile([0.0, 1.0, 0.0], (10, 1)) + rng.normal(0, 0.01, (10, 3))
    emb = np.vstack([d1, d2]).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    pos = np.vstack([np.tile([100.0, 200, 50], (10, 1)), np.tile([-300.0, 0, 60], (10, 1))])
    pos += rng.normal(0, 2.0, pos.shape)
    heading = np.concatenate([np.full(10, 90.0), np.full(10, 270.0)])
    db = LandmarkDB(emb, pos, heading)

    fix = db.query(np.array([1.0, 0, 0], dtype=np.float32), topk=5)
    assert np.linalg.norm(fix["pos_enu"][:2] - [100, 200]) < 5
    assert abs(fix["heading_deg"] - 90.0) < 10
    assert fix["spread_m"] < 10

    # Ambiguous query between clusters -> big spread (weak measurement).
    amb = np.array([0.707, 0.707, 0], dtype=np.float32)
    fix_amb = db.query(amb, topk=10)
    assert fix_amb["spread_m"] > 20


def test_landmark_db_save_load(tmp_path):
    emb = np.eye(4, dtype=np.float32)
    db = LandmarkDB(emb, np.arange(12).reshape(4, 3).astype(float), np.zeros(4))
    db.save(tmp_path / "db.npz")
    db2 = LandmarkDB.load(tmp_path / "db.npz")
    np.testing.assert_array_equal(db.embeddings, db2.embeddings)


def test_posenet_predict_units():
    net = PoseNet(width=8, pos_scale_m=400.0)
    out = net.predict(torch.rand(3, 3, 64, 64))
    assert out["pos_enu"].shape == (3, 3)
    assert ((out["heading_deg"] >= 0) & (out["heading_deg"] < 360)).all()
    assert (out["sigma_pos_m"] > 0).all()


def test_posenet_loss_uncertainty_weighting():
    net = PoseNet(width=8, pos_scale_m=100.0)
    img = torch.rand(4, 3, 64, 64)
    pos = torch.rand(4, 3) * 100
    h = torch.rand(4) * 2 * np.pi
    heading = torch.stack([torch.sin(h), torch.cos(h)], dim=1)
    out = net(img)
    loss, stats = net.loss(out, pos, heading)
    assert torch.isfinite(loss)
    loss.backward()  # gradients flow
    assert stats["pos_rmse_m"] > 0
