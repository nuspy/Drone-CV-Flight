"""Geofusion components: descriptor, constellation ratios, particle filter."""

from __future__ import annotations

import numpy as np

from dronecv.localization.geofusion import InvariantView, ParticleFuser, descriptor
from dronecv.localization.geofusion.constellation import constellation_score
from dronecv.localization.geofusion.retrieval import Candidate


def _view(depth, building, veg=None):
    h, w = depth.shape
    return InvariantView(
        depth=depth.astype(np.float32),
        building=building.astype(bool),
        vegetation=(veg if veg is not None else np.zeros_like(building)).astype(bool),
        points=np.zeros((h, w, 3)),
        pos=np.zeros(3), yaw_deg=0.0, pitch_down_deg=35.0, fov_deg=70.0,
    )


class TestDescriptor:
    def test_normalized_and_deterministic(self):
        rng = np.random.default_rng(0)
        depth = rng.uniform(10, 500, (48, 64))
        building = rng.random((48, 64)) > 0.7
        v = _view(depth, building)
        d1, d2 = descriptor(v), descriptor(v)
        assert np.allclose(d1, d2)
        assert abs(float(np.linalg.norm(d1)) - 1.0) < 1e-5

    def test_differs_between_different_scenes(self):
        rng = np.random.default_rng(1)
        a = _view(rng.uniform(10, 100, (48, 64)), rng.random((48, 64)) > 0.5)
        flat = _view(np.full((48, 64), 400.0), np.zeros((48, 64)))
        sim = float(descriptor(a) @ descriptor(flat))
        assert sim < 0.95  # town view vs empty plain must be distinguishable


class TestConstellation:
    LAYOUT = np.array([[0.0, 0.0], [50.0, 10.0], [20.0, 70.0], [90.0, 60.0]])

    def test_ratio_space_is_scale_invariant(self):
        # The same layout at 3x the size scores identically: only RATIOS matter.
        s1 = constellation_score(self.LAYOUT, self.LAYOUT)
        s3 = constellation_score(self.LAYOUT, self.LAYOUT * 3.0)
        assert s1 > 0.9 and abs(s1 - s3) < 1e-6

    def test_wrong_layout_scores_low(self):
        wrong = np.array([[0.0, 0.0], [5.0, 90.0], [95.0, 5.0], [50.0, 50.0]])
        right = constellation_score(self.LAYOUT, self.LAYOUT)
        bad = constellation_score(self.LAYOUT, wrong)
        assert right > bad

    def test_too_few_observed_is_neutral(self):
        assert constellation_score(self.LAYOUT[:2], self.LAYOUT) == 0.5

    def test_buildings_seen_but_model_empty_is_a_veto(self):
        assert constellation_score(self.LAYOUT, np.zeros((0, 2))) == 0.0


class TestParticleFuser:
    def test_converges_on_gaussian_fixes(self):
        pf = ParticleFuser(300, rng=np.random.default_rng(0))
        cands = [Candidate(pos=np.array([0.0, 60.0, 0.0]), yaw_deg=0.0, score=1.0),
                 Candidate(pos=np.array([500.0, 60.0, 500.0]), yaw_deg=180.0, score=1.0)]
        pf.init_from_candidates(cands)
        # repeated fixes at (500, 500, yaw 180): the wrong mode must die
        for _ in range(4):
            pf.predict(0.0, 0.0, 0.0)
            pf.weight_gaussian(np.array([500.0, 500.0]), 180.0)
            pf.resample_if_needed()
        xy, yaw, spread = pf.estimate()
        assert np.linalg.norm(xy - np.array([500.0, 500.0])) < 30.0
        assert abs((yaw - 180.0 + 180) % 360 - 180) < 15.0
        assert spread < 60.0

    def test_predict_moves_along_heading(self):
        pf = ParticleFuser(10, rng=np.random.default_rng(0))
        pf.xy[:] = 0.0
        pf.yaw[:] = 90.0  # facing +X (east)
        pf.predict(10.0, 0.0, 0.0, trans_noise_m=0.0, yaw_noise_deg=0.0)
        assert np.allclose(pf.xy[:, 0], 10.0, atol=1e-6)  # moved east
        assert np.allclose(pf.xy[:, 1], 0.0, atol=1e-6)

    def test_estimate_circular_yaw_mean(self):
        pf = ParticleFuser(2, rng=np.random.default_rng(0))
        pf.xy[:] = 0.0
        pf.yaw[:] = [350.0, 10.0]
        pf.w[:] = 0.5
        _, yaw, _ = pf.estimate()
        assert min(yaw, 360 - yaw) < 1e-6  # mean of 350 and 10 is 0, not 180
