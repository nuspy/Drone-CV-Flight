"""Shared domain-gap filter: identity, determinism, structure, plumbing."""

from __future__ import annotations

import numpy as np
import pytest

from dronecv.vision.preprocess_filter import FilterSpec, apply_filter


def _img(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 1, (48, 64, 3)).astype(np.float32)
    img[10:30, 20:40] = (0.9, 0.2, 0.1)  # a colored block -> edges at borders
    return img


class TestFilterSpec:
    def test_none_is_identity(self):
        img = _img()
        out = apply_filter(img, FilterSpec(mode="none"))
        assert out is img

    def test_none_spec_object_is_identity(self):
        img = _img()
        assert apply_filter(img, None) is img

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            FilterSpec(mode="sepia")

    def test_roundtrip_dict(self):
        spec = FilterSpec(mode="gray_edge", edge_weight=0.7, clahe=True)
        assert FilterSpec.from_dict(spec.to_dict()) == spec
        assert FilterSpec.from_dict(None) == FilterSpec()


class TestApply:
    @pytest.mark.parametrize("mode", ["gray", "edge", "gray_edge"])
    def test_shape_dtype_range(self, mode):
        out = apply_filter(_img(), FilterSpec(mode=mode))
        assert out.shape == (48, 64, 3) and out.dtype == np.float32
        assert 0.0 <= out.min() and out.max() <= 1.0

    @pytest.mark.parametrize("mode", ["gray", "edge", "gray_edge"])
    def test_deterministic(self, mode):
        spec = FilterSpec(mode=mode, clahe=True)
        np.testing.assert_array_equal(apply_filter(_img(), spec), apply_filter(_img(), spec))

    def test_gray_removes_color(self):
        out = apply_filter(_img(), FilterSpec(mode="gray"))
        np.testing.assert_allclose(out[..., 0], out[..., 1])
        np.testing.assert_allclose(out[..., 1], out[..., 2])

    def test_edge_fires_on_boundaries_not_flat(self):
        img = np.full((40, 40, 3), 0.5, dtype=np.float32)
        img[:, 20:] = 0.9
        out = apply_filter(img, FilterSpec(mode="edge"))
        assert out[20, 20, 0] > 0.3  # the step edge
        assert out[20, 5, 0] < 0.05  # flat region
        assert out[20, 35, 0] < 0.05

    def test_gray_edge_blends(self):
        img = _img()
        g = apply_filter(img, FilterSpec(mode="gray"))
        e = apply_filter(img, FilterSpec(mode="edge"))
        ge = apply_filter(img, FilterSpec(mode="gray_edge", edge_weight=0.5))
        np.testing.assert_allclose(ge, 0.5 * g + 0.5 * e, atol=1e-5)


class TestPlumbing:
    def test_dataset_view_applies_filter(self, tmp_path):
        from dronecv.capture.dataset import CaptureDataset, DatasetWriter, TorchCaptureView

        writer = DatasetWriter(tmp_path / "ds", info={"anchor": {}, "camera": {}})
        rng = np.random.default_rng(1)
        for _ in range(3):
            writer.add(
                (rng.uniform(0, 255, (32, 32, 3))).astype(np.uint8),
                {
                    "pos_sim": [0.0, 50.0, 0.0], "pos_enu": [0.0, 50.0, 0.0],
                    "yaw_sim_deg": 0.0, "heading_deg": 0.0, "pitch_deg": 20.0,
                    "agl_m": 50.0, "ground_y_sim": 0.0,
                    "utc": "2026-06-21T10:00:00+00:00", "sun_azimuth_deg": None,
                    "sun_elevation_deg": None,
                },
            )
        writer.close()
        ds = CaptureDataset(tmp_path / "ds")
        raw = TorchCaptureView(ds, [0, 1, 2])[0][0]
        filt = TorchCaptureView(ds, [0, 1, 2], filter_spec=FilterSpec(mode="gray"))[0][0]
        assert not np.allclose(raw.numpy(), filt.numpy())
        np.testing.assert_allclose(filt.numpy()[0], filt.numpy()[1])  # channels equal

    def test_bundle_spec_default_none(self):
        from dronecv.training.bundle import ModelBundle

        bundle = ModelBundle.__new__(ModelBundle)
        bundle.manifest = {}
        assert bundle.filter_spec.is_identity
        bundle.manifest = {"preprocess_filter": {"mode": "gray_edge", "edge_weight": 0.6,
                                                 "clahe": False}}
        assert bundle.filter_spec == FilterSpec(mode="gray_edge", edge_weight=0.6)
