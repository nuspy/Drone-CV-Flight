"""Cloud detection + multi-date compositing on synthetic ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from dronecv.gis.clouds import cloud_mask, composite_scenes
from dronecv.gis.providers.imagery import OrthoImage

RES = 10.0
SIZE = 240  # px -> 2.4 km at 10 m/px


def scene(cloud_boxes: list[tuple[int, int, int, int]], gain: float = 1.0,
          seed: int = 4) -> tuple[OrthoImage, np.ndarray]:
    """Textured ground + terracotta roofs + white clouds with SE shadows.
    Returns (ortho, true_invalid_mask)."""
    rng = np.random.default_rng(seed)
    rgb = np.empty((SIZE, SIZE, 3), np.float32)
    rgb[:] = np.array([0.42, 0.41, 0.38]) + rng.normal(0, 0.03, (SIZE, SIZE, 3))
    for r0 in range(20, SIZE - 20, 48):
        for c0 in range(16, SIZE - 20, 40):
            rgb[r0 : r0 + 10, c0 : c0 + 14] = (0.75, 0.50, 0.35)
    truth = np.zeros((SIZE, SIZE), bool)
    for r0, c0, hh, ww in cloud_boxes:
        rgb[r0 : r0 + hh, c0 : c0 + ww] = 0.88 + rng.normal(0, 0.008, (hh, ww, 1))
        truth[r0 : r0 + hh, c0 : c0 + ww] = True
        # Same-size shadow displaced west of the cloud (realistic offset).
        sr0, sr1 = min(r0 + hh // 2, SIZE - 1), min(r0 + hh // 2 + hh, SIZE)
        sc0, sc1 = max(c0 - ww - 6, 0), max(c0 - 6, 1)
        rgb[sr0:sr1, sc0:sc1] *= 0.35
        truth[sr0:sr1, sc0:sc1] = True
    rgb = np.clip(rgb * gain, 0, 1)
    gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    return OrthoImage(gray=gray, res_m=RES, e0=-1200, n0=-1200, rgb=rgb), truth


class TestCloudMask:
    def test_detects_clouds_and_shadows(self):
        ortho, truth = scene([(30, 100, 60, 80), (150, 40, 40, 50)])
        m = cloud_mask(ortho)
        recall = m[truth].mean()
        assert recall > 0.85, f"cloud/shadow recall {recall:.2f}"

    def test_clear_scene_mostly_untouched(self):
        ortho, _ = scene([])
        m = cloud_mask(ortho)
        assert m.mean() < 0.05, f"false cloud fraction {m.mean():.2%}"

    def test_roofs_not_flagged_as_clouds(self):
        ortho, truth = scene([(30, 100, 60, 80)])
        m = cloud_mask(ortho)
        roof = (ortho.rgb[..., 0] > 0.7) & (ortho.rgb[..., 1] < 0.6) & ~truth
        assert m[roof].mean() < 0.35, "terracotta roofs flagged as clouds"


class TestComposite:
    def test_holes_filled_from_other_dates(self):
        s1, t1 = scene([(30, 100, 60, 80)], gain=1.0, seed=4)
        s2, _ = scene([(160, 30, 50, 60)], gain=0.8, seed=4)  # different clouds, darker day
        s3, _ = scene([], gain=1.15, seed=4)  # clear but brightest
        out, stats = composite_scenes([(s1, "20260619"), (s2, "20260612"), (s3, "20260605")])
        assert stats.final_hole_fraction < 0.01
        assert stats.n_scenes_used >= 2
        assert stats.fills[0]["date"] == "20260619"
        # The filled cloud area now looks like ground, not like a cloud.
        assert out.gray[t1].mean() < 0.6
        # And it is radiometrically aligned: filled area vs clear area medians agree.
        clear = ~t1
        assert abs(np.median(out.gray[t1]) - np.median(out.gray[clear])) < 0.08

    def test_single_clear_scene_passthrough(self):
        s1, _ = scene([])
        out, stats = composite_scenes([(s1, "20260619")])
        assert stats.n_scenes_used == 1
        assert stats.final_hole_fraction < 0.05

    def test_mismatched_grids_rejected(self):
        s1, _ = scene([(30, 100, 60, 80)])
        small = OrthoImage(gray=np.zeros((10, 10), np.float32), res_m=RES, e0=0, n0=0,
                           rgb=np.zeros((10, 10, 3), np.float32))
        with pytest.raises(ValueError):
            composite_scenes([(s1, "a"), (small, "b")])


class TestMgrs:
    def test_budapest_tile(self):
        from dronecv.gis.providers.sentinel2 import mgrs_tile

        # Verified against the sentinel-cogs bucket: Budapest data lives
        # under sentinel-s2-l2a-cogs/34/T/CT/.
        assert mgrs_tile(47.5015, 19.0425) == (34, "T", "CT")

    def test_zone_and_band_basics(self):
        from dronecv.gis.providers.sentinel2 import mgrs_tile

        zone, band, sq = mgrs_tile(43.32, 11.33)  # Tuscany
        assert zone == 32 and band == "T" and len(sq) == 2
