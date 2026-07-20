"""Composite-mosaic radiometric normalization: same roofs, different strips."""

from __future__ import annotations

import numpy as np

from dronecv.gis.providers.imagery import OrthoImage
from dronecv.gis.radiometry import normalize_zones, radiometric_zones

RES = 0.5
SIZE = 480  # px -> 240 m
# Three vertical strips with different exposure gain and a slight tint.
STRIPS = [(0, 160, 1.00, (1.00, 1.00, 1.00)),
          (160, 320, 0.72, (1.02, 0.98, 0.96)),
          (320, 480, 1.25, (0.97, 1.00, 1.05))]
ROOF_RGB = np.array([0.78, 0.52, 0.36], np.float32)  # same sunlit terracotta everywhere


def composite_ortho() -> tuple[OrthoImage, np.ndarray]:
    """Synthetic composite + boolean mask of the true roof pixels."""
    rng = np.random.default_rng(11)
    rgb = np.empty((SIZE, SIZE, 3), np.float32)
    rgb[:] = np.array([0.45, 0.44, 0.42]) + rng.normal(0, 0.015, (SIZE, SIZE, 1))
    roof_mask = np.zeros((SIZE, SIZE), bool)
    for r0 in range(40, SIZE - 40, 96):
        for c0 in range(30, SIZE - 40, 72):
            rgb[r0 : r0 + 28, c0 : c0 + 40] = ROOF_RGB + rng.normal(0, 0.01, (28, 40, 3))
            roof_mask[r0 : r0 + 28, c0 : c0 + 40] = True
    for c0, c1, gain, tint in STRIPS:
        rgb[:, c0:c1] *= gain * np.asarray(tint, np.float32)[None, None, :]
    rgb = np.clip(rgb, 0, 1)
    gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    return OrthoImage(gray=gray, res_m=RES, e0=-120, n0=-120, rgb=rgb), roof_mask


class TestZones:
    def test_strips_detected(self):
        ortho, _ = composite_ortho()
        labels, stats = radiometric_zones(ortho.gray, ortho.rgb, RES, block_m=16.0)
        assert stats.n_zones == 3, f"expected 3 strips, found {stats.n_zones}"
        # Each strip center belongs to a distinct zone.
        centers = {int(labels[SIZE // 2, (c0 + c1) // 2]) for c0, c1, _, _ in STRIPS}
        assert len(centers) == 3

    def test_uniform_image_is_single_zone_noop(self):
        rng = np.random.default_rng(2)
        gray = (0.5 + rng.normal(0, 0.02, (300, 300))).astype(np.float32)
        ortho = OrthoImage(gray=np.clip(gray, 0, 1), res_m=RES, e0=0, n0=0)
        out, stats = normalize_zones(ortho, block_m=16.0)
        assert stats.n_zones == 1
        assert out is ortho  # exact no-op


class TestNormalization:
    def test_roof_medians_converge(self):
        ortho, roof_mask = composite_ortho()
        out, stats = normalize_zones(ortho, block_m=16.0)
        assert stats.n_zones == 3
        meds = []
        for c0, c1, _, _ in STRIPS:
            strip_roofs = roof_mask[:, c0:c1]
            meds.append(np.median(out.rgb[:, c0:c1][strip_roofs], axis=0))
        meds = np.asarray(meds)
        spread_before = _roof_spread(ortho, roof_mask)
        spread_after = float(np.abs(meds - meds.mean(0)).max())
        assert spread_after < 0.05, f"roof medians still spread {spread_after:.3f}"
        assert spread_after < spread_before / 3, (spread_before, spread_after)

    def test_no_new_seams(self):
        ortho, _ = composite_ortho()
        out, _ = normalize_zones(ortho, block_m=16.0)
        # Column-median profile across the seams must be smooth after.
        prof = np.median(out.gray, axis=0)
        jumps = np.abs(np.diff(prof))
        assert jumps.max() < 0.05, f"residual seam jump {jumps.max():.3f}"


def _roof_spread(ortho, roof_mask) -> float:
    meds = []
    for c0, c1, _, _ in STRIPS:
        strip_roofs = roof_mask[:, c0:c1]
        meds.append(np.median(ortho.rgb[:, c0:c1][strip_roofs], axis=0))
    meds = np.asarray(meds)
    return float(np.abs(meds - meds.mean(0)).max())


class TestDownstream:
    def test_building_mask_recall_parity_across_strips(self):
        from dronecv.gis.providers.footprints_from_imagery import building_mask

        ortho, roof_mask = composite_ortho()
        norm, _ = normalize_zones(ortho, block_m=16.0)
        m = building_mask(norm)
        recalls = []
        for c0, c1, _, _ in STRIPS:
            strip = roof_mask[:, c0:c1]
            recalls.append(m[:, c0:c1][strip].mean())
        assert min(recalls) > 0.4, f"strip recalls {recalls}"
        assert max(recalls) - min(recalls) < 0.35, f"recall imbalance {recalls}"

    def test_palette_single_roof_cluster(self):
        from dronecv.gis.palette import extract_palette

        ortho, roof_mask = composite_ortho()
        norm, _ = normalize_zones(ortho, block_m=16.0)
        pal = extract_palette([], roof_pixels=norm.rgb[roof_mask])
        r, g, b = pal.roof[0]
        assert r > g > b, f"dominant roof cluster not terracotta: {pal.roof[0]}"
        # The dominant cluster holds the (large) majority: its color is close
        # to the true roof color, not an average of two exposure variants.
        assert abs(r - 0.78) < 0.12 and abs(g - 0.52) < 0.12
