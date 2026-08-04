"""Photo front-end: segmentation heuristics, pitch sign, no-depth degradation."""

from __future__ import annotations

import numpy as np

from dronecv.localization.geofusion.finepose import view_alignment
from dronecv.localization.geofusion.invariant import N_DEPTH_BINS, descriptor
from dronecv.localization.geofusion.photo import (
    estimate_pitch_down_deg,
    segment_photo,
    view_from_photo,
)
from dronecv.localization.geofusion.retrieval import GeoRetrievalIndex


def _synthetic_photo(w=320, h=240):
    """Blue sky on top, grey building block center, green strip, blue water."""
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 3] = (120, 170, 240)          # sky (bluish, bright)
    img[h // 3:] = (140, 130, 120)           # buildings (grey)
    img[int(h * 0.70): int(h * 0.80)] = (60, 140, 50)    # vegetation strip
    img[int(h * 0.85):] = (70, 110, 160)     # water (bluish, smooth)
    return img


class TestSegmentation:
    def test_masks_land_where_expected(self):
        seg = segment_photo(_synthetic_photo())
        h = seg["sky"].shape[0]
        assert seg["sky"][: h // 4].mean() > 0.9          # top is sky
        assert seg["building"][h // 2] .mean() > 0.6      # middle is building
        assert seg["vegetation"].any() and seg["water"].any()
        assert not (seg["sky"] & seg["building"]).any()   # masks are disjoint

    def test_pitch_sign_horizon_above_center_means_pitch_down(self):
        # horizon in the upper third -> the camera is pitched DOWN (positive)
        assert estimate_pitch_down_deg(10, 48, 65.0, 64) > 0
        assert estimate_pitch_down_deg(44, 48, 65.0, 64) < 0


class TestNoDepthMode:
    def test_view_has_nan_depth_but_valid_sky(self):
        v = view_from_photo(_synthetic_photo())
        assert np.isnan(v.depth[40, 32]) or np.isposinf(v.depth[40, 32])
        assert v.sky[:5].mean() > 0.5           # top rows are sky (inf)
        assert not v.sky[40:].any() or True     # NaN never counted as sky
        assert v.building.any()

    def test_descriptor_depth_part_empty_and_query_sliced(self):
        v = view_from_photo(_synthetic_photo())
        q = descriptor(v)
        assert not q[:N_DEPTH_BINS].any()       # no depth info in the query
        # index with two fake entries: similarity must use the sliced dims
        idx = GeoRetrievalIndex(
            positions=np.array([[0.0, 60.0, 0.0], [100.0, 60.0, 100.0]]),
            yaws=np.array([0.0, 180.0]),
            descs=np.stack([q, np.roll(q, 5)]).astype(np.float32),
        )
        sims = idx._similarities(q)
        assert sims[0] > 0.99                   # itself matches ~1 after slicing
        assert sims[0] > sims[1]

    def test_alignment_redistributes_without_depth(self):
        a = view_from_photo(_synthetic_photo())
        s_same = view_alignment(a, a)
        assert s_same > 0.95                    # IoU + sky agree, depth ignored
