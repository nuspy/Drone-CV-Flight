"""Ortho -> ground albedo drape (imagery becomes the visible ground colour)."""

from __future__ import annotations

import numpy as np

from dronecv.gis.drape import _rgb_to_uint8, drape_albedo
from dronecv.gis.providers.imagery import OrthoImage


class _Meta:
    e0 = 0.0
    n0 = 0.0
    res_m = 1.0
    width = 4
    height = 4


class _Store:
    def __init__(self):
        self.meta = _Meta()
        self.class_id = np.zeros((4, 4), np.uint8)
        self.albedo = None

    def create_albedo(self):
        self.albedo = np.zeros((4, 4, 3), np.uint8)
        return self.albedo


def _ortho(rgb, e0=0.0, n0=0.0, valid=None):
    return OrthoImage(gray=rgb[..., 0], res_m=1.0, e0=e0, n0=n0, rgb=rgb, valid=valid)


def test_rgb_to_uint8_float_and_uint():
    assert _rgb_to_uint8(np.array([[[1.0, 0.0, 0.5]]])).tolist() == [[[255, 0, 128]]]
    u = np.array([[[10, 20, 30]]], np.uint8)
    assert _rgb_to_uint8(u) is u  # uint8 passed through


def test_full_coverage_uses_imagery():
    store = _Store()
    red = np.zeros((4, 4, 3), np.float32)
    red[..., 0] = 1.0
    cov = drape_albedo(store, _ortho(red))
    assert cov == 1.0
    assert store.albedo is not None
    assert (store.albedo[..., 0] == 255).all() and (store.albedo[..., 1:] == 0).all()


def test_uncovered_texels_fall_back_to_class_colour():
    from dronecv.gis.world import CLASS_COLORS

    store = _Store()
    # ortho covers only a 2x2 corner (its origin shifted so half the grid is out)
    red = np.zeros((2, 2, 3), np.float32)
    red[..., 0] = 1.0
    cov = drape_albedo(store, _ortho(red, e0=0.0, n0=0.0))
    assert 0.0 < cov < 1.0
    # covered corner is red; elsewhere is the class-0 synthetic colour
    assert store.albedo[0, 0].tolist() == [255, 0, 0]
    cls0 = tuple(int(round(c * 255)) for c in CLASS_COLORS[0])
    assert tuple(store.albedo[3, 3].tolist()) == cls0


def test_validity_mask_excludes_clouds():
    store = _Store()
    red = np.zeros((4, 4, 3), np.float32)
    red[..., 0] = 1.0
    valid = np.ones((4, 4), bool)
    valid[0, 0] = False  # a "cloud" texel -> keep class colour there
    drape_albedo(store, _ortho(red, valid=valid))
    assert store.albedo[0, 0].tolist() != [255, 0, 0]
    assert store.albedo[1, 1].tolist() == [255, 0, 0]
