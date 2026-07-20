"""Zone palette extraction + imagery vegetation on synthetic ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from dronecv.gis.palette import DEFAULT_ROOF, DEFAULT_WALL, ZonePalette, extract_palette


def _photo(tmp_path, name, blocks):
    """Write a photo made of horizontal color bands: [(rgb, rows), ...]."""
    import cv2

    rows = sum(r for _, r in blocks)
    img = np.zeros((rows, 200, 3), np.float32)
    y = 0
    rng = np.random.default_rng(5)
    for rgb, r in blocks:
        img[y : y + r] = np.array(rgb) + rng.normal(0, 0.01, (r, 200, 3))
        y += r
    cv2.imwrite(str(tmp_path / name), cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8),
                                                   cv2.COLOR_RGB2BGR))


class TestPalette:
    def test_wall_cluster_from_street_photo(self, tmp_path):
        # Sky on top, cream facade in the middle, dark road at the bottom.
        _photo(tmp_path, "street.jpg", [
            ((0.45, 0.65, 0.90), 80),   # sky (excluded: blue + top quarter)
            ((0.88, 0.82, 0.66), 200),  # cream walls <- should dominate
            ((0.20, 0.20, 0.22), 80),   # asphalt (excluded: too dark)
        ])
        pal = extract_palette([tmp_path])
        assert pal.n_wall_px > 400
        r, g, b = pal.wall[0]
        assert r > 0.75 and g > 0.70 and b > 0.55, f"wall cluster {pal.wall[0]}"
        assert r > b  # cream, not blue

    def test_roof_cluster_warm(self, tmp_path):
        _photo(tmp_path, "aerial.jpg", [
            ((0.66, 0.36, 0.24), 250),  # terracotta roofs
            ((0.50, 0.50, 0.50), 100),  # gray street
        ])
        pal = extract_palette([tmp_path])
        r, g, b = pal.roof[0]
        assert r > g > b, f"roof cluster not warm: {pal.roof[0]}"
        assert r > 0.5

    def test_ortho_roof_pixels_used(self):
        px = np.tile(np.array([[0.60, 0.34, 0.22]], np.float32), (1000, 1))
        pal = extract_palette([], roof_pixels=px)
        assert pal.n_roof_px == 1000
        np.testing.assert_allclose(pal.roof[0], (0.60, 0.34, 0.22), atol=0.02)

    def test_defaults_when_no_sources(self):
        pal = extract_palette([])
        assert pal.roof == list(DEFAULT_ROOF) and pal.wall == list(DEFAULT_WALL)

    def test_save_load_roundtrip(self, tmp_path):
        pal = extract_palette([], roof_pixels=np.full((500, 3), 0.5, np.float32))
        pal.save(tmp_path)
        again = ZonePalette.load(tmp_path)
        assert again is not None and again.roof == pal.roof and again.wall == pal.wall
        assert ZonePalette.load(tmp_path / "missing") is None


class TestImageryVegetation:
    @pytest.fixture
    def store(self, tmp_path):
        from dronecv.gis.store import GisMeta, GisStore

        meta = GisMeta(res_m=1.0, e0=-100, n0=-100, width=200, height=200,
                       anchor={}, ground_alt0=0.0, max_height=0.0)
        return GisStore.create(tmp_path / "gis", meta)

    def test_green_spots_become_vegetation(self, store):
        from dronecv.gis.providers.imagery import OrthoImage
        from dronecv.gis.vegetation import vegetation_from_imagery

        rgb = np.full((200, 200, 3), (0.55, 0.52, 0.48), np.float32)  # urban gray
        rgb[20:120, 30:130] = (0.20, 0.45, 0.18)  # a big park -> forest core
        rgb[150:156, 150:156] = (0.30, 0.50, 0.25)  # small green spot
        gray = rgb.mean(-1)
        ortho = OrthoImage(gray=gray, res_m=1.0, e0=-100, n0=-100, rgb=rgb)
        store.class_id[10:14, 10:14] = 2  # pre-existing water must survive
        stats = vegetation_from_imagery(store, ortho)
        cls = np.asarray(store.class_id)
        assert stats["imagery_green_texels"] > 5000
        assert stats["imagery_forest_texels"] > 2000
        assert (cls[60:80, 60:80] == 6).all()  # park core upgraded to forest
        assert (cls[10:14, 10:14] == 2).all()  # water untouched
        assert cls[0, 0] == 0  # gray urban stays ground

    def test_no_rgb_is_noop(self, store):
        from dronecv.gis.providers.imagery import OrthoImage
        from dronecv.gis.vegetation import vegetation_from_imagery

        ortho = OrthoImage(gray=np.zeros((10, 10), np.float32), res_m=1.0, e0=0, n0=0)
        assert vegetation_from_imagery(store, ortho)["imagery_green_texels"] == 0
        assert vegetation_from_imagery(store, None)["imagery_green_texels"] == 0
