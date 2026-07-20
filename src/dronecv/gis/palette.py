"""Per-environment chromatic palette extracted from photos of the area.

Building colors are usually uniform within a zone (municipal regulations fix
roof materials and facade tones), so a handful of photos pins the palette:

- ROOFS: warm saturated hues seen from above (aerial photos + the orthophoto
  texels that fall on rasterized buildings — those pixels ARE roofs);
- WALLS: bright desaturated tones from street-level photos (sky and
  vegetation hues are excluded before clustering).

Sources, all optional and merged: Wikimedia Commons POI photos (already
downloaded by the POI step), user-provided photos, orthophoto RGB. Result is
k-means cluster centers saved as `palette.json` in the GIS store; `GisWorld`
loads it to color roofs per class and to give facades their OWN colors (the
azimuth view color belongs to the roof only, never to the walls).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Fallbacks when a source yields too few pixels (central-European defaults).
DEFAULT_ROOF = [(0.64, 0.42, 0.30), (0.55, 0.36, 0.28), (0.48, 0.45, 0.44)]
DEFAULT_WALL = [(0.85, 0.80, 0.70), (0.90, 0.87, 0.80), (0.78, 0.70, 0.58)]

MIN_PIXELS = 400


@dataclass
class ZonePalette:
    roof: list[tuple[float, float, float]] = field(default_factory=lambda: list(DEFAULT_ROOF))
    wall: list[tuple[float, float, float]] = field(default_factory=lambda: list(DEFAULT_WALL))
    n_roof_px: int = 0
    n_wall_px: int = 0
    sources: list[str] = field(default_factory=list)

    def save(self, gis_dir: Path) -> None:
        (Path(gis_dir) / "palette.json").write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, gis_dir: Path) -> ZonePalette | None:
        p = Path(gis_dir) / "palette.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        d["roof"] = [tuple(c) for c in d["roof"]]
        d["wall"] = [tuple(c) for c in d["wall"]]
        return cls(**d)


def _kmeans(px: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> list[tuple[float, ...]]:
    """Plain numpy k-means on Nx3 RGB [0,1]; centers sorted by cluster size."""
    gen = np.random.default_rng(seed)
    centers = px[gen.choice(len(px), size=k, replace=False)]
    for _ in range(iters):
        d = ((px[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        lbl = d.argmin(1)
        for j in range(k):
            m = lbl == j
            if m.any():
                centers[j] = px[m].mean(0)
    counts = np.bincount(lbl, minlength=k)
    order = np.argsort(-counts)
    return [tuple(float(v) for v in centers[j]) for j in order if counts[j] > 0]


def _hsv(rgb: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / np.array(
        [180.0, 255.0, 255.0], dtype=np.float32
    )


def _photo_pixels(path: Path, max_side: int = 384) -> np.ndarray | None:
    import cv2

    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    scale = max_side / max(bgr.shape[:2])
    if scale < 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _wall_candidates(rgb: np.ndarray) -> np.ndarray:
    """Bright, desaturated, non-sky non-vegetation pixels of a photo."""
    hsv = _hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sky = (h > 0.52) & (h < 0.72) & (s > 0.15)  # blue hues
    veg = (h > 0.18) & (h < 0.45) & (s > 0.22)  # green hues
    m = (s < 0.38) & (v > 0.42) & (v < 0.97) & ~sky & ~veg
    m[: rgb.shape[0] // 4] = False  # top quarter is mostly sky/roofline
    return rgb[m].reshape(-1, 3)


def _roof_candidates(rgb: np.ndarray) -> np.ndarray:
    """Warm saturated pixels (tiles/terracotta bands)."""
    hsv = _hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    warm = (h < 0.12) | (h > 0.93)
    m = warm & (s > 0.25) & (v > 0.20) & (v < 0.95)
    return rgb[m].reshape(-1, 3)


def extract_palette(
    photo_dirs: list[Path],
    roof_pixels: np.ndarray | None = None,
    seed: int = 0,
) -> ZonePalette:
    """Cluster the zone's roof and wall colors from all available sources.

    `roof_pixels` (Nx3 float [0,1]): orthophoto colors sampled where
    buildings are rasterized — from above those pixels ARE roofs."""
    wall_px, roof_px, sources = [], [], []
    for d in photo_dirs:
        d = Path(d)
        if not d.exists():
            continue
        found = False
        for p in sorted(d.glob("*")):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            rgb = _photo_pixels(p)
            if rgb is None:
                continue
            found = True
            wall_px.append(_wall_candidates(rgb))
            roof_px.append(_roof_candidates(rgb))
        if found:
            sources.append(str(d))
    if roof_pixels is not None and len(roof_pixels):
        roof_px.append(np.asarray(roof_pixels, dtype=np.float32).reshape(-1, 3))
        sources.append("orthophoto")

    pal = ZonePalette(sources=sources)
    walls = np.concatenate(wall_px) if wall_px else np.empty((0, 3), np.float32)
    roofs = np.concatenate(roof_px) if roof_px else np.empty((0, 3), np.float32)
    pal.n_wall_px, pal.n_roof_px = len(walls), len(roofs)
    if len(walls) >= MIN_PIXELS:
        pal.wall = _kmeans(walls, k=3, seed=seed)
    if len(roofs) >= MIN_PIXELS:
        pal.roof = _kmeans(roofs, k=3, seed=seed)
    return pal
