"""Radiometric normalization of composite satellite mosaics.

Satellite orthophotos are usually COMPOSITES: strips acquired on different
days with different exposure, tone and white balance, joined along sharp
seams. The same terracotta roof can read bright in one strip and dark in the
next — every luminance-based consumer (footprint extraction, palette,
vegetation green-spots, shadow profiles) then misbehaves across the seam.

Two steps, both automatic:

1. `radiometric_zones` — segment the image into radiometrically homogeneous
   zones from per-block robust statistics (median + MAD of luminance,
   per-channel medians). Composite seams are sharp discontinuities of the
   block medians; region-growing over similar adjacent blocks recovers the
   strips (typically 1-6 zones; tiny zones are absorbed).
2. `normalize_zones` — map every zone onto the LARGEST one: shift+scale of
   luminance (align median and MAD) plus per-channel gain (align RGB
   medians), feathered near zone boundaries so the correction itself never
   introduces a new seam.

Honesty: the correction is RELATIVE — it aligns strips to the dominant one,
it does not recover "true" colors, and clipped/saturated areas cannot be
un-clipped. That is exactly what downstream consumers need: consistency,
not absolute radiometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dronecv.gis.providers.imagery import OrthoImage
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.radiometry")

LUM_SPLIT = 0.055  # |Δ block median| that separates two zones
CHAN_SPLIT = 0.045  # |Δ per-channel median| that separates two zones
MIN_ZONE_FRACTION = 0.01
FEATHER_M = 20.0


@dataclass
class ZoneStats:
    n_zones: int
    corrections: list[dict] = field(default_factory=list)


def _block_stats(gray: np.ndarray, rgb: np.ndarray | None, bs: int):
    h, w = gray.shape
    nr, nc = max(1, h // bs), max(1, w // bs)
    med = np.zeros((nr, nc), np.float32)
    mad = np.zeros((nr, nc), np.float32)
    cmed = np.zeros((nr, nc, 3), np.float32)
    for r in range(nr):
        for c in range(nc):
            tile = gray[r * bs : min((r + 1) * bs, h), c * bs : min((c + 1) * bs, w)]
            m = float(np.median(tile))
            med[r, c] = m
            mad[r, c] = float(np.median(np.abs(tile - m)))
            if rgb is not None:
                t = rgb[r * bs : min((r + 1) * bs, h), c * bs : min((c + 1) * bs, w)]
                cmed[r, c] = np.median(t.reshape(-1, 3), axis=0)
    return med, mad, cmed


def radiometric_zones(
    gray: np.ndarray,
    rgb: np.ndarray | None,
    res_m: float,
    block_m: float = 64.0,
) -> tuple[np.ndarray, ZoneStats]:
    """Label map (HxW int32, 0..K-1) of radiometrically homogeneous zones."""
    bs = max(8, int(block_m / max(res_m, 1e-6)))
    med, mad, cmed = _block_stats(gray, rgb, bs)
    nr, nc = med.shape

    # Union-find over adjacent blocks with compatible stats.
    parent = np.arange(nr * nc)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def similar(a, b):
        ra, ca = divmod(a, nc)
        rb, cb = divmod(b, nc)
        if abs(med[ra, ca] - med[rb, cb]) > LUM_SPLIT:
            return False
        if rgb is not None and np.abs(cmed[ra, ca] - cmed[rb, cb]).max() > CHAN_SPLIT:
            return False
        return True

    for r in range(nr):
        for c in range(nc):
            i = r * nc + c
            if c + 1 < nc and similar(i, i + 1):
                union(i, i + 1)
            if r + 1 < nr and similar(i, i + nc):
                union(i, i + nc)

    roots = np.array([find(i) for i in range(nr * nc)])
    # Absorb tiny zones into their most-similar neighbor zone.
    labels_of_root: dict[int, int] = {}
    counts: dict[int, int] = {}
    for root in roots:
        counts[root] = counts.get(root, 0) + 1
    big = {r for r, n in counts.items() if n >= max(1, MIN_ZONE_FRACTION * nr * nc)}
    if not big:
        big = {max(counts, key=counts.get)}
    for root in sorted(big, key=lambda r: -counts[r]):
        labels_of_root[root] = len(labels_of_root)
    # Tiny zones (roof-dominated blocks, small artifacts) are absorbed into
    # the SPATIALLY ADJACENT big zone with the most shared boundary — NOT the
    # most similar-looking one, which could sit in a different strip and
    # would drag the block under the wrong correction.
    block_lbl = np.full(nr * nc, -1, np.int32)
    for i, root in enumerate(roots):
        if root in labels_of_root:
            block_lbl[i] = labels_of_root[root]
    grid = block_lbl.reshape(nr, nc)
    for _ in range(nr + nc):  # propagate until every tiny block is claimed
        todo = np.argwhere(grid < 0)
        if not len(todo):
            break
        changed = False
        for ri, ci in todo:
            votes: dict[int, int] = {}
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                r2, c2 = ri + dr, ci + dc
                if 0 <= r2 < nr and 0 <= c2 < nc and grid[r2, c2] >= 0:
                    votes[int(grid[r2, c2])] = votes.get(int(grid[r2, c2]), 0) + 1
            if votes:
                grid[ri, ci] = max(votes, key=votes.get)
                changed = True
        if not changed:  # isolated region with no labeled neighbor (shouldn't happen)
            grid[grid < 0] = 0
            break
    block_lbl = grid

    labels = np.repeat(np.repeat(block_lbl, bs, axis=0), bs, axis=1)[: gray.shape[0], : gray.shape[1]]
    if labels.shape != gray.shape:  # bottom/right remainder blocks
        pad_r = gray.shape[0] - labels.shape[0]
        pad_c = gray.shape[1] - labels.shape[1]
        labels = np.pad(labels, ((0, pad_r), (0, pad_c)), mode="edge")
    return labels.astype(np.int32), ZoneStats(n_zones=len(labels_of_root))


def normalize_zones(ortho: OrthoImage, block_m: float = 64.0) -> tuple[OrthoImage, ZoneStats]:
    """Return a radiometrically consistent copy of `ortho` (+ stats).
    No-op (same object) when the mosaic is already homogeneous.

    The per-zone correction is applied SHARPLY at zone boundaries — that is
    what cancels the seam (a blended correction leaves the jump in place:
    halfway across the seam each side keeps half its offset). Sharp is safe
    here because zones only split where block medians jump by more than
    LUM_SPLIT: gradual drifts (vignettes, haze gradients) never split, so a
    boundary always sits on a real discontinuity of the data."""
    labels, stats = radiometric_zones(ortho.gray, ortho.rgb, ortho.res_m, block_m)
    if stats.n_zones <= 1:
        return ortho, stats

    zones = np.unique(labels)
    sizes = {int(z): int((labels == z).sum()) for z in zones}
    ref = max(sizes, key=sizes.get)
    g_ref = ortho.gray[labels == ref]
    ref_med = float(np.median(g_ref))
    ref_mad = float(np.median(np.abs(g_ref - ref_med))) or 1e-3

    n = int(zones.max()) + 1
    med_arr = np.zeros(n, np.float32)
    scale_arr = np.ones(n, np.float32)
    gain_arr = np.ones((n, 3), np.float32)
    ref_cmed = (
        np.median(ortho.rgb[labels == ref].reshape(-1, 3), axis=0)
        if ortho.rgb is not None else None
    )
    for z in zones:
        m = labels == z
        gz = ortho.gray[m]
        z_med = float(np.median(gz))
        z_mad = float(np.median(np.abs(gz - z_med))) or 1e-3
        med_arr[z] = z_med
        scale_arr[z] = float(np.clip(ref_mad / z_mad, 0.5, 2.0))
        if ref_cmed is not None:
            z_cmed = np.median(ortho.rgb[m].reshape(-1, 3), axis=0)
            gain_arr[z] = np.clip(ref_cmed / np.maximum(z_cmed, 1e-3), 0.5, 2.0)
        stats.corrections.append({
            "zone": int(z), "px": sizes[int(z)],
            "d_lum": round(ref_med - z_med, 4), "lum_scale": round(float(scale_arr[z]), 3),
        })

    gray_out = np.clip(
        (ortho.gray - med_arr[labels]) * scale_arr[labels] + ref_med, 0.0, 1.0
    ).astype(np.float32)
    rgb_out = None
    if ortho.rgb is not None:
        rgb_out = np.clip(ortho.rgb * gain_arr[labels], 0.0, 1.0).astype(np.float32)
        # Keep luminance consistent with the corrected gray: rescale rgb by
        # the same luminance shift ratio zone by zone.
        lum = rgb_out @ np.array([0.299, 0.587, 0.114], np.float32)
        ratio = np.clip(gray_out / np.maximum(lum, 1e-3), 0.5, 2.0)
        rgb_out = np.clip(rgb_out * ratio[..., None], 0.0, 1.0).astype(np.float32)
    log.info(
        f"radiometric normalization: {stats.n_zones} zones, "
        f"corrections {[c['d_lum'] for c in stats.corrections]}"
    )
    return (
        OrthoImage(gray=gray_out, res_m=ortho.res_m, e0=ortho.e0,
                   n0=ortho.n0, utc=ortho.utc, rgb=rgb_out),
        stats,
    )
