"""M3 — constellation verification: the judge against look-alike buildings.

Two buildings can look alike; the LAYOUT of five neighbors almost never
does. From the query view the visible buildings are extracted as ground
positions (connected components of the building mask, centroids of their
world hit points). The model side takes the footprint centroids from
buildings.json near the hypothesized pose. Both point sets are normalized by
their mean pairwise distance — so only RATIOS matter, never absolute meters
(monocular scale is unknown by design) — and greedily matched; the score is
matched-fraction damped by the mean residual.

A pose whose observed constellation does not embed in the model's is vetoed
regardless of how well a single facade matched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MIN_BLOB_PX = 6


def extract_buildings(view) -> np.ndarray:
    """Observed building ground positions (K, 2) in world XZ, one per visible
    building blob (connected components, 4-neighborhood, BFS)."""
    mask = view.building
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            pix: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pix.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            if len(pix) < MIN_BLOB_PX:
                continue
            rows = np.array([p[0] for p in pix])
            cols = np.array([p[1] for p in pix])
            pts = view.points[rows, cols]
            e, n = float(np.median(pts[:, 0])), float(np.median(pts[:, 2]))
            if np.isfinite(e) and np.isfinite(n):  # photos w/o depth: no 3D
                out.append(np.array([e, n]))
    return np.asarray(out) if out else np.zeros((0, 2))


def load_model_buildings(gis_dir: Path) -> np.ndarray:
    """Footprint centroids (N, 2) in ENU (= sim x, z) from buildings.json."""
    path = Path(gis_dir) / "buildings.json"
    if not path.exists():
        return np.zeros((0, 2))
    cents = []
    for b in json.loads(path.read_text())["buildings"]:
        ring = np.asarray(b["ring_enu"], dtype=np.float64)
        if len(ring) >= 3:
            cents.append(ring.mean(axis=0))
    return np.asarray(cents) if cents else np.zeros((0, 2))


def _normalize(pts: np.ndarray) -> np.ndarray | None:
    """Center and scale by mean pairwise distance -> ratio space."""
    if len(pts) < 3:
        return None
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    scale = d[np.triu_indices(len(pts), 1)].mean()
    if scale <= 1e-6:
        return None
    return (pts - pts.mean(axis=0)) / scale


def constellation_score(
    observed_xy: np.ndarray,
    model_xy: np.ndarray,
    near_xy: np.ndarray | None = None,
    near_radius_m: float = 400.0,
) -> float:
    """How well the observed building layout embeds in the model's, in
    ratio space. 1.0 = perfect; ~0 = the constellation is not there."""
    if near_xy is not None and len(model_xy):
        d = np.linalg.norm(model_xy - near_xy[None, :], axis=1)
        model_xy = model_xy[d <= near_radius_m]
    obs = _normalize(observed_xy)
    mod = _normalize(model_xy)
    if obs is None:
        return 0.5  # too few OBSERVED buildings to judge: neutral, never a veto
    if mod is None:
        # We SEE a constellation of buildings but the model has (almost) none
        # near this hypothesis — the strongest possible contradiction.
        return 0.0
    # greedy nearest-neighbor matching in ratio space
    used = np.zeros(len(mod), bool)
    residuals = []
    for p in obs:
        d = np.linalg.norm(mod - p[None, :], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if np.isfinite(d[j]):
            used[j] = True
            residuals.append(d[j])
    if not residuals:
        return 0.0
    matched = len(residuals) / len(obs)
    mean_res = float(np.mean(residuals))     # in mean-pairwise-distance units
    return float(matched * np.exp(-3.0 * mean_res))
