"""Geometric distinctiveness ("saliency") of a built GIS world.

Answers request 3 of the GIS extension: environments with recognizable macro
elements (mountains, unique towers) need FEWER visual references than
repetitive residential fabric. The map is a per-cell density multiplier for
the initial capture plan; the active loop's measured error map remains the
final judge and keeps refining after this prior.

Per cell (default 250 m):
- terrain relief (std of ground heights): mountains localize well;
- building height entropy: uniform rooftops = ambiguous, varied = distinctive;
- skyline rarity: how unique the cell's height histogram is vs ALL other
  cells (repetitive districts look like each other -> common -> dense POVs);
- landmark bonus: presence of out-of-scale structures.

Multiplier = clamp(base / distinctiveness), in [0.5, 2.2]: distinctive cells
get sparser initial coverage, repetitive ones denser.
"""

from __future__ import annotations

import numpy as np

from dronecv.gis.store import GisStore


def compute_saliency(
    store: GisStore, cell_m: float = 250.0
) -> dict:
    m = store.meta
    cell_px = max(8, int(cell_m / m.res_m))
    n_r = max(1, m.height // cell_px)
    n_c = max(1, m.width // cell_px)

    relief = np.zeros((n_r, n_c), dtype=np.float32)
    h_entropy = np.zeros((n_r, n_c), dtype=np.float32)
    landmark = np.zeros((n_r, n_c), dtype=np.float32)
    hists = np.zeros((n_r, n_c, 8), dtype=np.float32)

    bins = np.array([0.0, 3.0, 6.0, 10.0, 15.0, 25.0, 40.0, 70.0, 1e9])
    for r in range(n_r):
        for c in range(n_c):
            sl = np.s_[r * cell_px : (r + 1) * cell_px, c * cell_px : (c + 1) * cell_px]
            ground = np.asarray(store.ground[sl], dtype=np.float32)
            builds = np.asarray(store.build_h[sl], dtype=np.float32)
            relief[r, c] = float(ground.std())
            built = builds[builds > 1.0]
            if built.size:
                hist, _ = np.histogram(built, bins=bins)
                p = hist / hist.sum()
                nz = p[p > 0]
                h_entropy[r, c] = float(-(nz * np.log(nz)).sum())
                landmark[r, c] = float(built.max() > 3.0 * max(np.median(built), 1.0))
            hists[r, c] = np.histogram(builds, bins=bins)[0]

    # Skyline rarity: 1 - max cosine similarity of the cell's height histogram
    # with any OTHER cell (identical residential blocks -> similarity ~1).
    flat = hists.reshape(-1, hists.shape[-1])
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    normed = flat / np.maximum(norm, 1e-6)
    sim = normed @ normed.T
    np.fill_diagonal(sim, -1.0)
    rarity = (1.0 - sim.max(axis=1)).reshape(n_r, n_c)
    rarity[norm.reshape(n_r, n_c) < 1e-6] = 0.0  # empty cells are not "rare"

    # Distinctiveness in [0, ~1.5+], then density multiplier.
    distinct = (
        np.clip(relief / 15.0, 0, 1) * 0.35
        + np.clip(h_entropy / 1.6, 0, 1) * 0.30
        + np.clip(rarity / 0.5, 0, 1) * 0.20
        + landmark * 0.15
    )
    density = np.clip(1.35 - distinct, 0.5, 2.2)

    return {
        "cell_m": float(cell_px * m.res_m),
        "shape": [int(n_r), int(n_c)],
        "distinctiveness": distinct.tolist(),
        "density_multiplier": density.tolist(),
        "mean_density_multiplier": float(density.mean()),
    }


def orbit_points_from_store(store: GisStore, top_n: int = 20, min_sep_m: float = 150.0) -> list[list[float]]:
    """Anchor points for orbit capture: the tallest, well-separated structures
    (or terrain peaks when there are no buildings)."""
    m = store.meta
    build = np.asarray(store.build_h)
    source = build if build.max() > 2.0 else np.asarray(store.ground)
    flat = source.ravel()
    order = np.argsort(flat)[::-1]
    picked: list[tuple[float, float]] = []
    for idx in order[: top_n * 400]:
        if flat[idx] < 2.0 and source is build:
            break
        r, c = divmod(int(idx), m.width)
        e = m.e0 + (c + 0.5) * m.res_m
        n = m.n0 + (r + 0.5) * m.res_m
        if all((e - pe) ** 2 + (n - pn) ** 2 > min_sep_m**2 for pe, pn in picked):
            picked.append((e, n))
            if len(picked) >= top_n:
                break
    return [[float(e), float(n)] for e, n in picked]
