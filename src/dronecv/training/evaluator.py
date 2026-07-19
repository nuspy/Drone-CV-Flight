"""Evaluation: APR/retrieval accuracy, per-cell error map, confidence calibration.

Two evaluation sources:
- the spatial holdout cells of the dataset (never trained on);
- optionally a FRESH probe set of captures at random poses never seen at all
  (the active loop uses this to defeat any residual spatial leakage).

Also fits the APR sigma calibration scale: networks are systematically
overconfident, so we scale predicted sigma such that the empirical 90%
quantile of normalized errors matches the Rayleigh 90% quantile. The scale is
stored in the bundle manifest and applied by the localizer.
"""

from __future__ import annotations

import numpy as np
import torch

from dronecv.capture.dataset import CaptureDataset, TorchCaptureView
from dronecv.config import Config
from dronecv.geo.frames import wrap_deg
from dronecv.models.retrieval import embed_images
from dronecv.training.bundle import ModelBundle

RAYLEIGH_Q = {0.5: 1.1774, 0.68: 1.5096, 0.9: 2.1460, 0.95: 2.4477}


def _batched_predict(bundle: ModelBundle, imgs: torch.Tensor, batch: int = 64) -> dict[str, np.ndarray]:
    outs: dict[str, list[np.ndarray]] = {}
    bundle.pose_net.eval()
    for i in range(0, len(imgs), batch):
        pred = bundle.pose_net.predict(imgs[i : i + batch])
        for k, v in pred.items():
            outs.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def evaluate_bundle(
    cfg: Config,
    bundle: ModelBundle,
    ds: CaptureDataset,
    eval_indices: list[int],
    fit_calibration: bool = False,
) -> dict:
    """Returns a metrics dict; optionally fits and installs the sigma scale."""
    if not eval_indices:
        return {"n_eval": 0}
    view = TorchCaptureView(ds, eval_indices)
    imgs = torch.stack([view[i][0] for i in range(len(view))])
    gt_pos = ds.positions_enu()[eval_indices]
    gt_heading = ds.headings_deg()[eval_indices]

    # --- APR ---
    pred = _batched_predict(bundle, imgs)
    err_vec = pred["pos_enu"] - gt_pos
    err_h = np.linalg.norm(err_vec[:, :2], axis=1)
    err_alt = np.abs(err_vec[:, 2])
    err_heading = np.abs([wrap_deg(p - g) for p, g in zip(pred["heading_deg"], gt_heading, strict=True)])

    # --- retrieval ---
    embeddings = embed_images(bundle.embed_net, imgs)
    retr_err_h, recall_hits = [], []
    radius = cfg.training.pos_radius_m
    for i, desc in enumerate(embeddings):
        fix = bundle.landmark_db.query(desc, topk=cfg.localization.retrieval_topk)
        retr_err_h.append(float(np.linalg.norm(fix["pos_enu"][:2] - gt_pos[i, :2])))
        neigh = bundle.landmark_db.pos_enu[fix["indices"]]
        dists = np.linalg.norm(neigh[:, :2] - gt_pos[i, :2], axis=1)
        recall_hits.append(bool((dists < radius).any()))
    retr_err_h = np.array(retr_err_h)

    # --- calibration of APR sigma ---
    sigma = np.maximum(pred["sigma_pos_m"], 1e-3)
    z = err_h / sigma
    scale = 1.0
    if fit_calibration and len(z) >= 20:
        scale = float(np.quantile(z, 0.9) / RAYLEIGH_Q[0.9])
        scale = float(np.clip(scale, 0.2, 50.0))
        bundle.manifest["calibration"]["apr_sigma_scale"] = scale
    eff_sigma = sigma * bundle.manifest["calibration"]["apr_sigma_scale"]
    zc = err_h / np.maximum(eff_sigma, 1e-3)
    ece = float(
        np.mean([abs((zc < q).mean() - p) for p, q in RAYLEIGH_Q.items()])
    )

    # --- per-cell p95 error map (drives targeted capture) ---
    cell_m = cfg.training.val_cell_m
    cells: dict[tuple[int, int], list[float]] = {}
    for i, idx in enumerate(eval_indices):
        cell = ds.cell_of(idx, cell_m)
        cells.setdefault(cell, []).append(float(err_h[i]))
    cell_p95 = {
        f"{cx},{cz}": float(np.percentile(v, 95)) for (cx, cz), v in cells.items()
    }

    metrics = {
        "n_eval": len(eval_indices),
        "apr": {
            "pos_err_h_p50_m": float(np.percentile(err_h, 50)),
            "pos_err_h_p95_m": float(np.percentile(err_h, 95)),
            "pos_err_h_rmse_m": float(np.sqrt((err_h**2).mean())),
            "alt_err_p95_m": float(np.percentile(err_alt, 95)),
            "heading_err_p50_deg": float(np.percentile(err_heading, 50)),
        },
        "retrieval": {
            "recall_at_k": float(np.mean(recall_hits)),
            "fix_err_h_p50_m": float(np.percentile(retr_err_h, 50)),
            "fix_err_h_p95_m": float(np.percentile(retr_err_h, 95)),
        },
        "calibration": {
            "apr_sigma_scale": float(bundle.manifest["calibration"]["apr_sigma_scale"]),
            "ece": ece,
        },
        "cell_p95_m": cell_p95,
        "cell_size_m": cell_m,
    }
    return metrics


def combined_p95(metrics: dict) -> float:
    """Headline reliability number: the better cue's p95 horizontal error
    (the fused filter tracks the best available absolute cue)."""
    return min(metrics["apr"]["pos_err_h_p95_m"], metrics["retrieval"]["fix_err_h_p95_m"])
