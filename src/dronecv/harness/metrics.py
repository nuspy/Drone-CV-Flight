"""Reliability metrics computed by the automated flight test."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dronecv.geo.frames import wrap_deg


@dataclass
class FrameRecord:
    t: float
    est_pos: np.ndarray
    truth_pos: np.ndarray
    est_heading: float
    truth_heading: float
    est_vel: np.ndarray
    truth_vel: np.ndarray
    confidence: float
    initialized: bool
    lost: bool


@dataclass
class EpisodeRecord:
    episode: int
    target_source: str
    success: bool
    declared_arrival: bool
    true_dist_at_arrival_m: float | None
    min_true_dist_to_target_m: float
    standoff_violation: bool
    collided: bool
    duration_s: float
    path_length_m: float
    straight_dist_m: float
    extra: dict[str, Any] = field(default_factory=dict)


def accuracy_metrics(frames: list[FrameRecord], target_err_m: float) -> dict:
    if not frames:
        return {"n_frames": 0}
    init = [f for f in frames if f.initialized]
    err_h = np.array([np.linalg.norm(f.est_pos[:2] - f.truth_pos[:2]) for f in init])
    err_alt = np.array([abs(f.est_pos[2] - f.truth_pos[2]) for f in init])
    err_heading = np.array([abs(wrap_deg(f.est_heading - f.truth_heading)) for f in init])
    err_vel = np.array([np.linalg.norm(f.est_vel[:2] - f.truth_vel[:2]) for f in init])
    conf = np.array([f.confidence for f in init])
    within = (err_h < target_err_m).astype(float)

    # Confidence calibration: bin by confidence, compare against empirical
    # P(err < target). ECE = weighted mean absolute gap.
    bins = np.linspace(0, 1, 6)
    ece_terms = []
    reliability = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        sel = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if sel.sum() >= 5:
            gap = abs(float(conf[sel].mean()) - float(within[sel].mean()))
            ece_terms.append((sel.sum(), gap))
            reliability.append(
                {"bin": [float(lo), float(hi)], "mean_conf": float(conf[sel].mean()),
                 "empirical": float(within[sel].mean()), "n": int(sel.sum())}
            )
    total = sum(n for n, _ in ece_terms)
    ece = float(sum(n * g for n, g in ece_terms) / total) if total else None

    # Time to first fix: first frame whose error stays under 2x target.
    ttff = None
    for i, f in enumerate(frames):
        if f.initialized and np.linalg.norm(f.est_pos[:2] - f.truth_pos[:2]) < 2 * target_err_m:
            ttff = frames[i].t - frames[0].t
            break

    return {
        "n_frames": len(frames),
        "n_initialized": len(init),
        "pos_err_h_p50_m": float(np.percentile(err_h, 50)) if len(err_h) else None,
        "pos_err_h_p95_m": float(np.percentile(err_h, 95)) if len(err_h) else None,
        "pos_err_h_rmse_m": float(np.sqrt((err_h**2).mean())) if len(err_h) else None,
        "alt_err_mae_m": float(err_alt.mean()) if len(err_alt) else None,
        "heading_err_p50_deg": float(np.percentile(err_heading, 50)) if len(err_heading) else None,
        "vel_err_p50_ms": float(np.percentile(err_vel, 50)) if len(err_vel) else None,
        "mean_confidence": float(conf.mean()) if len(conf) else None,
        "within_target_frac": float(within.mean()) if len(within) else None,
        "confidence_ece": ece,
        "reliability_bins": reliability,
        "time_to_first_fix_s": ttff,
    }


def reach_metrics(episodes: list[EpisodeRecord]) -> dict:
    if not episodes:
        return {"n_episodes": 0}
    succ = [e for e in episodes if e.success]
    eff = [
        e.straight_dist_m / e.path_length_m
        for e in episodes
        if e.path_length_m > 1.0 and e.success
    ]
    return {
        "n_episodes": len(episodes),
        "success_rate": len(succ) / len(episodes),
        "declared_arrival_rate": sum(e.declared_arrival for e in episodes) / len(episodes),
        "mean_duration_s": float(np.mean([e.duration_s for e in succ])) if succ else None,
        "path_efficiency": float(np.mean(eff)) if eff else None,
        "standoff_violations": sum(e.standoff_violation for e in episodes),
        "collisions": sum(e.collided for e in episodes),
        "episodes": [
            {
                "episode": e.episode,
                "target_source": e.target_source,
                "success": e.success,
                "declared_arrival": e.declared_arrival,
                "true_dist_at_arrival_m": e.true_dist_at_arrival_m,
                "min_true_dist_to_target_m": e.min_true_dist_to_target_m,
                "standoff_violation": e.standoff_violation,
                "collided": e.collided,
                "duration_s": e.duration_s,
            }
            for e in episodes
        ],
    }
