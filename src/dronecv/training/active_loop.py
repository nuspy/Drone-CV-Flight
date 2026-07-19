"""The active training loop — auto-sizes the dataset for each environment.

    round 0: broad grid+orbit capture (initial_fraction of the budget)
    repeat:
        train (warm-started)  ->  evaluate on spatial holdout + FRESH probes
        stop if thresholds met / plateau / budget exhausted / max rounds
        else: spend growth_fraction more captures, allocated by softmax over
              per-cell excess error (TargetedPlan), and go again

"How much data does this environment need" is therefore a measured output,
not a guess; the error-vs-data curve lands in the training report.
"""

from __future__ import annotations

import json
import shutil

import numpy as np

from dronecv.capture import planner
from dronecv.capture.collector import Collector, open_sim_client
from dronecv.capture.dataset import CaptureDataset
from dronecv.commands.common import sim_endpoint
from dronecv.config import Config
from dronecv.training.bundle import ModelBundle
from dronecv.training.evaluator import combined_p95, evaluate_bundle
from dronecv.training.trainer import train_models
from dronecv.util.logging import get_logger

log = get_logger("dronecv.active")


def _targeted_cells(metrics: dict, threshold_m: float) -> list[tuple[float, float, float]]:
    """Cells whose p95 exceeds the threshold, weighted by the excess (softmax)."""
    cell_m = metrics["cell_size_m"]
    excess = {}
    for key, p95 in metrics["cell_p95_m"].items():
        if p95 > threshold_m:
            cx, cz = (int(v) for v in key.split(","))
            excess[(cx, cz)] = p95 - threshold_m
    if not excess:
        return []
    vals = np.array(list(excess.values()))
    weights = np.exp(vals / max(vals.max(), 1e-6))
    weights /= weights.sum()
    return [
        ((cx + 0.5) * cell_m, (cz + 0.5) * cell_m, float(w))
        for (cx, cz), w in zip(excess.keys(), weights, strict=True)
    ]


async def run_active_loop(cfg: Config, bounds_sim: tuple | None = None) -> dict:
    """`bounds_sim=(bmin, bmax)` restricts capture/eval to a sub-volume — the
    hierarchical tile trainer uses it to train one tile at a time (with the
    artifact paths scoped by cfg.env.name, which it suffixes per tile)."""
    al = cfg.active_loop
    history: list[dict] = []
    captures_used = 0
    stop_reason = "max_rounds"

    async with sim_endpoint(cfg) as (host, port):
        cfg.sim.host, cfg.sim.port = host, port
        client, anchor = await open_sim_client(cfg)
        try:
            collector = Collector(cfg, client, anchor)
            info = await collector.env_info()
            if bounds_sim is not None:
                bmin, bmax = np.array(bounds_sim[0]), np.array(bounds_sim[1])
            else:
                bmin, bmax = np.array(info.bounds_min_sim), np.array(info.bounds_max_sim)

            # ---- round 0 capture: broad coverage ----
            if (cfg.dataset_dir / "meta.jsonl").exists():
                log.info(f"resuming with existing dataset at {cfg.dataset_dir}")
            else:
                n0 = max(32, int(al.budget_captures * al.initial_fraction))
                density_fn, orbit_pts = _saliency_prior(cfg)
                poses = planner.grid_plan(bmin, bmax, cfg.capture, density=density_fn)
                # Orbit clusters: saliency anchors (GIS worlds: tallest/most
                # distinctive structures) or random interest points otherwise.
                if not orbit_pts:
                    gen = np.random.default_rng(cfg.env.seed + 101)
                    orbit_pts = [
                        (float(gen.uniform(bmin[0] * 0.6, bmax[0] * 0.6)), float(gen.uniform(bmin[2] * 0.6, bmax[2] * 0.6)))
                        for _ in range(6)
                    ]
                poses += planner.orbit_plan(orbit_pts[:12], cfg.capture)
                poses = planner.shuffle_and_cap(poses, n0, cfg.env.seed)
                captures_used += await collector.collect(poses, cfg.dataset_dir, "capture r0 (grid+orbit)")

            bundle: ModelBundle | None = None
            prev_p95: float | None = None
            plateau_count = 0

            for round_idx in range(1, al.max_rounds + 1):
                ds = CaptureDataset(cfg.dataset_dir)
                bundle, split = train_models(cfg, ds, warm_start=bundle)

                # Holdout evaluation fits the sigma calibration.
                val_metrics = evaluate_bundle(cfg, bundle, ds, split["val_idx"], fit_calibration=True)

                # Fresh probes: never-seen random poses (strongest generalization signal).
                probe_dir = cfg.artifacts_dir / f"probe_r{round_idx}"
                if probe_dir.exists():
                    shutil.rmtree(probe_dir)
                probe_poses = planner.random_plan(bmin, bmax, al.probe_captures, cfg.capture, cfg.env.seed + round_idx)
                await collector.collect(probe_poses, probe_dir, f"probes r{round_idx}")
                probe_ds = CaptureDataset(probe_dir)
                probe_metrics = evaluate_bundle(cfg, bundle, probe_ds, list(range(len(probe_ds))))

                p95 = combined_p95(probe_metrics)
                alt_p95 = probe_metrics["apr"]["alt_err_p95_m"]
                ece = probe_metrics["calibration"]["ece"]
                entry = {
                    "round": round_idx,
                    "captures_total": len(ds),
                    "probe_p95_m": p95,
                    "probe_alt_p95_m": alt_p95,
                    "probe_ece": ece,
                    "val": val_metrics,
                    "probe": probe_metrics,
                }
                history.append(entry)
                log.info(
                    f"round {round_idx}: {len(ds)} captures | probe p95 {p95:.1f} m | "
                    f"alt p95 {alt_p95:.1f} m | ECE {ece:.3f}"
                )

                thr = al.thresholds
                if p95 <= thr.p95_pos_err_m and alt_p95 <= thr.p95_alt_err_m and ece <= thr.max_ece:
                    stop_reason = "converged"
                    break
                if prev_p95 is not None:
                    improvement = (prev_p95 - p95) / max(prev_p95, 1e-6)
                    plateau_count = plateau_count + 1 if improvement < al.plateau_epsilon else 0
                    if plateau_count >= al.plateau_rounds:
                        stop_reason = "plateau"
                        break
                prev_p95 = p95
                if captures_used >= al.budget_captures:
                    stop_reason = "budget_exhausted"
                    break
                if round_idx == al.max_rounds:
                    break

                # ---- targeted refinement capture ----
                delta_n = min(
                    int(al.budget_captures * al.initial_fraction * al.growth_fraction),
                    al.budget_captures - captures_used,
                )
                cells = _targeted_cells(probe_metrics, thr.p95_pos_err_m)
                cells += _targeted_cells(val_metrics, thr.p95_pos_err_m)
                if cells:
                    poses = planner.targeted_plan(
                        _cells_to_sim(cells, anchor), val_metrics["cell_size_m"], delta_n, cfg.capture,
                        cfg.env.seed + 1000 + round_idx,
                    )
                else:  # errors are diffuse: densify uniformly
                    poses = planner.random_plan(bmin, bmax, delta_n, cfg.capture, cfg.env.seed + 500 + round_idx)
                poses = _clip_to_bounds(poses, bmin, bmax)
                captures_used += await collector.collect(
                    poses, cfg.dataset_dir, f"targeted capture r{round_idx}"
                )
        finally:
            await client.close()

    assert bundle is not None
    final_metrics = history[-1]["probe"]
    bundle.manifest["metrics"] = final_metrics
    bundle.manifest["training_history"] = [
        {k: e[k] for k in ("round", "captures_total", "probe_p95_m", "probe_alt_p95_m", "probe_ece")}
        for e in history
    ]
    bundle.manifest["stop_reason"] = stop_reason
    bundle.save(cfg.bundle_dir)

    report = {
        "env": cfg.env.name,
        "stop_reason": stop_reason,
        "rounds": len(history),
        "captures_used": history[-1]["captures_total"],
        "budget": al.budget_captures,
        "thresholds": al.thresholds.model_dump(),
        "history": history,
    }
    out = cfg.artifacts_dir / "training_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info(f"active loop finished ({stop_reason}) — bundle at {cfg.bundle_dir}")
    return {
        "stop_reason": stop_reason,
        "rounds": len(history),
        "captures_used": history[-1]["captures_total"],
        "final_metrics": final_metrics,
        "history": history,
    }


def _saliency_prior(cfg: Config):
    """(density_fn, orbit_points) from a GIS world's saliency map; (None, [])
    for procedural/unity environments."""
    if cfg.world.kind != "gis" or not cfg.world.gis_dir:
        return None, []
    import json
    from pathlib import Path

    gis_dir = Path(cfg.world.gis_dir)
    orbit: list[tuple[float, float]] = []
    density_fn = None
    meta_path = gis_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        orbit = [(float(e), float(n)) for e, n in meta.get("orbit_points", [])]
    sal_path = gis_dir / "saliency.json"
    if sal_path.exists():
        sal = json.loads(sal_path.read_text())
        dens = np.array(sal["density_multiplier"], dtype=float)
        cell = float(sal["cell_m"])
        meta = json.loads(meta_path.read_text())
        e0, n0 = float(meta["e0"]), float(meta["n0"])

        def density_fn(x: float, z: float) -> float:  # sim x=E, z=N (gis worlds)
            r = int((z - n0) // cell)
            c = int((x - e0) // cell)
            if 0 <= r < dens.shape[0] and 0 <= c < dens.shape[1]:
                return float(dens[r, c])
            return 1.0

        log.info(
            f"saliency prior active: mean density x{sal['mean_density_multiplier']:.2f}, "
            f"{len(orbit)} orbit anchors"
        )
    return density_fn, orbit


def _cells_to_sim(cells: list[tuple[float, float, float]], anchor) -> list[tuple[float, float, float]]:
    """Cell centers are in ENU (evaluation space); the planner emits sim-frame
    x/z, so convert each center through the anchor."""
    out = []
    for e, n, w in cells:
        sim = anchor.enu_to_sim(np.array([e, n, 0.0]))
        out.append((float(sim[0]), float(sim[2]), w))
    return out


def _clip_to_bounds(poses, bmin, bmax, margin: float = 10.0):
    from dronecv.capture.planner import CapturePose

    out = []
    for p in poses:
        x = float(np.clip(p.x, bmin[0] + margin, bmax[0] - margin))
        z = float(np.clip(p.z, bmin[2] + margin, bmax[2] - margin))
        out.append(CapturePose(x, z, p.agl_m, p.yaw_deg, p.pitch_deg))
    return out
