from __future__ import annotations

import json
from pathlib import Path

from dronecv.cli.console import console, metric_table
from dronecv.config import load_config


def run_evaluate(env: str, bundle_path: str | None) -> None:
    cfg = load_config(env)
    from dronecv.training.bundle import ModelBundle
    from dronecv.training.evaluator import evaluate_bundle
    from dronecv.training.trainer import load_dataset

    bundle = ModelBundle.load(Path(bundle_path) if bundle_path else cfg.bundle_dir)
    ds = load_dataset(cfg)
    _, val_idx = ds.spatial_split(cfg.training.val_cell_m, cfg.training.val_fraction, seed=cfg.env.seed)
    metrics = evaluate_bundle(cfg, bundle, ds, val_idx)
    out = cfg.artifacts_dir / "eval_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    console.print(
        metric_table(
            f"evaluation on {metrics['n_eval']} held-out captures",
            {
                "APR p50 / p95": f"{metrics['apr']['pos_err_h_p50_m']:.1f} / {metrics['apr']['pos_err_h_p95_m']:.1f} m",
                "APR alt p95": f"{metrics['apr']['alt_err_p95_m']:.1f} m",
                "retrieval p50 / p95": f"{metrics['retrieval']['fix_err_h_p50_m']:.1f} / {metrics['retrieval']['fix_err_h_p95_m']:.1f} m",
                "recall@k": f"{metrics['retrieval']['recall_at_k']:.2f}",
                "ECE": f"{metrics['calibration']['ece']:.3f}",
                "saved": str(out),
            },
        )
    )
