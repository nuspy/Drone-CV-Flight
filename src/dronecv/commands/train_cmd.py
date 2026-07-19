from __future__ import annotations

from dronecv.cli.console import console, metric_table
from dronecv.config import load_config


def run_train(env: str, rounds: int | None, budget: int | None, seed: int | None) -> None:
    overrides: dict = {}
    if seed is not None:
        overrides["env"] = {"seed": seed}
    al: dict = {}
    if rounds is not None:
        al["max_rounds"] = rounds
    if budget is not None:
        al["budget_captures"] = budget
    if al:
        overrides["active_loop"] = al
    cfg = load_config(env, overrides)

    import asyncio

    if cfg.active_loop.tile_m > 0:
        from dronecv.training.tiled import run_tiled_training

        result = asyncio.run(run_tiled_training(cfg))
    else:
        from dronecv.training.active_loop import run_active_loop

        result = asyncio.run(run_active_loop(cfg))
    metrics = result["final_metrics"]
    console.print(
        metric_table(
            f"training complete ({result['stop_reason']})",
            {
                "rounds": str(result["rounds"]),
                "captures used": str(result["captures_used"]),
                "APR p95 err": f"{metrics['apr']['pos_err_h_p95_m']:.1f} m",
                "retrieval p95 err": f"{metrics['retrieval']['fix_err_h_p95_m']:.1f} m",
                "recall@k": f"{metrics['retrieval']['recall_at_k']:.2f}",
                "ECE": f"{metrics['calibration']['ece']:.3f}",
                "bundle": str(cfg.bundle_dir),
            },
        )
    )
