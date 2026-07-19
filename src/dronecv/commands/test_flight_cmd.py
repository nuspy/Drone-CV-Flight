from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dronecv.cli.console import console, metric_table
from dronecv.config import Config, load_config
from dronecv.geo.anchor import GeoAnchor


def run_test_flight(env: str, bundle_path: str | None, episodes: int | None, seed: int | None) -> int:
    overrides: dict = {}
    if seed is not None:
        overrides["env"] = {"seed": seed}
    if episodes is not None:
        overrides["harness"] = {"episodes": episodes}
    cfg = load_config(env, overrides)
    return asyncio.run(run_test_flight_async(cfg, bundle_path))


async def run_test_flight_async(cfg: Config, bundle_path: str | None) -> int:
    from dronecv.commands.common import sim_endpoint
    from dronecv.harness.flight_test import FlightTestHarness
    from dronecv.harness.report import write_report
    from dronecv.localization.tile_router import is_tiled_bundle
    from dronecv.training.bundle import ModelBundle

    bundle_dir = Path(bundle_path) if bundle_path else cfg.bundle_dir
    bundle = bundle_dir if is_tiled_bundle(bundle_dir) else ModelBundle.load(bundle_dir)
    async with sim_endpoint(cfg) as (host, port):
        cfg.sim.host, cfg.sim.port = host, port
        harness = FlightTestHarness(cfg, bundle)
        results = await harness.run()

    training_report = None
    tr_path = cfg.artifacts_dir / "training_report.json"
    if tr_path.exists():
        tr = json.loads(tr_path.read_text())
        training_report = {
            "stop_reason": tr["stop_reason"],
            "rounds": tr["rounds"],
            "captures_used": tr["captures_used"],
            "budget": tr["budget"],
            "history_curve": [
                {
                    "captures_total": h["captures_total"],
                    "probe_p95_m": h["probe_p95_m"],
                    "probe_alt_p95_m": h["probe_alt_p95_m"],
                }
                for h in tr["history"]
            ],
        }

    path = write_report(
        cfg.report_dir,
        cfg.env.name or "?",
        (
            bundle.anchor.to_dict()
            if isinstance(bundle, ModelBundle)
            else GeoAnchor.resolve(cfg.env.anchor, None).to_dict()
        ),
        results,
        training_report,
    )
    acc, reach = results["accuracy"], results["reach"]
    console.print(
        metric_table(
            f"flight test {'PASSED' if results['passed'] else 'FAILED'}",
            {
                "pos err p50/p95": f"{acc.get('pos_err_h_p50_m'):.1f} / {acc.get('pos_err_h_p95_m'):.1f} m"
                if acc.get("pos_err_h_p95_m") is not None
                else "—",
                "alt MAE": f"{acc.get('alt_err_mae_m'):.1f} m" if acc.get("alt_err_mae_m") else "—",
                "kidnap recovery": f"{acc.get('kidnap_recovery_s')} s",
                "reach success": f"{reach.get('success_rate', 0):.0%} of {reach.get('n_episodes')}",
                "standoff violations": str(reach.get("standoff_violations")),
                "report": str(path),
            },
        )
    )
    return 0 if results["passed"] else 1
