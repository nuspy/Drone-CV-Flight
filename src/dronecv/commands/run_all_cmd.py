"""The one-command workflow: capture -> auto-sized training -> flight test ->
reliability report. Best case UX per the project brief: point it at an
environment and everything else is automatic."""

from __future__ import annotations

import asyncio

from dronecv.cli.console import console
from dronecv.config import load_config


def run_all_pipeline(
    env: str,
    budget: int | None,
    episodes: int | None,
    seed: int | None,
    bundle_path: str | None,
) -> int:
    overrides: dict = {}
    if seed is not None:
        overrides["env"] = {"seed": seed}
    if budget is not None:
        overrides["active_loop"] = {"budget_captures": budget}
    if episodes is not None:
        overrides["harness"] = {"episodes": episodes}
    cfg = load_config(env, overrides)

    console.rule(f"[bold]dronecv run-all — {env}")

    if bundle_path is None:
        console.print("[bold cyan]stage 1/2 — training (active loop, auto-sizing data)[/bold cyan]")
        from dronecv.training.active_loop import run_active_loop

        result = asyncio.run(run_active_loop(cfg))
        console.print(
            f"training finished: [bold]{result['stop_reason']}[/bold] after {result['rounds']} rounds, "
            f"{result['captures_used']} captures"
        )
    else:
        console.print(f"[bold cyan]stage 1/2 — skipped (using bundle {bundle_path})[/bold cyan]")

    console.print("[bold cyan]stage 2/2 — automated flight test[/bold cyan]")
    from dronecv.commands.test_flight_cmd import run_test_flight_async

    rc = asyncio.run(run_test_flight_async(cfg, bundle_path))
    console.rule("[bold green]run-all complete" if rc == 0 else "[bold red]run-all FAILED thresholds")
    return rc
