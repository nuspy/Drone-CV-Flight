from __future__ import annotations

import asyncio

from dronecv.cli.console import console
from dronecv.config import load_config


def run_sim(env: str, port: int | None) -> None:
    cfg = load_config(env)
    if cfg.env.kind != "headless":
        console.print(f"[red]'{env}' is a Unity environment — start the Unity scene instead.[/red]")
        raise SystemExit(2)
    from dronecv.sim.headless.server import run_server

    try:
        asyncio.run(run_server(cfg, port))
    except KeyboardInterrupt:
        console.print("sim stopped")
