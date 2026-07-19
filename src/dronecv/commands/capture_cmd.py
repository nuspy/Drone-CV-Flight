from __future__ import annotations

import asyncio

from dronecv.cli.console import console
from dronecv.config import load_config


def run_capture(env: str, plan: str, n: int | None, seed: int | None) -> None:
    overrides = {"env": {"seed": seed}} if seed is not None else {}
    cfg = load_config(env, overrides)
    asyncio.run(_capture(cfg, plan, n))


async def _capture(cfg, plan: str, n: int | None) -> int:
    from dronecv.capture import planner
    from dronecv.capture.collector import Collector, open_sim_client
    from dronecv.commands.common import sim_endpoint

    async with sim_endpoint(cfg) as (host, port):
        cfg.sim.host, cfg.sim.port = host, port
        client, anchor = await open_sim_client(cfg)
        try:
            collector = Collector(cfg, client, anchor)
            info = await collector.env_info()
            import numpy as np

            bmin, bmax = np.array(info.bounds_min_sim), np.array(info.bounds_max_sim)
            if plan == "grid":
                poses = planner.grid_plan(bmin, bmax, cfg.capture)
            elif plan == "orbit":
                gen = __import__("numpy").random.default_rng(cfg.env.seed)
                pts = [
                    (float(gen.uniform(bmin[0] * 0.7, bmax[0] * 0.7)), float(gen.uniform(bmin[2] * 0.7, bmax[2] * 0.7)))
                    for _ in range(8)
                ]
                poses = planner.orbit_plan(pts, cfg.capture)
            else:
                raise SystemExit(f"unknown plan '{plan}'")
            poses = planner.shuffle_and_cap(poses, n, cfg.env.seed)
            count = await collector.collect(poses, cfg.dataset_dir)
            console.print(f"[green]captured {count} views -> {cfg.dataset_dir}[/green]")
            return count
        finally:
            await client.close()
