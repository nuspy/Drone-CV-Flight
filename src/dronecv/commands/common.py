"""Shared helpers for CLI commands."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from dronecv.config import Config
from dronecv.util.logging import get_logger

log = get_logger("dronecv")


@contextlib.asynccontextmanager
async def sim_endpoint(cfg: Config) -> AsyncIterator[tuple[str, int]]:
    """Yield (host, port) of a live simulator.

    Headless env: start the in-process sim server on an ephemeral port.
    Unity env: expect the Unity bridge to be listening at the configured
    host:port (same protocol, indistinguishable from here on).
    """
    if cfg.env.kind == "headless":
        from dronecv.sim.headless.server import HeadlessSimServer

        server = HeadlessSimServer(cfg)
        host, port = await server.start(port=0)
        try:
            yield host, port
        finally:
            await server.stop()
    else:
        log.info(
            f"waiting for Unity bridge at {cfg.sim.host}:{cfg.sim.port} "
            f"(open the scene with the DroneCV SimLoop and press Play)"
        )
        yield cfg.sim.host, cfg.sim.port


def apply_overrides(cfg_env: str, **kv) -> dict:
    """Build a config override dict from non-None CLI flags."""
    overrides: dict = {}
    for dotted, value in kv.items():
        if value is None:
            continue
        keys = dotted.split("__")
        node = overrides
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return overrides
