from __future__ import annotations

import asyncio
from pathlib import Path

from dronecv.cli.console import console
from dronecv.config import load_config


def run_localize(env: str, host: str, port: int, bundle_path: str | None) -> None:
    cfg = load_config(env)
    from dronecv.localization.service import LocalizationService
    from dronecv.training.bundle import ModelBundle

    bundle = ModelBundle.load(Path(bundle_path) if bundle_path else cfg.bundle_dir)
    service = LocalizationService(cfg, bundle)

    async def main() -> None:
        await service.connect(host, port)
        try:
            async for frame_id, est in service.estimates():
                status = "LOST" if est.lost else ("ok" if est.initialized else "init")
                console.print(
                    f"[{frame_id:6d}] {est.lat:.6f}, {est.lon:.6f}  alt {est.alt_msl:7.1f} m "
                    f"(agl {est.alt_agl if est.alt_agl is not None else float('nan'):5.1f}) "
                    f"v {est.speed_ms:4.1f} m/s hdg {est.heading_deg:5.1f}  "
                    f"conf {est.confidence:.2f} [{status}]"
                )
        finally:
            await service.close()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("localizer stopped")
