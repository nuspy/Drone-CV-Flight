from __future__ import annotations

from pathlib import Path

from dronecv.cli.console import console, metric_table
from dronecv.config import load_config


def run_localize_photo(env: str, image_path: str, bundle_path: str | None) -> None:
    import cv2

    from dronecv.localization.single_shot import SingleShotLocalizer
    from dronecv.training.bundle import ModelBundle

    cfg = load_config(env)
    bundle = ModelBundle.load(Path(bundle_path) if bundle_path else cfg.bundle_dir)
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        console.print(f"[red]could not read image {image_path}[/red]")
        raise SystemExit(2)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fix = SingleShotLocalizer(bundle).localize(rgb)
    console.print(
        metric_table(
            "photo localization",
            {
                "latitude": f"{fix.lat:.6f}",
                "longitude": f"{fix.lon:.6f}",
                "altitude MSL": f"{fix.alt_msl:.1f} m",
                "heading": f"{fix.heading_deg:.1f} deg",
                "confidence": f"{fix.confidence:.2f}",
                "sigma": f"{fix.sigma_h_m:.1f} m",
                "maps": f"https://maps.google.com/?q={fix.lat:.6f},{fix.lon:.6f}",
            },
        )
    )
