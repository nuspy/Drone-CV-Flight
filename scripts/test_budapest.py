"""Automated real-world test: Budapest, Danube banks + Buda Castle hill.

    python scripts/test_budapest.py --source osm --photos /path/to/photos
        [--budget 1200] [--skip-train] [--skip-build] [--out DIR]

Stages (each skippable/resumable):
  1. build the GIS environment 'budapest_test' (~2.0 x 2.0 km around the
     Parliament / Castle hill reach of the Danube) from real data:
     --source osm      Overpass + Copernicus DEM (free networks)
     --source overture Overture Maps on S3 (works behind AWS-only proxies)
  2. render 10 random aerial views: 5 from the Danube centerline, 5 from the
     Castle-hill area -> renders/*.png
  3. train the localization models (active loop, small budget) unless
     --skip-train
  4. localize every photo in --photos (single-shot: retrieval+APR) and render
     the synthetic view from each ESTIMATED pose -> localization/*.png +
     results.json with lat/lon/alt/heading/confidence and a Google Maps link.

Honest expectation for stage 4: the models are trained on shape-first
synthetic renders (class colors); real photos differ radically in appearance,
so confidence SHOULD come out low — that number being honest is part of what
the test verifies. The renders-from-estimated-pose make the comparison visual.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# AOI: Danube from below Castle hill to above Parliament, banks + Buda hill.
BBOX_TEXT = "47.4925,19.0290,47.5105,19.0560"
ENV_NAME = "budapest_test"
# Danube centerline (approx): south point -> north point.
DANUBE = [(47.4940, 19.0425), (47.5090, 19.0468)]
CASTLE = (47.4960, 19.0320, 47.5030, 19.0400)  # lat/lon box of the hill area


def build(source: str, res_m: float) -> None:
    from dronecv.gis.geometry import parse_bbox
    from dronecv.gis.pipeline import BuildSources, build_environment
    from dronecv.gis.providers.dem import CopernicusDem

    bbox = parse_bbox(BBOX_TEXT)
    if source == "overture":
        from dronecv.gis.providers.overture import (
            OvertureBuildingsProvider,
            OvertureLandcoverProvider,
        )

        sources = BuildSources(
            dem=CopernicusDem(),
            buildings=OvertureBuildingsProvider(),
            landcover=OvertureLandcoverProvider(),
        )
    else:
        from dronecv.gis.providers.buildings import OverpassBuildings
        from dronecv.gis.providers.landcover import OverpassLandcover
        from dronecv.gis.providers.poi import OverpassPoi

        sources = BuildSources(
            dem=CopernicusDem(),
            buildings=OverpassBuildings(),
            landcover=OverpassLandcover(),
            poi=OverpassPoi(),
        )
    build_environment(
        bbox, ENV_NAME, out_root=ROOT / "artifacts" / "gis", configs_root=ROOT,
        sources=sources, res_m=res_m,
    )


def _world():
    from dronecv.gis.world import GisWorld

    return GisWorld.open(ROOT / "artifacts" / "gis" / ENV_NAME)


def _anchor():
    from dronecv.geo.anchor import GeoAnchor
    from dronecv.gis.store import GisStore

    return GeoAnchor.from_dict(GisStore.open(ROOT / "artifacts" / "gis" / ENV_NAME).meta.anchor)


def _render(world, pos_enu, heading_deg, tilt_deg, w=512, h=384):
    from datetime import UTC, datetime

    from dronecv.geo import celestial
    from dronecv.sim.headless import rasterizer

    anchor = _anchor()
    utc = datetime(2026, 6, 21, 10, 0, tzinfo=UTC)
    sun = celestial.sun_position(utc, anchor.lat0, anchor.lon0)
    rgb, _ = rasterizer.render(
        world, np.asarray(pos_enu, float), heading_deg, tilt_deg, w, h, 70.0,
        sun.azimuth_deg, sun.elevation_deg,
    )
    return rgb


def random_renders(out_dir: Path, n_danube: int = 5, n_castle: int = 5) -> list[dict]:
    import cv2

    world = _world()
    anchor = _anchor()
    gen = np.random.default_rng(42)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n_danube + n_castle):
        if i < n_danube:
            t = gen.uniform(0, 1)
            lat = DANUBE[0][0] + t * (DANUBE[1][0] - DANUBE[0][0])
            lon = DANUBE[0][1] + t * (DANUBE[1][1] - DANUBE[0][1])
            label = "danube"
        else:
            lat = gen.uniform(CASTLE[0], CASTLE[2])
            lon = gen.uniform(CASTLE[1], CASTLE[3])
            label = "castle"
        enu = anchor.geodetic_to_enu(lat, lon, anchor.alt0)
        ground = float(world.height_at(np.array(enu[0]), np.array(enu[1])))
        alt_agl = float(gen.uniform(70, 150))
        heading = float(gen.uniform(0, 360))
        tilt = float(gen.uniform(15, 35))
        pos = np.array([enu[0], ground + alt_agl, enu[1]])
        rgb = _render(world, pos, heading, tilt)
        name = f"{i:02d}_{label}_h{heading:.0f}_agl{alt_agl:.0f}.png"
        cv2.imwrite(str(out_dir / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        records.append({
            "file": name, "lat": round(lat, 6), "lon": round(lon, 6),
            "alt_agl_m": round(alt_agl, 1), "heading_deg": round(heading, 1),
            "area": label,
        })
        print(f"render {name}  ({lat:.5f}, {lon:.5f})")
    (out_dir / "renders.json").write_text(json.dumps(records, indent=1))
    return records


def train(budget: int) -> None:
    from dronecv.config import load_config
    from dronecv.training.active_loop import run_active_loop

    cfg = load_config(ENV_NAME, overrides={
        "active_loop": {"budget_captures": budget, "max_rounds": 2, "probe_captures": 60},
        "training": {"epochs_per_round": 10},
    }, root=ROOT)
    result = asyncio.run(run_active_loop(cfg))
    print(f"training: {result['stop_reason']} after {result['captures_used']} captures")


def localize_photos(photos_dir: Path, out_dir: Path) -> None:
    import cv2

    from dronecv.config import load_config
    from dronecv.localization.single_shot import SingleShotLocalizer
    from dronecv.training.bundle import ModelBundle

    cfg = load_config(ENV_NAME, root=ROOT)
    bundle = ModelBundle.load(cfg.bundle_dir)
    localizer = SingleShotLocalizer(bundle)
    world = _world()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for photo in sorted(photos_dir.glob("*.[jJpP]*[gG]")):
        bgr = cv2.imread(str(photo))
        if bgr is None:
            continue
        fix = localizer.localize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        # Render from the ESTIMATED pose.
        pos = np.array([fix.pos_enu[0], fix.pos_enu[2], fix.pos_enu[1]])
        render = _render(world, pos, fix.heading_deg, 20.0)
        side = _side_by_side(bgr, render)
        out_name = f"loc_{photo.stem}.png"
        cv2.imwrite(str(out_dir / out_name), side)
        rec = {
            "photo": photo.name,
            "lat": round(fix.lat, 6), "lon": round(fix.lon, 6),
            "alt_msl": round(fix.alt_msl, 1),
            "heading_deg": round(fix.heading_deg, 1),
            "confidence": round(fix.confidence, 3),
            "sigma_h_m": round(fix.sigma_h_m, 1),
            "maps": f"https://maps.google.com/?q={fix.lat:.6f},{fix.lon:.6f}",
            "comparison": out_name,
        }
        results.append(rec)
        print(json.dumps(rec))
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))


def _side_by_side(bgr_photo, rgb_render):
    import cv2

    h = 384
    photo = cv2.resize(bgr_photo, (int(bgr_photo.shape[1] * h / bgr_photo.shape[0]), h))
    render = cv2.cvtColor(rgb_render, cv2.COLOR_RGB2BGR)
    render = cv2.resize(render, (int(render.shape[1] * h / render.shape[0]), h))
    return np.hstack([photo, render])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["osm", "overture"], default="osm")
    ap.add_argument("--photos", type=Path, default=None)
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--res", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / ENV_NAME / "test_output")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    if not args.skip_build:
        build(args.source, args.res)
    random_renders(args.out / "renders")
    if not args.skip_train:
        train(args.budget)
    if args.photos and args.photos.exists():
        localize_photos(args.photos, args.out / "localization")
    print(f"done — outputs in {args.out}")


if __name__ == "__main__":
    main()
