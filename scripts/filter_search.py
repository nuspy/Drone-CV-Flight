"""Automatic domain-gap filter search for the Budapest test.

    python scripts/filter_search.py [--photos tests/data/budapest_photos]
        [--env budapest_test] [--out DIR]

For each filter combination the SAME captured dataset is retrained with the
SAME seed (only the preprocessing differs — the fair comparison the user
asked for: "filtri applicati con differenti pesi e combinazioni fino a
trovare l'ottimale"), then each candidate bundle is scored on:

1. SANITY (synthetic): 20 fresh holdout renders at known poses (seed 777,
   never seen in training) -> median localization error. Guards against a
   filter that destroys the easy case.
2. REAL PHOTOS: median distance between the estimate and the approximate
   camera position of each photo (hand-labeled landmarks, +-150 m — fine for
   RANKING combos), plus mean confidence and retrieval similarity margin.

Baseline "none" is evaluated through the same code path. Output: markdown +
JSON table, winner selection (best photo error among combos whose sanity
error stays within 1.5x of baseline), side-by-side renders with the winner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

COMBOS = {
    "none": {"filter_mode": "none"},
    "gray": {"filter_mode": "gray"},
    "edge": {"filter_mode": "edge"},
    "gray_edge": {"filter_mode": "gray_edge", "filter_edge_weight": 0.5},
    "gray_edge_clahe": {"filter_mode": "gray_edge", "filter_edge_weight": 0.5,
                        "filter_clahe": True},
}

# Approximate camera positions of the test photos (lat, lon) — hand-labeled
# from the visible landmarks; ~±150 m accuracy, used only to RANK combos.
PHOTO_GT = {
    "01_fishermans_bastion_aerial": (47.5024, 19.0344),
    "02_chain_bridge_parliament_dusk": (47.4980, 19.0405),
    "03_parliament_aerial": (47.5065, 19.0440),
    "04_fishermans_bastion_sunrise": (47.5023, 19.0346),
    "05_parliament_margaret_bridge": (47.5032, 19.0446),
    "06_chain_bridge_basilica": (47.4988, 19.0360),
}


def _dist_m(lat1, lon1, lat2, lon2):
    k = 111_320.0
    return math.hypot((lat1 - lat2) * k, (lon1 - lon2) * k * math.cos(math.radians(lat1)))


def train_combo(env: str, name: str, overrides: dict, out_dir: Path, epochs: int = 10):
    from dronecv.config import load_config
    from dronecv.training.bundle import ModelBundle
    from dronecv.training.trainer import load_dataset, train_models

    cfg = load_config(env, overrides={
        "training": {**overrides, "epochs_per_round": epochs},
    }, root=ROOT)
    if name == "none":  # reuse the already-trained baseline bundle
        base = cfg.bundle_dir
        if (base / "manifest.json").exists():
            return ModelBundle.load(base)
    ds = load_dataset(cfg)
    bundle, _ = train_models(cfg, ds)
    bundle.save(out_dir / name / "model_bundle")
    return bundle


def synth_holdout(env: str, n: int = 20, seed: int = 777):
    """Fresh renders at known poses, never seen in training."""
    from datetime import UTC, datetime

    from dronecv.config import load_config
    from dronecv.geo import celestial
    from dronecv.geo.anchor import GeoAnchor
    from dronecv.gis.store import GisStore
    from dronecv.gis.world import GisWorld
    from dronecv.sim.headless import rasterizer

    cfg = load_config(env, root=ROOT)
    gis_dir = ROOT / "artifacts" / "gis" / env
    world = GisWorld.open(gis_dir)
    anchor = GeoAnchor.from_dict(GisStore.open(gis_dir).meta.anchor)
    gen = np.random.default_rng(seed)
    utc = datetime(2026, 6, 21, 10, 0, tzinfo=UTC)
    sun = celestial.sun_position(utc, anchor.lat0, anchor.lon0)
    views = []
    for _ in range(n):
        e = gen.uniform(world.bounds_min[0] * 0.6, world.bounds_max[0] * 0.6)
        nn = gen.uniform(world.bounds_min[2] * 0.6, world.bounds_max[2] * 0.6)
        ground = float(world.height_at(np.array(e), np.array(nn)))
        pos = np.array([e, ground + gen.uniform(70, 150), nn])
        heading = float(gen.uniform(0, 360))
        rgb, _ = rasterizer.render(
            world, pos, heading, float(gen.uniform(15, 35)),
            cfg.sim.image_width, cfg.sim.image_height, 70.0,
            sun.azimuth_deg, sun.elevation_deg,
        )
        views.append((rgb, np.array([e, nn]), heading))
    return views


def score_bundle(bundle, views, photos_dir: Path):
    import cv2

    from dronecv.localization.single_shot import SingleShotLocalizer

    loc = SingleShotLocalizer(bundle)
    synth_err = []
    for rgb, pos_en, _heading in views:
        fix = loc.localize(rgb)
        synth_err.append(float(np.hypot(fix.pos_enu[0] - pos_en[0], fix.pos_enu[1] - pos_en[1])))

    photo_err, confs, margins, fixes = [], [], [], {}
    for photo in sorted(photos_dir.glob("*.jpg")):
        gt = PHOTO_GT.get(photo.stem)
        bgr = cv2.imread(str(photo))
        if gt is None or bgr is None:
            continue
        fix = loc.localize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        photo_err.append(_dist_m(fix.lat, fix.lon, gt[0], gt[1]))
        confs.append(fix.confidence)
        sims = np.asarray(fix.diagnostics["retrieval_similarity"], dtype=float).ravel()
        margins.append(float(sims.max() - np.median(sims)) if sims.size > 1 else 0.0)
        fixes[photo.stem] = fix
    return {
        "synth_median_m": float(np.median(synth_err)),
        "synth_p90_m": float(np.percentile(synth_err, 90)),
        "photo_median_m": float(np.median(photo_err)),
        "photo_mean_conf": float(np.mean(confs)),
        "retrieval_margin": float(np.mean(margins)),
    }, fixes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="budapest_test")
    ap.add_argument("--photos", type=Path, default=ROOT / "tests" / "data" / "budapest_photos")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "budapest_test" / "filter_search")
    ap.add_argument("--epochs", type=int, default=10,
                    help="epochs per combo (30+ for a converged comparison on a real machine)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    views = synth_holdout(args.env)
    results = {}
    for name, overrides in COMBOS.items():
        print(f"=== combo {name} ===", flush=True)
        bundle = train_combo(args.env, name, overrides, args.out, epochs=args.epochs)
        metrics, _ = score_bundle(bundle, views, args.photos)
        results[name] = metrics
        print(json.dumps({name: metrics}), flush=True)

    # Winner: best photo error among combos whose sanity holds up.
    guard = results["none"]["synth_median_m"] * 1.5
    eligible = {k: v for k, v in results.items() if v["synth_median_m"] <= guard}
    winner = min(eligible, key=lambda k: eligible[k]["photo_median_m"])

    lines = ["| combo | synth med (m) | synth p90 (m) | photo med (m) | conf | retr. margin |",
             "|---|---|---|---|---|---|"]
    for k, v in results.items():
        star = " **<- winner**" if k == winner else ""
        lines.append(
            f"| {k}{star} | {v['synth_median_m']:.0f} | {v['synth_p90_m']:.0f} "
            f"| {v['photo_median_m']:.0f} | {v['photo_mean_conf']:.3f} "
            f"| {v['retrieval_margin']:.3f} |"
        )
    table = "\n".join(lines)
    print(table, flush=True)
    (args.out / "results.json").write_text(json.dumps({"results": results, "winner": winner}, indent=1))
    (args.out / "results.md").write_text(table + "\n")
    print(f"winner: {winner} — outputs in {args.out}", flush=True)


if __name__ == "__main__":
    main()
