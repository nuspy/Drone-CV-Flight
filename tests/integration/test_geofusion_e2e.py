"""End-to-end geofusion: build a real env from fixtures, then localize by
geometry alone — proving color and sun cannot break it, look-alike poses are
vetoed, and a trajectory disambiguates."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dronecv.gis.geometry import BBox
from dronecv.gis.pipeline import BuildSources, build_environment
from dronecv.gis.providers.buildings import OverpassBuildings
from dronecv.gis.providers.dem import CopernicusDem
from dronecv.gis.providers.landcover import OverpassLandcover
from dronecv.gis.providers.poi import OverpassPoi
from dronecv.gis.world import GisWorld
from dronecv.localization.geofusion import (
    GeoFusionConfig,
    GeoFusionLocalizer,
    InvariantView,
    constellation_score,
    extract_buildings,
)
from dronecv.localization.geofusion.constellation import load_model_buildings
from dronecv.sim.headless.rasterizer import render

ROOT = Path(__file__).parent.parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "gis"
BBOX = BBox(south=43.3145, west=11.3225, north=43.3255, east=11.3375)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("geofusion")
    gis_dir = build_environment(
        BBOX, "gf_e2e", out_root=root / "art", configs_root=root,
        sources=BuildSources(
            dem=CopernicusDem(tile_dir=FIXTURES),
            buildings=OverpassBuildings(fixture_path=FIXTURES / "overpass_buildings.json"),
            landcover=OverpassLandcover(fixture_path=FIXTURES / "overpass_landcover.json"),
            poi=OverpassPoi(fixture_path=FIXTURES / "overpass_poi.json"),
        ),
        res_m=2.0,
    )
    world = GisWorld.open(gis_dir)
    loc = GeoFusionLocalizer(world, gis_dir=gis_dir,
                             cfg=GeoFusionConfig(grid_step_m=140.0, n_yaws=6))
    return world, loc, gis_dir


def _gt_pose(world, gis_dir):
    """A ground-truth pose 220 m from the town, camera facing its center."""
    b = load_model_buildings(gis_dir)
    center = b.mean(axis=0)
    xy = center + np.array([30.0, -220.0])
    d = center - xy
    yaw = math.degrees(math.atan2(d[0], d[1])) % 360.0
    ground = float(world.height_at(np.array([xy[0]]), np.array([xy[1]]))[0])
    return xy, yaw, np.array([xy[0], ground + 60.0, xy[1]])


def test_invariant_channels_ignore_sun_and_color(env):
    world, _, _ = env
    pos = np.array([0.0, float(world.height_at(np.array([0.0]), np.array([0.0]))[0]) + 60.0, 0.0])
    rgb_a, rng_a = render(world, pos, 45.0, 35.0, 64, 48, 70.0, 120.0, 60.0)
    rgb_b, rng_b = render(world, pos, 45.0, 35.0, 64, 48, 70.0, 300.0, 15.0)
    # the PHOTOMETRIC image changes a lot with the sun...
    assert float(np.abs(rgb_a.astype(int) - rgb_b.astype(int)).mean()) > 5.0
    # ...but the geometry the localizer consumes does not change at all
    assert np.allclose(rng_a, rng_b)


def test_single_shot_localizes_without_prior(env):
    world, loc, gis_dir = env
    gt_xy, gt_yaw, gt_pos = _gt_pose(world, gis_dir)
    view = InvariantView.from_pose(world, gt_pos, gt_yaw)
    fix = loc.localize(view)
    err = float(np.linalg.norm(fix.pos[[0, 2]] - gt_xy))
    yaw_err = abs((fix.yaw_deg - gt_yaw + 180) % 360 - 180)
    assert err < 40.0, f"position error {err:.1f} m"
    assert yaw_err < 15.0, f"yaw error {yaw_err:.1f} deg"
    assert fix.confidence > 0.2
    assert fix.diagnostics["n_observed_buildings"] >= 3


def test_prior_restricts_candidates(env):
    world, loc, gis_dir = env
    gt_xy, gt_yaw, gt_pos = _gt_pose(world, gis_dir)
    view = InvariantView.from_pose(world, gt_pos, gt_yaw)
    cands = loc.index.query(view, k=8, prior_xy=gt_xy, prior_radius_m=250.0)
    assert cands, "prior query returned no candidates"
    for c in cands:
        assert np.linalg.norm(c.pos[[0, 2]] - gt_xy) <= 250.0 + 1e-6


def test_constellation_vetoes_lookalike_area(env):
    world, loc, gis_dir = env
    gt_xy, gt_yaw, gt_pos = _gt_pose(world, gis_dir)
    view = InvariantView.from_pose(world, gt_pos, gt_yaw)
    observed = extract_buildings(view)
    assert len(observed) >= 3
    model = loc.model_buildings
    s_gt = constellation_score(observed, model, near_xy=gt_pos[[0, 2]])
    s_far = constellation_score(observed, model, near_xy=np.array([450.0, 480.0]))
    assert s_gt > 0.3, f"GT constellation {s_gt:.2f}"
    assert s_far < loc.cfg.veto_threshold, f"far constellation {s_far:.2f} not vetoed"


def test_sequence_stays_locked_with_noisy_odometry(env):
    world, loc, gis_dir = env
    gt_xy, gt_yaw, _ = _gt_pose(world, gis_dir)
    rng = np.random.default_rng(7)
    xy, yaw = gt_xy.copy(), gt_yaw
    poses = [(xy.copy(), yaw)]
    odom = [(0.0, 0.0, 0.0)]
    for _ in range(5):
        fwd, dyaw = 40.0, 8.0
        yr = math.radians(yaw)
        xy = xy + fwd * np.array([math.sin(yr), math.cos(yr)])
        yaw = (yaw + dyaw) % 360.0
        poses.append((xy.copy(), yaw))
        odom.append((fwd + rng.normal(0, 1.5), rng.normal(0, 1.5),
                     dyaw + rng.normal(0, 1.0)))
    views = []
    for p, yw in poses:
        g = float(world.height_at(np.array([p[0]]), np.array([p[1]]))[0])
        views.append(InvariantView.from_pose(world, np.array([p[0], g + 60.0, p[1]]), yw))

    fixes = loc.localize_sequence(views, odom, rng=np.random.default_rng(3))
    errs = [float(np.linalg.norm(f.pos[[0, 2]] - p))
            for f, (p, _) in zip(fixes, poses, strict=True)]
    assert max(errs) < 60.0, f"errors {errs}"
    assert float(np.mean(errs)) < 40.0, f"mean error {np.mean(errs):.1f} m"
    assert errs[-1] < 50.0
