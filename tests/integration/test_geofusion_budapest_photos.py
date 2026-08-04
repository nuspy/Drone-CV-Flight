"""Real-photo e2e: localize six REAL photographs of central Budapest
(Parliament, Chain Bridge, Buda Castle hill, Fisherman's Bastion) against an
environment built from open data — no color is ever compared, no model is
trained.

Needs network (Copernicus DEM S3 + Overture S3) and several minutes, so it
only runs when DRONECV_RUN_BUDAPEST_E2E=1. Ground-truth camera positions are
approximate (estimated from the photo content, ±~100 m).

MEASURED BASELINE (2026-08, heuristic segmentation, NO depth model, Overture
heights ~2% so the model skylines are nearly flat): per-photo error 439-1392 m
over a 2.5 x 2.4 km area, median ~1.15 km. The assertions below pin THAT
baseline (better-than-random, inside the modeled area) — they are the floor
the designed upgrades must beat: a real segmentation + monocular-depth model
plugged into `view_from_photo`, OSM-tagged heights instead of Overture
defaults, and an EKF/odometry prior instead of a cold city-wide search.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.skipif(
    os.environ.get("DRONECV_RUN_BUDAPEST_E2E") != "1",
    reason="needs network + minutes; set DRONECV_RUN_BUDAPEST_E2E=1")]

ROOT = Path(__file__).parent.parent.parent
PHOTOS = ROOT / "tests" / "fixtures" / "photos_budapest"
BBOX = (47.492, 19.028, 47.514, 19.062)
LAT0, LON0 = (BBOX[0] + BBOX[2]) / 2, (BBOX[1] + BBOX[3]) / 2

# file -> approximate GT camera position (lat, lon), estimated from content
GT = {
    "parl_margaret.jpg": (47.5025, 19.0425),
    "parl_aerial.jpg": (47.5068, 19.0430),
    "bastion_sunset.jpg": (47.5017, 19.0342),
    "chain_basilica.jpg": (47.4975, 19.0385),
    "chain_parl.jpg": (47.4965, 19.0398),
    "bastion_aerial.jpg": (47.5012, 19.0350),
}


def _to_enu(lat: float, lon: float) -> np.ndarray:
    return np.array([(lon - LON0) * 111320 * math.cos(math.radians(LAT0)),
                     (lat - LAT0) * 111320.0])


@pytest.fixture(scope="module")
def localizer(tmp_path_factory):
    from dronecv.gis.geometry import BBox
    from dronecv.gis.pipeline import BuildSources, build_environment
    from dronecv.gis.providers.dem import CopernicusDem
    from dronecv.gis.providers.overture import (
        OvertureBuildingsProvider,
        OvertureLandcoverProvider,
        OverturePoiProvider,
    )
    from dronecv.gis.world import GisWorld
    from dronecv.localization.geofusion import GeoFusionConfig, GeoFusionLocalizer

    root = tmp_path_factory.mktemp("budapest")
    gis_dir = build_environment(
        BBox(*BBOX), "bud_center", out_root=root / "art", configs_root=root,
        sources=BuildSources(dem=CopernicusDem(),
                             buildings=OvertureBuildingsProvider(),
                             landcover=OvertureLandcoverProvider(),
                             poi=OverturePoiProvider()),
        res_m=2.0, on_cell_fail="defer",
    )
    world = GisWorld.open(gis_dir)
    return GeoFusionLocalizer(world, gis_dir=gis_dir,
                              cfg=GeoFusionConfig(grid_step_m=140.0, n_yaws=6))


def test_budapest_photos_localize_to_the_right_district(localizer):
    import cv2

    from dronecv.localization.geofusion.photo import view_from_photo

    errs = {}
    for fn, (glat, glon) in GT.items():
        img = cv2.cvtColor(cv2.imread(str(PHOTOS / fn)), cv2.COLOR_BGR2RGB)
        best = None
        for fov in (35.0, 55.0, 70.0):  # unknown lens: sweep, keep best conf
            fix = localizer.localize(view_from_photo(img, fov_deg=fov))
            if best is None or fix.confidence > best.confidence:
                best = fix
        errs[fn] = float(np.linalg.norm(best.pos[[0, 2]] - _to_enu(glat, glon)))

    med = float(np.median(list(errs.values())))
    # baseline floor (see module docstring): inside the area, beats chance
    assert med < 1500.0, f"median error {med:.0f} m — errors: {errs}"
    assert max(errs.values()) < 2500.0, f"errors: {errs}"
    # at least two photos land in the right neighborhood even with the
    # no-depth heuristic front-end
    assert sum(e < 700.0 for e in errs.values()) >= 2, f"errors: {errs}"
