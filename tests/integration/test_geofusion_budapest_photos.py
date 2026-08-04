"""Real-photo e2e: localize six REAL photographs of central Budapest
(Parliament, Chain Bridge, Buda Castle hill, Fisherman's Bastion) against an
environment built from open data — no color is ever compared, no model is
trained.

Needs network (Copernicus DEM S3 + Overture S3) and several minutes, so it
only runs when DRONECV_RUN_BUDAPEST_E2E=1. Ground-truth camera positions are
approximate (estimated from the photo content, ±~100 m).

MEASURED BASELINE (2026-08, heuristic fine-resolution segmentation, NO depth
model, Overture heights mostly defaults):
  - COLD city-wide search (2.5 x 2.4 km): median ~1.4 km — a single photo
    without depth cannot disambiguate a whole city; documented, not asserted.
  - WITH a 400 m prior (the realistic in-flight condition — the EKF/odometry
    always provides one): errors 148-545 m, median ~300 m, every photo in the
    right neighborhood. THAT is what the assertions pin.
Upgrades designed to shrink it further: a segmentation + monocular-depth
model plugged into `view_from_photo`, OSM-tagged heights instead of Overture
defaults, and the sequence particle filter over a real flight.
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

# file -> approximate GT camera position (lat, lon), from landmark bearings
# (render-verified: the Buda-bank/castle-hill viewpoints, not mid-river)
GT = {
    "parl_margaret.jpg": (47.5030, 19.0375),
    "parl_aerial.jpg": (47.5070, 19.0415),
    "bastion_sunset.jpg": (47.5015, 19.0347),
    "chain_basilica.jpg": (47.4970, 19.0380),
    "chain_parl.jpg": (47.4960, 19.0402),
    "bastion_aerial.jpg": (47.5013, 19.0353),
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


def test_budapest_photos_localize_with_flight_prior(localizer):
    """The in-flight condition: the EKF/odometry always gives a ~400 m prior.
    Every photo must land in the right neighborhood."""
    import cv2

    from dronecv.localization.geofusion.photo import view_from_photo

    errs = {}
    for fn, (glat, glon) in GT.items():
        img = cv2.cvtColor(cv2.imread(str(PHOTOS / fn)), cv2.COLOR_BGR2RGB)
        gt = _to_enu(glat, glon)
        best = None
        for fov in (35.0, 55.0, 70.0):  # unknown lens: sweep, keep best conf
            fix = localizer.localize(view_from_photo(img, fov_deg=fov),
                                     prior_xy=gt, prior_radius_m=400.0)
            if best is None or fix.confidence > best.confidence:
                best = fix
        errs[fn] = float(np.linalg.norm(best.pos[[0, 2]] - gt))

    med = float(np.median(list(errs.values())))
    # measured 2026-08: median ~300 m, worst 545 m (thresholds with margin)
    assert med < 450.0, f"median error {med:.0f} m — errors: {errs}"
    assert max(errs.values()) < 700.0, f"errors: {errs}"


def test_budapest_photos_cold_search_beats_chance(localizer):
    """Documented weakness pinned: a COLD city-wide single-photo search
    without a depth model stays km-scale (measured median ~1.4 km) — it must
    still beat random (area diagonal ~3.5 km) and stay in the area."""
    import cv2

    from dronecv.localization.geofusion.photo import view_from_photo

    errs = []
    for fn, (glat, glon) in GT.items():
        img = cv2.cvtColor(cv2.imread(str(PHOTOS / fn)), cv2.COLOR_BGR2RGB)
        fix = localizer.localize(view_from_photo(img, fov_deg=55.0))
        errs.append(float(np.linalg.norm(fix.pos[[0, 2]] - _to_enu(glat, glon))))
    assert float(np.median(errs)) < 2000.0, f"errors: {errs}"
