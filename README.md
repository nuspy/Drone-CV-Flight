# Drone-CV-Flight

**GPS-denied autonomous drone navigation by visual recognition of the
surrounding environment.** The system estimates the drone's exact position
(WGS84 latitude/longitude), altitude, velocity and a calibrated confidence —
using only camera imagery, a (mocked) lidar altimeter and the clock. No GPS,
no radio. Models are trained per-environment in simulation (Unity 6.5 or the
built-in headless simulator), with the amount of training data **auto-sized**
for each environment, and the whole pipeline is validated by an automated
blind flight test.

```
one command:
    dronecv run-all --env <environment>
= capture -> training (auto-sized) -> blind localization test
  -> autonomous target-reach test -> reliability report (HTML + JSON)
```

## How it localizes (redundant cues, fused)

| Cue | What it provides | Module |
|---|---|---|
| Visual place recognition | absolute position from a database of recognizable POVs (top-k consensus + spread covariance) | `models/retrieval.py` |
| Absolute pose regression (APR) | absolute position + heading with learned, calibrated uncertainty | `models/pose_net.py` |
| Visual odometry | frame-to-frame yaw change + metric ground-plane displacement (robust 2D Kabsch, no learned parts) | `models/vo.py` |
| Sun/moon | absolute heading from the detected disc vs the NOAA ephemeris | `geo/celestial.py`, `localization/measurements.py` |
| Mock lidar + terrain prior | absolute altitude | `localization/terrain.py` |

All cues feed a 7-state EKF (position, velocity, yaw) with chi-square gating,
lost-mode detection and re-initialization (`localization/fusion.py`). The
confidence output is `P(horizontal error < target)` under the filter
covariance, scaled by cue health — and its calibration (ECE) is itself a
tested, reported metric.

Guidance to a target — given as **coordinates or a photo** — outputs the true-
north bearing (degrees), horizontal distance and altitude change, holding a
**10 m safety standoff** at the target's altitude (`guidance/`).

## The three components

1. **Training platform** — a capture module (Unity package or headless sim)
   teleports a virtual camera through the environment, harvesting multi-POV
   views with ground-truth poses, depth and sun angles. The **active loop**
   (`training/active_loop.py`) trains, evaluates on spatially held-out cells
   *and* fresh probe poses, and keeps capturing where the error map says the
   model is weak — until thresholds, plateau, or budget. "How much data does
   this environment need" is a measured output (see the error-vs-data curve
   in the report).
2. **Localization / guidance module** — `dronecv localize` streams estimates
   from any live sim; the same `Localizer` class is what would run on a real
   companion computer (its only inputs are frames, lidar and time).
3. **Automated test system** — `dronecv test-flight` runs a scripted-flight
   accuracy phase (including a mid-flight *kidnap* teleport) and autonomous
   target-reach episodes. The simulator streams ground truth **only** to the
   test harness: the protocol refuses `truth` subscriptions to the localizer
   role, so the localizer is blind by construction. Reports land in
   `artifacts/<env>/report/` with a pass/fail verdict (CI gate).

## Quick start (no Unity needed)

```bash
pip install -e ".[dev]"          # torch CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu
dronecv envs                     # list environments
dronecv run-all --env headless_ci --seed 0
dronecv report open --env headless_ci
```

Individual stages: `dronecv sim | capture | train | evaluate | localize |
test-flight | protocol verify`.

## Unity 6.5

The `unity/com.dronecv.flight` package makes any Unity scene speak the same
wire protocol — the Python side cannot tell Unity and the headless sim apart.
Setup (2 minutes) and validation:

```bash
# in Unity: GameObject > DroneCV > Create Sim Rig, press Play. Then:
dronecv protocol verify --host <unity-machine> --port 7601
dronecv run-all --env unity_example
```

See `unity/com.dronecv.flight/Documentation~/setup.md` and
`docs/unity_setup.md`. Cross-language correctness is pinned by shared fixture
files (frame conversions, solar ephemeris, protocol golden bytes) consumed by
both pytest and Unity EditMode tests.

## Georeferencing (dual mode)

The sim's local frame is tied to the world either by **metadata embedded in
the environment** (`dronecv_geo.json` sidecar / GeoAnchorAsset, reported in
the protocol handshake) or by the **per-environment YAML** fallback
(`configs/envs/*.yaml`: origin lat/lon/alt + true-north offset). The resolved
anchor and its provenance are stamped into every model bundle; a bundle
refuses to run against a sim with a different anchor.

## Repository map

```
src/dronecv/        config, geo, protocol, sim/headless, capture, models,
                    training, localization, guidance, harness, cli, commands
unity/com.dronecv.flight/   Unity 6.5 UPM package (C#) + EditMode tests
configs/            default.yaml + per-environment files
tests/              unit / integration / e2e (the e2e is the CI gate)
docs/               architecture, protocol, coordinate frames, Unity setup
artifacts/<env>/    datasets, model bundles, training + reliability reports
```

## Development

```bash
make lint  # ruff
make unit  # fast tests
make test  # unit + integration
make e2e   # full pipeline on headless_ci (the CI gate)
```

After any protocol change: bump `PROTOCOL_VERSION`, run
`python scripts/gen_protocol_fixtures.py`, commit the regenerated fixtures
(they validate the C# codec too).

## v1 limitations (deliberate)

Kinematic drone model, single downward lidar beam, fixed camera tilt, no
obstacle avoidance (terrain contact fails the test), yaw-only attitude,
unimodal EKF (lost-mode re-init instead of a particle filter), celestial cue
is heading-only and opportunistic. See `docs/architecture.md` for the full
list and the upgrade paths.
