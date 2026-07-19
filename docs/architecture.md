# Architecture

## Processes and data flow

```
+---------------------------+       TCP, dronecv protocol v1
|  Simulator (either):      |<------------------------------------------+
|   a) Unity 6.5 package    |                                           |
|   b) headless Python sim  |        +--------------------------+       |
+---------------------------+        |  dronecv Python side     |       |
        | truth channel              |  capture collector       |       |
        | (role=harness only)        |  active training loop    |-------+
        v                            |  localizer + EKF fusion  |
+---------------------------+        |  guidance + controller   |
|  Test harness (CLI)       |<------>+--------------------------+
|  metrics, report, verdict |
+---------------------------+
```

One wire protocol, two interchangeable simulators. The localizer's only
inputs are `sensor_frame` messages (image, lidar range, UTC) — the protocol
refuses the `truth` channel to anything but the harness role, so "the sim
does not give coordinates to the localizer" is enforced structurally, not by
convention.

## Modules

- `dronecv.config` — layered YAML (default ← env ← CLI flags), pydantic,
  unknown keys rejected.
- `dronecv.geo` — `frames.py` (sim ↔ ENU ↔ WGS84, heading conventions),
  `anchor.py` (dual-mode geo anchor with provenance), `celestial.py`
  (dependency-free NOAA sun + low-precision moon; C# port in the Unity
  package, pinned by shared fixtures).
- `dronecv.protocol` — pydantic message schemas (`messages.py`), binary
  framing (`framing.py`), asyncio transport, high-level `SimClient`.
  Golden-bytes fixtures pin the encoding for the C# codec.
- `dronecv.sim.headless` — seeded procedural world (value-noise terrain with
  high-frequency albedo detail + distinctive striped landmarks), pure-numpy
  heightfield raycaster (RGB + exact range, Lambert shading from the real sun
  direction, sun/moon discs), kinematic drone, mock lidar, full protocol
  server.
- `dronecv.capture` — pose plans (grid / orbit / targeted / random probes),
  collector (probes terrain height via depth captures — works against any
  simulator), sharded dataset with **spatial whole-cell holdout** (random
  splits leak near-duplicate views and overstate accuracy).
- `dronecv.models` — retrieval embedding (batch-hard triplet over
  anchor-positive pair batches), `LandmarkDB` consensus queries, APR with
  Kendall-style heteroscedastic uncertainty, classical VO (ORB + robust
  ground-plane 2D Kabsch: joint yaw + metric displacement, immune to the
  rotation/translation ambiguity and to essential-matrix degeneracy on pure
  rotation).
- `dronecv.training` — trainer (decoupled retrieval/APR loops, photometric
  augmentation), evaluator (per-cell error map, retrieval recall, ECE, APR
  sigma calibration), **active loop**, model bundle (weights + LandmarkDB +
  terrain prior + manifest with anchor provenance and training history).
- `dronecv.localization` — measurement models (each cue → value + covariance
  + gate), 7-state EKF (Joseph updates, per-dof chi-square gates with
  initial-transient bypass, lost-mode), `Localizer`, streaming service.
- `dronecv.guidance` — target resolution (coordinates or visual),
  standoff-aware guidance, minimal confidence-scaled flight controller.
- `dronecv.harness` — blind accuracy phase (+ kidnap recovery), autonomous
  reach episodes judged on TRUE distance/standoff/collision, metrics, HTML +
  JSON report with pass/fail verdict.

## The active loop (auto-sizing the dataset)

```
round 0: grid (2 altitude bands x N yaw bins) + orbit capture
repeat:
  train (warm-started) -> evaluate on holdout cells AND fresh random probes
  stop if thresholds met | plateau (2 rounds < eps improvement) | budget | max rounds
  else: allocate ~half the initial batch to cells with excess error
        (softmax weights, yaw/altitude diversity) and capture there
outputs: model bundle + training_report.json (error-vs-data curve)
```

The stopping thresholds gate *capturing more data*, not the pipeline: a
plateau/budget stop is flagged in the report and the flight test still runs —
its own pass thresholds are the reliability verdict.

## Confidence

`confidence = P(horizontal error < target | filter covariance) * cue_health`
(Rayleigh CDF on the filter's horizontal sigma; health = fraction of recent
absolute fixes accepted by the gates). Calibration is measured in the flight
test by binning confidence against empirical accuracy (reliability diagram +
ECE in the report).

## v1 simplifications and upgrade paths

| Simplification | Why acceptable in v1 | Upgrade path |
|---|---|---|
| Yaw-only attitude, kinematic drone | camera tilt fixed, no aerobatics | full quaternion EKF + IMU |
| VO assumes locally flat ground | residual trimming + generous sigma | stochastic-cloning relative updates |
| Unimodal EKF | consensus + gating make fixes unimodal; lost-mode re-init handles kidnap | `ParticleFuser` behind the same interface |
| APR interpolates poorly far from training views | active loop densifies; retrieval is redundant | features + PnP against a 3D landmark map |
| Celestial cue = heading only | position from elevation is ~100 km class | (none needed) |
| No obstacle avoidance | test environment; collision = test failure | planner + depth-based avoidance |
