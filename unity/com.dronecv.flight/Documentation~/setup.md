# DroneCV Flight — Unity 6.5 setup

This package has two independent uses that share nothing at runtime:

- **GIS Environment Builder** (menu **DroneCV > GIS Environment Builder…**) — the
  3D-product side: generate a real-world area and import terrain + buildings +
  trees directly into your scene, export to other formats, and optionally merge a
  district into one object + one material. It needs no ML and installs its own
  Python (see below).
- **Sim bridge** (menu **GameObject > DroneCV > Create Sim Rig**) — the
  navigation side: turns any Unity scene into a dronecv simulator over TCP. This
  is unchanged and documented further down.

## GIS Environment Builder

Open **DroneCV > GIS Environment Builder…**. The window drives the same Python
`dronecv` GIS pipeline the CLI uses — it does **not** reimplement anything.

1. **Python environment (auto-installed).** Press *Install environment*. The
   package provisions a private Python under `Library/DroneCV/py` (never
   system-wide): it uses a system Python ≥3.11 if present, otherwise downloads a
   standalone CPython automatically, creates a venv, and `pip install`s
   `dronecv[gis]`. No manual Python/pip/venv setup is required. Set
   `DRONECV_PY_SOURCE` to a repo checkout or a wheel if you install offline.
2. **Build the area.** Enter an env name and a place name or bounding box, pick
   resolution/sources, press *Build*. Progress streams into the log.
3. **Import into Unity.** Choose *Current scene* or *New empty scene* and press
   *Export scene + import into Unity* — the terrain (with collider), buildings
   (glTF/OBJ with colliders), trees and a GeoAnchorAsset are created via the same
   `DroneCV > Import GIS Scene…` importer.
4. **Export** to OBJ (native, includes the Terrain), glTF/GLB, a Unity Prefab +
   TerrainData, or FBX (only if `com.unity.formats.fbx` is installed).

**Render pipeline / materials.** The import works in **URP, HDRP and Built-in**:
the Terrain gets the pipeline's terrain material and every material uses the
active pipeline's lit shader (no magenta). Buildings import from `scene.glb`
with LoD2 roofs + facade textures if a glTF importer (`com.unity.cloud.gltfast`)
is installed; **otherwise they are built natively from `buildings.json` +
`palette.json`** with per-class materials always assigned — so terrain, trees
and buildings render correctly out of the box with no extra packages.
5. **Merge district** (optional) collapses the imported buildings into one mesh
   + one material per unit (whole scene / per class / 500 m cell), baking each
   building's colour into vertex colours over one shared tiling facade texture
   (shader `DroneCV/DistrictMerged`) — the standard way to keep a repeatable
   material working across a merged object while cutting draw calls.

The tooltips in the window explain every field (resolution, sources, terrain
resolution, cell size, …).

---

This package also turns any Unity scene into a dronecv simulator: the Python side
(training, localization, automated flight test) talks to it over TCP exactly
as it talks to the built-in headless simulator, and cannot tell them apart.

## Requirements

- Unity 6.x (tested target: Unity 6.5)
- The imported 3D environment must have **colliders** (a Terrain, or mesh
  colliders on the ground/buildings): the drone's terrain collision, the mock
  lidar and spawn-height probing all use `Physics.Raycast`.
- `com.unity.nuget.newtonsoft-json` (declared as a package dependency,
  installed automatically).

## Install

Copy or reference `unity/com.dronecv.flight` from this repository in your
project's `Packages/` folder (or add it via *Package Manager > Install
package from disk*).

## Scene setup (2 minutes)

1. Open the scene containing your imported environment (the environment comes
   from your existing import pipeline — this package does not import assets).
2. Menu **GameObject > DroneCV > Create Sim Rig**. This creates:
   - `DroneCV Sim` — `SimLoop` (the bridge), `GeoAnchorProvider`, `SunController`
   - `Drone` — `DroneBody` (kinematic dynamics identical to the headless sim)
     + `MockLidar`
   - `CaptureRig` — the camera used for teleport-captures and sensor frames
3. On **SimLoop**:
   - set `Bounds Min/Max` to the flyable volume (a gizmo shows it when selected)
   - set `Env Id` to your environment name (e.g. `unity_example`)
   - camera resolution/FOV/tilt must match `configs/envs/<env>.yaml`
4. On **SunController**: assign the scene's directional light. The light is
   then driven by the real NOAA solar ephemeris from the sim clock and the
   geo anchor, so the celestial heading cue works.
5. Georeferencing (choose one — this is the dual mode):
   - **Embedded metadata**: place a `dronecv_geo.json` next to the project
     (or set an absolute path on `GeoAnchorProvider`):
     ```json
     { "lat0": 45.4642, "lon0": 9.1900, "alt0": 120.0, "true_north_offset_deg": 0.0 }
     ```
   - **Asset fallback**: create *Assets > Create > DroneCV > Geo Anchor* and
     assign it. If neither exists, the Python side falls back to the anchor
     in `configs/envs/<env>.yaml`.
   `true_north_offset_deg` is the compass bearing of the Unity **+Z** axis.
6. Press **Play**.

## Validate the bridge (no Unity knowledge needed on the Python side)

```bash
dronecv protocol verify --host <machine running Unity> --port 7601
```

Every protocol behavior the pipeline needs is checked: handshake and version,
camera info, capture rgb+depth+sun with determinism, `truth` denied to
`role=localizer`, reset/streaming/commands for `role=harness`.

## Run the full pipeline against Unity

```bash
# configs/envs/unity_example.yaml: kind: unity, host/port of the machine
dronecv run-all --env unity_example
```

`run-all` waits for the bridge, then performs capture → active-loop training
(auto-sizing the dataset for your environment) → the automated blind
localization + autonomous target-reach test → the reliability report in
`artifacts/unity_example/report/`.

## EditMode tests (cross-language guarantees)

*Window > General > Test Runner > EditMode* runs:

- `FramesConversionTests` — Unity↔ENU conversions pinned to
  `Tests/Fixtures/frames_cases.json` (same file pytest uses)
- `CelestialTests` — the C# NOAA port vs Python-generated sun vectors
- `CodecGoldenTests` — the C# codec decodes byte-identical golden frames
  produced by the Python encoder

Regenerate fixtures after any protocol change with
`python scripts/gen_protocol_fixtures.py` (run from the repo root, commit the
result).

## Troubleshooting

- **`protocol verify` fails on capture depth**: the environment has no
  colliders, or the far plane clips the ground — raise `farClipPlane` on the
  CaptureRig camera.
- **Lidar always null**: no collider under the drone, or `MaxRange` too small.
- **Sun checks fail**: `SunController.StartUtc` diverges from the env YAML's
  `env.start_utc`, or no directional light assigned.
- **Determinism check fails**: disable in-scene animation/particles that
  change between identical captures (or accept it — training still works,
  the check is strict on purpose).
