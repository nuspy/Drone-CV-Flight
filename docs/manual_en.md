# End-to-End Manual

*(Versione italiana: [manuale_it.md](manuale_it.md))*

This manual walks through both uses of the project, start to finish:

- **Part A — High-quality 3D environment generation** for Unity / Blender /
  visualization (the "3D product").
- **Part B — Manual recognition test "from an aircraft"**: aerial photo →
  geographic coordinates, live streaming mode, and the automated reliability
  test (the navigation product).

Both parts share the same first step: building a real-world environment from
open GIS data.

---

## 0. Prerequisites

| What | Notes |
|---|---|
| Python **3.11 or 3.12** | `python --version` — 3.10 is NOT supported (the code uses 3.11+ features). On Windows: `py -3.12 -m venv .venv` then `.venv\Scripts\Activate.ps1` |
| Project install | `git clone … && cd Drone-CV-Flight && pip install -e ".[dev,gis]" -c constraints.txt` (the constraints file pins tested versions and prevents pip resolver backtracking) |
| Desktop GUI | extra install: `pip install -e ".[dev,gis,gui]" -c constraints.txt` (PySide6, ~500 MB) |
| GPU | optional, auto-detected. No GPU → nothing to do, everything runs on CPU. NVIDIA GPU → `pip install torch --index-url https://download.pytorch.org/whl/cu124`; training then logs `training on cuda (…)` and runs 10-30× faster. GPU accelerates training/inference only — GIS build and coverage checks are network/CPU bound either way |
| Blender 3.6+/4.x | only for the native `.blend` export (must be on `PATH`) |
| Unity 2022.3+ | only for the Unity scene; install the **glTFast** package (`com.unity.cloud.gltfast`) for materials |
| Network | Overpass/OSM works on normal networks (with automatic mirror fallback). On filtered networks (AWS-only proxies) use the Overture / Sentinel-2 providers — their data comes from public S3 |
| Disk | ~1 GB per built environment at 1 m/px (2×2 km); training datasets add a few hundred MB |

Every command below is run from the repository root, with the virtualenv
active (`dronecv` is installed as a console script by `pip install -e`).

---

## Part A — High-quality 3D export, end to end

### A1. Choose the area and the sources

Desktop GUI:

```bash
dronecv gis gui
```

Workflow in the window: search the place by name (the result is only a
dashed **preview** frame — it never becomes the selection), then **draw the
exact AOI** yourself (rectangle or polygon; deleting the shape clears the
selection). Drawing triggers the coverage check (~1 minute: it queries OSM,
scans Overture and lists Sentinel-2 scenes) and the panel then shows, side
by side: DEM tiles, **OSM vs Overture building counts and height
coverage**, and recent Sentinel-2 dates. The Buildings/Imagery selectors
are pre-set to the recommendation (the source that actually has more
buildings; `s2` when scenes exist) — override them freely, add the
*reconstruct buildings from imagery* checkbox and a *palette photos*
folder, then Build.

CLI equivalent of the coverage check:

```bash
dronecv gis info --bbox 47.4925,19.0290,47.5105,19.0560
```

**Choosing between sources, honestly:**

| Source | Strengths | Weaknesses | Pick it when |
|---|---|---|---|
| Buildings: **OSM/Overpass** | richest tags (`roof:shape`, `building:part` LoD for monuments, POIs with wikidata → Commons photos), fresh to the minute | coverage = mapper effort; public-server usage policy | well-mapped areas (most European cities) |
| Buildings: **Overture** | OSM + Microsoft ML footprints + Esri merged: better coverage where OSM is thin, often better heights; S3 = no rate limits | monthly releases, poorer tags (no `building:part`), 30-60 s scan | sparsely mapped areas, filtered networks |
| Imagery: **`s2`** (Sentinel-2) | dated scenes every ~5 days, automatic cloud-free compositing, free incl. commercial | 10 m/px | palette + vegetation + large-structure reconstruction |
| Imagery: **`eox`** | pretty cloudless mosaic | no date (no shadow heights), **non-commercial** | quick previews only |
| Imagery: **`xyz:URL`** | 0.3-0.6 m/px | provider ToS is on you | serious footprint reconstruction |
| Imagery: **your GeoTIFF** | best of all: sub-meter + known time → shadow heights | you must have one | whenever you do |

### A2. Build at maximum quality

```bash
dronecv gis build \
    --bbox 47.4925,19.0290,47.5105,19.0560 \
    --env-name budapest_hq \
    --res 1.0 \
    --imagery s2 \
    --palette-photos my_photos/ \
    --ortho my_ortho.tif --ortho-utc 2025-06-21T10:00:00Z
```

Every flag is a quality lever — use what you have, everything is optional
except `--bbox`/`--place` and `--env-name`:

| Lever | Effect | When to use |
|---|---|---|
| `--res 1.0` | 1 m/px mosaic: crisp building edges (default is already 1.0; 2.0 halves memory for big areas) | always for quality |
| `--palette-photos DIR` | extracts the zone's roof/wall color clusters from your photos of the area — walls get their own plaster tones, roofs the real tile colors | 3-10 photos are enough; aerial + street level mix is best |
| `--reconstruct-buildings` | extracts extra building footprints from satellite imagery where OSM/Overture have nothing | sparse/unmapped areas (NOT well-mapped centers — at 10 m it only adds noise there) |
| `--imagery s2` | Sentinel-2 multi-date **cloud-free composite** (~10 m): clouds and their shadows are detected per scene and holes filled from other dates automatically | the default imagery choice — palette, vegetation, large structures |
| `--imagery eox` | Sentinel-2 cloudless mosaic (no dates, non-commercial) | quick previews |
| `--imagery "xyz:URL"` | high-res XYZ tiles (~0.3-0.6 m/px) — you are responsible for the provider's terms of service | serious reconstruction |
| `--ortho file.tif --ortho-utc …` | your own georeferenced orthophoto **with acquisition time** → shadow-based height inference for untagged buildings + roof palette + vegetation green-spots | best single upgrade if you have regional orthophotos |
| *(automatic)* `--no-ortho-normalize` to disable | composite-mosaic strips (different exposure/tone per acquisition) are detected and radiometrically aligned before any use — the same roof is recognized whether its strip is bright or dark | leave on; disable only for single-acquisition imagery you trust |
| *(automatic)* | untagged buildings inherit the median height of tagged neighbors within ~250 m; POIs stamp landmark archetypes (domes, spires, crenellations); Overture/OSM green+forest classes become 3D canopy | — |

The build ends with a stats summary (buildings, heights by source,
reconstructed footprints, vegetation, POIs, palette pixel counts). Full
detail in `artifacts/gis/budapest_hq/meta.json`.

### A3. Visual check before exporting

Render a few aerial views straight from the built store (no training
needed) — adapted from `scripts/test_budapest.py`:

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
import numpy as np, cv2
from dronecv.gis.world import GisWorld
from dronecv.sim.headless import rasterizer
w = GisWorld.open("artifacts/gis/budapest_hq")
rgb, _ = rasterizer.render(w, np.array([0.0, 150.0, 0.0]), 45.0, 25.0,
                           960, 720, 70.0, 160.0, 55.0)
cv2.imwrite("check.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
EOF
```

Look for: terracotta roofs vs light walls, flat dark water, green where
parks are, landmark shapes where you expect them.

Or, for a full realtime walk-through with sky, sun and shadows:

```bash
dronecv gis view --env budapest_hq
```

This builds `scene.glb` if needed and opens a browser viewer. Fly with the
mouse (drag to look, click to center), **W/S/A/D** to move, **Q/E** for
down/up, **G** to switch between global and view-relative movement, the wheel
to zoom and **Tab** to reset. In the GUI the same thing is one click: **Open
3D viewer**.

### A4. Export

```bash
# Unity Terrain assets + scene.glb (glTF 2.0, PBR materials, facade texture)
dronecv gis export-scene --env budapest_hq

# native .blend — runs Blender headless if it is on PATH,
# otherwise prints the exact command to run on a machine that has it
dronecv gis export-blender --env budapest_hq --blend budapest.blend
```

The export folder (`artifacts/gis/budapest_hq/scene_export/`) contains:

- `scene.glb` — terrain (class vertex colors) + LoD2 buildings grouped by
  class, palette PBR materials, procedural window texture on facades.
  Opens directly in Blender, any glTF viewer, three.js.
- `terrain.raw`, `splatmap.png`, `trees.json` — Unity Terrain assets.
- `buildings.obj` — plain OBJ fallback (no materials), CAD-friendly.
- `blender_build_scene.py`, `scene_meta.json` (anchor, extents, real sun
  position, attribution).

**Unity**: open your project → install glTFast → menu **DroneCV > Import
GIS Scene…** → pick the export folder. You get a Terrain with orography +
splat layers + typed trees, and the glb scene with materials.

**Blender**: `export-blender` already produced the `.blend` (LoD2 buildings
with materials, instanced forests, sun at the real solar position, sky).
Manual alternative:
`blender --background --python scene_export/blender_build_scene.py -- scene_export out.blend`.

### A5. Commercial use — read this

Output licensing is inherited from the data sources (details in
[3d_product.md](3d_product.md)): OSM/Overture data is ODbL (keep the
attribution strings from `scene_meta.json`), Copernicus DEM and Sentinel-2
(`--imagery s2`) allow commercial use with credit ("contains modified
Copernicus Sentinel data"), **EOX imagery is non-commercial** — for a
sellable product use `s2`, your own or licensed imagery.

---

## Part B — Manual recognition test "from an aircraft", end to end

Goal: prove the localizer can turn what a drone camera sees into WGS84
coordinates + heading + honest confidence, without GPS and without ever
telling it where it is.

### B1. Build the environment

Same build as Part A (A2). From a normal network the OSM path also brings
POIs and landcover in one shot:

```bash
dronecv gis build --bbox 47.4925,19.0290,47.5105,19.0560 --env-name budapest_test --res 1.0
```

### B2. Train

```bash
# everything in one command (capture -> auto-sized training -> flight test -> report):
dronecv run-all --env budapest_test --budget 3000

# or step by step:
dronecv train --env budapest_test --budget 3000     # active loop, auto-sizes rounds
dronecv evaluate --env budapest_test                # fresh-probe metrics
```

Budget guidance (captures, not epochs): **1200 = smoke test** (expect
hundreds of meters of error), **3000+ = serious** for a ~2×2 km dense city,
less for areas with strong landmarks (the saliency prior already
concentrates captures where they matter). For higher fidelity raise
`sim.image_width/image_height` to 192 and `training.backbone_width` to 64
in `configs/envs/budapest_test.yaml`.

### B3. The manual photo test

Take any aerial-style photo of the area (or use the committed samples in
`tests/data/budapest_photos/`) and ask for coordinates:

```bash
dronecv localize-photo tests/data/budapest_photos/03_parliament_aerial.jpg \
    --env budapest_test
```

Output: `lat, lon, alt (MSL), heading (true north), confidence [0-1],
sigma (m)` and a Google Maps link. The fully automated version of this test
(build + 10 random renders + training + all photos + side-by-side
comparisons + `results.json`):

```bash
python scripts/test_budapest.py --source osm --budget 3000 --res 1.0 \
    --photos tests/data/budapest_photos
```

**Domain gap**: models trained on synthetic renders see real photos as a
different world. The shared preprocessing filter narrows the gap and is
applied identically at training and inference (never drifts — the spec is
recorded in the model bundle). Find the best combination for your area:

```bash
python scripts/filter_search.py --epochs 30
```

then retrain with the winner, e.g. `dronecv train --env budapest_test`
after setting in the env YAML:

```yaml
training:
  filter_mode: gray_edge   # none | gray | edge | gray_edge
  filter_edge_weight: 0.5
```

### B4. Live "companion computer" mode

Terminal 1 — the world (headless sim, or a Unity scene speaking the same
protocol):

```bash
dronecv sim --env budapest_test
```

Terminal 2 — the localizer, streaming fixes exactly as it would on the
aircraft (camera frames in, coordinates out; it never receives a position):

```bash
dronecv localize --env budapest_test
```

Phone-in-the-loop variant: `dronecv serve --env budapest_test` starts the
HTTP inference server; the Android app (`android/`) photographs, uploads,
and shows the estimated position on Google Maps — or runs fully on-device
after `dronecv export --env budapest_test` (ONNX bundle).

### B5. The automated reliability verdict

```bash
dronecv test-flight --env budapest_test --episodes 12
```

The harness flies scripted + autonomous episodes; ground truth goes ONLY to
the test harness (the localizer is blind by construction — the protocol
refuses the truth channel to non-harness roles). The report
(`artifacts/budapest_test/report/`) gives horizontal/vertical error
percentiles, heading error, confidence calibration (ECE), target-reach
success, and a PASS/FAIL verdict against the env thresholds.

### B6. Reading the numbers honestly

- **confidence** is calibrated: 0.9 means "90% of fixes this confident are
  within the target radius". On real photos with a small training budget
  expect LOW confidence — that is the system telling the truth, not a bug.
- **sigma_h_m** is floored by the bundle's measured generalization error:
  it cannot claim precision it never demonstrated on probes.
- Trust a fix when: confidence high AND retrieval/APR agree
  (`disagreement_m` small in the diagnostics) AND the rendered view from
  the estimated pose matches the photo.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| pip stuck for hours "backtracking" | you forgot `-c constraints.txt`, or Python is not 3.11/3.12 (recreate the venv: `py -3.12 -m venv .venv`) |
| `the desktop GUI requires PySide6` | install the gui extra: `pip install -e ".[dev,gis,gui]" -c constraints.txt` |
| Overpass 406 / 403 / 429 / 504 / timeouts | identifying User-Agent + public mirror fallback (busiest server tried last). 504 = the public servers are overloaded (worst at European evening peak) — retry shortly. On total failure the build STOPS with a clear message (no silent source switch); to continue anyway pick Buildings→`overture` in the GUI, or pass `--allow-overture-fallback` / the `overture` sources on the CLI |
| `HTTPFileSystem requires requests and aiohttp` | reinstall the gis extra: `pip install -e ".[dev,gis]" -c constraints.txt` (Overture reads GeoParquet over HTTPS via fsspec, which needs aiohttp+requests — now declared) |
| `--imagery eox` cannot connect | prefer `--imagery s2`; or supply `--ortho` |
| Training slow on CPU | lower `--budget`, keep `image_width` at 128; or install CUDA torch (`pip install torch --index-url https://download.pytorch.org/whl/cu124`) — auto-detected, no code change |
| `export-blender` says Blender not found | install Blender and re-run, or copy the printed `blender --background …` command to a machine that has it |
| pyarrow crash while scanning Overture | already mitigated (crash-isolated worker processes); if it persists, re-run — partial scans retry file by file |
| Unity imports glb without materials | install `com.unity.cloud.gltfast` before importing |
