# GIS environments: real-world areas without Unity

`dronecv gis` reconstructs a flyable 3D environment from open GIS data —
terrain + extruded buildings — and plugs it into the standard pipeline: the
same raycaster renders it, the same active loop trains on it, the same
harness certifies it, and the localizer outputs **real GPS coordinates**
(anchor = area center, ENU-aligned). Shape-first by design: geometry carries
the signal, colors are class bands (lighting-invariant), an orthophoto drape
is optional.

## Quick start

```bash
pip install -e ".[gis]"
dronecv gis info  --bbox 43.31,11.32,43.33,11.35          # source coverage
dronecv gis build --place "Siena" --env-name siena        # or --bbox / --mask area.geojson
dronecv run-all   --env siena                             # train + certify on the real area
dronecv gis gui                                           # desktop GUI (pip install -e ".[gui]")
```

## Data sources

| Data | Default (no keys) | Notes |
|---|---|---|
| Elevation | [Copernicus GLO-30 DEM](https://registry.opendata.aws/copernicus-dem/) — public S3, COG GeoTIFF | 30 m grid; a few national tiles unreleased ([readme](https://copernicus-dem-30m.s3.amazonaws.com/readme.html)) |
| Buildings + heights | OSM via Overpass (`height`, `building:levels` tags) | height coverage varies by region |
| Ground classes | OSM via Overpass (roads/water/green/rail/parking) | |
| Better heights (optional) | [Overture Maps buildings](https://docs.overturemaps.org/guides/buildings/) GeoParquet ([AWS](https://registry.opendata.aws/overture/)) | best in the US; `duckdb` extra |
| Orthophoto (optional) | user GeoTIFF (best <1 m/px with acquisition time); Sentinel-2 via Copernicus Data Space (free account); EOX cloudless (non-commercial) | enables shadow heights + albedo drape |
| Geocoding | Nominatim | GUI/`--place` |

**Attribution obligations** (also stamped into every build's `meta.json`):
building/landcover data © OpenStreetMap contributors (ODbL); elevation ©
European Union, ESA, Airbus (Copernicus GLO-30). OSM raster *tiles* are only
used interactively in the GUI map per the OSMF tile policy — bulk data comes
from Overpass.

## Building heights: the resolution chain

1. `height` tag (meters) — real data;
2. `building:levels` × 3 m;
3. **shadow inference** (`--ortho photo.tif --ortho-utc 2025-06-21T10:00:00Z`):
   shadow direction from the NOAA sun ephemeris at the photo's time and the
   area's location; luminance profiles from the down-sun footprint edges give
   the shadow length L, and `h = L·tan(elevation)`. Robust median over edge
   rays, occlusion-aware, contrast-gated. Accuracy tracks the orthophoto:
   0.2–1 m/px resolves houses; Sentinel-2's 10 m/px only ~15 m+ structures;
4. **neighbor median**: an untagged building inherits the median height of
   tagged buildings within ~250 m (a fixed default flattens dense centers);
5. per-class defaults (residential 7 m, industrial 9 m, landmark 25 m, …).

## Parcelled downloads: a fixed cell grid with per-cell cache

Vector data (OSM/Overpass, Overture) is downloaded cell by cell on a **fixed
global 500 m grid** (anchored at 0,0 — the same ground square is always the
same cell, in any request or session), not as one giant bbox request:

- a failed cell loses only ~500 m, and Overpass per-cell queries are small
  enough that they rarely 504;
- every successful cell is **cached** under
  `~/.cache/dronecv/gis/cells/<source-kind>/<version>/<cell>.json` — a
  re-run only fetches what's missing (aborting a build keeps what was
  downloaded; a second build of the same area is instant), and overlapping
  selections resolve to the same cells and are processed once;
- Overture is scanned once over the union of missing cells and partitioned
  into the per-cell caches (no per-cell parquet re-scan).

Overpass queries are **hedged** across the public mirrors: the fastest one to
answer wins, with a short (40 s) timeout and a 7 s stagger between mirrors, so
a hung instance (which changes minute to minute) no longer costs a full
timeout — a tiny per-cell query returns in seconds instead of minutes.

A cell whose data of one kind failed is **not** cached (buildings, landcover
and POIs are cached separately), so reusing that cell later refetches only the
missing kind — a build that got roads but not buildings for a cell will
retry the buildings. Only an explicit **Skip data** choice marks a cell final
(a `.skip` sentinel). To finish a partially-built area, just re-run the build:
it resumes from the cache and fetches the deferred cells.

On a cell that fails after all mirrors, there is **no silent source switch**.
The CLI policy is `--on-cell-fail {defer|skip|abort|fallback}` (default
`defer`: collect the failures, write `download_manifest.json`, finish the
reachable cells, resume on the next run). The desktop GUI draws the grid on
the map colored live by status (cached green / fetched blue / deferred orange
/ skipped grey / failed red) and, on a failure, asks per cell — skip, use the
other source, retry at end, or abandon — with an "apply to all cells with the
same error" option; at the end it offers to retry the deferred cells.
`dronecv gis cells --bbox …` reports the cache status of an area.

## Clouds: detection + multi-date cloud-free compositing (`--imagery s2`)

Satellite photos are frequently obstructed by clouds. Two mechanisms:

1. **Any imagery source** gets a cloud/shadow mask automatically (bright +
   desaturated blobs, plus paired dark shadows within plausible reach) —
   masked texels are excluded from footprint extraction, palette and
   vegetation, and the build warns when obstruction exceeds 2%.
2. **`--imagery s2`** navigates the DATES: Sentinel-2 L2A publishes every
   acquisition (~5 days) as public COGs; the provider lists recent scenes
   for the AOI's MGRS tile, reads only the AOI window of each, detects
   clouds per scene, and fills the holes of the best scene from other dates
   (radiometrically aligned before filling) until coverage is complete — a
   process that may take several photos, all automatic. The composite mixes
   dates, so it deliberately carries no acquisition time (shadow-based
   height inference is skipped on it). Scene usage is recorded in
   `meta.json` (`cloudfree_composite`). Attribution: "contains modified
   Copernicus Sentinel data" (free, incl. commercial use).

## Composite mosaics: automatic radiometric normalization

Satellite orthophotos are usually composites of strips acquired on different
days — different exposure, tone and white balance, joined along sharp seams;
the same roof reads bright in one strip and dark in the next. The build
detects the radiometric zones automatically (per-block robust statistics;
seams are sharp discontinuities, so gradual drifts never split) and aligns
every zone to the dominant one (luminance shift+scale, per-channel gain,
applied sharply at the seam — which is exactly what cancels it). On by
default; `--no-ortho-normalize` disables it; zone count and corrections are
recorded in `meta.json`. Downstream, footprint extraction thresholds are in
local-dispersion units and the roof palette clusters in chromaticity, so
residual exposure differences collapse onto the same material.

## Footprint reconstruction from imagery (`--reconstruct-buildings`)

GIS vectors can have gaps (unmapped districts, new construction). With
`--reconstruct-buildings` the build extracts additional footprints from an
orthophoto and merges them **only where GIS has nothing** (vector data stays
authoritative; dedup by overlap):

```
dronecv gis build --bbox ... --env-name x --reconstruct-buildings \
    --ortho photo.tif --ortho-utc 2025-06-21T10:00:00Z   # best: your own <1 m/px
dronecv gis build --bbox ... --env-name x --reconstruct-buildings \
    --imagery eox --imagery-res 10                        # Sentinel-2 cloudless (free)
dronecv gis build ... --reconstruct-buildings --imagery "xyz:https://.../{z}/{y}/{x}"
                                                          # high-res tiles, YOU own the ToS
```

Chain: imagery → building mask (denoise, brightness vs large-scale local
reference — a bright-roof detector; a learned segmentation model can be
plugged via `mask_fn`) → morphology → polygonization (rectangularity gate) →
optional shadow validation (a real building casts a shadow on the anti-solar
side at the photo's time) → merge. Reconstructed footprints enter the normal
height chain (shadow → neighbor-median → default). Honesty: at 10 m/px only
large structures survive the area gate; real footprint quality needs ≤1 m/px
imagery (regional orthophotos, or XYZ tiles under their provider's terms).

## Adaptive POV concentration (saliency)

The build computes a geometric distinctiveness map (terrain relief, building
height entropy, skyline rarity vs the rest of the area, landmark presence).
Distinctive cells (a mountain flank, an Eiffel-class tower) start with ~0.5×
the base capture density; repetitive residential fabric gets up to ~2.2×.
The tallest well-separated structures become orbit-capture anchors. This is
only the *prior*: the active loop keeps measuring per-cell error on fresh
probes and reallocates captures where the models actually struggle.

## Large areas (>~5 km): hierarchical tiles

Set `active_loop.tile_m` (e.g. 2000) in the env config: training splits the
AOI into tiles, runs the active loop per tile (budget applies per tile) and
adds a coarse global place-recognition model. At flight time `TiledLocalizer`
routes: coarse "which tile" → fine tile bundle (LRU-loaded) → precise fix,
with automatic hand-off at tile borders. Storage: mosaics are disk memmaps
(~13 GB at 1 m/px for 40×40 km — use `--res 2` to quarter it).

## POIs, famous buildings and online photos

`dronecv gis build` also queries OSM POIs (`historic`, `tourism`,
`man_made=tower/...`) and `building:part` elements:

- **Landmark archetypes**: a POI matched to its footprint reshapes the
  building in the height mosaic — spires on towers, domes on churches,
  crenellations on castles — so the TRAINING renders see the same
  recognizable silhouette the exports reproduce. Real `building:part`
  heights (where mapped) win over archetypes.
- **Online photos**: with `fetch_photos` enabled, the primary Wikimedia
  Commons image of each `wikidata`-tagged POI is downloaded to
  `photos/` with attribution records (free licenses). Google Maps/Places
  photos are deliberately NOT integrated: their ToS forbid offline storage
  and texture use.

## 3D territory features

- **Forests**: OSM forest/wood polygons (typology from `leaf_type`) become a
  noisy canopy height layer — visible geometry in the training renders and
  individual tree instances (position, type, height) in the exports.
  Density comes from per-texel deterministic gaps (~18% clearings).
- **Bridges**: `bridge=yes` ways get a raised deck spanning the end-point
  terrain heights; **railways** get a small embankment ridge; roads/rivers/
  water/parking remain ground classes carved into the albedo/splat.
- Mountains/orography come from the DEM as before.

> The export pipeline is also a standalone product — see
> [3d_product.md](3d_product.md) for the Unity/Blender/CAD profile, formats
> and output licensing, and [manual_en.md](manual_en.md) /
> [manuale_it.md](manuale_it.md) for the step-by-step walkthrough.

## Scene export (Unity Terrain / Blender)

```bash
dronecv gis export-scene --env siena          # -> artifacts/gis/siena/scene_export/
```

Produces `terrain.raw` (16-bit orography), `splatmap.png` (ground/vegetation/
hard/water weights), `trees.json` (instances with density+typology),
`buildings.obj` (extruded footprints + landmark meshes, Z-up ENU meters) and
`scene_meta.json` (anchor, scales, attribution).

- **Unity**: menu **DroneCV > Import GIS Scene…** (in the
  com.dronecv.flight package) builds a Terrain with the real orography,
  splat layers, TreeInstances by type/density, the building meshes with
  colliders and a GeoAnchorAsset — ready for the SimLoop rig
  (*GameObject > DroneCV > Create Sim Rig*) and `dronecv run-all`.
- **Blender**: `blender --python blender_build_scene.py -- <export dir>`
  assembles terrain mesh, buildings and dupli-vert instanced forests.

## Realtime 3D viewer

```bash
dronecv gis view --env siena      # builds scene.glb if needed, opens a browser
```

A self-contained three.js/WebGL viewer for `scene.glb`: PBR materials, a real
sky with the sun placed at the scene's solar position (from
`scene_meta.json`), soft shadows and ACES tone mapping. It is served over a
local HTTP server (browsers block `file://` fetches of the model) and opened
in your browser; `--port` fixes the port and `--no-browser` just serves it.
Free-fly controls: **mouse drag** look, **click** to center the view on a
point, **W/S/A/D** move, **Q/E** down/up, **G** toggles global vs
view-relative movement, **wheel** zooms, **Tab** resets, **Shift** moves
faster.

- **Open any model**: press **O** (or the *Open…* button, or drag-and-drop) to
  load a different `.glb`/`.gltf` — not just the built scene. The desktop GUI
  also has an **Open a .glb file…** button for arbitrary files, alongside
  **Open 3D viewer** for the current environment.
- **Screenshot**: press **P** (or the *Screenshot* button) to capture the
  render. It is saved server-side under
  `<scene_export>/screenshots/<model-name>/`, and the **filename encodes the
  camera's latitude, longitude and height** (e.g.
  `scene_lat47.500000_lon19.040000_h150.0m.png`) — the live HUD shows the same
  coordinates. Coordinates come from `scene_meta.json`'s anchor; for a model
  with no meta the name falls back to ENU east/north. Standalone (no server),
  it downloads instead.

The GUI's non-obvious controls all carry tooltips (what *budget* and *terrain
resolution* mean, how to set them, and what raising or lowering them does).

## Limits (v1, stated)

- Buildings are vertical extrusions (LoD1): no roof shapes, no overhangs.
- Colors are synthetic classes unless an orthophoto is draped — by design
  (shapes don't change with lighting; colors do).
- DEM is 30 m: sharp cliffs/embankments are smoothed; buildings are NOT in
  the DEM (added separately), but bridges/trees are absent entirely.
- Shadow heights need a photo timestamp; without one the class defaults kick
  in (the build report says exactly which chain filled each count).
- A model trained on the reconstruction localizes real photos best where
  geometry dominates the view (skylines, building patterns); texture-only
  scenes (open fields) rely on terrain relief.
