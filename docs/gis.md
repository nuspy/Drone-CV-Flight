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
4. per-class defaults (residential 7 m, industrial 9 m, landmark 25 m, …).

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
