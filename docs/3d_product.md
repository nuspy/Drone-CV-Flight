# Two products, one pipeline

> Step-by-step walkthrough for both: [manual_en.md](manual_en.md) ·
> italiano: [manuale_it.md](manuale_it.md).

The repository serves two separately sellable purposes built on the same
GIS→3D core:

**A — GPS-denied visual navigation platform.** Build a real-world area,
train the localization models on shape-first renders, test end-to-end with
ground truth the localizer never sees, deploy to the inference server /
Android. Consumers: drone integrators, defense/SAR, research.

**B — 3D environment generator for Unity / Blender / visualization.** The
same build step produces a georeferenced, semantically classed city model —
terrain orography, LoD2 buildings (real footprints + heights + shaped
roofs), vegetation with density/typology, roads/water — exported with
materials. Consumers: game studios, archviz, simulation, urban planning.

## The B pipeline

```
dronecv gis build --bbox ... --env-name city \
    [--reconstruct-buildings --imagery eox] [--palette-photos fotos/]
dronecv gis export-scene   --env city         # Unity Terrain assets + scene.glb
dronecv gis export-blender --env city         # native .blend (runs Blender headless)
```

What makes the output look right:

| Ingredient | Source |
|---|---|
| Terrain orography | Copernicus GLO-30 DEM |
| Building footprints + heights | OSM/Overture tags → shadow inference → neighbor-median → class defaults |
| Extra footprints where GIS is empty | satellite imagery segmentation (`--reconstruct-buildings`) |
| Roof shapes (LoD2) | `roof:shape`/`roof_shape` tags; heuristic gabled for narrow residential |
| Colors | **zone palette extracted from photos of the area** (roofs = warm clusters seen from above, walls = bright plaster clusters at street level) — walls never inherit roof colors |
| Facades | procedural window atlas (one bay / one floor per 3 m UV tile) multiplied by the palette wall color |
| Vegetation | OSM/Overture green+forest classes + green spots from color imagery; exported as typed tree instances |
| Landmarks | POIs (Overpass/Overture places) matched to buildings → dome/spire/crenellation archetypes |
| Lighting | real solar position (NOAA ephemeris) written into `scene_meta.json` |

## Formats

- `scene.glb` — glTF 2.0, PBR materials, embedded textures, Y-up. Imports
  directly into Blender, Unity (glTFast), three.js, any glTF viewer.
- `<env>.blend` — native Blender file (assembled by
  `blender_build_scene.py`: glb import + instanced forests + sun + sky).
- Unity Terrain assets — `terrain.raw` + `splatmap.png` + `trees.json` via
  the *DroneCV > Import GIS Scene* editor menu (heightmap terrain with
  splat layers; buildings from scene.glb or buildings.obj).
- `buildings.obj` — plain OBJ fallback (no materials), CAD-friendly.

## Level of detail, honestly

- LoD1: exact footprints (with courtyards) extruded to resolved heights.
- LoD2: shaped roofs on the min-rotated-rect where it fits (IoU ≥ 0.8);
  tagged shapes win, heuristics fill the rest. Irregular footprints keep
  flat roofs rather than inventing wrong geometry.
- Facade realism is texture-level (procedural windows), not geometry-level;
  photogrammetric detail is out of scope — the value is *the whole city,
  georeferenced, in minutes, from open data*.

## Licensing of the OUTPUT

- OSM-derived data (buildings, landcover, POIs): **ODbL** — derivative
  databases must credit OpenStreetMap contributors and remain
  ODbL-compatible. The attribution strings are embedded in
  `scene_meta.json`; keep them in shipped products.
- Overture Maps: CDLA-Permissive-2.0 / ODbL per theme (see release notes).
- Copernicus DEM: free including commercial use, credit required
  ("© European Union, ESA, Airbus").
- EOX Sentinel-2 cloudless imagery: **non-commercial** (CC BY-NC-SA) — for
  commercial products use your own orthophotos or a licensed provider for
  the palette/reconstruction inputs.
- Wikimedia Commons photos: per-file licenses (mostly CC BY-SA) — they only
  influence extracted *palette statistics*, not shipped pixels.
