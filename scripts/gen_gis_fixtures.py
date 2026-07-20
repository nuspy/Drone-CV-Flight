"""Generate offline GIS fixtures (tests/fixtures/gis/) so CI never touches
the network:

- a Copernicus-named DEM tile (valid GeoTIFF, 1x1 degree extent at reduced
  resolution) with a deterministic hill + valley;
- an Overpass buildings JSON (ways with height tags / levels / nothing, one
  multipolygon relation with a courtyard, one tall landmark tower);
- an Overpass landcover JSON (roads, water, park).

Deterministic: re-running produces identical files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "tests" / "fixtures" / "gis"

# Fixture AOI: ~1.1 x 1.1 km around (43.320, 11.330) — inside tile N43 E011.
LAT0, LON0 = 43.320, 11.330


def gen_dem() -> None:
    import rasterio
    from rasterio.transform import from_bounds

    OUT.mkdir(parents=True, exist_ok=True)
    size = 240  # px over 1 degree -> ~460 m/px; enough for a smooth hill
    ys, xs = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij")
    # Hill centered near the AOI + gentle north slope. Row 0 = north (GeoTIFF).
    lat = 44.0 - ys  # row -> latitude
    lon = 11.0 + xs
    d2 = ((lat - LAT0) / 0.02) ** 2 + ((lon - (LON0 + 0.006)) / 0.02) ** 2
    elev = 260.0 + 90.0 * np.exp(-d2) + (44.0 - lat) * 40.0
    transform = from_bounds(11.0, 43.0, 12.0, 44.0, size, size)
    name = "Copernicus_DSM_COG_10_N43_00_E011_00_DEM"
    with rasterio.open(
        OUT / f"{name}.tif",
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(elev.astype(np.float32), 1)
    print(f"wrote {name}.tif")


def _way(wid, ring, tags):
    return {
        "type": "way",
        "id": wid,
        "tags": tags,
        "geometry": [{"lat": la, "lon": lo} for lo, la in ring],
    }


def _rect(lon, lat, w_deg, h_deg):
    return [
        (lon, lat), (lon + w_deg, lat), (lon + w_deg, lat + h_deg), (lon, lat + h_deg), (lon, lat),
    ]


def gen_buildings() -> None:
    d = 0.00035  # ~30 m footprint side
    elements = []
    # Residential block (heights via levels), regular grid = repetitive fabric.
    wid = 100
    for i in range(4):
        for j in range(3):
            lon = LON0 - 0.004 + i * 0.0012
            lat = LAT0 - 0.003 + j * 0.0010
            elements.append(_way(wid, _rect(lon, lat, d, d * 0.8), {
                "building": "residential", "building:levels": "2",
            }))
            wid += 1
    # Industrial shed with explicit height.
    elements.append(_way(200, _rect(LON0 + 0.002, LAT0 - 0.0035, 0.0012, 0.0007), {
        "building": "industrial", "height": "11.5",
    }))
    # Tall landmark tower (explicit height 55 m).
    elements.append(_way(201, _rect(LON0 + 0.0055, LAT0 + 0.0028, 0.00018, 0.00018), {
        "building": "tower", "height": "55",
    }))
    # Untagged buildings (no height at all -> shadow/default chain).
    elements.append(_way(202, _rect(LON0 + 0.0035, LAT0 + 0.0005, 0.0006, 0.0005), {
        "building": "yes",
    }))
    elements.append(_way(203, _rect(LON0 - 0.0005, LAT0 + 0.0025, 0.0007, 0.0004), {
        "building": "commercial",
    }))
    # Multipolygon with a courtyard.
    outer = _rect(LON0 - 0.0045, LAT0 + 0.0018, 0.0014, 0.0012)
    inner = _rect(LON0 - 0.0041, LAT0 + 0.0021, 0.0006, 0.0005)
    elements.append({
        "type": "relation",
        "id": 300,
        "tags": {"type": "multipolygon", "building": "apartments", "building:levels": "3"},
        "members": [
            {"type": "way", "role": "outer", "geometry": [{"lat": la, "lon": lo} for lo, la in outer]},
            {"type": "way", "role": "inner", "geometry": [{"lat": la, "lon": lo} for lo, la in inner]},
        ],
    })
    (OUT / "overpass_buildings.json").write_text(json.dumps({"elements": elements}, indent=1))
    print(f"wrote overpass_buildings.json ({len(elements)} elements)")


def gen_landcover() -> None:
    elements = [
        _way(400, [(LON0 - 0.006, LAT0), (LON0 + 0.006, LAT0)], {"highway": "secondary"}),
        _way(401, [(LON0, LAT0 - 0.005), (LON0, LAT0 + 0.005)], {"highway": "residential"}),
        _way(402, _rect(LON0 - 0.0058, LAT0 - 0.0048, 0.002, 0.0015), {"natural": "water"}),
        _way(403, _rect(LON0 + 0.003, LAT0 - 0.0018, 0.0025, 0.002), {"leisure": "park"}),
        # Forests with typology (vegetation layer) + a bridge over the water.
        _way(404, _rect(LON0 - 0.0058, LAT0 + 0.001, 0.0022, 0.003),
             {"landuse": "forest", "leaf_type": "broadleaved"}),
        _way(405, _rect(LON0 + 0.0015, LAT0 + 0.0032, 0.002, 0.0018),
             {"natural": "wood", "leaf_type": "needleleaved"}),
        _way(406, [(LON0 - 0.0058, LAT0 - 0.004), (LON0 - 0.003, LAT0 - 0.004)],
             {"highway": "secondary", "bridge": "yes"}),
        _way(407, [(LON0 - 0.006, LAT0 - 0.0052), (LON0 + 0.006, LAT0 - 0.0055)],
             {"railway": "rail"}),
    ]
    (OUT / "overpass_landcover.json").write_text(json.dumps({"elements": elements}, indent=1))
    print(f"wrote overpass_landcover.json ({len(elements)} elements)")


def gen_poi() -> None:

    elements = [
        # POI node on the tall tower (way 201 in the buildings fixture).
        {
            "type": "node", "id": 500,
            "lat": LAT0 + 0.0029, "lon": LON0 + 0.0056,
            "tags": {"man_made": "tower", "name": "Torre Alta", "wikidata": "Q999999"},
        },
        # Castle POI near the courtyard relation -> crenellated archetype.
        {
            "type": "node", "id": 501,
            "lat": LAT0 + 0.0024, "lon": LON0 - 0.0038,
            "tags": {"historic": "castle", "name": "Castello di Prova"},
        },
        # A building:part adding a taller wing on the industrial shed.
        _way(502, _rect(LON0 + 0.0022, LAT0 - 0.0033, 0.0004, 0.0004),
             {"building:part": "yes", "height": "18"}),
    ]
    (OUT / "overpass_poi.json").write_text(json.dumps({"elements": elements}, indent=1))
    print(f"wrote overpass_poi.json ({len(elements)} elements)")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    gen_dem()
    gen_buildings()
    gen_landcover()
    gen_poi()
