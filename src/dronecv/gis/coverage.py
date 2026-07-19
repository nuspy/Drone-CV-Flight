"""Source coverage report for an AOI — feeds `dronecv gis info` and the GUI
source-selection panel (auto-preselect by coverage, manual confirmation)."""

from __future__ import annotations

from dronecv.gis.geometry import BBox
from dronecv.gis.providers.buildings import OverpassBuildings, height_stats
from dronecv.gis.providers.dem import CopernicusDem, tile_name, tiles_for_bbox


def dem_coverage(bbox: BBox, dem: CopernicusDem | None = None) -> dict:
    tiles = tiles_for_bbox(bbox)
    report = {"provider": "copernicus_glo30", "tiles": [], "available": 0}
    if dem is None:
        import httpx

        for la, lo in tiles:
            name = tile_name(la, lo)
            url = f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
            try:
                ok = httpx.head(url, timeout=15.0).status_code == 200
            except httpx.HTTPError:
                ok = False
            report["tiles"].append({"tile": name, "available": ok})
            report["available"] += ok
    else:  # fixture mode
        for la, lo in tiles:
            name = tile_name(la, lo)
            ok = (dem.tile_dir / f"{name}.tif").exists() if dem.tile_dir else False
            report["tiles"].append({"tile": name, "available": ok})
            report["available"] += ok
    report["coverage"] = report["available"] / max(len(tiles), 1)
    return report


def buildings_coverage(bbox: BBox, provider: OverpassBuildings | None = None) -> dict:
    provider = provider or OverpassBuildings()
    buildings = provider.fetch(bbox)
    stats = height_stats(buildings)
    return {"provider": "osm_overpass", **stats}


def coverage_report(
    bbox: BBox,
    dem: CopernicusDem | None = None,
    buildings: OverpassBuildings | None = None,
) -> dict:
    report = {
        "bbox": [bbox.south, bbox.west, bbox.north, bbox.east],
        "dem": dem_coverage(bbox, dem),
        "buildings": buildings_coverage(bbox, buildings),
    }
    # Auto-selection hint for the GUI: prefer sources whose coverage is real.
    report["recommended"] = {
        "dem": "copernicus_glo30" if report["dem"]["coverage"] > 0 else None,
        "buildings": "osm_overpass" if report["buildings"]["n_buildings"] > 0 else None,
        "heights": (
            "tags" if report["buildings"]["height_coverage"] > 0.5
            else "shadow+defaults"
        ),
    }
    return report
