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
    """OSM/Overpass building coverage; unreachable network -> available=False
    instead of an exception (filtered networks must still see Overture)."""
    provider = provider or OverpassBuildings()
    try:
        buildings = provider.fetch(bbox)
    except Exception as e:  # noqa: BLE001
        return {"provider": "osm_overpass", "available": False, "error": str(e),
                "n_buildings": 0, "height_coverage": 0.0}
    stats = height_stats(buildings)
    return {"provider": "osm_overpass", "available": True, **stats}


def overture_buildings_coverage(bbox: BBox) -> dict:
    """Overture building count + height coverage for the AOI (row-group scan
    + a light per-row count, no geometry decoding)."""
    try:
        import numpy as np

        from dronecv.gis.providers.overture import (
            BUCKET,
            _FooterIndex,
            latest_release,
            list_theme_files,
            scan_files_parallel,
        )

        release = latest_release()
        files = list_theme_files(release, "buildings", "building")
        index = _FooterIndex()
        matches = scan_files_parallel(index, files, bbox)
        n, with_h = 0, 0
        for key, rgs in matches:
            table = index.read_rows(f"{BUCKET}/{key}", rgs, ["height"], bbox)
            heights = np.asarray(table.column("height").to_pylist(), dtype=object)
            n += len(heights)
            with_h += int(sum(1 for h in heights if h is not None and h > 0))
        return {"provider": "overture", "available": True, "release": release,
                "n_buildings": n, "height_coverage": with_h / max(n, 1)}
    except Exception as e:  # noqa: BLE001
        return {"provider": "overture", "available": False, "error": str(e),
                "n_buildings": 0, "height_coverage": 0.0}


def sentinel2_coverage(bbox: BBox) -> dict:
    """Recent Sentinel-2 dates available for the AOI's MGRS tile."""
    try:
        from dronecv.gis.providers.sentinel2 import list_scenes, mgrs_tile

        zone, band, sq = mgrs_tile((bbox.south + bbox.north) / 2, (bbox.west + bbox.east) / 2)
        scenes = list_scenes(zone, band, sq, months_back=2)
        return {"provider": "sentinel2", "available": len(scenes) > 0,
                "tile": f"{zone}{band}{sq}", "n_recent_scenes": len(scenes),
                "newest": scenes[0][1] if scenes else None}
    except Exception as e:  # noqa: BLE001
        return {"provider": "sentinel2", "available": False, "error": str(e),
                "n_recent_scenes": 0, "newest": None}


def coverage_report(
    bbox: BBox,
    dem: CopernicusDem | None = None,
    buildings: OverpassBuildings | None = None,
    include_overture: bool = True,
    include_imagery: bool = True,
) -> dict:
    report = {
        "bbox": [bbox.south, bbox.west, bbox.north, bbox.east],
        "dem": dem_coverage(bbox, dem),
        "buildings": buildings_coverage(bbox, buildings),
    }
    if include_overture:
        report["buildings_overture"] = overture_buildings_coverage(bbox)
    if include_imagery:
        report["imagery_s2"] = sentinel2_coverage(bbox)

    # Auto-selection hint for the GUI: prefer sources whose coverage is real;
    # between OSM and Overture pick whichever actually has more buildings.
    osm = report["buildings"]
    ovt = report.get("buildings_overture", {"available": False, "n_buildings": 0})
    best_buildings = None
    if osm.get("available") or ovt.get("available"):
        best_buildings = (
            "overture"
            if ovt.get("n_buildings", 0) > osm.get("n_buildings", 0)
            else ("osm_overpass" if osm.get("available") else "overture")
        )
    height_cov = max(osm.get("height_coverage", 0.0), ovt.get("height_coverage", 0.0))
    report["recommended"] = {
        "dem": "copernicus_glo30" if report["dem"]["coverage"] > 0 else None,
        "buildings": best_buildings,
        "heights": "tags" if height_cov > 0.5 else "shadow+defaults",
        "imagery": "s2" if report.get("imagery_s2", {}).get("available") else None,
    }
    return report
