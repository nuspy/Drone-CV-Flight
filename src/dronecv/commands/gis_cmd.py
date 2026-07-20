from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dronecv.cli.console import console, metric_table


def run_gis_build(
    place: str | None,
    bbox_text: str | None,
    mask_path: str | None,
    env_name: str,
    ortho: str | None,
    ortho_utc: str | None,
    res_m: float,
    reconstruct: bool = False,
    imagery: str | None = None,
    imagery_res: float = 10.0,
    palette_photos: str | None = None,
) -> None:
    from dronecv.config import find_config_root
    from dronecv.gis.geometry import parse_bbox
    from dronecv.gis.pipeline import build_environment

    if bbox_text:
        bbox = parse_bbox(bbox_text)
    elif place:
        bbox = geocode_place(place)
    elif mask_path:
        bbox = bbox_of_geojson(Path(mask_path))
    else:
        console.print("[red]provide --place, --bbox or --mask[/red]")
        raise SystemExit(2)

    root = find_config_root()
    gis_dir = build_environment(
        bbox,
        env_name,
        out_root=root / "artifacts" / "gis",
        configs_root=root,
        ortho_path=Path(ortho) if ortho else None,
        ortho_utc=datetime.fromisoformat(ortho_utc) if ortho_utc else None,
        res_m=res_m,
        reconstruct_buildings=reconstruct,
        imagery=imagery,
        imagery_res_m=imagery_res,
        palette_photos_dir=Path(palette_photos) if palette_photos else None,
    )
    meta = json.loads((gis_dir / "meta.json").read_text())
    stats = meta["stats"]
    console.print(
        metric_table(
            f"GIS environment '{env_name}' built",
            {
                "area": f"{meta['width'] * meta['res_m']:.0f} x {meta['height'] * meta['res_m']:.0f} m @ {meta['res_m']} m/px",
                "anchor": f"{meta['anchor']['lat0']:.5f}, {meta['anchor']['lon0']:.5f}",
                "buildings": f"{stats.get('n_buildings', 0)} ({stats.get('n_with_height', 0)} tagged, "
                f"{stats.get('n_shadow_heights', 0)} shadow, {stats.get('n_reconstructed', 0)} "
                f"reconstructed, {stats.get('n_class_default', 0)} default)",
                "terrain": f"{stats.get('min_ground', 0):.0f}..{stats.get('max_ground', 0):.0f} m rel",
                "saliency": f"mean density x{stats.get('saliency', {}).get('mean_density_multiplier', 1):.2f}",
                "next": f"dronecv run-all --env {env_name}",
            },
        )
    )
    for line in meta.get("attribution", []):
        console.print(f"[dim]{line}[/dim]")


def run_gis_info(bbox_text: str) -> None:
    from dronecv.gis.coverage import coverage_report
    from dronecv.gis.geometry import parse_bbox

    report = coverage_report(parse_bbox(bbox_text))
    console.print_json(json.dumps(report))


def geocode_place(place: str):
    """Nominatim place -> bbox (respecting the usage policy: single request,
    identifying user agent)."""
    import httpx

    from dronecv.gis.geometry import BBox

    resp = httpx.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place, "format": "json", "limit": 1},
        headers={"User-Agent": "dronecv-gis/0.1 (open-source drone navigation research)"},
        timeout=30.0,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise SystemExit(f"place not found: {place}")
    bb = results[0]["boundingbox"]  # [south, north, west, east]
    return BBox(float(bb[0]), float(bb[2]), float(bb[1]), float(bb[3]))


def bbox_of_geojson(path: Path):
    from dronecv.gis.geometry import BBox

    gj = json.loads(path.read_text())
    coords: list[tuple[float, float]] = []

    def collect(geom):
        if geom["type"] == "Polygon":
            coords.extend((p[0], p[1]) for p in geom["coordinates"][0])
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                coords.extend((p[0], p[1]) for p in poly[0])

    if gj.get("type") == "FeatureCollection":
        for f in gj["features"]:
            collect(f["geometry"])
    elif gj.get("type") == "Feature":
        collect(gj["geometry"])
    else:
        collect(gj)
    if not coords:
        raise SystemExit("no polygon in mask GeoJSON")
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return BBox(min(lats), min(lons), max(lats), max(lons))
