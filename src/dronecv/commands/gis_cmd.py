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
    ortho_normalize: bool = True,
    allow_overture_fallback: bool = False,
    cell_m: float = 500.0,
    on_cell_fail: str = "defer",
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
        ortho_normalize=ortho_normalize,
        allow_overture_fallback=allow_overture_fallback,
        cell_m=cell_m,
        on_cell_fail=on_cell_fail,
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


def run_gis_cells(bbox_text: str, source: str, cell_m: float) -> None:
    from dronecv.gis.geometry import parse_bbox
    from dronecv.gis.parcel_cache import CellCache
    from dronecv.gis.parcels import cells_for_bbox

    cells = cells_for_bbox(parse_bbox(bbox_text), cell_m)
    version = "osm" if source == "osm" else "latest"
    caches = {k: CellCache(f"{source}-{k}", version) for k in ("buildings", "landcover", "poi")}
    rows = {}
    for kind, cache in caches.items():
        cached = sum(1 for c in cells if cache.has(c.id))
        skipped = sum(1 for c in cells if cache.is_skipped(c.id))
        rows[kind] = f"{cached} cached, {skipped} skipped, {len(cells) - cached - skipped} missing"
    console.print(
        metric_table(
            f"Cell cache ({source}, {cell_m:.0f} m) — {len(cells)} cells in area",
            {**rows, "cache_dir": str(CellCache(f"{source}-buildings", version).dir.parent.parent)},
        )
    )


def _viewer_handler_class():
    """A SimpleHTTPRequestHandler that also accepts POST /screenshot: it saves
    the PNG under `<served-dir>/screenshots/<model>/<filename>` so the browser
    viewer can write screenshots directly to disk (filenames encode the camera
    coordinates + height)."""
    import base64
    import json as _json
    import re
    from http.server import SimpleHTTPRequestHandler

    def _safe(name: str, default: str) -> str:
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name or "").strip("._") or default
        return name

    class ViewerHandler(SimpleHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (http.server naming)
            if self.path.rstrip("/") != "/screenshot":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = _json.loads(self.rfile.read(length))
                model = _safe(payload.get("model", ""), "model")
                fname = _safe(payload.get("filename", ""), "shot.png")
                if not fname.lower().endswith(".png"):
                    fname += ".png"
                data = payload["data"].split(",", 1)[-1]
                png = base64.b64decode(data)
            except Exception:  # noqa: BLE001
                self.send_error(400, "bad screenshot payload")
                return
            out_dir = Path(self.directory) / "screenshots" / model
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / fname).write_bytes(png)
            body = _json.dumps({"path": str(out_dir / fname)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the console quiet
            pass

    return ViewerHandler


def serve_scene_dir(scene_dir: Path, port: int = 0, model: str | None = None) -> tuple:
    """Start a background HTTP server rooted at `scene_dir` (the browser blocks
    file:// fetches of scene.glb, so it must be served) and return
    (httpd, url_of_viewer). `model` selects a specific file to open (default
    scene.glb). Caller keeps the httpd to shut it down."""
    import functools
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.parse import quote

    handler = functools.partial(_viewer_handler_class(), directory=str(scene_dir))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    real_port = httpd.server_address[1]
    url = f"http://127.0.0.1:{real_port}/viewer.html"
    if model:
        url += f"?model={quote(model)}"
    return httpd, url


def prepare_scene_viewer(env: str, terrain_resolution: int = 513) -> Path:
    """Ensure `<env>/scene_export/` holds scene.glb + viewer.html, building the
    scene if needed. Returns the scene_export directory."""
    from dronecv.config import load_config
    from dronecv.gis.export.scene_export import export_scene
    from dronecv.gis.export.viewer_html import write_viewer

    cfg = load_config(env)
    if cfg.world.kind != "gis" or not cfg.world.gis_dir:
        raise SystemExit(f"'{env}' is not a GIS environment")
    gis_dir = Path(cfg.world.gis_dir)
    out_dir = gis_dir / "scene_export"
    if not (out_dir / "scene.glb").exists():
        export_scene(gis_dir, out_dir, terrain_resolution)
    write_viewer(out_dir)
    return out_dir


def run_gis_view(env: str, port: int = 0, no_browser: bool = False) -> None:
    """Serve the realtime 3D viewer for a built environment and open a browser.
    Blocks until Ctrl+C."""
    import webbrowser

    out_dir = prepare_scene_viewer(env)
    httpd, url = serve_scene_dir(out_dir, port)
    console.print(f"[green]serving 3D viewer at[/green] [bold]{url}[/bold]  (Ctrl+C to stop)")
    if not no_browser:
        webbrowser.open(url)
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        console.print("\n[dim]viewer stopped[/dim]")
    finally:
        httpd.shutdown()


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
