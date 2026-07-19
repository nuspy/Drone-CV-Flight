"""HTTP inference server for photo localization (`dronecv serve`).

Endpoints (consumed by the Android app and by curl for quick tests):

    GET  /health              liveness + env id
    GET  /info                model/anchor/camera summary
    POST /localize            multipart "image" (jpeg/png) -> geographic fix
    GET  /mobile-bundle.zip   the exported on-device model bundle (the app
                              downloads it once for NPU mode)

Run:  dronecv serve --env headless_ci [--host 0.0.0.0 --port 8000]
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from dronecv import __version__
from dronecv.export.mobile import export_mobile_bundle
from dronecv.localization.single_shot import SingleShotLocalizer
from dronecv.training.bundle import ModelBundle


def create_app(bundle: ModelBundle, mobile_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="dronecv photo localization", version=__version__)
    localizer = SingleShotLocalizer(bundle)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "env": bundle.manifest.get("env"), "version": __version__}

    @app.get("/info")
    def info() -> dict:
        cam = bundle.manifest.get("camera") or {}
        return {
            "env": bundle.manifest.get("env"),
            "anchor": bundle.manifest["anchor"],
            "camera": {"width": cam.get("width"), "height": cam.get("height")},
            "metrics": bundle.manifest.get("metrics", {}),
            "version": __version__,
        }

    @app.post("/localize")
    async def localize(image: UploadFile = File(...)) -> dict:  # noqa: B008
        raw = await image.read()
        buf = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="could not decode image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fix = localizer.localize(rgb)
        return {
            "lat": fix.lat,
            "lon": fix.lon,
            "alt_msl": fix.alt_msl,
            "heading_deg": fix.heading_deg,
            "confidence": fix.confidence,
            "sigma_h_m": fix.sigma_h_m,
            "pos_enu": fix.pos_enu.tolist(),
            "diagnostics": fix.diagnostics,
        }

    @app.get("/mobile-bundle.zip")
    def mobile_bundle() -> Response:
        nonlocal mobile_dir
        if mobile_dir is None or not (mobile_dir / "manifest.json").exists():
            # Export lazily on first request.
            mobile_dir = Path(mobile_dir or "mobile_bundle_export")
            export_mobile_bundle(bundle, mobile_dir)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(mobile_dir.iterdir()):
                if f.is_file():
                    zf.write(f, f.name)
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=mobile_bundle.zip"},
        )

    return app


def run_server(bundle_dir: Path, host: str, port: int, mobile_dir: Path | None) -> None:
    import uvicorn

    bundle = ModelBundle.load(bundle_dir)
    app = create_app(bundle, mobile_dir)
    print(json.dumps({"serving": bundle.manifest.get("env"), "host": host, "port": port}))
    uvicorn.run(app, host=host, port=port, log_level="info")
