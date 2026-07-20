"""Desktop GUI for building GIS environments (`dronecv gis gui`).

Left panel: place search (Nominatim), source coverage with auto-preselection
and manual confirmation, environment name/resolution, build with progress.
Right: Leaflet map (QWebEngineView) where the exact AOI is drawn as a
rectangle or free polygon mask.

Logic that matters (state transitions, geojson->bbox, coverage handling,
build orchestration) lives in `GuiState`/worker classes so it is testable
offscreen without a display; the Qt widgets are a thin shell.
"""

from __future__ import annotations

import json
import queue
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dronecv.gis.geometry import BBox
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gui")

# Cell status -> map color (kept out of the Qt shell so it is testable).
CELL_COLORS = {
    "pending": "#bbbbbb", "fetching": "#e0c020", "fetched": "#2b7bd0",
    "cached": "#3fa34d", "deferred": "#e08a1e", "skipped": "#666666",
    "fallback": "#8e44ad", "aborted": "#c0392b", "failed": "#c0392b",
}


class CellDecisionController:
    """Per-cell failure decisions with an "apply to all cells with the same
    error" memo. Pure logic (no Qt): `ask(cell, error, sig) -> (Decision,
    apply_all: bool)` is supplied by the caller (a dialog in the GUI, a fake
    in tests)."""

    def __init__(self):
        self._memo: dict = {}  # error signature -> Decision

    def decide(self, cell, error, sig, ask):
        if sig in self._memo:
            return self._memo[sig]
        decision, apply_all = ask(cell, error, sig)
        if apply_all:
            self._memo[sig] = decision
        return decision

    def reset(self):
        self._memo.clear()


@dataclass
class GuiState:
    """Display-independent GUI state machine."""

    mask_geojson: dict | None = None
    bbox: BBox | None = None
    coverage: dict | None = None
    selected_sources: dict = field(default_factory=dict)
    env_name: str = ""
    res_m: float = 1.0
    ortho_path: str | None = None
    ortho_utc: str | None = None
    imagery: str = "none"  # none | s2 | eox
    reconstruct: bool = False
    palette_photos: str | None = None
    allow_overture_fallback: bool = False
    cell_m: float = 500.0

    def set_mask(self, geojson_text: str) -> BBox:
        gj = json.loads(geojson_text)
        geom = gj.get("geometry", gj)
        ring = geom["coordinates"][0]
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        self.mask_geojson = gj
        self.bbox = BBox(min(lats), min(lons), max(lats), max(lons))
        return self.bbox

    def cells_json(self) -> str:
        """GeoJSON-ish payload for the map grid: the fixed 500 m cells that
        cover the current AOI (drawn grey, colored live during the build)."""
        from dronecv.gis.parcels import cells_for_bbox

        if self.bbox is None:
            return "[]"
        cells = cells_for_bbox(self.bbox, self.cell_m)
        return json.dumps([
            {"id": c.id, "s": c.bbox.south, "w": c.bbox.west,
             "n": c.bbox.north, "e": c.bbox.east}
            for c in cells
        ])

    def apply_coverage(self, report: dict) -> dict:
        """Store the coverage report and auto-preselect sources (the user can
        still override in the panel before building)."""
        self.coverage = report
        rec = report.get("recommended", {})
        self.selected_sources = {
            "dem": rec.get("dem") or "copernicus_glo30",
            "buildings": rec.get("buildings") or "osm_overpass",
            "heights": rec.get("heights") or "shadow+defaults",
        }
        if rec.get("imagery") and self.imagery == "none":
            self.imagery = rec["imagery"]
        return self.selected_sources

    def can_build(self) -> tuple[bool, str]:
        if self.bbox is None:
            return False, "draw the area of interest on the map first"
        if not self.env_name.strip():
            return False, "set an environment name"
        if not self.env_name.replace("_", "").replace("-", "").isalnum():
            return False, "environment name must be alphanumeric/_/-"
        if self.coverage and self.coverage["dem"]["coverage"] == 0:
            return False, "no DEM coverage for this area"
        return True, "ready"

    def build_kwargs(self) -> dict:
        kwargs = {
            "bbox": self.bbox,
            "env_name": self.env_name.strip(),
            "res_m": self.res_m,
            "ortho_path": Path(self.ortho_path) if self.ortho_path else None,
            "ortho_utc": datetime.fromisoformat(self.ortho_utc) if self.ortho_utc else None,
            "imagery": None if self.imagery == "none" else self.imagery,
            "reconstruct_buildings": self.reconstruct,
            "palette_photos_dir": Path(self.palette_photos) if self.palette_photos else None,
            "allow_overture_fallback": self.allow_overture_fallback,
        }
        if self.selected_sources.get("buildings") == "overture":
            from dronecv.gis.pipeline import BuildSources
            from dronecv.gis.providers.dem import CopernicusDem
            from dronecv.gis.providers.overture import (
                OvertureBuildingsProvider,
                OvertureLandcoverProvider,
                OverturePoiProvider,
            )

            kwargs["sources"] = BuildSources(
                dem=CopernicusDem(),
                buildings=OvertureBuildingsProvider(),
                landcover=OvertureLandcoverProvider(),
                poi=OverturePoiProvider(),
            )
        return kwargs


def run_gui() -> None:  # pragma: no cover - requires a display
    try:
        from PySide6.QtCore import QObject, QThread, QUrl, Signal, Slot
        from PySide6.QtWebChannel import QWebChannel
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QProgressBar,
            QPushButton,
            QSplitter,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as e:  # noqa: F841
        raise SystemExit(
            "the desktop GUI requires PySide6: pip install 'dronecv[gui]'"
        ) from e

    from dronecv.config import find_config_root
    from dronecv.gui.map_page import MAP_HTML

    class Bridge(QObject):
        mask_drawn = Signal(str)
        mask_deleted = Signal()
        map_moved = Signal(float, float, int)
        map_ready = Signal()

        @Slot(str)
        def maskDrawn(self, geojson: str) -> None:  # noqa: N802 (JS naming)
            self.mask_drawn.emit(geojson)

        @Slot()
        def maskDeleted(self) -> None:  # noqa: N802 (JS naming)
            self.mask_deleted.emit()

        @Slot(float, float, int)
        def mapMoved(self, lat: float, lon: float, zoom: int) -> None:  # noqa: N802
            self.map_moved.emit(lat, lon, zoom)

        @Slot()
        def mapReady(self) -> None:  # noqa: N802
            self.map_ready.emit()

    class CoverageWorker(QThread):
        done = Signal(dict)
        failed = Signal(str)

        def __init__(self, bbox: BBox, parent=None):
            super().__init__(parent)
            self.bbox = bbox

        def run(self) -> None:
            try:
                from dronecv.gis.coverage import coverage_report

                self.done.emit(coverage_report(self.bbox))
            except Exception as e:  # noqa: BLE001
                self.failed.emit(str(e))

    class BuildWorker(QThread):
        done = Signal(str)
        failed = Signal(str)
        progress = Signal(str)
        cell_status = Signal(str, str)           # (cell_id, status)
        cell_failed = Signal(str, str, str)      # (cell_id, reason, error_sig)

        def __init__(self, kwargs: dict, controller, parent=None):
            super().__init__(parent)
            self.kwargs = kwargs
            self.controller = controller
            self._answers: queue.Queue = queue.Queue()  # main thread -> worker

        def answer(self, decision, apply_all: bool) -> None:
            """Called from the GUI thread with the user's dialog choice."""
            self._answers.put((decision, apply_all))

        def _ask(self, cell, error, sig):
            # Runs in the worker thread: surface the dialog, block for the answer.
            self.cell_failed.emit(cell.id, str(error), sig)
            return self._answers.get()

        def _decide(self, cell, error, sig):
            self.cell_status.emit(cell.id, "failed")
            return self.controller.decide(cell, error, sig, self._ask)

        def run(self) -> None:
            try:
                from dronecv.gis.pipeline import build_environment

                root = find_config_root()
                self.progress.emit("downloading cells and building — this can take a while…")
                gis_dir = build_environment(
                    out_root=root / "artifacts" / "gis", configs_root=root,
                    decide=self._decide,
                    on_cell_status=lambda cell, s: self.cell_status.emit(cell.id, s),
                    **self.kwargs,
                )
                self.done.emit(str(gis_dir))
            except Exception as e:  # noqa: BLE001
                self.failed.emit(str(e))

    class ActionWorker(QThread):
        """Runs one post-build CLI function (train / test / photo / export)
        off the UI thread."""

        done = Signal(str)
        failed = Signal(str)

        def __init__(self, fn, label: str, parent=None):
            super().__init__(parent)
            self.fn = fn
            self.label = label

        def run(self) -> None:
            try:
                msg = self.fn() or ""
                self.done.emit(f"{self.label}: done. {msg}")
            except Exception as e:  # noqa: BLE001
                self.failed.emit(f"{self.label} failed: {e}")

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("DroneCV — GIS environment builder")
            self.resize(1280, 800)
            self.state = GuiState()
            self._threads: list = []  # every QThread ever started, pruned lazily
            self._cov_generation = 0

            # ---- left panel ----
            panel = QWidget()
            form = QVBoxLayout(panel)
            search_row = QHBoxLayout()
            self.search_box = QLineEdit(placeholderText="search place (Nominatim)…")
            search_btn = QPushButton("Search")
            search_btn.clicked.connect(self.on_search)
            self.search_box.returnPressed.connect(self.on_search)
            search_row.addWidget(self.search_box)
            search_row.addWidget(search_btn)
            form.addLayout(search_row)

            form.addWidget(QLabel("Draw the exact area on the map (rectangle or polygon)."))

            cfg_form = QFormLayout()
            self.env_edit = QLineEdit(placeholderText="environment name, e.g. siena_centro")
            self.env_edit.setToolTip(
                "A short name (letters, digits, _ or -) for this environment. It "
                "becomes the folder under artifacts/gis/<name> and the --env used "
                "by every other command (train, run-all, view, export)."
            )
            cfg_form.addRow("Env name", self.env_edit)
            self.res_spin = QDoubleSpinBox(minimum=0.5, maximum=10.0, value=1.0, singleStep=0.5)
            self.res_spin.setToolTip(
                "Ground sampling of the imagery/orthophoto mosaic, in METERS PER "
                "PIXEL.\n\nLOWER value = finer detail, but the mosaic grows "
                "quadratically (0.5 m/px has 4× the pixels — and RAM/disk — of "
                "1 m/px) and the build is slower.\nHIGHER value = coarser texture "
                "but fast and light.\n\n1 m is a good default; use 2–4 for large "
                "areas. Building shapes come from vector data, so this mainly "
                "affects the draped texture, not the geometry."
            )
            cfg_form.addRow("Resolution m/px", self.res_spin)
            self.dem_combo = QComboBox()
            self.dem_combo.addItems(["copernicus_glo30"])
            self.dem_combo.setToolTip(
                "Elevation source for the terrain relief. Copernicus GLO-30 is a "
                "free global 30 m digital elevation model (no account needed)."
            )
            cfg_form.addRow("DEM", self.dem_combo)
            self.bld_combo = QComboBox()
            self.bld_combo.addItems(["osm_overpass", "overture"])
            self.bld_combo.currentTextChanged.connect(self.on_buildings_source)
            self.bld_combo.setToolTip(
                "Source of building footprints and heights.\n\n"
                "• osm_overpass — OpenStreetMap via Overpass: best coverage in "
                "Europe, height/levels tags where mapped.\n"
                "• overture — Overture Maps: often better height coverage in the "
                "US.\n\nThe coverage report above compares both for your area."
            )
            cfg_form.addRow("Buildings", self.bld_combo)
            self.heights_label = QLabel("—")
            self.heights_label.setToolTip(
                "How building heights will be filled: real height/levels tags "
                "first, then shadow inference (needs a timed orthophoto), then "
                "neighbor-median, then per-class defaults."
            )
            cfg_form.addRow("Heights", self.heights_label)
            self.imagery_combo = QComboBox()
            self.imagery_combo.addItems([
                "none",
                "s2 (Sentinel-2 cloud-free, 10 m)",
                "eox (Sentinel-2 mosaic, non-commercial)",
            ])
            self.imagery_combo.currentIndexChanged.connect(self._refresh_ready)
            self.imagery_combo.setToolTip(
                "Satellite imagery to drape as color (and, when reconstruction is "
                "on, to detect extra buildings).\n\n"
                "• none — synthetic class colors only (shape-first; lighting-"
                "invariant).\n"
                "• s2 — Sentinel-2 multi-date CLOUD-FREE composite (~10 m/px, "
                "free, incl. commercial).\n"
                "• eox — Sentinel-2 cloudless mosaic (non-commercial use).\n\n"
                "Colors help texture-poor scenes; geometry still drives "
                "localization."
            )
            cfg_form.addRow("Imagery", self.imagery_combo)
            self.reconstruct_check = QCheckBox("reconstruct buildings from imagery")
            self.reconstruct_check.stateChanged.connect(self._refresh_ready)
            self.reconstruct_check.setToolTip(
                "Extract EXTRA building footprints from the imagery, but ONLY "
                "where the vector data has none (GIS stays authoritative). Useful "
                "for unmapped districts or new construction. Needs an imagery "
                "source; quality tracks its resolution (10 m/px catches only "
                "large structures)."
            )
            cfg_form.addRow("", self.reconstruct_check)
            self.fallback_check = QCheckBox("if OSM/Overpass fails, use Overture automatically")
            self.fallback_check.stateChanged.connect(self._refresh_ready)
            self.fallback_check.setToolTip(
                "OFF by default so you stay in control of the data source. When "
                "ON, a cell that OSM/Overpass can't deliver is silently retried "
                "from Overture instead of asking you per cell."
            )
            cfg_form.addRow("", self.fallback_check)
            self.palette_edit = QLineEdit(
                placeholderText="folder of area photos for the color palette (optional)"
            )
            self.palette_edit.setToolTip(
                "Optional: a folder of ground/aerial photos of the area. Their "
                "dominant roof/wall colors are sampled to tint the synthetic "
                "materials so the scene looks closer to the real place."
            )
            cfg_form.addRow("Palette photos", self.palette_edit)
            form.addLayout(cfg_form)

            self.coverage_view = QPlainTextEdit(readOnly=True, maximumBlockCount=400)
            self.coverage_view.setPlaceholderText("coverage report appears after drawing the area")
            form.addWidget(self.coverage_view, stretch=1)

            self.build_btn = QPushButton("Build environment")
            self.build_btn.setEnabled(False)
            self.build_btn.clicked.connect(self.on_build)
            form.addWidget(self.build_btn)
            self.progress = QProgressBar(minimum=0, maximum=0, visible=False)
            form.addWidget(self.progress)
            self.status = QLabel("draw the area of interest on the map first")
            self.status.setWordWrap(True)
            form.addWidget(self.status)

            # ---- after-build actions (train / test / photo / export) ----
            form.addWidget(QLabel("After build (uses the Env name above):"))
            act = QFormLayout()
            self.budget_spin = QDoubleSpinBox(minimum=200, maximum=100000, value=3000,
                                              singleStep=500, decimals=0)
            self.budget_spin.setToolTip(
                "BUDGET = the maximum number of TRAINING CAPTURES the active loop "
                "may collect. A capture is one synthetic photo rendered from the "
                "3D scene (a viewpoint) that the models learn to localize.\n\n"
                "HIGHER budget = more viewpoints → better accuracy and robustness, "
                "but longer training and more disk.\nLOWER budget = fast, but the "
                "model may struggle in repetitive or large areas.\n\n"
                "How to set it: ~3000 for a small city district; 6000–10000 for "
                "large or visually repetitive areas. The loop stops early once "
                "extra captures stop improving error, so the budget is a ceiling, "
                "not a fixed cost."
            )
            train_row = QHBoxLayout()
            self.train_btn = QPushButton("Train")
            self.train_btn.clicked.connect(self.on_train)
            self.train_btn.setToolTip(
                "Run the active training loop for this environment (capture → "
                "train → measure → capture where error is high), up to Budget "
                "captures. Progress prints to the console."
            )
            self.eval_btn = QPushButton("Evaluate")
            self.eval_btn.clicked.connect(self.on_evaluate)
            self.eval_btn.setToolTip(
                "Evaluate the trained model on fresh probe captures and print the "
                "localization error metrics."
            )
            train_row.addWidget(self.train_btn)
            train_row.addWidget(self.eval_btn)
            act.addRow("Budget", self.budget_spin)
            act.addRow("Model", self._row_widget(train_row))
            self.episodes_spin = QDoubleSpinBox(minimum=1, maximum=100, value=12, decimals=0)
            self.episodes_spin.setToolTip(
                "How many autonomous test flights the reliability test runs. More "
                "episodes = a more stable pass/fail verdict but a longer test."
            )
            self.test_btn = QPushButton("Reliability test")
            self.test_btn.clicked.connect(self.on_reliability)
            self.test_btn.setToolTip(
                "Run the automated reliability test (localization accuracy + "
                "autonomous target-reach) and report PASSED/FAILED with a report "
                "link."
            )
            act.addRow("Episodes", self.episodes_spin)
            act.addRow("", self.test_btn)
            self.photo_btn = QPushButton("Localize a photo…")
            self.photo_btn.clicked.connect(self.on_localize_photo)
            self.photo_btn.setToolTip(
                "Pick a real photo of the area and estimate its GPS coordinates, "
                "heading and confidence against the trained model."
            )
            act.addRow("", self.photo_btn)
            self.export_combo = QComboBox()
            self.export_combo.addItems(["glTF/GLB scene", "OBJ", "Blender .blend", "Unity assets"])
            self.export_combo.setToolTip(
                "3D export format:\n"
                "• glTF/GLB scene — self-contained scene.glb (PBR, opens in the "
                "3D viewer, Blender, Unity, any glTF tool).\n"
                "• OBJ — buildings mesh only.\n"
                "• Blender .blend — native .blend (runs Blender if installed).\n"
                "• Unity assets — terrain.raw + splat + meshes for the Unity "
                "importer."
            )
            self.terrain_spin = QDoubleSpinBox(minimum=129, maximum=2049, value=513, decimals=0)
            self.terrain_spin.setToolTip(
                "TERRAIN RESOLUTION = the side length, in samples, of the square "
                "heightmap grid that stores the terrain relief. It must be "
                "2^n+1 (129, 257, 513, 1025, 2049) because Unity Terrain requires "
                "it.\n\nHIGHER = finer relief and smoother slopes, but larger "
                "files and slower import (2049×2049 ≈ 16× the samples of 513).\n"
                "LOWER = coarser, lighter, faster.\n\n"
                "How to set it: 513 is a good default; raise to 1025/2049 for "
                "mountainous terrain where you want crisp ridgelines, lower to "
                "257 for flat areas or to save memory. It does not affect "
                "buildings, only the ground surface."
            )
            self.export_btn = QPushButton("Export 3D")
            self.export_btn.clicked.connect(self.on_export)
            self.export_btn.setToolTip(
                "Write the selected 3D format under artifacts/gis/<env>/"
                "scene_export/."
            )
            act.addRow("Export", self.export_combo)
            act.addRow("Terrain res", self.terrain_spin)
            act.addRow("", self.export_btn)
            self.viewer_btn = QPushButton("Open 3D viewer")
            self.viewer_btn.clicked.connect(self.on_open_viewer)
            self.viewer_btn.setToolTip(
                "Open the realtime 3D viewer in your browser: sky + sun at the "
                "scene's real solar position, soft shadows, and free-fly "
                "controls (mouse look, WASD move, Q/E down/up, wheel zoom, Tab "
                "reset). Builds the scene.glb first if it doesn't exist yet."
            )
            act.addRow("", self.viewer_btn)
            form.addLayout(act)

            # ---- map ----
            self.web = QWebEngineView()
            self.bridge = Bridge()
            self.channel = QWebChannel()
            self.channel.registerObject("bridge", self.bridge)
            self.web.page().setWebChannel(self.channel)
            self.web.setHtml(MAP_HTML, baseUrl=QUrl("https://dronecv.local/"))
            self.bridge.mask_drawn.connect(self.on_mask)
            self.bridge.mask_deleted.connect(self.on_mask_deleted)
            self.bridge.map_moved.connect(self.on_map_moved)
            self.bridge.map_ready.connect(self.on_map_ready)

            from dronecv.gui.session import load_session

            self._session = load_session()

            split = QSplitter()
            split.addWidget(panel)
            split.addWidget(self.web)
            split.setStretchFactor(1, 1)
            split.setSizes([380, 900])
            self.setCentralWidget(split)

            self.env_edit.textChanged.connect(self._refresh_ready)
            self.palette_edit.textChanged.connect(self._refresh_ready)

        # ------------------------------------------------------------ handlers

        def on_search(self) -> None:
            text = self.search_box.text().strip()
            if not text:
                return
            try:
                from dronecv.commands.gis_cmd import geocode_place

                bbox = geocode_place(text)
                self.web.page().runJavaScript(
                    f"showBBox({bbox.south},{bbox.west},{bbox.north},{bbox.east})"
                )
            except Exception as e:  # noqa: BLE001
                self.status.setText(f"search failed: {e}")

        def on_mask(self, geojson: str) -> None:
            bbox = self.state.set_mask(geojson)
            self._save_session()  # remember the selection for next time
            # Draw the fixed 500 m download grid over the AOI (grey; colored
            # live during the build).
            self.web.page().runJavaScript(f"drawCells({self.state.cells_json()!r})")
            self.status.setText(
                f"area: {bbox.south:.4f},{bbox.west:.4f} → {bbox.north:.4f},{bbox.east:.4f} — "
                "checking coverage (OSM + Overture + Sentinel-2, can take ~1 min)…"
            )
            # A redraw while a previous check is still running must NOT
            # destroy that thread (Qt aborts the whole app): keep every
            # worker referenced until it finishes, and use a generation
            # counter so only the LATEST result updates the panel.
            self._cov_generation += 1
            gen = self._cov_generation
            worker = CoverageWorker(bbox, parent=self)
            self._register(worker)
            worker.done.connect(
                lambda rep, g=gen: self.on_coverage(rep) if g == self._cov_generation else None
            )
            worker.failed.connect(
                lambda e, g=gen: self.status.setText(f"coverage failed: {e}")
                if g == self._cov_generation else None
            )
            worker.start()

        def on_mask_deleted(self) -> None:
            self.state.mask_geojson = None
            self.state.bbox = None
            self.state.coverage = None
            self._cov_generation += 1  # invalidate any in-flight check
            self.coverage_view.setPlainText("")
            self._refresh_ready()

        def _register(self, worker) -> None:
            """Track EVERY QThread (coverage AND build). References are kept
            until the thread has finished — Qt aborts the whole app if a
            running QThread is destroyed — and pruned lazily, never from a
            finished-signal handler."""
            self._threads = [t for t in self._threads if t.isRunning()]
            self._threads.append(worker)

        def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
            self._save_session()
            for httpd in getattr(self, "_viewer_servers", []):
                httpd.shutdown()
            # Give running threads a moment; then detach hard so a long
            # network call or build cannot block the window from closing.
            for w in list(self._threads):
                if w.isRunning() and not w.wait(1500):
                    w.terminate()
                    w.wait(500)
            super().closeEvent(event)

        def on_buildings_source(self, text: str) -> None:
            self.state.selected_sources["buildings"] = text
            self._refresh_ready()

        def on_coverage(self, report: dict) -> None:
            selected = self.state.apply_coverage(report)
            self.heights_label.setText(selected["heights"])
            self.bld_combo.setCurrentText(selected["buildings"] or "osm_overpass")
            if self.state.imagery == "s2":
                self.imagery_combo.setCurrentIndex(1)
            dem = report["dem"]
            osm = report["buildings"]
            ovt = report.get("buildings_overture", {})
            s2 = report.get("imagery_s2", {})
            osm_line = (
                f"OSM/Overpass: {osm['n_buildings']} buildings "
                f"({osm['height_coverage']:.0%} tagged heights)"
                if osm.get("available") else "OSM/Overpass: UNREACHABLE from this network"
            )
            ovt_line = (
                f"Overture {ovt.get('release', '')}: {ovt.get('n_buildings', 0)} buildings "
                f"({ovt.get('height_coverage', 0.0):.0%} with heights)"
                if ovt.get("available") else "Overture: unavailable"
            )
            s2_line = (
                f"Sentinel-2 tile {s2.get('tile')}: {s2.get('n_recent_scenes', 0)} recent "
                f"scenes (newest {s2.get('newest')}) — cloud-free compositing available"
                if s2.get("available") else "Sentinel-2: unavailable"
            )
            self.coverage_view.setPlainText(
                f"DEM tiles available: {dem['available']}/{len(dem['tiles'])}\n"
                f"{osm_line}\n{ovt_line}\n{s2_line}\n"
                f"Recommended: buildings={selected['buildings']}, "
                f"heights={selected['heights']}\n\n"
                + json.dumps(report, indent=1)
            )
            self._refresh_ready()

        def _refresh_ready(self) -> None:
            self.state.env_name = self.env_edit.text()
            self.state.res_m = float(self.res_spin.value())
            self.state.imagery = self.imagery_combo.currentText().split(" ")[0]
            self.state.reconstruct = self.reconstruct_check.isChecked()
            self.state.allow_overture_fallback = self.fallback_check.isChecked()
            self.state.palette_photos = self.palette_edit.text().strip() or None
            self.state.selected_sources["buildings"] = self.bld_combo.currentText()
            ok, why = self.state.can_build()
            self.build_btn.setEnabled(ok)
            self.status.setText(why)

        def on_build(self) -> None:
            self.build_btn.setEnabled(False)
            self.progress.setVisible(True)
            self.web.page().runJavaScript(f"drawCells({self.state.cells_json()!r})")
            self._controller = CellDecisionController()
            worker = BuildWorker(self.state.build_kwargs(), self._controller, parent=self)
            self._build_worker = worker
            self._register(worker)
            worker.progress.connect(self.status.setText)
            worker.cell_status.connect(self.on_cell_status)
            worker.cell_failed.connect(self.on_cell_failed)
            worker.done.connect(self.on_built)
            worker.failed.connect(self.on_build_failed)
            worker.start()

        def on_cell_status(self, cell_id: str, status: str) -> None:
            color = CELL_COLORS.get(status, "#bbbbbb")
            self.web.page().runJavaScript(f"setCellColor({cell_id!r}, {color!r})")

        def on_cell_failed(self, cell_id: str, reason: str, sig: str) -> None:
            """All mirrors failed for this cell: ask the user what to do. Runs
            in the GUI thread; the worker is blocked waiting for the answer."""
            from dronecv.gis.parcel_fetch import Decision

            box = QMessageBox(self)
            box.setWindowTitle("Cell download failed")
            box.setText(f"Cell {cell_id} could not be downloaded.\n\n{reason}")
            b_retry = box.addButton("Fallback other source", QMessageBox.AcceptRole)
            b_skip = box.addButton("Skip (never retry)", QMessageBox.DestructiveRole)
            b_defer = box.addButton("Retry at end", QMessageBox.ActionRole)
            b_abort = box.addButton("Abandon build", QMessageBox.RejectRole)
            apply_all = QCheckBox("apply to all cells with the same error in this run")
            box.setCheckBox(apply_all)
            box.exec()
            clicked = box.clickedButton()
            decision = {
                b_retry: Decision.FALLBACK, b_skip: Decision.SKIP,
                b_defer: Decision.DEFER, b_abort: Decision.ABORT,
            }.get(clicked, Decision.DEFER)
            self._build_worker.answer(decision, apply_all.isChecked())

        def on_built(self, gis_dir: str) -> None:
            self.progress.setVisible(False)
            self.build_btn.setEnabled(True)
            name = self.state.env_name.strip()
            # Offer to retry the deferred cells (resumes from cache).
            import json as _json

            manifest = Path(gis_dir) / "download_manifest.json"
            deferred = _json.loads(manifest.read_text()).get("deferred", []) if manifest.exists() else []
            if deferred:
                total = self.state.cells_json().count('"id"')
                ans = QMessageBox.question(
                    self, "Retry deferred cells",
                    f"{len(deferred)} of {total} cells were deferred. Retry them now?",
                )
                if ans == QMessageBox.Yes:
                    self.on_build()  # resume: cached cells are skipped
                    return
            self.status.setText(
                f"environment built at {gis_dir}\nNext: dronecv run-all --env {name}"
            )

        def on_build_failed(self, err: str) -> None:
            self.progress.setVisible(False)
            self.build_btn.setEnabled(True)
            self.status.setText(f"build failed: {err}")

        # -------------------------------------------------- session persistence

        @staticmethod
        def _row_widget(layout):
            from PySide6.QtWidgets import QWidget as _W

            w = _W()
            w.setLayout(layout)
            return w

        def _save_session(self) -> None:
            from dronecv.gui.session import save_session

            self._session.update({
                "env_name": self.env_edit.text(),
                "buildings": self.bld_combo.currentText(),
                "imagery": self.imagery_combo.currentIndex(),
                "res_m": float(self.res_spin.value()),
                "selection": self.state.mask_geojson,
            })
            try:
                save_session(self._session)
            except OSError:
                pass

        def on_map_moved(self, lat: float, lon: float, zoom: int) -> None:
            self._session["map"] = {"lat": lat, "lon": lon, "zoom": zoom}
            self._save_session()

        def on_map_ready(self) -> None:
            s = self._session
            if s.get("map"):
                m = s["map"]
                self.web.page().runJavaScript(f"restoreView({m['lat']},{m['lon']},{m['zoom']})")
            if s.get("selection"):
                self.web.page().runJavaScript(f"restoreSelection({json.dumps(s['selection'])!r})")
            if s.get("env_name"):
                self.env_edit.setText(s["env_name"])
            if s.get("buildings"):
                self.bld_combo.setCurrentText(s["buildings"])
            if s.get("res_m"):
                self.res_spin.setValue(float(s["res_m"]))

        # ----------------------------------------------------- post-build actions

        def _run_action(self, label: str, fn) -> None:
            env = self.env_edit.text().strip()
            if not env:
                self.status.setText("set the Env name first")
                return
            self.status.setText(f"{label}… (see console for progress)")
            worker = ActionWorker(fn, label, parent=self)
            self._register(worker)
            worker.done.connect(self.status.setText)
            worker.failed.connect(self.status.setText)
            worker.start()

        def on_train(self) -> None:
            env, budget = self.env_edit.text().strip(), int(self.budget_spin.value())

            def fn():
                from dronecv.commands.train_cmd import run_train

                run_train(env, None, budget, None)
                return f"budget {budget}"

            self._run_action("training", fn)

        def on_evaluate(self) -> None:
            env = self.env_edit.text().strip()

            def fn():
                from dronecv.commands.evaluate_cmd import run_evaluate

                run_evaluate(env, None)

            self._run_action("evaluate", fn)

        def on_reliability(self) -> None:
            env, episodes = self.env_edit.text().strip(), int(self.episodes_spin.value())

            def fn():
                from dronecv.commands.test_flight_cmd import run_test_flight

                code = run_test_flight(env, None, episodes, None)
                return "PASSED" if code == 0 else "FAILED (see report)"

            self._run_action("reliability test", fn)

        def on_localize_photo(self) -> None:
            env = self.env_edit.text().strip()
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose a photo", "", "Images (*.jpg *.jpeg *.png)"
            )
            if not path:
                return

            def fn():
                from dronecv.commands.photo_cmd import run_localize_photo

                run_localize_photo(env, path, None)
                return "see console for coordinates + Maps link"

            self._run_action("localize-photo", fn)

        def on_open_viewer(self) -> None:
            env = self.env_edit.text().strip()
            if not env:
                self.status.setText("set the Env name first")
                return
            res = int(self.terrain_spin.value())
            self.status.setText("preparing 3D scene… (first time builds scene.glb)")

            def fn():
                from dronecv.commands.gis_cmd import prepare_scene_viewer

                return str(prepare_scene_viewer(env, res))

            worker = ActionWorker(fn, "3D viewer", parent=self)
            self._register(worker)
            worker.done.connect(self._open_viewer_ready)
            worker.failed.connect(self.status.setText)
            worker.start()

        def _open_viewer_ready(self, msg: str) -> None:
            # msg = "3D viewer: done. <scene_export dir>"
            import webbrowser

            from dronecv.commands.gis_cmd import serve_scene_dir

            scene_dir = Path(msg.split("done.", 1)[1].strip())
            if not hasattr(self, "_viewer_servers"):
                self._viewer_servers = []
            httpd, url = serve_scene_dir(scene_dir)
            self._viewer_servers.append(httpd)  # keep alive for the session
            webbrowser.open(url)
            self.status.setText(f"3D viewer open at {url}")

        def on_export(self) -> None:
            env = self.env_edit.text().strip()
            fmt = self.export_combo.currentText()
            res = int(self.terrain_spin.value())

            def fn():
                from pathlib import Path as _P

                from dronecv.config import load_config
                from dronecv.gis.export.scene_export import export_scene

                cfg = load_config(env)
                if cfg.world.kind != "gis" or not cfg.world.gis_dir:
                    raise RuntimeError(f"'{env}' is not a GIS environment")
                gis_dir = _P(cfg.world.gis_dir)
                out = export_scene(gis_dir, gis_dir / "scene_export", res)
                if "blend" in fmt:
                    import shutil
                    import subprocess

                    if shutil.which("blender"):
                        subprocess.run(
                            ["blender", "--background", "--python",
                             str(out / "blender_build_scene.py"), "--",
                             str(out), str(out / f"{env}.blend")], check=True)
                        return f"{out}/{env}.blend"
                    return f"assets in {out}; Blender not on PATH (run blender_build_scene.py there)"
                return str(out)

            self._run_action(f"export ({fmt})", fn)

    app = QApplication([])
    win = MainWindow()
    win.show()
    raise SystemExit(app.exec())
