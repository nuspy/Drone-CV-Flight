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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dronecv.gis.geometry import BBox
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gui")


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

    def set_mask(self, geojson_text: str) -> BBox:
        gj = json.loads(geojson_text)
        geom = gj.get("geometry", gj)
        ring = geom["coordinates"][0]
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        self.mask_geojson = gj
        self.bbox = BBox(min(lats), min(lons), max(lats), max(lons))
        return self.bbox

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
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
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

        @Slot(str)
        def maskDrawn(self, geojson: str) -> None:  # noqa: N802 (JS naming)
            self.mask_drawn.emit(geojson)

        @Slot()
        def maskDeleted(self) -> None:  # noqa: N802 (JS naming)
            self.mask_deleted.emit()

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

        def __init__(self, kwargs: dict, parent=None):
            super().__init__(parent)
            self.kwargs = kwargs

        def run(self) -> None:
            try:
                from dronecv.gis.pipeline import build_environment

                root = find_config_root()
                self.progress.emit("downloading and building — this can take a while…")
                gis_dir = build_environment(
                    out_root=root / "artifacts" / "gis", configs_root=root, **self.kwargs
                )
                self.done.emit(str(gis_dir))
            except Exception as e:  # noqa: BLE001
                self.failed.emit(str(e))

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
            cfg_form.addRow("Env name", self.env_edit)
            self.res_spin = QDoubleSpinBox(minimum=0.5, maximum=10.0, value=1.0, singleStep=0.5)
            cfg_form.addRow("Resolution m/px", self.res_spin)
            self.dem_combo = QComboBox()
            self.dem_combo.addItems(["copernicus_glo30"])
            cfg_form.addRow("DEM", self.dem_combo)
            self.bld_combo = QComboBox()
            self.bld_combo.addItems(["osm_overpass", "overture"])
            self.bld_combo.currentTextChanged.connect(self.on_buildings_source)
            cfg_form.addRow("Buildings", self.bld_combo)
            self.heights_label = QLabel("—")
            cfg_form.addRow("Heights", self.heights_label)
            self.imagery_combo = QComboBox()
            self.imagery_combo.addItems([
                "none",
                "s2 (Sentinel-2 cloud-free, 10 m)",
                "eox (Sentinel-2 mosaic, non-commercial)",
            ])
            self.imagery_combo.currentIndexChanged.connect(self._refresh_ready)
            cfg_form.addRow("Imagery", self.imagery_combo)
            self.reconstruct_check = QCheckBox("reconstruct buildings from imagery")
            self.reconstruct_check.stateChanged.connect(self._refresh_ready)
            cfg_form.addRow("", self.reconstruct_check)
            self.fallback_check = QCheckBox("if OSM/Overpass fails, use Overture automatically")
            self.fallback_check.stateChanged.connect(self._refresh_ready)
            cfg_form.addRow("", self.fallback_check)
            self.palette_edit = QLineEdit(
                placeholderText="folder of area photos for the color palette (optional)"
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

            # ---- map ----
            self.web = QWebEngineView()
            self.bridge = Bridge()
            self.channel = QWebChannel()
            self.channel.registerObject("bridge", self.bridge)
            self.web.page().setWebChannel(self.channel)
            self.web.setHtml(MAP_HTML, baseUrl=QUrl("https://dronecv.local/"))
            self.bridge.mask_drawn.connect(self.on_mask)
            self.bridge.mask_deleted.connect(self.on_mask_deleted)

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
            worker = BuildWorker(self.state.build_kwargs(), parent=self)
            self._register(worker)
            worker.progress.connect(self.status.setText)
            worker.done.connect(self.on_built)
            worker.failed.connect(self.on_build_failed)
            worker.start()

        def on_built(self, gis_dir: str) -> None:
            self.progress.setVisible(False)
            name = self.state.env_name.strip()
            self.status.setText(
                f"environment built at {gis_dir}\nNext: dronecv run-all --env {name}"
            )

        def on_build_failed(self, err: str) -> None:
            self.progress.setVisible(False)
            self.build_btn.setEnabled(True)
            self.status.setText(f"build failed: {err}")

    app = QApplication([])
    win = MainWindow()
    win.show()
    raise SystemExit(app.exec())
