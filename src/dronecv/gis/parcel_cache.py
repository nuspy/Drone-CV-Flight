"""Per-cell download cache (shared across environments and sessions).

Layout: `~/.cache/dronecv/gis/cells/<provider>/<source_version>/<cell_id>.json`
plus `<cell_id>.skip` sentinels ("do not retry this cell with this source" —
mirrors the DEM provider's `.missing` sentinel). `source_version` keeps
different Overture releases (and `osm`) from colliding.

The cache stores JSON blobs; the dataclass <-> dict codecs live in
`gis.parcel_fetch` so this module stays type-agnostic and testable on its
own. The cache dir is overridable (tests point it at tmp_path).
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def default_cache_root() -> Path:
    env = os.environ.get("DRONECV_CACHE_DIR")
    base = Path(env) if env else Path.home() / ".cache" / "dronecv" / "gis"
    return base / "cells"


class CellCache:
    def __init__(self, provider: str, source_version: str, root: Path | None = None):
        self.dir = (root or default_cache_root()) / provider / _safe(source_version)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, cell_id: str) -> Path:
        return self.dir / f"{cell_id}.json"

    def _skip_path(self, cell_id: str) -> Path:
        return self.dir / f"{cell_id}.skip"

    def has(self, cell_id: str) -> bool:
        return self._path(cell_id).exists()

    def get(self, cell_id: str):
        p = self._path(cell_id)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, cell_id: str, obj) -> None:
        # Atomic write so an interrupted run never leaves a truncated cell.
        tmp = self._path(cell_id).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj))
        tmp.replace(self._path(cell_id))

    def mark_skip(self, cell_id: str, reason: str = "") -> None:
        self._skip_path(cell_id).write_text(reason)

    def is_skipped(self, cell_id: str) -> bool:
        return self._skip_path(cell_id).exists()

    def clear_skip(self, cell_id: str) -> None:
        self._skip_path(cell_id).unlink(missing_ok=True)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)
