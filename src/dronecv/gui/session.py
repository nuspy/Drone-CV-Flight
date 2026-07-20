"""Persist the desktop GUI session between runs.

Saves the map view (center + zoom), the drawn selection layers (GeoJSON) and
the last-used build parameters to `~/.config/dronecv/gui_state.json`, so
reopening the GUI restores the previous map position and selection. Pure I/O
(no Qt) so it is unit-testable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def session_path() -> Path:
    env = os.environ.get("DRONECV_CONFIG_DIR")
    base = Path(env) if env else Path.home() / ".config" / "dronecv"
    return base / "gui_state.json"


def load_session(path: Path | None = None) -> dict:
    p = path or session_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_session(data: dict, path: Path | None = None) -> None:
    p = path or session_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(p)
