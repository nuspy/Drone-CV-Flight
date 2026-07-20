"""Standardized global cell grid for parcelled downloads.

Vector data (OSM/Overpass, Overture) is fetched cell by cell instead of one
giant bbox request, so a single failure loses only ~500 m of territory, the
result is cached per cell (reused across sessions — aborting a build keeps
what was already downloaded), and overlapping selections resolve to the SAME
cells and are processed once.

The grid is GLOBAL and FIXED: cells are anchored at (0, 0) with a constant
angular step derived from `cell_m` in latitude, so the same ground square is
always the same cell — in any request, any session, any AOI. Cells are
`cell_m` tall (N-S) and ~`cell_m`*cos(lat) wide (E-W); the slight E-W
narrowing at high latitude is irrelevant to download/cache/merge, which all
reproject into local ENU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dronecv.gis.geometry import BBox

M_PER_DEG_LAT = 111_320.0
DEFAULT_CELL_M = 500.0


def cell_deg(cell_m: float = DEFAULT_CELL_M) -> float:
    """Angular step of the fixed grid for a given cell size in meters."""
    return cell_m / M_PER_DEG_LAT


@dataclass(frozen=True)
class Cell:
    """One fixed-grid cell, identified by integer indices from origin (0,0)."""

    ilat: int
    ilon: int
    cell_m: float = DEFAULT_CELL_M

    @property
    def id(self) -> str:
        return f"c{int(self.cell_m)}_{self.ilat}_{self.ilon}"

    @property
    def bbox(self) -> BBox:
        d = cell_deg(self.cell_m)
        return BBox(self.ilat * d, self.ilon * d, (self.ilat + 1) * d, (self.ilon + 1) * d)

    @property
    def center(self) -> tuple[float, float]:
        return self.bbox.center


def cell_of(lat: float, lon: float, cell_m: float = DEFAULT_CELL_M) -> Cell:
    d = cell_deg(cell_m)
    return Cell(int(math.floor(lat / d)), int(math.floor(lon / d)), cell_m)


def cells_for_bbox(bbox: BBox, cell_m: float = DEFAULT_CELL_M) -> list[Cell]:
    """Every grid cell that intersects the bbox (snapped to the fixed grid)."""
    d = cell_deg(cell_m)
    i0 = int(math.floor(bbox.south / d))
    i1 = int(math.floor((bbox.north - 1e-12) / d))
    j0 = int(math.floor(bbox.west / d))
    j1 = int(math.floor((bbox.east - 1e-12) / d))
    return [Cell(i, j, cell_m) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]


def _geojson_bbox(gj: dict) -> BBox:
    """Bounding box of a GeoJSON geometry / Feature (rings of [lon, lat])."""
    geom = gj.get("geometry", gj)
    coords = geom["coordinates"]

    def walk(x):
        if isinstance(x, (int, float)):
            return
        if x and isinstance(x[0], (int, float)):
            pts.append(x)
            return
        for sub in x:
            walk(sub)

    pts: list = []
    walk(coords)
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return BBox(min(lats), min(lons), max(lats), max(lons))


def cells_for_selection(
    selections: list, cell_m: float = DEFAULT_CELL_M
) -> list[Cell]:
    """Deduplicated UNION of the cells covering several selections (BBox or
    GeoJSON dict). Overlapping selections yield the same cells only once, so
    downstream fetching never processes an overlap twice. Returned sorted for
    determinism."""
    seen: dict[str, Cell] = {}
    for sel in selections:
        bbox = sel if isinstance(sel, BBox) else _geojson_bbox(sel)
        for c in cells_for_bbox(bbox, cell_m):
            seen[c.id] = c
    return [seen[k] for k in sorted(seen)]


def cells_bbox(cells: list[Cell]) -> BBox:
    """Bounding box covering a set of cells (their union extent)."""
    if not cells:
        raise ValueError("no cells")
    boxes = [c.bbox for c in cells]
    return BBox(
        min(b.south for b in boxes), min(b.west for b in boxes),
        max(b.north for b in boxes), max(b.east for b in boxes),
    )
