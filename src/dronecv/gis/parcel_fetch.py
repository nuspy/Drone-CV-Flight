"""Cell-by-cell fetch orchestrator with per-cell failure decisions.

Fetches a set of grid cells from a provider, caching each successful cell.
When a cell fails (after the provider's own retries/mirror-fallback), a
`decide(cell, error) -> Decision` callback chooses what to do — the GUI wires
this to a dialog, the CLI to a policy:

    SKIP      leave the cell empty and never retry it with this source
              (writes a .skip sentinel)
    FALLBACK  refetch THIS cell from the other source, cache under it
    DEFER     add the cell to a "retry at end of run" list, move on
    ABORT     stop now (already-cached cells stay on disk -> next run resumes)

`decide` also receives an error SIGNATURE so the GUI's "apply to all cells
with the same error" checkbox can memoize a choice for the run.

Provider efficiency: Overpass is queried one small request per cell (light,
avoids 504s). Overture is scanned ONCE over the union bbox of the missing
cells and the result partitioned into per-cell caches (re-scanning the
parquet footers per 500 m cell would be wasteful) — providers advertise
this via an optional `fetch_region(cells)`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum

from dronecv.gis.parcels import Cell
from dronecv.gis.providers.buildings import Building
from dronecv.gis.providers.landcover import LandcoverFeature
from dronecv.gis.providers.poi import BuildingPart, Poi
from dronecv.util.logging import get_logger

log = get_logger("dronecv.gis.cells")


class Decision(Enum):
    SKIP = "skip"
    FALLBACK = "fallback"
    DEFER = "defer"
    ABORT = "abort"


class BuildAborted(RuntimeError):
    """Raised when a decision aborts the whole build."""


# --------------------------------------------------------------- codecs

def _dump(kind: str, obj):
    """Serialize a provider result (list, or the POI (pois, parts) tuple)."""
    if kind == "poi":
        pois, parts = obj
        return {"pois": [asdict(p) for p in pois], "parts": [asdict(p) for p in parts]}
    return [asdict(x) for x in obj]


def _load(kind: str, blob):
    if kind == "buildings":
        return [Building(**d) for d in blob]
    if kind == "landcover":
        return [LandcoverFeature(**d) for d in blob]
    if kind == "poi":
        return (
            [Poi(**d) for d in blob["pois"]],
            [BuildingPart(**d) for d in blob["parts"]],
        )
    raise ValueError(f"unknown kind {kind}")


def _empty(kind: str):
    return ([], []) if kind == "poi" else []


# -------------------------------------------------------------- reports

@dataclass
class CellReport:
    kind: str
    cached: list[str] = field(default_factory=list)
    fetched: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    fallback: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # cell_id -> reason
    n_total: int = 0

    def summary(self) -> dict:
        return {
            "kind": self.kind, "n_total": self.n_total,
            "cached": len(self.cached), "fetched": len(self.fetched),
            "skipped": len(self.skipped), "fallback": len(self.fallback),
            "deferred": self.deferred, "failed": self.failed,
        }


# --------------------------------------------------------------- merge

def merge_dedup(kind: str, per_cell: list):
    """Concatenate per-cell results and drop boundary duplicates (a feature
    straddling two cells is returned by both)."""
    if kind == "poi":
        pois: dict = {}
        parts: dict = {}
        for pois_i, parts_i in per_cell:
            for p in pois_i:
                pois[p.wikidata or f"{p.lonlat}:{p.name}"] = p
            for pt in parts_i:
                parts[_ring_key(pt.footprint_lonlat)] = pt
        return list(pois.values()), list(parts.values())
    if kind == "buildings":
        out: dict = {}
        for cell_list in per_cell:
            for b in cell_list:
                key = f"osm:{b.osm_id}" if b.osm_id is not None else _ring_key(b.footprint_lonlat)
                out[key] = b
        return list(out.values())
    # landcover
    out = {}
    for cell_list in per_cell:
        for f in cell_list:
            ring = f.ring_lonlat or f.line_lonlat or []
            out[f"{f.kind}:{_ring_key(ring)}"] = f
    return list(out.values())


def _ring_key(ring) -> str:
    if not ring:
        return "∅"
    return f"{len(ring)}:{round(ring[0][0], 6)},{round(ring[0][1], 6)}:" \
           f"{round(ring[-1][0], 6)},{round(ring[-1][1], 6)}"


# ----------------------------------------------------------- orchestrator

def _err_signature(error: Exception) -> str:
    """Coarse fingerprint so 'apply to all same errors' can group failures."""
    text = str(error)
    for token in ("504", "503", "502", "429", "timed out", "timeout", "Connection"):
        if token in text:
            return token
    return type(error).__name__


def fetch_cells(
    kind: str,
    cells: list[Cell],
    provider,
    source_version: str,
    cache,
    decide: Callable[[Cell, Exception, str], Decision] | None = None,
    fallback_provider=None,
    fallback_source_version: str | None = None,
    fallback_cache=None,
    on_status: Callable[[Cell, str], None] | None = None,
) -> tuple[list, CellReport]:
    """Fetch every cell (skipping cached / skipped ones). Returns
    (list-of-per-cell-results, CellReport). `decide` defaults to DEFER."""
    report = CellReport(kind=kind, n_total=len(cells))
    per_cell: list = []
    decide = decide or (lambda cell, err, sig: Decision.DEFER)

    def status(cell: Cell, s: str):
        if on_status:
            on_status(cell, s)

    # Overture-style batch: fetch the whole missing-region once, partition.
    to_fetch = [c for c in cells if not cache.has(c.id) and not cache.is_skipped(c.id)]
    prefetched = _try_region(kind, to_fetch, provider, cache, status)

    for cell in cells:
        if cache.is_skipped(cell.id):
            report.skipped.append(cell.id)
            per_cell.append(_empty(kind))
            status(cell, "skipped")
            continue
        blob = cache.get(cell.id)
        if blob is not None:
            per_cell.append(_load(kind, blob))
            report.cached.append(cell.id)
            status(cell, "cached")
            continue
        if cell.id in prefetched:  # filled by the batch pre-fetch
            per_cell.append(_load(kind, cache.get(cell.id)))
            report.fetched.append(cell.id)
            status(cell, "fetched")
            continue
        status(cell, "fetching")
        try:
            result = provider.fetch(cell.bbox)
        except Exception as e:  # noqa: BLE001 — a FETCH failure -> decide
            sig = _err_signature(e)
            d = decide(cell, e, sig)
            if d == Decision.ABORT:
                report.failed[cell.id] = str(e)
                status(cell, "aborted")
                raise BuildAborted(f"build aborted at cell {cell.id}: {e}") from e
            if d == Decision.SKIP:
                cache.mark_skip(cell.id, str(e))
                report.skipped.append(cell.id)
                per_cell.append(_empty(kind))
                status(cell, "skipped")
            elif d == Decision.FALLBACK and fallback_provider is not None:
                per_cell.append(
                    _fallback_one(kind, cell, fallback_provider, fallback_cache, status)
                )
                report.fallback.append(cell.id)
            else:  # DEFER (or FALLBACK without a fallback provider)
                report.deferred.append(cell.id)
                report.failed[cell.id] = str(e)
                per_cell.append(_empty(kind))
                status(cell, "deferred")
        else:
            # Fetch SUCCEEDED. Persist best-effort: a cache-write failure must
            # never masquerade as a fetch failure (which would defer + discard
            # good data and force a re-download next run). Use the data anyway.
            try:
                cache.put(cell.id, _dump(kind, result))
            except Exception as ce:  # noqa: BLE001
                log.warning(f"cell {cell.id} fetched but could NOT be cached "
                            f"({ce}); it will be refetched next run. Cache dir: {cache.dir}")
            per_cell.append(result)
            report.fetched.append(cell.id)
            status(cell, "fetched")
    n_reuse = len(report.cached) + len(report.skipped)
    if n_reuse:
        log.info(f"cells[{kind}]: resumed {n_reuse}/{report.n_total} from cache "
                 f"(dir: {cache.dir})")
    log.info(f"cells[{kind}]: {report.summary()}")
    return per_cell, report


def _try_region(kind, cells, provider, cache, status) -> set[str]:
    """If the provider supports fetch_region, fetch the union of missing cells
    in one shot and cache each. Returns the ids that got filled."""
    fn = getattr(provider, "fetch_region", None)
    if fn is None or not cells:
        return set()
    try:
        per_cell = fn(cells)  # {cell_id: result}
    except Exception as e:  # noqa: BLE001 — fall back to per-cell fetching
        log.warning(f"region fetch failed ({e}); falling back to per-cell")
        return set()
    filled = set()
    for cell in cells:
        if cell.id in per_cell:
            cache.put(cell.id, _dump(kind, per_cell[cell.id]))
            filled.add(cell.id)
            status(cell, "fetched")
    return filled


def _fallback_one(kind, cell, fallback_provider, fallback_cache, status):
    status(cell, "fallback")
    cached = fallback_cache.get(cell.id) if fallback_cache else None
    if cached is not None:
        return _load(kind, cached)
    result = fallback_provider.fetch(cell.bbox)
    if fallback_cache:
        fallback_cache.put(cell.id, _dump(kind, result))
    return result


# ---------------------------------------------------- pipeline-facing glue

def bucket_by_cell(kind: str, result, cells: list[Cell]) -> dict:
    """Assign each feature to the grid cell containing its centroid — lets a
    provider fetch a whole region once and fill per-cell caches (Overture).
    Returns {cell_id: per-cell-result}."""
    from dronecv.gis.parcels import cell_of

    ids = {c.id for c in cells}
    cm = cells[0].cell_m if cells else 500.0

    def assign(out, pts, feat):
        if not pts:
            return
        clon = sum(p[0] for p in pts) / len(pts)
        clat = sum(p[1] for p in pts) / len(pts)
        cid = cell_of(clat, clon, cm).id
        if cid in ids:
            out[cid].append(feat)

    if kind == "poi":
        out: dict = {c.id: ([], []) for c in cells}
        pois, parts = result
        for p in pois:
            cid = cell_of(p.lonlat[1], p.lonlat[0], cm).id
            if cid in ids:
                out[cid][0].append(p)
        for pt in parts:
            pts = pt.footprint_lonlat
            clon = sum(x[0] for x in pts) / len(pts)
            clat = sum(x[1] for x in pts) / len(pts)
            cid = cell_of(clat, clon, cm).id
            if cid in ids:
                out[cid][1].append(pt)
        return out

    out = {c.id: [] for c in cells}
    for f in result:
        pts = f.footprint_lonlat if kind == "buildings" else (f.ring_lonlat or f.line_lonlat)
        assign(out, pts, f)
    return out


def source_identity(provider) -> tuple[str, str]:
    """(source_name, source_version) for cache keying. Overpass -> ('osm',
    'osm'); Overture -> ('overture', release or 'latest')."""
    from dronecv.gis.providers.buildings import OverpassBuildings
    from dronecv.gis.providers.landcover import OverpassLandcover
    from dronecv.gis.providers.poi import OverpassPoi

    if isinstance(provider, (OverpassBuildings, OverpassLandcover, OverpassPoi)):
        return "osm", "osm"
    return "overture", str(getattr(provider, "release", None) or "latest")


def opposite_provider(kind: str, provider):
    """Build the other-source provider for a FALLBACK decision."""
    to_overture = source_identity(provider)[0] == "osm"
    import dronecv.gis.providers.overture as ov
    from dronecv.gis.providers import buildings as ob
    from dronecv.gis.providers import landcover as ol
    from dronecv.gis.providers import poi as op

    if kind == "buildings":
        return ov.OvertureBuildingsProvider() if to_overture else ob.OverpassBuildings()
    if kind == "landcover":
        return ov.OvertureLandcoverProvider() if to_overture else ol.OverpassLandcover()
    if kind == "poi":
        return ov.OverturePoiProvider() if to_overture else op.OverpassPoi()
    raise ValueError(kind)


def policy_decider(on_cell_fail: str) -> Callable[[Cell, Exception, str], Decision]:
    """Non-interactive decision policy from a flag (CLI / headless builds)."""
    mapping = {
        "defer": Decision.DEFER, "skip": Decision.SKIP,
        "abort": Decision.ABORT, "fallback": Decision.FALLBACK,
    }
    if on_cell_fail not in mapping:
        raise ValueError(f"on_cell_fail must be one of {sorted(mapping)}")
    choice = mapping[on_cell_fail]
    return lambda cell, err, sig: choice


def fetch_vector(
    kind: str,
    provider,
    bbox,
    cell_m: float = 500.0,
    cache_root=None,
    decide: Callable[[Cell, Exception, str], Decision] | None = None,
    on_status: Callable[[Cell, str], None] | None = None,
):
    """Cell-based fetch of one vector kind for an AOI bbox. Returns
    (merged_result, CellReport). Handles cache setup, the FALLBACK provider,
    and boundary dedup — the pipeline's single entry point."""
    from dronecv.gis.parcel_cache import CellCache
    from dronecv.gis.parcels import cells_for_bbox

    cells = cells_for_bbox(bbox, cell_m)
    name, version = source_identity(provider)
    # The cache dir is keyed by KIND too, else buildings/landcover/poi under
    # the same source ('osm') would collide on identical cell filenames.
    cache = CellCache(f"{name}-{kind}", version, root=cache_root)

    fb_provider = opposite_provider(kind, provider)
    fb_name, fb_version = source_identity(fb_provider)
    fb_cache = CellCache(f"{fb_name}-{kind}", fb_version, root=cache_root)

    per_cell, report = fetch_cells(
        kind, cells, provider, version, cache, decide=decide,
        fallback_provider=fb_provider, fallback_source_version=fb_version,
        fallback_cache=fb_cache, on_status=on_status,
    )
    return merge_dedup(kind, per_cell), report
