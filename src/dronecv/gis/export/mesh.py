"""Mesh generation for scene exports (Unity / Blender).

Coordinate convention for all exported geometry: right-handed Z-up ENU
(x=East, y=North, z=Up) in meters, origin at the AOI anchor. Blender uses
this natively; the Unity importer rotates/mirrors on import.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mapbox_earcut
import numpy as np
from shapely.geometry import Polygon


@dataclass
class Mesh:
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)
    # Optional per-vertex UVs (facade/roof texturing). When present, kept
    # parallel to `vertices`; `add` pads with zeros if the source has none.
    uvs: list[tuple[float, float]] = field(default_factory=list)

    def add(self, verts, faces, uvs=None) -> None:
        off = len(self.vertices)
        if self.uvs or uvs:
            self.uvs.extend([(0.0, 0.0)] * (off - len(self.uvs)))
            self.uvs.extend(uvs if uvs else [(0.0, 0.0)] * len(verts))
        self.vertices.extend(verts)
        self.faces.extend((a + off, b + off, c + off) for a, b, c in faces)


def _cap_triangulation(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Earcut triangulation of a polygon with holes -> (verts2d, tri indices)."""
    rings = [np.array(poly.exterior.coords[:-1], dtype=np.float64)]
    rings += [np.array(r.coords[:-1], dtype=np.float64) for r in poly.interiors]
    verts = np.vstack(rings)
    ring_ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tris = mapbox_earcut.triangulate_float64(verts, ring_ends).reshape(-1, 3)
    return verts, tris


def extrude_polygon(poly: Polygon, base_z: float, top_z: float) -> Mesh:
    """Prism from a (possibly holed) footprint: walls + top cap + bottom cap."""
    mesh = Mesh()
    verts2d, tris = _cap_triangulation(poly)
    n = len(verts2d)
    # bottom ring then top ring
    for x, y in verts2d:
        mesh.vertices.append((float(x), float(y), float(base_z)))
    for x, y in verts2d:
        mesh.vertices.append((float(x), float(y), float(top_z)))
    # caps (top CCW up, bottom flipped)
    for a, b, c in tris:
        mesh.faces.append((int(a) + n, int(b) + n, int(c) + n))
        mesh.faces.append((int(c), int(b), int(a)))
    # walls per ring
    start = 0
    rings = [len(poly.exterior.coords) - 1] + [len(r.coords) - 1 for r in poly.interiors]
    for ring_len in rings:
        for i in range(ring_len):
            a = start + i
            b = start + (i + 1) % ring_len
            mesh.faces.append((a, b, b + n))
            mesh.faces.append((a, b + n, a + n))
        start += ring_len
    return mesh


def _rect_approx(poly: Polygon) -> tuple[np.ndarray, float, float, float]:
    """Min-rotated-rect corners ordered [c0, c0+L, c0+L+W, c0+W] + (L, W, IoU)."""
    rect = poly.minimum_rotated_rectangle
    if rect.geom_type != "Polygon":
        return np.zeros((4, 2)), 0.0, 0.0, 0.0
    c = np.array(rect.exterior.coords[:-1], dtype=np.float64)
    e01, e03 = c[1] - c[0], c[3] - c[0]
    if np.linalg.norm(e01) < np.linalg.norm(e03):
        c = c[[0, 3, 2, 1]]  # make edge 0->1 the LONG axis
    ll = float(np.linalg.norm(c[1] - c[0]))
    ww = float(np.linalg.norm(c[3] - c[0]))
    iou = poly.intersection(rect).area / max(poly.union(rect).area, 1e-9)
    return c, ll, ww, float(iou)


UV_M = 3.0  # texture tile size in meters (one facade window / floor per tile)


def _quad(mesh: Mesh, a, b, c, d) -> None:
    """Quad a->b->c->d with automatic UVs: u along the a->b horizontal run,
    v from the height of each corner (in UV_M tiles)."""
    pts = [tuple(map(float, p)) for p in (a, b, c, d)]
    u = float(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])) / UV_M
    uvs = [(0.0, pts[0][2] / UV_M), (u, pts[1][2] / UV_M),
           (u, pts[2][2] / UV_M), (0.0, pts[3][2] / UV_M)]
    mesh.add(pts, [(0, 1, 2), (0, 2, 3)], uvs)


def _tri(mesh: Mesh, a, b, c) -> None:
    pts = [tuple(map(float, p)) for p in (a, b, c)]
    u = float(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])) / UV_M
    uvs = [(0.0, pts[0][2] / UV_M), (u, pts[1][2] / UV_M), (u / 2.0, pts[2][2] / UV_M)]
    mesh.add(pts, [(0, 1, 2)], uvs)


def building_meshes(
    poly: Polygon,
    base_z: float,
    height_m: float,
    roof_shape: str | None,
    roof_height_m: float | None,
    building_class: str = "generic",
) -> tuple[Mesh, Mesh]:
    """LoD2 building -> (walls, roof) meshes with real roof geometry.

    Shaped roofs (gabled/hipped/pyramidal/skillion) are built on the
    min-rotated-rect approximation when it fits the footprint (IoU>=0.80);
    irregular or holed footprints fall back to a flat roof on the true
    footprint. Untagged shapes get a heuristic: narrow residential ->
    gabled, everything else flat."""
    corners, ll, ww, iou = _rect_approx(poly)
    shape = (roof_shape or "").lower()
    if shape in ("", "no", "unknown", None):
        shape = "gabled" if (
            building_class in ("residential", "generic")
            and ww < 20.0 and poly.area < 650.0 and not poly.interiors
        ) else "flat"
    if shape not in ("flat",) and (iou < 0.80 or poly.interiors or ww < 4.0):
        shape = "flat"
    roof_h = roof_height_m if roof_height_m else float(np.clip(0.30 * ww, 2.0, 7.0))
    roof_h = min(roof_h, max(height_m - 2.0, 0.0)) if shape != "flat" else 0.0
    top = base_z + height_m
    wall_top = top - roof_h

    walls, roof = Mesh(), Mesh()
    if shape == "flat":
        # True footprint: walls edge by edge (UV per facade), earcut roof cap.
        rings = [list(poly.exterior.coords[:-1])] + [list(r.coords[:-1]) for r in poly.interiors]
        for ring in rings:
            for i in range(len(ring)):
                a, b = ring[i], ring[(i + 1) % len(ring)]
                _quad(walls, (a[0], a[1], base_z), (b[0], b[1], base_z),
                      (b[0], b[1], top), (a[0], a[1], top))
        verts2d, tris = _cap_triangulation(poly)
        roof.add(
            [(float(x), float(y), top) for x, y in verts2d],
            [(int(a), int(b), int(c)) for a, b, c in tris],
            [(float(x) / UV_M, float(y) / UV_M) for x, y in verts2d],
        )
        return walls, roof

    z = lambda p, h: (p[0], p[1], h)  # noqa: E731
    c0, c1, c2, c3 = corners
    for a, b in ((c0, c1), (c1, c2), (c2, c3), (c3, c0)):
        _quad(walls, z(a, base_z), z(b, base_z), z(b, wall_top), z(a, wall_top))
    _quad(walls, z(c3, base_z), z(c2, base_z), z(c1, base_z), z(c0, base_z))  # bottom

    if shape in ("gabled", "gable"):
        ma, mb = (c0 + c3) / 2.0, (c1 + c2) / 2.0  # ridge under the long axis
        _tri(walls, z(c0, wall_top), z(c3, wall_top), z(ma, top))
        _tri(walls, z(c2, wall_top), z(c1, wall_top), z(mb, top))
        _quad(roof, z(c0, wall_top), z(c1, wall_top), z(mb, top), z(ma, top))
        _quad(roof, z(c2, wall_top), z(c3, wall_top), z(ma, top), z(mb, top))
    elif shape in ("hipped", "hip", "half-hipped"):
        axis = (c1 - c0) / max(ll, 1e-9)
        inset = min(ww / 2.0, ll / 2.0 - 0.1)
        ma = (c0 + c3) / 2.0 + axis * inset
        mb = (c1 + c2) / 2.0 - axis * inset
        _quad(roof, z(c0, wall_top), z(c1, wall_top), z(mb, top), z(ma, top))
        _quad(roof, z(c2, wall_top), z(c3, wall_top), z(ma, top), z(mb, top))
        _tri(roof, z(c3, wall_top), z(c0, wall_top), z(ma, top))
        _tri(roof, z(c1, wall_top), z(c2, wall_top), z(mb, top))
    elif shape in ("pyramidal", "pyramid"):
        apex = corners.mean(axis=0)
        for a, b in ((c0, c1), (c1, c2), (c2, c3), (c3, c0)):
            _tri(roof, z(a, wall_top), z(b, wall_top), z(apex, top))
    elif shape in ("skillion", "lean_to", "shed"):
        # single slope rising along the short axis: edge c0-c1 low, c3-c2 high
        _quad(roof, z(c0, wall_top), z(c1, wall_top), z(c2, top), z(c3, top))
        _tri(walls, z(c1, wall_top), z(c2, wall_top), z(c2, top))
        _tri(walls, z(c3, wall_top), z(c0, wall_top), z(c3, top))
        _quad(walls, z(c3, wall_top), z(c2, wall_top), z(c2, top), z(c3, top))
    elif shape in ("dome", "onion"):
        d = dome((c0[0] + c2[0]) / 2.0, (c0[1] + c2[1]) / 2.0, wall_top,
                 min(ll, ww) / 2.0)
        roof.add(d.vertices, d.faces)
    else:  # unrecognized tag -> flat cap on the rect
        _quad(roof, z(c0, wall_top), z(c1, wall_top), z(c2, wall_top), z(c3, wall_top))
    return walls, roof


def spire(cx: float, cy: float, base_z: float, radius: float, height: float, segments: int = 12) -> Mesh:
    """Cone (landmark spire) centered at (cx, cy)."""
    mesh = Mesh()
    apex = (cx, cy, base_z + height)
    ring = [
        (cx + radius * np.cos(2 * np.pi * i / segments),
         cy + radius * np.sin(2 * np.pi * i / segments), base_z)
        for i in range(segments)
    ]
    mesh.vertices = [*ring, apex]
    for i in range(segments):
        mesh.faces.append((i, (i + 1) % segments, segments))
    return mesh


def dome(cx: float, cy: float, base_z: float, radius: float, segments: int = 12, rings: int = 5) -> Mesh:
    """Hemisphere (landmark dome)."""
    mesh = Mesh()
    for j in range(rings):
        phi = (np.pi / 2) * j / rings
        for i in range(segments):
            th = 2 * np.pi * i / segments
            mesh.vertices.append((
                cx + radius * np.cos(phi) * np.cos(th),
                cy + radius * np.cos(phi) * np.sin(th),
                base_z + radius * np.sin(phi),
            ))
    top = len(mesh.vertices)
    mesh.vertices.append((cx, cy, base_z + radius))
    for j in range(rings - 1):
        for i in range(segments):
            a = j * segments + i
            b = j * segments + (i + 1) % segments
            c = (j + 1) * segments + (i + 1) % segments
            d = (j + 1) * segments + i
            mesh.faces.append((a, b, c))
            mesh.faces.append((a, c, d))
    last = (rings - 1) * segments
    for i in range(segments):
        mesh.faces.append((last + i, last + (i + 1) % segments, top))
    return mesh


def grid_terrain(ground: np.ndarray, res_m: float, e0: float, n0: float, step: int = 1) -> Mesh:
    """Terrain surface mesh from the ground raster (decimate with `step`)."""
    g = ground[::step, ::step]
    h, w = g.shape
    mesh = Mesh()
    for r in range(h):
        for c in range(w):
            mesh.vertices.append((e0 + c * step * res_m, n0 + r * step * res_m, float(g[r, c])))
    for r in range(h - 1):
        for c in range(w - 1):
            a = r * w + c
            b = a + 1
            d = a + w
            e = d + 1
            mesh.faces.append((a, b, e))
            mesh.faces.append((a, e, d))
    return mesh


def write_obj(path, objects: dict[str, Mesh], materials: dict | None = None) -> None:
    """Minimal OBJ writer: one `o` group per named mesh. If `materials`
    ({group_name: (r, g, b) in [0,1]}) is given, a sibling `.mtl` is written and
    each group gets `usemtl` — so importers (Unity/Blender/DCC) assign per-group
    colours instead of one default material."""
    from pathlib import Path as _P

    path = _P(path)
    mtl_name = path.with_suffix(".mtl").name if materials else None
    with open(path, "w") as fh:
        fh.write("# dronecv gis export (Z-up ENU meters)\n")
        if mtl_name:
            fh.write(f"mtllib {mtl_name}\n")
        offset = 1
        for name, mesh in objects.items():
            fh.write(f"o {name}\n")
            if materials and name in materials:
                fh.write(f"usemtl {name}\n")
            for v in mesh.vertices:
                fh.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            for f in mesh.faces:
                fh.write(f"f {f[0] + offset} {f[1] + offset} {f[2] + offset}\n")
            offset += len(mesh.vertices)
    if materials:
        with open(path.with_suffix(".mtl"), "w") as fh:
            for name, rgb in materials.items():
                r, g, b = (float(c) for c in rgb)
                fh.write(f"newmtl {name}\nKd {r:.3f} {g:.3f} {b:.3f}\nKa 0 0 0\nd 1\nillum 1\n\n")
