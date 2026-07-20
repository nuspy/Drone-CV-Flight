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

    def add(self, verts, faces) -> None:
        off = len(self.vertices)
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


def write_obj(path, objects: dict[str, Mesh]) -> None:
    """Minimal OBJ writer: one `o` group per named mesh."""
    with open(path, "w") as fh:
        fh.write("# dronecv gis export (Z-up ENU meters)\n")
        offset = 1
        for name, mesh in objects.items():
            fh.write(f"o {name}\n")
            for v in mesh.vertices:
                fh.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            for f in mesh.faces:
                fh.write(f"f {f[0] + offset} {f[1] + offset} {f[2] + offset}\n")
            offset += len(mesh.vertices)
