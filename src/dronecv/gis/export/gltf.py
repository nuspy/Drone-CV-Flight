"""Minimal glTF 2.0 (GLB) writer — no external dependencies.

Produces a single .glb with PBR materials (base color factor and/or embedded
PNG texture), optional per-vertex UVs and colors, and a root node that
rotates our Z-up ENU meters into glTF's Y-up convention. Imports cleanly in
Blender, Unity (glTFast/UnityGLTF) and any glTF viewer.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_ZUP_TO_YUP = [-0.7071067811865476, 0.0, 0.0, 0.7071067811865476]  # -90 deg about X


class GlbBuilder:
    def __init__(self, name: str = "dronecv-scene"):
        self._bin = bytearray()
        self._buffer_views: list[dict] = []
        self._accessors: list[dict] = []
        self._images: list[dict] = []
        self._textures: list[dict] = []
        self._materials: list[dict] = []
        self._meshes: list[dict] = []
        self._nodes: list[dict] = []
        self._name = name

    # ------------------------------------------------------------ low level

    def _push(self, data: bytes, target: int | None) -> int:
        while len(self._bin) % 4:
            self._bin.append(0)
        view = {"buffer": 0, "byteOffset": len(self._bin), "byteLength": len(data)}
        if target:
            view["target"] = target
        self._bin.extend(data)
        self._buffer_views.append(view)
        return len(self._buffer_views) - 1

    def _accessor(self, view: int, ctype: int, count: int, type_: str,
                  vmin=None, vmax=None) -> int:
        acc = {"bufferView": view, "componentType": ctype, "count": count, "type": type_}
        if vmin is not None:
            acc["min"], acc["max"] = vmin, vmax
        self._accessors.append(acc)
        return len(self._accessors) - 1

    # ----------------------------------------------------------- public API

    def add_material(self, name: str, rgb: tuple[float, float, float],
                     texture_png: bytes | None = None, roughness: float = 0.92) -> int:
        mat = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [*[float(c) for c in rgb], 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": roughness,
            },
            "doubleSided": True,
        }
        if texture_png is not None:
            img_view = self._push(texture_png, None)
            self._images.append({"bufferView": img_view, "mimeType": "image/png"})
            self._textures.append({"source": len(self._images) - 1})
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {
                "index": len(self._textures) - 1
            }
        self._materials.append(mat)
        return len(self._materials) - 1

    def add_mesh(self, name: str, positions: np.ndarray, indices: np.ndarray,
                 material: int, uvs: np.ndarray | None = None,
                 colors: np.ndarray | None = None) -> None:
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        idx = np.ascontiguousarray(indices, dtype=np.uint32).ravel()
        pv = self._push(pos.tobytes(), 34962)
        pa = self._accessor(pv, 5126, len(pos), "VEC3",
                            [float(v) for v in pos.min(0)], [float(v) for v in pos.max(0)])
        iv = self._push(idx.tobytes(), 34963)
        ia = self._accessor(iv, 5125, len(idx), "SCALAR")
        attrs = {"POSITION": pa}
        if uvs is not None and len(uvs) == len(pos):
            uv = np.ascontiguousarray(uvs, dtype=np.float32)
            attrs["TEXCOORD_0"] = self._accessor(self._push(uv.tobytes(), 34962), 5126,
                                                 len(uv), "VEC2")
        if colors is not None and len(colors) == len(pos):
            col = np.ascontiguousarray(colors, dtype=np.float32)
            attrs["COLOR_0"] = self._accessor(self._push(col.tobytes(), 34962), 5126,
                                              len(col), "VEC3")
        self._meshes.append({
            "name": name,
            "primitives": [{"attributes": attrs, "indices": ia, "material": material}],
        })
        self._nodes.append({"name": name, "mesh": len(self._meshes) - 1})

    def save(self, path: Path) -> Path:
        root = {"name": self._name, "rotation": _ZUP_TO_YUP,
                "children": list(range(1, len(self._nodes) + 1))}
        gltf = {
            "asset": {"version": "2.0", "generator": "dronecv-gis"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [root, *self._nodes],
            "meshes": self._meshes,
            "materials": self._materials,
            "accessors": self._accessors,
            "bufferViews": self._buffer_views,
            "buffers": [{"byteLength": len(self._bin)}],
        }
        if self._images:
            gltf["images"] = self._images
            gltf["textures"] = self._textures
            gltf["samplers"] = [{"wrapS": 10497, "wrapT": 10497}]
            for t in gltf["textures"]:
                t["sampler"] = 0
        payload = json.dumps(gltf).encode()
        payload += b" " * ((4 - len(payload) % 4) % 4)
        binc = bytes(self._bin) + b"\0" * ((4 - len(self._bin) % 4) % 4)
        out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(payload) + 8 + len(binc))
        out += struct.pack("<II", len(payload), 0x4E4F534A) + payload
        out += struct.pack("<II", len(binc), 0x004E4942) + binc
        Path(path).write_bytes(out)
        return Path(path)


def window_atlas_png(size: int = 128) -> bytes:
    """One facade tile: white plaster with a dark window — multiplied by the
    wall base color it becomes 'wall with windows', tiling every UV_M meters
    horizontally (one window bay) and vertically (one floor)."""
    import io

    from PIL import Image

    img = np.full((size, size, 3), 255, np.uint8)
    x0, x1 = int(size * 0.30), int(size * 0.70)
    y0, y1 = int(size * 0.28), int(size * 0.78)
    img[y0:y1, x0:x1] = (72, 84, 98)
    img[y0 + 2 : y1 - 2, x0 + 2 : x0 + 4] = 230  # frame highlights
    mid = (x0 + x1) // 2
    img[y0:y1, mid : mid + 2] = 200
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    return buf.getvalue()
