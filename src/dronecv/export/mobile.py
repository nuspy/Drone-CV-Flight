"""Export a trained model bundle for on-device (Android NPU) inference.

Produces `artifacts/<env>/mobile_bundle/`:

    embed.onnx          image [1,3,H,W] float32 [0,1] -> descriptor [1,D] (L2-normed)
    posenet.onnx        image -> pos_scaled [1,3], heading_vec [1,2],
                        logvar_pos [1], logvar_heading [1]
    landmark_db.bin     binary descriptor index (layout below)
    manifest.json       anchor, camera, pos_scale_m, calibration, versions

landmark_db.bin layout (little-endian):
    int32 magic = 0x444C4442 ("DLDB")   int32 version = 1
    int32 N (entries)                   int32 D (descriptor dim)
    float32[N*D]  embeddings (L2-normalized, row-major)
    float32[N*3]  pos_enu
    float32[N]    heading_deg

The Android app (and any other client) runs the two ONNX graphs with ONNX
Runtime (NNAPI/NPU execution provider on-device), does the same top-k cosine
consensus + inverse-variance fusion as
`dronecv.localization.single_shot`, and converts ENU -> WGS84 with the
manifest anchor.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch

from dronecv.training.bundle import ModelBundle

MOBILE_BUNDLE_VERSION = "1.0"
DB_MAGIC = 0x444C4442


def write_landmark_db_bin(bundle: ModelBundle, path: Path) -> None:
    emb = bundle.landmark_db.embeddings.astype("<f4")
    pos = bundle.landmark_db.pos_enu.astype("<f4")
    heading = bundle.landmark_db.heading_deg.astype("<f4")
    n, d = emb.shape
    with path.open("wb") as fh:
        fh.write(struct.pack("<iiii", DB_MAGIC, 1, n, d))
        fh.write(emb.tobytes())
        fh.write(pos.tobytes())
        fh.write(heading.tobytes())


def read_landmark_db_bin(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference reader (also documents the format for the Kotlin port)."""
    data = path.read_bytes()
    magic, version, n, d = struct.unpack_from("<iiii", data, 0)
    if magic != DB_MAGIC or version != 1:
        raise ValueError("not a landmark_db.bin v1 file")
    off = 16
    emb = np.frombuffer(data, dtype="<f4", count=n * d, offset=off).reshape(n, d)
    off += n * d * 4
    pos = np.frombuffer(data, dtype="<f4", count=n * 3, offset=off).reshape(n, 3)
    off += n * 3 * 4
    heading = np.frombuffer(data, dtype="<f4", count=n, offset=off)
    return emb.copy(), pos.copy(), heading.copy()


class _PoseNetExport(torch.nn.Module):
    """Flattens the PoseNet dict output into ONNX-friendly tensors."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return (
            out["pos_scaled"],
            out["heading_vec"],
            out["logvar_pos"],
            out["logvar_heading"],
        )


def export_mobile_bundle(bundle: ModelBundle, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam = bundle.manifest.get("camera") or {}
    width, height = int(cam.get("width", 96)), int(cam.get("height", 96))
    dummy = torch.zeros(1, 3, height, width)

    bundle.embed_net.eval()
    bundle.pose_net.eval()
    torch.onnx.export(
        bundle.embed_net,
        dummy,
        str(out_dir / "embed.onnx"),
        input_names=["image"],
        output_names=["descriptor"],
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        _PoseNetExport(bundle.pose_net),
        dummy,
        str(out_dir / "posenet.onnx"),
        input_names=["image"],
        output_names=["pos_scaled", "heading_vec", "logvar_pos", "logvar_heading"],
        opset_version=17,
        dynamo=False,
    )
    write_landmark_db_bin(bundle, out_dir / "landmark_db.bin")

    manifest = {
        "mobile_bundle_version": MOBILE_BUNDLE_VERSION,
        "env": bundle.manifest.get("env"),
        "anchor": bundle.manifest["anchor"],
        "camera": {"width": width, "height": height},
        "pos_scale_m": bundle.manifest["hyperparams"]["pos_scale_m"],
        "embedding_dim": bundle.manifest["hyperparams"]["embedding_dim"],
        "apr_sigma_scale": bundle.manifest.get("calibration", {}).get("apr_sigma_scale", 1.0),
        "trained_metrics": bundle.manifest.get("metrics", {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir
