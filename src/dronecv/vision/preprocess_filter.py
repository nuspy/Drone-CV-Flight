"""Shared domain-gap preprocessing filter.

The models are trained on shape-first synthetic renders; real photos differ
radically in appearance (colors, textures, light). A filter that keeps
structure (edges, luminance) and discards appearance shrinks that gap — but
ONLY if the exact same transform is applied on BOTH sides: to every training
image AND to every inference input (single-shot photos, live flight frames,
evaluation probes). This module is that single definition; the trained
bundle records its spec in the manifest so inference can never drift from
what the model was trained on.

    out = (1 - edge_weight) * grayscale + edge_weight * sobel_edges

with `mode` selecting the terms and optional CLAHE contrast normalization
before the edge extraction. Output keeps 3 channels so model architectures
are untouched; `mode="none"` is the exact identity (mobile/ONNX contract
unchanged).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass(frozen=True)
class FilterSpec:
    mode: str = "none"  # none | gray | edge | gray_edge
    edge_weight: float = 0.5  # only used by gray_edge
    clahe: bool = False

    def __post_init__(self):
        if self.mode not in ("none", "gray", "edge", "gray_edge"):
            raise ValueError(f"unknown filter mode '{self.mode}'")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> FilterSpec:
        return cls(**d) if d else cls()

    @property
    def is_identity(self) -> bool:
        return self.mode == "none"


def apply_filter(img: np.ndarray, spec: FilterSpec | None) -> np.ndarray:
    """HxWx3 float32 [0,1] -> HxWx3 float32 [0,1]. Deterministic; identity
    for mode="none"."""
    if spec is None or spec.is_identity:
        return img
    import cv2

    gray = np.clip(img.astype(np.float32) @ LUMA, 0.0, 1.0)
    if spec.clahe:
        u8 = (gray * 255.0).astype(np.uint8)
        u8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8)
        gray = u8.astype(np.float32) / 255.0

    if spec.mode == "gray":
        out = gray
    else:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.clip(np.sqrt(gx * gx + gy * gy) / 2.0, 0.0, 1.0)
        if spec.mode == "edge":
            out = edges
        else:  # gray_edge
            w = float(np.clip(spec.edge_weight, 0.0, 1.0))
            out = (1.0 - w) * gray + w * edges
    return np.repeat(out[..., None], 3, axis=-1).astype(np.float32)
