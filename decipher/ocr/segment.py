"""Stage 1 — segment an image of unknown script into individual glyph bounding boxes.

Vision work, stubbed. This is the well-studied, tractable part of the whole project and
needs no LLM: connected-component / projection-profile segmentation for clean scripts,
a learned detector for cursive/connected ones.

TODO: implement. Suggested path: start with OpenCV connected components + projection
profiles; escalate to a small learned segmenter only if the script is connected/cursive.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GlyphBox:
    """A detected glyph region in an image."""
    x: int
    y: int
    w: int
    h: int
    page: int = 0


def segment_page(image_path: str) -> list[GlyphBox]:  # pragma: no cover - stub
    raise NotImplementedError(
        "segment_page: implement glyph detection (connected components / projection profiles)."
    )
