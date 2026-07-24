"""Stage 1 — cluster glyph crops so repeated symbols get the same cluster ID.

Unsupervised: the model has no idea what any symbol *is*; it only decides "these two
ghirigori are the same sign". Visual embedding (small CNN / autoencoder / a siamese net) +
clustering (DBSCAN handles an unknown number of clusters and outliers well).

TODO: implement. The output cluster IDs feed ``transliterate.build_transliteration``.
"""
from __future__ import annotations

from decipher.ocr.segment import GlyphBox


def cluster_glyphs(boxes: list[GlyphBox], image_paths: list[str]) -> list[int]:  # pragma: no cover - stub
    """Return a cluster ID per box, in reading order. Same sign -> same ID."""
    raise NotImplementedError(
        "cluster_glyphs: embed glyph crops (CNN/siamese) then cluster (DBSCAN/k-means)."
    )
