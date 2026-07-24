"""Stage 1 — glyph clusters -> neutral Unicode Private Use Area transliteration.

We map each discovered glyph cluster to a PUA codepoint (U+E000..U+F8FF) so that no
Latin/known-alphabet association leaks into the structural or hypothesis stages. The unknown
stays unknown until stage 3 grounds it (or fails to, honestly).

Segmentation (isolating glyphs from images) and clustering (grouping occurrences of the same
sign) live in ``segment.py`` / ``cluster.py`` — vision work, stubbed. This module owns only
the symbol->PUA bookkeeping, which needs no vision dependency.
"""
from __future__ import annotations

from decipher.types import Glyph, Transliteration

PUA_START = 0xE000
PUA_END = 0xF8FF


def build_transliteration(
    cluster_sequence: list[int],
    word_breaks: list[int] | None = None,
    occurrences: dict[int, int] | None = None,
) -> Transliteration:
    """Turn a sequence of cluster IDs (from OCR clustering) into a PUA Transliteration.

    ``word_breaks`` is a list of indices after which a word boundary was detected. When None,
    the text is left unsegmented and ``word_break_confident`` is False — downstream stats must
    treat token counts as provisional.
    """
    unique = sorted(set(cluster_sequence))
    if len(unique) > (PUA_END - PUA_START + 1):
        raise ValueError("more glyph clusters than PUA codepoints available")
    mapping = {cid: chr(PUA_START + i) for i, cid in enumerate(unique)}

    occ = occurrences or {}
    glyphs = {
        mapping[cid]: Glyph(cluster_id=cid, pua_char=mapping[cid], n_occurrences=occ.get(cid, 0))
        for cid in unique
    }

    breaks = set(word_breaks or [])
    chars: list[str] = []
    tokens: list[str] = []
    current: list[str] = []
    for i, cid in enumerate(cluster_sequence):
        ch = mapping[cid]
        chars.append(ch)
        current.append(ch)
        if i in breaks:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))

    return Transliteration(
        text="".join(chars),
        tokens=tokens if tokens else ["".join(chars)],
        glyphs=glyphs,
        word_break_confident=word_breaks is not None,
    )
