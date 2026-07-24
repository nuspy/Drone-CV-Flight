"""Stage 3a — structure -> typological fingerprint (WALS/Grambank-comparable).

This is the *moderately* reliable jump. Everything here stays probabilistic: each
feature carries the uncertainty of the structural evidence behind it.

REMEMBER: typology != genealogy. SOV+agglutinative describes Turkish, Japanese, Quechua,
Korean, Dravidian — unrelated languages. This module narrows the search space; it does not
identify a family. Callers must treat the output as a prior, never a match.
"""
from __future__ import annotations

import math

from decipher.types import Fingerprint, Morphology, RoleAnalysis, TypologicalFingerprint


def synthesis_index(morph: Morphology, n_tokens: int) -> float | None:
    """Greenberg morphemes-per-word. ~1 isolating, 2-3 agglutinative/fusional, higher
    polysynthetic. Approximated as (roots + affix attachments) / word count."""
    if n_tokens == 0:
        return None
    morphs = sum(morph.roots.values()) + sum(morph.affixes.values())
    return morphs / n_tokens if morphs else None


def _morphological_type(si: float | None, fp: Fingerprint) -> dict[str, float]:
    """Soft classification into isolating/agglutinative/fusional/polysynthetic.

    Uses the synthesis index plus a proxy for fusion: high type/token ratio with a moderate
    synthesis index suggests fusion (one affix = several features); many affix types with
    regular attachment suggests agglutination. This is deliberately coarse — a placeholder
    for a trained classifier calibrated on WALS-labelled corpora (TODO).
    """
    if si is None:
        return {}
    if si < 1.3:
        return {"isolating": 0.7, "agglutinative": 0.2, "fusional": 0.1}
    if si < 2.5:
        # distinguish agglutinative vs fusional by regularity proxy (ttr)
        fusion = min(0.6, fp.type_token_ratio)
        return {"agglutinative": 1 - fusion, "fusional": fusion}
    return {"agglutinative": 0.4, "polysynthetic": 0.6}


def build(fp: Fingerprint, morph: Morphology, roles: RoleAnalysis) -> TypologicalFingerprint:
    """Assemble the typological fingerprint from the stage-2 structures."""
    si = synthesis_index(morph, fp.n_tokens)

    word_order: dict[str, float] = {}
    if roles.dominant_order:
        # spread the remaining mass over the alternatives so nothing is treated as certain
        conf = max(0.0, min(1.0, roles.order_confidence))
        word_order[roles.dominant_order] = conf
        others = [o for o in ("SOV", "SVO", "VSO", "VOS", "OVS", "OSV") if o != roles.dominant_order]
        for o in others:
            word_order[o] = (1 - conf) / len(others)

    tf = TypologicalFingerprint(
        word_order=word_order,
        morphological_type=_morphological_type(si, fp),
    )
    if si is not None:
        tf.notes.append(f"synthesis_index~{si:.2f}")
    tf.notes.append("typology constrains the candidate set; it does NOT identify a family")
    return tf


def feature_vector(tf: TypologicalFingerprint) -> dict[str, float]:
    """Flatten to a numeric vector for distance computation against a typological DB.
    Missing features are simply absent (handled as max-uncertainty by the distance fn)."""
    vec: dict[str, float] = {}
    for order, p in tf.word_order.items():
        vec[f"order:{order}"] = p
    for mt, p in tf.morphological_type.items():
        vec[f"morph:{mt}"] = p
    for name in ("head_directionality", "has_case_marking", "adposition_order"):
        val = getattr(tf, name)
        if val is not None:
            vec[name] = float(val)
    return vec


def cosine_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """1 - cosine similarity over shared keys; 1.0 (max distance) when they share nothing."""
    keys = set(a) & set(b)
    if not keys:
        return 1.0
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)
