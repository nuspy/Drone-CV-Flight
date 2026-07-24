"""Stage 3b — typological fingerprint -> ranked candidate languages.

The WEAK jump. Output is always a probability distribution over candidates, never a single
winner. Extra-textual priors (geography, epoch, material, script type) are worth MORE than
raw typology and are combined here when supplied.
"""
from __future__ import annotations

from typing import Iterable

from decipher.hypothesis import typology
from decipher.types import Candidate, TypologicalFingerprint


class TypologyDB:
    """Interface to a typological database (WALS / Grambank).

    TODO: back this with real WALS/Grambank feature tables. The scaffold ships an in-memory
    stub so the pipeline runs end-to-end; ``entries`` maps language -> (family, feature_vec)
    in the same flattened form as ``typology.feature_vector``.
    """

    def __init__(self, entries: dict[str, tuple[str, dict[str, float]]] | None = None):
        self.entries = entries or {}

    def all(self) -> Iterable[tuple[str, str, dict[str, float]]]:
        for lang, (family, vec) in self.entries.items():
            yield lang, family, vec


def rank(
    tf: TypologicalFingerprint,
    db: TypologyDB,
    priors: dict[str, float] | None = None,
    top_k: int = 10,
) -> list[Candidate]:
    """Rank candidates by combining typological similarity with extra-textual priors.

    ``priors`` maps language -> a boost in [0, 1] (geography/epoch/material/script). Because
    typology alone is too weak to identify a family, these priors should dominate whenever
    they exist; the combination here is intentionally prior-heavy.
    """
    priors = priors or {}
    query = typology.feature_vector(tf)
    cands: list[Candidate] = []
    for lang, family, vec in db.all():
        dist = typology.cosine_distance(query, vec)
        boost = priors.get(lang, 0.0)
        # posterior: closeness (1 - dist) blended with prior, prior-weighted 2:1
        posterior = ((1 - dist) + 2 * boost) / 3
        cands.append(
            Candidate(
                language=lang,
                family=family,
                typological_distance=dist,
                prior_boost=boost,
                posterior=posterior,
            )
        )
    cands.sort(key=lambda c: c.posterior, reverse=True)
    return cands[:top_k]
