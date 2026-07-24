"""Stage 2 — distributional role induction: find verb-like / subject-like / object-like
classes and the dominant constituent order, WITHOUT knowing meaning.

Method: unsupervised POS induction (Brown clustering / HMM) groups tokens by distributional
context; argument structure and dominant order come from the positions of the induced classes
around the verb-like class. The class NAMES are labels of convenience — the induction knows
"this class behaves like X", not that it *is* X.

The stub gives a frequency/position heuristic so the pipeline runs; replace with Brown
clustering or an HMM POS inducer (TODO).
"""
from __future__ import annotations

from collections import Counter

from decipher.types import Role, RoleAnalysis


def induce(tokens: list[str], sentences: list[list[str]] | None = None) -> RoleAnalysis:
    """Heuristic roles + a placeholder order estimate.

    Without real POS induction we cannot reliably fix S/V/O order, so ``dominant_order`` is
    left None with 0 confidence here — honestly reporting "unknown" rather than guessing.
    Wire Brown/HMM induction to populate it (TODO).
    """
    freq = Counter(tokens)
    roles: dict[str, Role] = {}
    if freq:
        cutoff = freq.most_common(max(1, len(freq) // 20))[-1][1]
        for tok, c in freq.items():
            # very frequent short tokens behave like function words; rare ones like content
            if c >= cutoff and len(tok) <= 2:
                roles[tok] = Role.FUNCTION_LIKE
            else:
                roles[tok] = Role.UNKNOWN

    return RoleAnalysis(token_roles=roles, dominant_order=None, order_confidence=0.0)
