"""Stage 3d — internal scoring of a hypothesis: coverage, consistency, and MDL.

MDL (Minimum Description Length) is the unifying criterion: the best decipherment is the one
that *compresses* the text best —
    L(grammar) + L(lexicon) + L(text | grammar, lexicon)   -> minimize.
A true decipherment captures real regularities and compresses; a pseudo-decipherment needs
as many ad-hoc exceptions as it "explains" and does not compress. Occam's razor as a cost.
"""
from __future__ import annotations

import math
from collections import Counter

from decipher.types import Hypothesis, Transliteration


def coverage(hyp: Hypothesis, tr: Transliteration) -> float:
    """Fraction of token occurrences the lexicon assigns a gloss to."""
    if not tr.tokens:
        return 0.0
    covered = sum(1 for t in tr.tokens if t in hyp.lexicon)
    return covered / len(tr.tokens)


def consistency(hyp: Hypothesis) -> float:
    """Stability of the mapping: penalize one gloss reused for many distinct tokens (a
    flexible many->one mapping can 'explain' anything). 1.0 = injective, ->0 = degenerate."""
    if not hyp.lexicon:
        return 0.0
    glosses = Counter(hyp.lexicon.values())
    reuse = sum(c - 1 for c in glosses.values())
    return 1.0 / (1.0 + reuse)


def description_length_bits(hyp: Hypothesis, tr: Transliteration, k: float = 0.1) -> float:
    """A tractable MDL proxy in bits:

        L(lexicon)               ~ |lexicon| * bits-per-entry
      + L(text | lexicon)        ~ code length of the token stream under a smoothed unigram
                                   model whose *covered* tokens are cheaper (they carry the
                                   lexicon's structure), uncovered tokens pay full price.

    Lower is better. This is a placeholder for a real grammar+lexicon coder (TODO) but has the
    right shape: adding lexicon entries costs bits, and only pays off if it shortens the text
    encoding more than it costs — which is exactly what blocks over-fitting.
    """
    bits_per_entry = 16.0   # crude fixed cost per lexicon entry
    l_lexicon = bits_per_entry * len(hyp.lexicon)

    counts = Counter(tr.tokens)
    total = sum(counts.values())
    v = len(counts)
    l_text = 0.0
    for tok in tr.tokens:
        p = (counts[tok] + k) / (total + k * v)
        cost = -math.log2(p)
        if tok in hyp.lexicon:
            cost *= 0.5     # covered tokens are cheaper: the lexicon is "paying" for them
        l_text += cost
    return l_lexicon + l_text


def score(hyp: Hypothesis, tr: Transliteration) -> Hypothesis:
    """Fill the hypothesis' internal scores. Falsification (null_model/heldout) is applied
    separately and is what ultimately admits or rejects it."""
    hyp.coverage = coverage(hyp, tr)
    hyp.consistency = consistency(hyp)
    hyp.mdl_bits = description_length_bits(hyp, tr)
    return hyp
