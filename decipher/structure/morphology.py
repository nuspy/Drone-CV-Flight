"""Stage 2 — unsupervised morphological decomposition.

Separates lexical roots from grammatical affixes without knowing meaning. Classic
approaches: Morfessor (MDL-based segmentation), or an unsupervised affix-frequency method.

The stub ships a crude frequent-affix heuristic so the pipeline runs; replace with Morfessor
or an MDL segmenter (TODO). Note the MDL theme recurs here and in hypothesis.score — it is
the same razor applied at two levels (morphology, and full decipherment).
"""
from __future__ import annotations

from collections import Counter

from decipher.types import Morphology


def analyze(tokens: list[str], max_affix_len: int = 3, min_freq: int = 3) -> Morphology:
    """Heuristic: candidate affixes are frequent prefixes/suffixes shared across many types;
    the residue after stripping is the candidate root. Deliberately simple."""
    types = set(tokens)
    prefixes: Counter = Counter()
    suffixes: Counter = Counter()
    for t in types:
        for L in range(1, min(max_affix_len, len(t)) + 1):
            prefixes[t[:L]] += 1
            suffixes[t[-L:]] += 1

    affixes = {a: c for a, c in {**{f"-{s}": c for s, c in suffixes.items()},
                                 **{f"{p}-": c for p, c in prefixes.items()}}.items()
               if c >= min_freq}

    roots: Counter = Counter()
    for t in tokens:
        stripped = t
        for s, _ in suffixes.most_common():
            if len(s) < len(stripped) and stripped.endswith(s):
                stripped = stripped[: -len(s)]
                break
        roots[stripped] += 1

    return Morphology(roots=dict(roots), affixes=affixes, paradigms=[])
