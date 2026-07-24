"""The half that makes the project honest: try hard to prove the signal is *not* there.

Two roles:
  * ``is_language_like`` — decide whether a corpus behaves like language at all, by
    comparing its conditional entropy against shuffled and frequency-matched nulls.
  * ``null_margin`` — score how far a decipherment Hypothesis beats those same nulls.

A hypothesis that does not beat its nulls is meaningless, regardless of how coherent its
readings look. This is what demolishes most "solved it!" claims.
"""
from __future__ import annotations

from collections import Counter

from decipher.structure.stats import conditional_entropy


def word_shuffled(tokens: list[str], rng) -> list[str]:
    """Null (i): destroys syntax, preserves the unigram frequency distribution.

    ``rng`` must be an explicit random.Random — no implicit global RNG, so runs are
    reproducible (and so workflow-style resumability is not broken by hidden randomness).
    """
    shuffled = list(tokens)
    rng.shuffle(shuffled)
    return shuffled


def frequency_matched_random(tokens: list[str], rng) -> list[str]:
    """Null (ii): keeps only the Zipf shape, draws each position i.i.d. from the unigram dist."""
    counts = Counter(tokens)
    population = list(counts.keys())
    weights = [counts[t] for t in population]
    return rng.choices(population, weights=weights, k=len(tokens))


def is_language_like(tokens: list[str], rng, margin_bits: float = 0.15) -> tuple[bool, dict]:
    """A corpus is language-like if its *conditional* entropy is meaningfully lower than
    that of its own shuffled/random nulls: real syntax makes the next token more
    predictable from the previous one than chance allows.

    Returns (verdict, detail-dict) so the caller can log the actual numbers — silent
    truncation of evidence is not allowed.
    """
    h2_real = conditional_entropy(tokens)
    h2_shuf = conditional_entropy(word_shuffled(tokens, rng))
    h2_rand = conditional_entropy(frequency_matched_random(tokens, rng))
    best_null = min(h2_shuf, h2_rand)
    verdict = (best_null - h2_real) >= margin_bits
    return verdict, {
        "h2_real": h2_real,
        "h2_shuffled": h2_shuf,
        "h2_random": h2_rand,
        "margin_bits": best_null - h2_real,
        "threshold": margin_bits,
    }


def null_margin(hypothesis_score: float, null_scores: list[float]) -> float:
    """How far a hypothesis' internal score exceeds the best score any null model achieved
    under the *same* scoring procedure. Must be > 0 for the hypothesis to be admissible.

    ``hypothesis_score`` and ``null_scores`` are "higher is better" (e.g. negative MDL bits,
    or a coherence score). The caller is responsible for running the identical scorer on the
    nulls — that identity is the whole point of the test.
    """
    if not null_scores:
        raise ValueError("null_margin requires at least one null score; the test is the point")
    return hypothesis_score - max(null_scores)
