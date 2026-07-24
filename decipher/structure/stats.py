"""Stage 2 — statistical fingerprint of a token stream.

Real, dependency-free implementations. These answer the first question of the whole
project: *is this even language-like, or is it noise/glossolalia?* — a falsification-first
stance (we try to disprove "it's a language" before spending effort decoding it).
"""
from __future__ import annotations

import math
from collections import Counter

from decipher.types import Fingerprint


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


def unigram_entropy(tokens: list[str]) -> float:
    """h1: bits per token from the unigram distribution."""
    return _entropy(Counter(tokens))


def conditional_entropy(tokens: list[str]) -> float:
    """h2 = H(w_i | w_{i-1}), bits/token. Language sits well below the unigram entropy;
    random text with the same unigram distribution does not."""
    if len(tokens) < 2:
        return 0.0
    bigrams = Counter(zip(tokens[:-1], tokens[1:]))
    prev = Counter(tokens[:-1])
    total = sum(bigrams.values())
    h = 0.0
    for (a, b), c_ab in bigrams.items():
        p_ab = c_ab / total
        p_b_given_a = c_ab / prev[a]
        h -= p_ab * math.log2(p_b_given_a)
    return h


def zipf_slope(tokens: list[str]) -> float:
    """Least-squares slope of log(rank) vs log(freq). Natural language clusters near -1."""
    freqs = sorted(Counter(tokens).values(), reverse=True)
    if len(freqs) < 2:
        return 0.0
    xs = [math.log(r + 1) for r in range(len(freqs))]
    ys = [math.log(f) for f in freqs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def fingerprint(tokens: list[str]) -> Fingerprint:
    """Compute the full statistical fingerprint. Verdict on ``is_language_like`` is left to
    ``falsification.null_model`` — a single corpus's numbers only mean something relative to
    its own null models."""
    n_tokens = len(tokens)
    n_types = len(set(tokens))
    ttr = n_types / n_tokens if n_tokens else 0.0
    mean_len = sum(len(t) for t in tokens) / n_tokens if n_tokens else 0.0
    return Fingerprint(
        n_tokens=n_tokens,
        n_types=n_types,
        type_token_ratio=ttr,
        zipf_slope=zipf_slope(tokens),
        unigram_entropy=unigram_entropy(tokens),
        conditional_entropy=conditional_entropy(tokens),
        mean_word_length=mean_len,
    )
