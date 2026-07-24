"""Predictive validation: an induced grammar+lexicon must *generalize* to unseen text.

A real structure predicts held-out text better than a naive baseline; an overfit
pseudo-decipherment does not. We express this as a perplexity ratio (< 1.0 required).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable


def _bigram_perplexity(train: list[str], test: list[str], k: float = 0.1) -> float:
    """Add-k smoothed bigram perplexity of ``test`` under a model estimated on ``train``.
    Serves as the neutral baseline any hypothesis must beat."""
    vocab = set(train) | set(test)
    v = len(vocab)
    prev = Counter(train[:-1])
    bigrams = Counter(zip(train[:-1], train[1:]))
    log_p = 0.0
    for a, b in zip(test[:-1], test[1:]):
        p = (bigrams[(a, b)] + k) / (prev[a] + k * v)
        log_p += math.log2(p)
    n = max(len(test) - 1, 1)
    return 2 ** (-log_p / n)


def heldout_ppl_ratio(
    train: list[str],
    test: list[str],
    model_logprob: Callable[[list[str], list[str]], float] | None = None,
) -> float:
    """Ratio = perplexity(test | hypothesis model) / perplexity(test | bigram baseline).

    ``model_logprob(train, test) -> total log2 prob of test`` plugs in the hypothesis'
    own generative model (grammar+lexicon). If not provided, the ratio is 1.0 by
    construction (baseline vs baseline) — a hypothesis that supplies no predictive model
    trivially fails ``Hypothesis.generalizes()``, which is the correct default.
    """
    baseline_ppl = _bigram_perplexity(train, test)
    if model_logprob is None:
        return 1.0
    lp = model_logprob(train, test)
    n = max(len(test) - 1, 1)
    model_ppl = 2 ** (-lp / n)
    return model_ppl / baseline_ppl if baseline_ppl else float("inf")
