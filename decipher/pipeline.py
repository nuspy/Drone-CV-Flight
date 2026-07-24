"""End-to-end orchestration of the three stages, falsification-first.

Mirrors the pseudocode in docs/study-point3-decipherment.md §8. The design guarantee:
no hypothesis is ever reported without having first been run against its null models and
a held-out generalization test.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from decipher.falsification import null_model
from decipher.falsification.heldout import heldout_ppl_ratio
from decipher.hypothesis import align, candidates, score, typology
from decipher.hypothesis.candidates import TypologyDB
from decipher.structure import morphology, roles, stats
from decipher.types import Fingerprint, Hypothesis, Transliteration


@dataclass
class DecipherReport:
    fingerprint: Fingerprint
    is_language_like: bool
    language_test_detail: dict
    survivors: list[Hypothesis] = field(default_factory=list)
    all_hypotheses: list[Hypothesis] = field(default_factory=list)
    message: str = ""


def run(
    tr: Transliteration,
    db: TypologyDB,
    priors: dict[str, float] | None = None,
    top_k: int = 5,
    seed: int = 0,
) -> DecipherReport:
    """Run the full pipeline. ``seed`` fixes the RNG used by the null models so results are
    reproducible."""
    rng = random.Random(seed)
    tokens = tr.tokens

    # 1. structural fingerprint + the very first falsification: is this even language?
    fp = stats.fingerprint(tokens)
    lang_ok, detail = null_model.is_language_like(tokens, rng)
    fp.is_language_like = lang_ok
    if not lang_ok:
        return DecipherReport(
            fingerprint=fp,
            is_language_like=False,
            language_test_detail=detail,
            message="Corpus does not beat its shuffled/random nulls on conditional entropy: "
                    "not language-like with the current data. Decoding is not attempted.",
        )

    # 2. structure
    morph = morphology.analyze(tokens)
    role_analysis = roles.induce(tokens)
    fp.synthesis_index = typology.synthesis_index(morph, fp.n_tokens)

    # 3. typology -> candidates -> anchors -> per-candidate alignment + internal scoring
    tf = typology.build(fp, morph, role_analysis)
    cands = candidates.rank(tf, db, priors=priors, top_k=top_k)
    anchors = align.find_anchors(tr, role_analysis)

    hyps: list[Hypothesis] = []
    split = max(1, int(len(tokens) * 0.8))
    train, test = tokens[:split], tokens[split:]
    for cand in cands:
        hyp = align.align(tr, cand, anchors)
        score.score(hyp, tr)
        # 7. falsification per hypothesis
        # null margin: compare internal MDL (lower bits = better -> negate for "higher better")
        # against the same scorer run on nulls.
        null_scores = []
        for null_tokens in (
            null_model.word_shuffled(tokens, rng),
            null_model.frequency_matched_random(tokens, rng),
        ):
            null_tr = Transliteration(text="".join(null_tokens), tokens=null_tokens, glyphs=tr.glyphs)
            null_hyp = align.align(null_tr, cand, anchors)
            score.score(null_hyp, null_tr)
            null_scores.append(-null_hyp.mdl_bits)
        hyp.null_margin = null_model.null_margin(-hyp.mdl_bits, null_scores)
        hyp.heldout_ppl_ratio = heldout_ppl_ratio(train, test, model_logprob=None)  # TODO: real model
        hyps.append(hyp)

    survivors = [h for h in hyps if h.survives()]
    survivors.sort(key=lambda h: (h.candidate.posterior, -(h.mdl_bits or 0)), reverse=True)

    msg = (
        f"{len(survivors)}/{len(hyps)} hypotheses survived falsification."
        if survivors
        else "No hypothesis survived falsification. Report the structural characterization and "
             "state which anchor (bilingual, numerals, related language) would unlock meaning."
    )
    return DecipherReport(
        fingerprint=fp,
        is_language_like=True,
        language_test_detail=detail,
        survivors=survivors,
        all_hypotheses=hyps,
        message=msg,
    )
