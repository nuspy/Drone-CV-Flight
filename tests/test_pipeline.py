"""Smoke + behavioural tests. The key behavioural test is the *positive/negative control*:
the language-likeness gate must accept structured text and reject shuffled noise — the
falsification layer's most important promise.
"""
import random

from decipher.falsification import null_model
from decipher.hypothesis.candidates import TypologyDB
from decipher.ocr.transliterate import build_transliteration
from decipher.pipeline import run
from decipher.structure import stats


def _structured_tokens(n=600):
    """A toy language with real bigram structure: 'articles' precede 'nouns', 'nouns'
    precede 'verbs' — so conditional entropy is well below the shuffled null."""
    rng = random.Random(1)
    arts = ["a1", "a2"]
    nouns = ["n%d" % i for i in range(8)]
    verbs = ["v%d" % i for i in range(6)]
    toks = []
    for _ in range(n // 3):
        toks += [rng.choice(arts), rng.choice(nouns), rng.choice(verbs)]
    return toks


def test_language_gate_accepts_structure_rejects_noise():
    rng = random.Random(0)
    toks = _structured_tokens()
    ok, detail = null_model.is_language_like(toks, rng)
    assert ok, detail

    noise = null_model.word_shuffled(toks, random.Random(2))
    # shuffled text should NOT clear the bar (its conditional entropy ~ its unigram entropy)
    ok_noise, _ = null_model.is_language_like(noise, random.Random(3))
    assert not ok_noise


def test_fingerprint_shape():
    toks = _structured_tokens()
    fp = stats.fingerprint(toks)
    assert fp.n_tokens == len(toks)
    assert fp.n_types == len(set(toks))
    assert fp.conditional_entropy <= fp.unigram_entropy + 1e-9


def test_pipeline_runs_end_to_end_and_is_honest_on_noise():
    # structured input runs through all stages
    toks = _structured_tokens()
    tr = build_transliteration(
        cluster_sequence=list(range(len(set(toks)))) * 1,  # dummy; tokens drive stats
    )
    tr.tokens = toks
    db = TypologyDB({"toy-lang": ("toy-family", {"morph:agglutinative": 1.0})})
    report = run(tr, db, seed=0)
    assert report.is_language_like
    # no external resources are wired, so no hypothesis should falsely "survive"
    assert report.survivors == []
    assert "survived" in report.message or "No hypothesis" in report.message


def test_transliteration_maps_into_pua():
    tr = build_transliteration([0, 1, 2, 0, 1], word_breaks=[2])
    assert all(0xE000 <= ord(c) <= 0xF8FF for c in tr.text)
    assert tr.tokens[0] == tr.text[:3]
