"""Stage 3c — candidate language -> alignment -> partial lexicon.

Four methods, each with a PRECONDITION. The orchestrator runs those whose preconditions
hold and lets them vote (convergence across independent methods is the signal; see
falsification). Anchors are tried first because their meaning is constrained by structure,
not by knowing the language.

Most method bodies are typed contracts with explicit TODOs: they require external resources
(a known related language's phonology, comparable-corpus embeddings, an LLM). The anchor
finder for numerals is implemented, since it needs nothing external.
"""
from __future__ import annotations

from collections import Counter

from decipher.types import Anchor, Candidate, Hypothesis, RoleAnalysis, Transliteration


# --- (a) anchors -------------------------------------------------------------------------

def find_numeral_anchors(tokens: list[str]) -> list[Anchor]:
    """Detect candidate numerals from *internal structure* alone: numeral systems build
    larger values by composing a small set of atoms (base 10/20/60), so number words tend to
    share sub-strings in regular, additive/multiplicative patterns and cluster at certain
    frequencies. This is a heuristic seed, not a decision.

    Heuristic here: tokens that are frequent AND share a common suffix/prefix with several
    other frequent tokens (a compositional family) are flagged low-confidence numeral
    candidates. Real work: fit a positional/base model (TODO).
    """
    freq = Counter(tokens)
    common = [t for t, _ in freq.most_common(50) if len(t) >= 2]
    anchors: list[Anchor] = []
    for t in common:
        family = [u for u in common if u != t and (u.startswith(t[:2]) or u.endswith(t[-2:]))]
        if len(family) >= 3:
            anchors.append(
                Anchor(
                    kind="numeral",
                    tokens=[t, *family[:3]],
                    evidence=f"compositional family of {len(family)} share-affix frequent tokens",
                    confidence=0.2,
                )
            )
    return anchors


def find_formula_anchors(tokens: list[str], min_len: int = 3) -> list[Anchor]:
    """Repeated fixed multi-token sequences (colophons, invocations, headings) are strong
    positional anchors. TODO: proper repeated-substring mining (suffix automaton); this stub
    catches exact repeated trigrams."""
    trigrams = Counter(tuple(tokens[i:i + min_len]) for i in range(len(tokens) - min_len + 1))
    anchors: list[Anchor] = []
    for seq, c in trigrams.items():
        if c >= 3:
            anchors.append(
                Anchor(kind="formula", tokens=list(seq), evidence=f"repeated {c}x", confidence=0.4)
            )
    return anchors


def find_anchors(tr: Transliteration, roles: RoleAnalysis | None = None) -> list[Anchor]:
    anchors = find_numeral_anchors(tr.tokens) + find_formula_anchors(tr.tokens)
    # TODO: proper-name anchors from low-freq tokens in fixed high-freq frames (needs roles)
    return anchors


# --- (b) cognate-based, needs a known relative -------------------------------------------

def align_cognates(tr: Transliteration, candidate: Candidate) -> dict[str, str]:
    """Neural/character-level alignment with a phonetic prior against a KNOWN RELATED
    language (the Ugaritic<->Hebrew, Linear-B<->Greek line of work).

    PRECONDITION: candidate has a known relative AND the script recovers phonology.
    TODO: implement minimum-cost-flow / phonetic-prior alignment; requires the relative's
    lexicon+phonology. Returns {} when the precondition is unmet.
    """
    return {}


# --- (c) unsupervised bilingual lexicon induction ---------------------------------------

def align_embeddings(tr: Transliteration, candidate: Candidate) -> dict[str, str]:
    """Align the unknown language's distributional embedding space to a candidate's
    (VecMap/MUSE family), no dictionary.

    PRECONDITION: comparable-domain corpora and roughly isomorphic distributions. FRAGILE on
    the small, narrow-domain corpora typical of ancient texts — the caller should weight this
    method low when the corpus is small. TODO: train embeddings + unsupervised mapping.
    """
    return {}


# --- (d) LLM proposer, on a short leash --------------------------------------------------

def propose_readings_llm(tr: Transliteration, candidate: Candidate, lexicon: dict[str, str]) -> dict[str, str]:
    """Ask an LLM to extend the partial lexicon with grammatical, coherent readings under the
    current hypothesis.

    The LLM PROPOSES; falsification DISPOSES. Never let the model judge its own reading — it
    hallucinates coherence by design. Every gloss returned here must still pass null_model and
    heldout downstream. TODO: wire to the API with a strict "propose, don't evaluate" prompt.
    """
    return {}


# --- orchestration of the methods --------------------------------------------------------

def align(
    tr: Transliteration,
    candidate: Candidate,
    anchors: list[Anchor],
    methods: tuple[str, ...] = ("cognate", "embedding", "llm"),
) -> Hypothesis:
    """Run the applicable methods, record each one's proposed lexicon separately (so their
    convergence can be measured), and merge into a single hypothesis lexicon."""
    hyp = Hypothesis(candidate=candidate, anchors_used=anchors)
    dispatch = {
        "cognate": align_cognates,
        "embedding": align_embeddings,
    }
    for m in methods:
        if m == "llm":
            votes = propose_readings_llm(tr, candidate, hyp.lexicon)
        else:
            votes = dispatch[m](tr, candidate)
        hyp.method_votes[m] = votes

    # merge: a gloss is adopted for a token by majority vote across methods that proposed it
    tally: dict[str, Counter] = {}
    for votes in hyp.method_votes.values():
        for tok, gloss in votes.items():
            tally.setdefault(tok, Counter())[gloss] += 1
    for tok, c in tally.items():
        gloss, n = c.most_common(1)[0]
        if n >= 2:                      # require convergence of >=2 independent methods
            hyp.lexicon[tok] = gloss
    return hyp
