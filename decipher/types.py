"""Shared data contracts for the decipherment pipeline.

These dataclasses are the interface between the three stages:
  1. OCR / transliteration  (decipher.ocr)
  2. Structural analysis     (decipher.structure)
  3. Hypothesis generation   (decipher.hypothesis) + falsification (decipher.falsification)

Nothing here knows what any symbol *means* — that is the whole point. Meaning only
ever enters through an ``Anchor`` or an externally grounded ``Hypothesis``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --- Stage 1 output: transliteration -------------------------------------------------

@dataclass(frozen=True)
class Glyph:
    """A clustered symbol, mapped to a neutral Unicode Private Use Area codepoint.

    We deliberately map unknown symbols into the PUA (U+E000..U+F8FF) so no Latin/known
    associations leak into downstream models.
    """
    cluster_id: int
    pua_char: str              # e.g. ""
    n_occurrences: int
    exemplar_ref: Optional[str] = None   # pointer to a representative image crop


@dataclass
class Transliteration:
    """A text rendered as a sequence of PUA characters, segmented into tokens."""
    text: str                  # the raw PUA string (may include whitespace as word breaks)
    tokens: list[str]          # segmented "words"
    glyphs: dict[str, Glyph]   # pua_char -> Glyph
    word_break_confident: bool = False   # did segmentation find reliable word boundaries?


# --- Stage 2 output: structure -------------------------------------------------------

@dataclass
class Fingerprint:
    """Language-likeness and gross morphological signature of a corpus."""
    n_tokens: int
    n_types: int
    type_token_ratio: float
    zipf_slope: float                 # slope of log-rank vs log-freq fit; ~ -1 for language
    unigram_entropy: float            # h1, bits/token
    conditional_entropy: float        # h2 given previous token, bits/token
    mean_word_length: float
    synthesis_index: Optional[float] = None   # Greenberg morphemes-per-word, if morphology run
    is_language_like: Optional[bool] = None   # verdict from falsification.null_model

    def summary(self) -> str:
        return (
            f"types={self.n_types} tokens={self.n_tokens} ttr={self.type_token_ratio:.3f} "
            f"zipf={self.zipf_slope:.2f} h1={self.unigram_entropy:.2f} h2={self.conditional_entropy:.2f}"
        )


@dataclass
class Morphology:
    """Unsupervised morphological decomposition."""
    roots: dict[str, int] = field(default_factory=dict)      # root -> frequency
    affixes: dict[str, int] = field(default_factory=dict)    # affix -> frequency
    paradigms: list[list[str]] = field(default_factory=list) # groups of co-varying forms


class Role(str, Enum):
    """Distributionally-induced coarse role. Names are *labels of convenience*: the
    induction knows "this class behaves like X", not that it *is* X."""
    VERB_LIKE = "verb_like"
    SUBJECT_LIKE = "subject_like"
    OBJECT_LIKE = "object_like"
    MODIFIER_LIKE = "modifier_like"
    FUNCTION_LIKE = "function_like"     # articles, particles, adpositions
    UNKNOWN = "unknown"


@dataclass
class RoleAnalysis:
    token_roles: dict[str, Role] = field(default_factory=dict)   # token -> dominant role
    dominant_order: Optional[str] = None    # e.g. "SOV", "SVO", "VSO", or None if unclear
    order_confidence: float = 0.0


# --- Stage 3: typology, candidates, hypotheses ---------------------------------------

@dataclass
class TypologicalFingerprint:
    """Observed typological features with uncertainty, in a WALS/Grambank-comparable form.

    Values are probabilities/scores, never hard categoricals — every feature carries the
    uncertainty of the structural evidence it came from.
    """
    word_order: dict[str, float] = field(default_factory=dict)   # {"SOV":0.6,"SVO":0.3,...}
    morphological_type: dict[str, float] = field(default_factory=dict)  # isolating/agglutinative/...
    head_directionality: Optional[float] = None   # -1 head-final .. +1 head-initial
    has_case_marking: Optional[float] = None
    adposition_order: Optional[float] = None      # -1 postposition .. +1 preposition
    notes: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    """A candidate language/family with a probability, NEVER a single winner alone."""
    language: str
    family: Optional[str]
    typological_distance: float      # lower = better typological match
    prior_boost: float = 0.0         # from geography/epoch/material/script
    posterior: float = 0.0           # combined score, filled by candidates.rank()


@dataclass
class Anchor:
    """A structure-constrained fixed point whose meaning does not depend on knowing the
    language: numerals, proper names, dates, repeated formulae, units."""
    kind: str                        # "numeral" | "proper_name" | "formula" | "unit" | ...
    tokens: list[str]
    evidence: str
    confidence: float


@dataclass
class Hypothesis:
    """A full decipherment hypothesis for one candidate language."""
    candidate: Candidate
    lexicon: dict[str, str] = field(default_factory=dict)   # PUA token -> proposed gloss
    grammar_notes: list[str] = field(default_factory=list)
    anchors_used: list[Anchor] = field(default_factory=list)
    method_votes: dict[str, dict[str, str]] = field(default_factory=dict)  # method -> lexicon
    # scores (filled in by hypothesis.score / falsification):
    mdl_bits: Optional[float] = None
    coverage: Optional[float] = None
    consistency: Optional[float] = None
    null_margin: Optional[float] = None      # how far it beats the null models (>0 required)
    heldout_ppl_ratio: Optional[float] = None  # heldout perplexity / baseline (<1 required)

    def beats_nulls(self, min_margin: float = 0.0) -> bool:
        return self.null_margin is not None and self.null_margin > min_margin

    def generalizes(self, max_ratio: float = 1.0) -> bool:
        return self.heldout_ppl_ratio is not None and self.heldout_ppl_ratio < max_ratio

    def survives(self) -> bool:
        return self.beats_nulls() and self.generalizes()
