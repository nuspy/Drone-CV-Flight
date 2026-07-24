# decipher — structuring and cautiously decoding unknown scripts

A scaffold for going from an image of an **unknown / undeciphered script** to *ranked,
falsifiable hypotheses* about its structure and — where an anchor to the known world exists —
its meaning.

Design study: **[`docs/study-point3-decipherment.md`](docs/study-point3-decipherment.md)**.

## The one idea that makes it serious

Any sufficiently flexible symbol→meaning mapping can produce *plausible-looking* readings
(the graveyard of "Voynich solved!" claims). So this project is **two halves of equal weight**:

1. **Generation** — structure → typology → candidate languages → alignment → readings.
2. **Falsification** — null models, held-out prediction, positive controls, cross-method
   convergence, MDL. **No hypothesis is reported without first beating its null models and a
   generalization test.**

Without the second half, a decipherment tool is a machine for coherent illusions.

## The three stages

| Stage | Package | Status |
|---|---|---|
| 1. OCR: unknown symbols → neutral Unicode PUA transliteration | `decipher/ocr` | transliteration real; segmentation/clustering (vision) stubbed |
| 2. Structure: statistics, morphology, role induction | `decipher/structure` | stats real; morphology/roles heuristic stubs |
| 3. Hypothesis: typology → candidates → alignment → MDL scoring | `decipher/hypothesis` | contracts + logic; external resources (WALS, embeddings, LLM) stubbed |
| ⊹ Falsification (cross-cutting) | `decipher/falsification` | null models + held-out real |

Why unknown symbols are mapped into the **Private Use Area** (U+E000–U+F8FF): so no
Latin/known-alphabet association leaks into the downstream models. The unknown stays unknown
until stage 3 grounds it — or honestly fails to.

## What it can and cannot conclude

- Known language, unknown script → likely a verifiable reading.
- Unknown language, known relative → partial lexicon with confidence.
- Isolate with some anchor (numerals, fragmentary bilingual) → islands of meaning.
- Isolate, no anchor → **full structure, no meaning.** Saying so — and naming which anchor
  would unlock it — is the correct scientific output, not a failure.

## Quick start

```bash
pip install -e ".[dev]"
pytest
```

The pipeline runs end-to-end on toy input today (see `tests/test_pipeline.py`); the `TODO`
markers flag exactly where real resources plug in (a vision segmenter, corpus embeddings, a
WALS/Grambank table, an LLM proposer). The LLM, when wired, **proposes** readings and never
**judges** them — every gloss still passes the falsification layer.
