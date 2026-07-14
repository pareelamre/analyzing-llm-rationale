# Manual Validation: Inter-Judge Agreement Report

**Sample**: 100 rationales from Qwen2.5-7B, temperature=0, variant0_neutral_baseline
  - Correct predictions: 49
  - Incorrect predictions: 51

**Third judge**: Gemma-4-31B-it with 5 targeted criteria (criterion_alignment,
evidence_consistency, hallucination, usefulness, probability_support).

**Existing judges**: Gemma-4-31B-it and Kimi-K2.5 with 6 criteria
(plausibility, completeness, source_consistency, non_hallucination,
informativeness, conciseness).

---

## Gemma-4-31B vs Kimi-K2.5 (existing 6 criteria)

| Criterion | n | Spearman ρ | Cohen κ | Mean |diff| | Mean A | Mean B |
|-----------|---|-----------|---------|----------|--------|--------|
| plausibility | 100 | 0.633 | 0.419 | 0.232 | 0.735 | 0.679 |
| completeness | 100 | 0.726 | 0.421 | 0.239 | 0.727 | 0.539 |
| source_consistency | 100 | 0.780 | 0.662 | 0.208 | 0.717 | 0.542 |
| non_hallucination | 100 | 0.746 | 0.624 | 0.203 | 0.689 | 0.554 |
| informativeness | 100 | 0.664 | 0.429 | 0.243 | 0.716 | 0.559 |
| conciseness | 100 | 0.448 | nan | 0.138 | 0.978 | 0.841 |

## Gemma-4-31B (third judge, 5 criteria) vs Gemma-4-31B (primary, 6 criteria)

| Criterion | n | Spearman ρ | Cohen κ | Mean |diff| | Mean A | Mean B |
|-----------|---|-----------|---------|----------|--------|--------|
| hallucination ↔ non_hallucination | 100 | 0.609 | 0.301 | 0.409 | 0.296 | 0.689 |
| evidence_consistency ↔ source_consistency | 100 | 0.434 | 0.227 | 0.396 | 0.415 | 0.717 |
| usefulness ↔ informativeness | 100 | 0.433 | 0.267 | 0.386 | 0.398 | 0.716 |

## Gemma-4-31B (third judge, 5 criteria) vs Kimi-K2.5 (overlapping concepts)

| Criterion | n | Spearman ρ | Cohen κ | Mean |diff| | Mean A | Mean B |
|-----------|---|-----------|---------|----------|--------|--------|
| hallucination ↔ non_hallucination | 100 | 0.564 | 0.313 | 0.362 | 0.296 | 0.554 |
| evidence_consistency ↔ source_consistency | 100 | 0.398 | 0.168 | 0.383 | 0.415 | 0.542 |
| usefulness ↔ informativeness | 100 | 0.282 | 0.150 | 0.385 | 0.398 | 0.559 |

---

## Interpretation notes

- **Spearman ρ > 0.6**: moderate-to-strong ordinal agreement
- **Cohen κ > 0.4**: moderate agreement on binarised scores (threshold 0.5)
- **Mean |diff| < 0.15**: acceptable absolute score spread
- `probability_support` has no direct analogue in the Gemma/Kimi criteria;
  it is reported only for the GPT-OSS-120B third-judge output.
- `hallucination` (third judge) corresponds to `non_hallucination` (existing judges);
  the scale is inverted — high hallucination score = *clean* rationale.
