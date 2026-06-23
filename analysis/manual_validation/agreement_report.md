# Manual Validation: Inter-Judge Agreement Report

**Sample**: 150 rationales from Qwen2.5-7B, temperature=0, variant0_neutral_baseline
  - Correct predictions: 75
  - Incorrect predictions: 75

**Third judge**: Gemma-4-31B-it with 5 targeted criteria (criterion_alignment,
evidence_consistency, hallucination, usefulness, probability_support).

**Existing judges**: Gemma-4-31B-it and Kimi-K2.5 with 6 criteria
(plausibility, completeness, source_consistency, non_hallucination,
informativeness, conciseness).

---

## Gemma-4-31B vs Kimi-K2.5 (existing 6 criteria)

| Criterion | n | Spearman ρ | Cohen κ | Mean |diff| | Mean A | Mean B |
|-----------|---|-----------|---------|----------|--------|--------|
| plausibility | 150 | 0.653 | 0.408 | 0.229 | 0.767 | 0.677 |
| completeness | 150 | 0.714 | 0.431 | 0.245 | 0.750 | 0.549 |
| source_consistency | 150 | 0.773 | 0.558 | 0.217 | 0.741 | 0.554 |
| non_hallucination | 150 | 0.741 | 0.579 | 0.210 | 0.697 | 0.566 |
| informativeness | 150 | 0.660 | 0.475 | 0.251 | 0.742 | 0.563 |
| conciseness | 150 | 0.489 | nan | 0.144 | 0.983 | 0.839 |

## Gemma-4-31B (third judge, 5 criteria) vs Gemma-4-31B (primary, 6 criteria)

| Criterion | n | Spearman ρ | Cohen κ | Mean |diff| | Mean A | Mean B |
|-----------|---|-----------|---------|----------|--------|--------|
| hallucination ↔ non_hallucination | 150 | 0.593 | 0.309 | 0.397 | 0.337 | 0.697 |
| evidence_consistency ↔ source_consistency | 150 | 0.493 | 0.253 | 0.367 | 0.453 | 0.741 |
| usefulness ↔ informativeness | 150 | 0.480 | 0.295 | 0.363 | 0.443 | 0.742 |

## Gemma-4-31B (third judge, 5 criteria) vs Kimi-K2.5 (overlapping concepts)

| Criterion | n | Spearman ρ | Cohen κ | Mean |diff| | Mean A | Mean B |
|-----------|---|-----------|---------|----------|--------|--------|
| hallucination ↔ non_hallucination | 150 | 0.508 | 0.284 | 0.373 | 0.337 | 0.566 |
| evidence_consistency ↔ source_consistency | 150 | 0.429 | 0.267 | 0.361 | 0.453 | 0.554 |
| usefulness ↔ informativeness | 150 | 0.342 | 0.262 | 0.362 | 0.443 | 0.563 |

---

## Interpretation notes

- **Spearman ρ > 0.6**: moderate-to-strong ordinal agreement
- **Cohen κ > 0.4**: moderate agreement on binarised scores (threshold 0.5)
- **Mean |diff| < 0.15**: acceptable absolute score spread
- `probability_support` has no direct analogue in the Gemma/Kimi criteria;
  it is reported only for the GPT-OSS-120B third-judge output.
- `hallucination` (third judge) corresponds to `non_hallucination` (existing judges);
  the scale is inverted — high hallucination score = *clean* rationale.