# Human validation: human labels vs LLM judges

Human-annotated rationales: **100** (of 100 in the sample).

Cohen κ binarises scores at 0.5; % agree is the binary match rate; Spearman ρ uses the raw 0/0.5/1 scale. Higher = the LLM judge tracks the human better.

## Human vs GPT/Gemma third judge (same 5 criteria)

| Criterion | n | Spearman ρ | Cohen κ | % agree | Mean \|diff\| | Human | Judge |
|-----------|---|-----------|---------|---------|------------|-------|-------|
| criterion_alignment | 100 | 0.220 | 0.144 | 61% | 0.408 | 0.87 | 0.534 |
| evidence_consistency | 100 | 0.168 | 0.115 | 53% | 0.515 | 0.89 | 0.415 |
| hallucination | 100 | 0.204 | 0.081 | 39% | 0.614 | 0.91 | 0.296 |
| usefulness | 100 | 0.152 | 0.148 | 56% | 0.493 | 0.765 | 0.398 |
| probability_support | 100 | 0.170 | 0.083 | 50% | 0.574 | 0.945 | 0.3735 |

## Human vs Gemma judge (mappable criteria)

| Criterion | n | Spearman ρ | Cohen κ | % agree | Mean \|diff\| | Human | Judge |
|-----------|---|-----------|---------|---------|------------|-------|-------|
| hallucination ↔ non_hallucination | 100 | 0.229 | 0.177 | 71% | 0.287 | 0.91 | 0.689 |
| evidence_consistency ↔ source_consistency | 100 | 0.170 | 0.178 | 73% | 0.279 | 0.89 | 0.717 |
| usefulness ↔ informativeness | 100 | 0.161 | 0.218 | 74% | 0.321 | 0.765 | 0.716 |

## Human vs Kimi judge (mappable criteria)

| Criterion | n | Spearman ρ | Cohen κ | % agree | Mean \|diff\| | Human | Judge |
|-----------|---|-----------|---------|---------|------------|-------|-------|
| hallucination ↔ non_hallucination | 100 | 0.228 | 0.090 | 59% | 0.408 | 0.91 | 0.554 |
| evidence_consistency ↔ source_consistency | 100 | 0.168 | 0.095 | 59% | 0.431 | 0.89 | 0.542 |
| usefulness ↔ informativeness | 100 | 0.273 | 0.170 | 63% | 0.408 | 0.765 | 0.559 |

## Gemma vs Kimi (the two judges, full sample)

| Criterion | n | Spearman ρ | Cohen κ | % agree | Mean \|diff\| | Human | Judge |
|-----------|---|-----------|---------|---------|------------|-------|-------|
| plausibility | 100 | 0.598 | 0.419 | 80% | 0.232 | 0.735 | 0.6795 |
| completeness | 100 | 0.709 | 0.421 | 74% | 0.239 | 0.727 | 0.5395 |
| source_consistency | 100 | 0.754 | 0.662 | 84% | 0.208 | 0.717 | 0.542 |
| non_hallucination | 100 | 0.713 | 0.624 | 82% | 0.203 | 0.689 | 0.554 |
| informativeness | 100 | 0.648 | 0.429 | 75% | 0.243 | 0.716 | 0.559 |
| conciseness | 100 | 0.185 | nan | 100% | 0.138 | 0.978 | 0.8405 |

## Human pass-rate per criterion

| Criterion | Pass-rate (score ≥ 0.5) |
|-----------|--------------------------|
| criterion_alignment | 87% |
| evidence_consistency | 89% |
| hallucination | 91% |
| usefulness | 77% |
| probability_support | 95% |

## Reading these numbers

- κ ≥ 0.6 = substantial, 0.4–0.6 = moderate, < 0.4 = weak human↔judge agreement. Weak agreement on a criterion is evidence the LLM judge is *not* a safe stand-in for a human on that dimension.
- `criterion_alignment` and `probability_support` have no analogue in the Gemma/Kimi 6-criteria rubric, so they are only validated against the third judge.
