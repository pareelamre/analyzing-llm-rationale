# Partial SHAP Analysis

This analysis uses the currently available LLM-judge outputs and predicts `forecast_correct` from the judged rationale attributes. The rationale-evaluation sample is `42660` model-variant-question rows when missing rationales are included; SHAP metrics below use the rows with complete judge scores.

## combined_mean

- Rows: `42660` sample rows including missing rationales (`42652` complete scored rows)
- Positive rate: `0.755`
- CV ROC-AUC: `0.726`
- CV Accuracy: `0.646`

| Feature | Mean | Correct - Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.742 | 0.138 | 0.07380 | 0.779 |
| plausibility | 0.792 | 0.128 | 0.05963 | 0.863 |
| informativeness | 0.740 | 0.119 | 0.02688 | 0.664 |
| non_hallucination | 0.686 | 0.117 | 0.02378 | -0.891 |
| conciseness | 0.911 | 0.024 | 0.01939 | 0.628 |
| source_consistency | 0.719 | 0.134 | 0.01083 | 0.402 |

Top SHAP feature: `completeness` with mean |SHAP| `0.07380`.

## gemma-4-31b-it

- Rows: `42659`
- Positive rate: `0.755`
- CV ROC-AUC: `0.666`
- CV Accuracy: `0.672`

| Feature | Mean | Correct - Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| plausibility | 0.862 | 0.116 | 0.07253 | 0.778 |
| completeness | 0.848 | 0.109 | 0.04180 | 0.748 |
| informativeness | 0.841 | 0.109 | 0.02732 | 0.037 |
| source_consistency | 0.803 | 0.105 | 0.01811 | 0.788 |
| non_hallucination | 0.739 | 0.084 | 0.01187 | -0.526 |
| conciseness | 0.987 | 0.008 | 0.00547 | -0.645 |

Top SHAP feature: `plausibility` with mean |SHAP| `0.07253`.

## kimi-k2.5

- Rows: `42653`
- Positive rate: `0.754`
- CV ROC-AUC: `0.713`
- CV Accuracy: `0.630`

| Feature | Mean | Correct - Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.637 | 0.168 | 0.08073 | 0.895 |
| plausibility | 0.721 | 0.140 | 0.05086 | 0.870 |
| informativeness | 0.639 | 0.129 | 0.02309 | 0.782 |
| conciseness | 0.835 | 0.040 | 0.02005 | 0.594 |
| non_hallucination | 0.632 | 0.150 | 0.01541 | -0.726 |
| source_consistency | 0.635 | 0.164 | 0.00974 | -0.310 |

Top SHAP feature: `completeness` with mean |SHAP| `0.08073`.
