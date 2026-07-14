# SHAP Analysis (revised — grouped CV + controls)

Cross-validation is grouped by question ID so no question appears in both
train and test. Two models are fit per judge dataset:
- **judge_only**: 6 judge attributes only (grouped CV)
- **full**: judge attributes + model, variant, temperature, rationale length,
  evidence length, output validity, category flags (grouped CV)
- Question ID is controlled through the grouped split rather than used as a
  feature, because held-out questions have unseen IDs.

ROC-AUC difference between judge_only and full reveals confound magnitude.
When run with `--skip-shap`, feature rankings are RandomForest impurity
importances rather than SHAP values.

## combined_mean__full  (full model with controls)

- Rows: `42652` across `1580` questions
- SHAP explanation rows: `12000`
- Positive rate: `0.755`
- CV ROC-AUC (grouped): `0.709`
- CV Accuracy (grouped): `0.639`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.742 | 0.138 | 0.04602 | 0.787 |
| plausibility | 0.792 | 0.128 | 0.04548 | 0.827 |
| informativeness | 0.740 | 0.119 | 0.02981 | 0.753 |
| conciseness | 0.911 | 0.024 | 0.01886 | 0.730 |
| source_consistency | 0.719 | 0.134 | 0.01634 | 0.749 |
| evidence_words | 251.405 | 13.522 | 0.01516 | 0.391 |
| temperature | 0.083 | 0.028 | 0.01281 | 0.918 |
| model_Qwen3-32B | 0.333 | -0.076 | 0.00843 | -0.956 |
| non_hallucination | 0.686 | 0.117 | 0.00799 | -0.281 |
| rationale_words | 52.229 | -2.439 | 0.00780 | -0.938 |
| cat_finance | 0.217 | 0.079 | 0.00615 | 0.941 |
| cat_artificial-intelligence | 0.092 | -0.044 | 0.00387 | -0.952 |
| cat_politics | 0.139 | 0.020 | 0.00232 | 0.897 |
| cat_economy-business | 0.199 | -0.032 | 0.00209 | -0.725 |
| cat_health-pandemics | 0.071 | -0.026 | 0.00158 | -0.834 |

Top feature: `completeness` (Mean |SHAP| = `0.04602`).

## combined_mean__judge_only  (judge attributes only)

- Rows: `42652` across `1580` questions
- SHAP explanation rows: `12000`
- Positive rate: `0.755`
- CV ROC-AUC (grouped): `0.702`
- CV Accuracy (grouped): `0.633`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.742 | 0.138 | 0.07325 | 0.780 |
| plausibility | 0.792 | 0.128 | 0.05962 | 0.863 |
| informativeness | 0.740 | 0.119 | 0.02677 | 0.666 |
| non_hallucination | 0.686 | 0.117 | 0.02366 | -0.892 |
| conciseness | 0.911 | 0.024 | 0.01946 | 0.630 |
| source_consistency | 0.719 | 0.134 | 0.01077 | 0.396 |

Top feature: `completeness` (Mean |SHAP| = `0.07325`).

