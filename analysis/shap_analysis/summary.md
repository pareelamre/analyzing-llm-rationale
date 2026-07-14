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

## gemma-4-31b-it__full  (full model with controls)

- Rows: `42659` across `1580` questions
- SHAP explanation rows: `12000`
- Positive rate: `0.755`
- CV ROC-AUC (grouped): `0.698`
- CV Accuracy (grouped): `0.683`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| plausibility | 0.862 | 0.116 | 0.05271 | 0.757 |
| completeness | 0.848 | 0.109 | 0.02659 | 0.710 |
| informativeness | 0.841 | 0.109 | 0.02620 | 0.583 |
| source_consistency | 0.803 | 0.105 | 0.01883 | 0.841 |
| evidence_words | 251.413 | 13.514 | 0.01835 | 0.266 |
| rationale_words | 52.230 | -2.440 | 0.01325 | -0.934 |
| cat_finance | 0.217 | 0.079 | 0.01235 | 0.951 |
| temperature | 0.083 | 0.028 | 0.01009 | 0.903 |
| model_Qwen3-32B | 0.333 | -0.076 | 0.00920 | -0.971 |
| cat_artificial-intelligence | 0.092 | -0.044 | 0.00480 | -0.927 |
| non_hallucination | 0.739 | 0.084 | 0.00355 | -0.124 |
| cat_politics | 0.139 | 0.020 | 0.00303 | 0.845 |
| cat_economy-business | 0.199 | -0.032 | 0.00256 | -0.716 |
| model_Qwen2.5-7b-instruct | 0.333 | -0.037 | 0.00206 | 0.665 |
| cat_sports-entertainment | 0.046 | 0.012 | 0.00163 | 0.933 |

Top feature: `plausibility` (Mean |SHAP| = `0.05271`).

## gemma-4-31b-it__judge_only  (judge attributes only)

- Rows: `42659` across `1580` questions
- SHAP explanation rows: `12000`
- Positive rate: `0.755`
- CV ROC-AUC (grouped): `0.649`
- CV Accuracy (grouped): `0.668`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| plausibility | 0.862 | 0.116 | 0.07237 | 0.780 |
| completeness | 0.848 | 0.109 | 0.04185 | 0.745 |
| informativeness | 0.841 | 0.109 | 0.02717 | 0.046 |
| source_consistency | 0.803 | 0.105 | 0.01828 | 0.790 |
| non_hallucination | 0.739 | 0.084 | 0.01186 | -0.519 |
| conciseness | 0.987 | 0.008 | 0.00549 | -0.635 |

Top feature: `plausibility` (Mean |SHAP| = `0.07237`).

## kimi-k2.5__full  (full model with controls)

- Rows: `42653` across `1580` questions
- SHAP explanation rows: `12000`
- Positive rate: `0.754`
- CV ROC-AUC (grouped): `0.701`
- CV Accuracy (grouped): `0.635`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.637 | 0.168 | 0.04784 | 0.892 |
| plausibility | 0.721 | 0.140 | 0.03999 | 0.842 |
| informativeness | 0.639 | 0.129 | 0.02656 | 0.850 |
| conciseness | 0.835 | 0.040 | 0.01972 | 0.752 |
| temperature | 0.083 | 0.028 | 0.01558 | 0.931 |
| evidence_words | 251.408 | 13.508 | 0.01408 | 0.353 |
| source_consistency | 0.635 | 0.164 | 0.01220 | 0.820 |
| model_Qwen3-32B | 0.333 | -0.076 | 0.00808 | -0.928 |
| non_hallucination | 0.632 | 0.150 | 0.00785 | 0.204 |
| rationale_words | 52.230 | -2.442 | 0.00691 | -0.927 |
| cat_finance | 0.217 | 0.079 | 0.00626 | 0.943 |
| cat_artificial-intelligence | 0.092 | -0.044 | 0.00406 | -0.970 |
| cat_economy-business | 0.199 | -0.032 | 0.00235 | -0.744 |
| cat_politics | 0.139 | 0.020 | 0.00188 | 0.932 |
| cat_health-pandemics | 0.071 | -0.026 | 0.00152 | -0.863 |

Top feature: `completeness` (Mean |SHAP| = `0.04784`).

## kimi-k2.5__judge_only  (judge attributes only)

- Rows: `42653` across `1580` questions
- SHAP explanation rows: `12000`
- Positive rate: `0.754`
- CV ROC-AUC (grouped): `0.694`
- CV Accuracy (grouped): `0.622`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.637 | 0.168 | 0.08051 | 0.895 |
| plausibility | 0.721 | 0.140 | 0.05076 | 0.868 |
| informativeness | 0.639 | 0.129 | 0.02308 | 0.778 |
| conciseness | 0.835 | 0.040 | 0.01997 | 0.597 |
| non_hallucination | 0.632 | 0.150 | 0.01537 | -0.722 |
| source_consistency | 0.635 | 0.164 | 0.00967 | -0.311 |

Top feature: `completeness` (Mean |SHAP| = `0.08051`).

