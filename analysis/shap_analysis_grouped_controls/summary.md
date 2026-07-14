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
- SHAP explanation rows: `0`
- Positive rate: `0.755`
- CV ROC-AUC (grouped): `0.709`
- CV Accuracy (grouped): `0.639`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | RF importance | Value-Importance Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.742 | 0.138 | 0.21605 | 0.000 |
| plausibility | 0.792 | 0.128 | 0.21537 | 0.000 |
| informativeness | 0.740 | 0.119 | 0.14801 | 0.000 |
| source_consistency | 0.719 | 0.134 | 0.09320 | 0.000 |
| evidence_words | 251.405 | 13.522 | 0.07418 | 0.000 |
| conciseness | 0.911 | 0.024 | 0.07190 | 0.000 |
| non_hallucination | 0.686 | 0.117 | 0.05159 | 0.000 |
| temperature | 0.083 | 0.028 | 0.02582 | 0.000 |
| rationale_words | 52.229 | -2.439 | 0.02104 | 0.000 |
| model_Qwen3-32B | 0.333 | -0.076 | 0.01361 | 0.000 |
| cat_finance | 0.217 | 0.079 | 0.01197 | 0.000 |
| cat_artificial-intelligence | 0.092 | -0.044 | 0.01002 | 0.000 |
| cat_economy-business | 0.199 | -0.032 | 0.00651 | 0.000 |
| cat_politics | 0.139 | 0.020 | 0.00474 | 0.000 |
| cat_geopolitics | 0.169 | -0.007 | 0.00432 | 0.000 |

Top feature: `completeness` (RF importance = `0.21605`).

## combined_mean__judge_only  (judge attributes only)

- Rows: `42652` across `1580` questions
- SHAP explanation rows: `0`
- Positive rate: `0.755`
- CV ROC-AUC (grouped): `0.702`
- CV Accuracy (grouped): `0.633`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | RF importance | Value-Importance Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.742 | 0.138 | 0.36246 | 0.000 |
| plausibility | 0.792 | 0.128 | 0.26036 | 0.000 |
| informativeness | 0.740 | 0.119 | 0.16932 | 0.000 |
| source_consistency | 0.719 | 0.134 | 0.08794 | 0.000 |
| conciseness | 0.911 | 0.024 | 0.06522 | 0.000 |
| non_hallucination | 0.686 | 0.117 | 0.05469 | 0.000 |

Top feature: `completeness` (RF importance = `0.36246`).

## gemma-4-31b-it__full  (full model with controls)

- Rows: `42659` across `1580` questions
- SHAP explanation rows: `0`
- Positive rate: `0.755`
- CV ROC-AUC (grouped): `0.698`
- CV Accuracy (grouped): `0.683`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | RF importance | Value-Importance Corr |
| --- | ---: | ---: | ---: | ---: |
| plausibility | 0.862 | 0.116 | 0.24419 | 0.000 |
| informativeness | 0.841 | 0.109 | 0.15088 | 0.000 |
| evidence_words | 251.413 | 13.514 | 0.13393 | 0.000 |
| completeness | 0.848 | 0.109 | 0.12198 | 0.000 |
| source_consistency | 0.803 | 0.105 | 0.09115 | 0.000 |
| rationale_words | 52.230 | -2.440 | 0.04083 | 0.000 |
| non_hallucination | 0.739 | 0.084 | 0.03353 | 0.000 |
| cat_finance | 0.217 | 0.079 | 0.03137 | 0.000 |
| temperature | 0.083 | 0.028 | 0.02837 | 0.000 |
| model_Qwen3-32B | 0.333 | -0.076 | 0.01854 | 0.000 |
| cat_artificial-intelligence | 0.092 | -0.044 | 0.01614 | 0.000 |
| conciseness | 0.987 | 0.008 | 0.01148 | 0.000 |
| cat_economy-business | 0.199 | -0.032 | 0.01057 | 0.000 |
| cat_politics | 0.139 | 0.020 | 0.00959 | 0.000 |
| model_Qwen2.5-7b-instruct | 0.333 | -0.037 | 0.00842 | 0.000 |

Top feature: `plausibility` (RF importance = `0.24419`).

## gemma-4-31b-it__judge_only  (judge attributes only)

- Rows: `42659` across `1580` questions
- SHAP explanation rows: `0`
- Positive rate: `0.755`
- CV ROC-AUC (grouped): `0.649`
- CV Accuracy (grouped): `0.668`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | RF importance | Value-Importance Corr |
| --- | ---: | ---: | ---: | ---: |
| plausibility | 0.862 | 0.116 | 0.40160 | 0.000 |
| informativeness | 0.841 | 0.109 | 0.21873 | 0.000 |
| completeness | 0.848 | 0.109 | 0.21028 | 0.000 |
| source_consistency | 0.803 | 0.105 | 0.10511 | 0.000 |
| non_hallucination | 0.739 | 0.084 | 0.04700 | 0.000 |
| conciseness | 0.987 | 0.008 | 0.01729 | 0.000 |

Top feature: `plausibility` (RF importance = `0.40160`).

## kimi-k2.5__full  (full model with controls)

- Rows: `42653` across `1580` questions
- SHAP explanation rows: `0`
- Positive rate: `0.754`
- CV ROC-AUC (grouped): `0.701`
- CV Accuracy (grouped): `0.635`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | RF importance | Value-Importance Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.637 | 0.168 | 0.22068 | 0.000 |
| plausibility | 0.721 | 0.140 | 0.20194 | 0.000 |
| informativeness | 0.639 | 0.129 | 0.13269 | 0.000 |
| source_consistency | 0.635 | 0.164 | 0.09011 | 0.000 |
| evidence_words | 251.408 | 13.508 | 0.07734 | 0.000 |
| conciseness | 0.835 | 0.040 | 0.07520 | 0.000 |
| non_hallucination | 0.632 | 0.150 | 0.05248 | 0.000 |
| temperature | 0.083 | 0.028 | 0.03536 | 0.000 |
| rationale_words | 52.230 | -2.442 | 0.02147 | 0.000 |
| model_Qwen3-32B | 0.333 | -0.076 | 0.01532 | 0.000 |
| cat_finance | 0.217 | 0.079 | 0.01414 | 0.000 |
| cat_artificial-intelligence | 0.092 | -0.044 | 0.01063 | 0.000 |
| cat_economy-business | 0.199 | -0.032 | 0.00728 | 0.000 |
| model_Qwen2.5-7b-instruct | 0.333 | -0.037 | 0.00465 | 0.000 |
| cat_health-pandemics | 0.071 | -0.026 | 0.00429 | 0.000 |

Top feature: `completeness` (RF importance = `0.22068`).

## kimi-k2.5__judge_only  (judge attributes only)

- Rows: `42653` across `1580` questions
- SHAP explanation rows: `0`
- Positive rate: `0.754`
- CV ROC-AUC (grouped): `0.694`
- CV Accuracy (grouped): `0.622`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | RF importance | Value-Importance Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.637 | 0.168 | 0.38270 | 0.000 |
| plausibility | 0.721 | 0.140 | 0.25903 | 0.000 |
| informativeness | 0.639 | 0.129 | 0.15629 | 0.000 |
| conciseness | 0.835 | 0.040 | 0.07956 | 0.000 |
| source_consistency | 0.635 | 0.164 | 0.07347 | 0.000 |
| non_hallucination | 0.632 | 0.150 | 0.04896 | 0.000 |

Top feature: `completeness` (RF importance = `0.38270`).

