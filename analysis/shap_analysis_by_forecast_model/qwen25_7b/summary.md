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

- Rows: `14220` across `1580` questions
- SHAP explanation rows: `12000`
- Positive rate: `0.734`
- CV ROC-AUC (grouped): `0.710`
- CV Accuracy (grouped): `0.636`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.687 | 0.158 | 0.04994 | 0.844 |
| plausibility | 0.747 | 0.161 | 0.04695 | 0.905 |
| informativeness | 0.685 | 0.136 | 0.02850 | 0.769 |
| source_consistency | 0.683 | 0.158 | 0.01943 | 0.732 |
| conciseness | 0.910 | 0.024 | 0.01589 | 0.815 |
| evidence_words | 251.416 | 1.421 | 0.01503 | 0.331 |
| rationale_words | 44.386 | -2.541 | 0.01294 | -0.864 |
| cat_finance | 0.217 | 0.092 | 0.01001 | 0.958 |
| non_hallucination | 0.667 | 0.134 | 0.00771 | 0.019 |
| cat_artificial-intelligence | 0.092 | -0.039 | 0.00455 | -0.887 |
| cat_politics | 0.139 | 0.018 | 0.00257 | 0.851 |
| cat_economy-business | 0.199 | -0.039 | 0.00240 | -0.777 |
| cat_space | 0.039 | -0.016 | 0.00222 | -0.865 |
| cat_geopolitics | 0.169 | -0.027 | 0.00143 | -0.512 |
| cat_health-pandemics | 0.071 | -0.027 | 0.00122 | -0.883 |

Top feature: `completeness` (Mean |SHAP| = `0.04994`).

## combined_mean__judge_only  (judge attributes only)

- Rows: `14220` across `1580` questions
- SHAP explanation rows: `12000`
- Positive rate: `0.734`
- CV ROC-AUC (grouped): `0.700`
- CV Accuracy (grouped): `0.632`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.687 | 0.158 | 0.07652 | 0.814 |
| plausibility | 0.747 | 0.161 | 0.06836 | 0.905 |
| informativeness | 0.685 | 0.136 | 0.02065 | 0.485 |
| non_hallucination | 0.667 | 0.134 | 0.02052 | -0.845 |
| source_consistency | 0.683 | 0.158 | 0.01792 | 0.568 |
| conciseness | 0.910 | 0.024 | 0.01222 | 0.704 |

Top feature: `completeness` (Mean |SHAP| = `0.07652`).

