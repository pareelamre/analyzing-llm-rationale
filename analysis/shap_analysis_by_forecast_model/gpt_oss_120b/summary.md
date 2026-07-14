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

- Rows: `14213` across `1580` questions
- SHAP explanation rows: `12000`
- Positive rate: `0.817`
- CV ROC-AUC (grouped): `0.654`
- CV Accuracy (grouped): `0.640`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.811 | 0.077 | 0.04406 | 0.754 |
| plausibility | 0.851 | 0.062 | 0.03025 | 0.740 |
| informativeness | 0.805 | 0.062 | 0.02726 | 0.559 |
| evidence_words | 251.393 | 26.947 | 0.02641 | 0.475 |
| conciseness | 0.915 | 0.017 | 0.01534 | 0.648 |
| non_hallucination | 0.738 | 0.049 | 0.01241 | -0.529 |
| rationale_words | 54.020 | -1.868 | 0.01165 | -0.889 |
| source_consistency | 0.786 | 0.068 | 0.00687 | 0.573 |
| cat_economy-business | 0.199 | -0.056 | 0.00424 | -0.763 |
| cat_finance | 0.217 | 0.046 | 0.00394 | 0.779 |
| cat_technology | 0.048 | 0.016 | 0.00366 | 0.869 |
| cat_geopolitics | 0.169 | 0.024 | 0.00356 | 0.637 |
| cat_politics | 0.139 | 0.023 | 0.00318 | 0.815 |
| cat_sports-entertainment | 0.046 | 0.019 | 0.00288 | 0.954 |
| cat_environment-climate | 0.028 | -0.028 | 0.00198 | -0.927 |

Top feature: `completeness` (Mean |SHAP| = `0.04406`).

## combined_mean__judge_only  (judge attributes only)

- Rows: `14213` across `1580` questions
- SHAP explanation rows: `12000`
- Positive rate: `0.817`
- CV ROC-AUC (grouped): `0.640`
- CV Accuracy (grouped): `0.607`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.811 | 0.077 | 0.06353 | 0.786 |
| plausibility | 0.851 | 0.062 | 0.03883 | 0.738 |
| informativeness | 0.805 | 0.062 | 0.03181 | 0.361 |
| non_hallucination | 0.738 | 0.049 | 0.03123 | -0.854 |
| conciseness | 0.915 | 0.017 | 0.01751 | 0.408 |
| source_consistency | 0.786 | 0.068 | 0.01079 | 0.064 |

Top feature: `completeness` (Mean |SHAP| = `0.06353`).

