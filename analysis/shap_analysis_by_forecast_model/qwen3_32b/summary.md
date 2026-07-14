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

- Rows: `14219` across `1580` questions
- SHAP explanation rows: `12000`
- Positive rate: `0.712`
- CV ROC-AUC (grouped): `0.719`
- CV Accuracy (grouped): `0.640`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.729 | 0.140 | 0.05497 | 0.690 |
| plausibility | 0.777 | 0.122 | 0.04926 | 0.754 |
| informativeness | 0.731 | 0.120 | 0.03214 | 0.672 |
| conciseness | 0.909 | 0.028 | 0.02596 | 0.673 |
| evidence_words | 251.406 | 15.668 | 0.01476 | 0.351 |
| source_consistency | 0.689 | 0.134 | 0.01002 | 0.181 |
| non_hallucination | 0.653 | 0.128 | 0.00792 | -0.452 |
| cat_finance | 0.217 | 0.093 | 0.00763 | 0.946 |
| cat_artificial-intelligence | 0.092 | -0.065 | 0.00677 | -0.963 |
| rationale_words | 58.283 | -2.919 | 0.00548 | -0.814 |
| cat_politics | 0.139 | 0.021 | 0.00279 | 0.893 |
| cat_economy-business | 0.199 | -0.009 | 0.00274 | -0.152 |
| cat_space | 0.039 | -0.019 | 0.00154 | -0.821 |
| cat_health-pandemics | 0.071 | -0.031 | 0.00147 | -0.861 |
| cat_sports-entertainment | 0.046 | 0.010 | 0.00138 | 0.874 |

Top feature: `completeness` (Mean |SHAP| = `0.05497`).

## combined_mean__judge_only  (judge attributes only)

- Rows: `14219` across `1580` questions
- SHAP explanation rows: `12000`
- Positive rate: `0.712`
- CV ROC-AUC (grouped): `0.714`
- CV Accuracy (grouped): `0.646`
- Max train/test question overlap across folds: `0`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.729 | 0.140 | 0.07832 | 0.657 |
| plausibility | 0.777 | 0.122 | 0.06518 | 0.770 |
| informativeness | 0.731 | 0.120 | 0.03724 | 0.622 |
| conciseness | 0.909 | 0.028 | 0.02711 | 0.557 |
| non_hallucination | 0.653 | 0.128 | 0.02333 | -0.846 |
| source_consistency | 0.689 | 0.134 | 0.01353 | -0.633 |

Top feature: `completeness` (Mean |SHAP| = `0.07832`).

