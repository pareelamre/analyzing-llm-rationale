# SHAP Analysis (revised — grouped CV + controls)

Cross-validation is grouped by question ID so no question appears in both
train and test. Two models are fit per judge dataset:
- **judge_only**: 6 judge attributes only (grouped CV)
- **full**: judge attributes + model, variant, temperature, rationale length,
  evidence length, output validity, category flags (grouped CV)

ROC-AUC difference between judge_only and full reveals confound magnitude.

## combined_mean__full  (full model with controls)

- Rows: `42652` across `1580` questions
- Positive rate: `0.755`
- CV ROC-AUC (grouped): `0.715`
- CV Accuracy (grouped): `0.643`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.742 | 0.138 | 0.05891 | 0.787 |
| plausibility | 0.792 | 0.128 | 0.04834 | 0.834 |
| informativeness | 0.740 | 0.119 | 0.02573 | 0.714 |
| evidence_words | 251.405 | 13.522 | 0.01759 | 0.433 |
| conciseness | 0.911 | 0.024 | 0.01681 | 0.726 |
| temperature | 0.083 | 0.028 | 0.01393 | 0.905 |
| source_consistency | 0.719 | 0.134 | 0.01281 | 0.668 |
| rationale_words | 52.229 | -2.439 | 0.01032 | -0.950 |
| model_Qwen3-32B | 0.333 | -0.076 | 0.00992 | -0.949 |
| non_hallucination | 0.686 | 0.117 | 0.00900 | -0.599 |
| model_Qwen2.5-7b-instruct | 0.333 | -0.037 | 0.00149 | 0.377 |
| variant_variant4_credibility | 0.111 | 0.006 | 0.00105 | 0.872 |
| variant_variant5_key_conditions | 0.111 | -0.008 | 0.00026 | -0.436 |
| variant_variant8_temporal_anchors | 0.111 | 0.004 | 0.00021 | 0.608 |
| variant_variant7_uncertainty_language | 0.111 | -0.002 | 0.00019 | 0.667 |

Top SHAP feature: `completeness` (mean |SHAP| = `0.05891`).

## combined_mean__judge_only  (judge attributes only)

- Rows: `42652` across `1580` questions
- Positive rate: `0.755`
- CV ROC-AUC (grouped): `0.702`
- CV Accuracy (grouped): `0.632`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.742 | 0.138 | 0.07379 | 0.779 |
| plausibility | 0.792 | 0.128 | 0.05965 | 0.864 |
| informativeness | 0.740 | 0.119 | 0.02690 | 0.663 |
| non_hallucination | 0.686 | 0.117 | 0.02363 | -0.891 |
| conciseness | 0.911 | 0.024 | 0.01944 | 0.628 |
| source_consistency | 0.719 | 0.134 | 0.01082 | 0.395 |

Top SHAP feature: `completeness` (mean |SHAP| = `0.07379`).

## gemma-4-31b-it__full  (full model with controls)

- Rows: `42659` across `1580` questions
- Positive rate: `0.755`
- CV ROC-AUC (grouped): `0.692`
- CV Accuracy (grouped): `0.679`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| plausibility | 0.862 | 0.116 | 0.05974 | 0.753 |
| completeness | 0.848 | 0.109 | 0.03482 | 0.703 |
| evidence_words | 251.413 | 13.514 | 0.02209 | 0.209 |
| informativeness | 0.841 | 0.109 | 0.02067 | 0.458 |
| source_consistency | 0.803 | 0.105 | 0.01812 | 0.834 |
| rationale_words | 52.230 | -2.440 | 0.01650 | -0.928 |
| temperature | 0.083 | 0.028 | 0.01159 | 0.914 |
| model_Qwen3-32B | 0.333 | -0.076 | 0.01151 | -0.971 |
| non_hallucination | 0.739 | 0.084 | 0.00411 | -0.357 |
| model_Qwen2.5-7b-instruct | 0.333 | -0.037 | 0.00384 | 0.759 |
| conciseness | 0.987 | 0.008 | 0.00114 | -0.525 |
| variant_variant4_credibility | 0.111 | 0.006 | 0.00057 | 0.917 |
| variant_variant5_key_conditions | 0.111 | -0.008 | 0.00031 | -0.456 |
| variant_variant3_reasoning_type | 0.111 | -0.004 | 0.00029 | -0.685 |
| variant_variant8_temporal_anchors | 0.111 | 0.004 | 0.00028 | 0.284 |

Top SHAP feature: `plausibility` (mean |SHAP| = `0.05974`).

## gemma-4-31b-it__judge_only  (judge attributes only)

- Rows: `42659` across `1580` questions
- Positive rate: `0.755`
- CV ROC-AUC (grouped): `0.644`
- CV Accuracy (grouped): `0.670`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| plausibility | 0.862 | 0.116 | 0.07262 | 0.779 |
| completeness | 0.848 | 0.109 | 0.04174 | 0.745 |
| informativeness | 0.841 | 0.109 | 0.02722 | 0.037 |
| source_consistency | 0.803 | 0.105 | 0.01815 | 0.790 |
| non_hallucination | 0.739 | 0.084 | 0.01193 | -0.520 |
| conciseness | 0.987 | 0.008 | 0.00548 | -0.647 |

Top SHAP feature: `plausibility` (mean |SHAP| = `0.07262`).

## kimi-k2.5__full  (full model with controls)

- Rows: `42653` across `1580` questions
- Positive rate: `0.754`
- CV ROC-AUC (grouped): `0.705`
- CV Accuracy (grouped): `0.636`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.637 | 0.168 | 0.06029 | 0.895 |
| plausibility | 0.721 | 0.140 | 0.04234 | 0.845 |
| conciseness | 0.835 | 0.040 | 0.02088 | 0.736 |
| informativeness | 0.639 | 0.129 | 0.02032 | 0.831 |
| temperature | 0.083 | 0.028 | 0.01747 | 0.924 |
| evidence_words | 251.408 | 13.508 | 0.01671 | 0.406 |
| model_Qwen3-32B | 0.333 | -0.076 | 0.00973 | -0.928 |
| rationale_words | 52.230 | -2.442 | 0.00961 | -0.930 |
| source_consistency | 0.635 | 0.164 | 0.00926 | 0.689 |
| non_hallucination | 0.632 | 0.150 | 0.00811 | -0.144 |
| model_Qwen2.5-7b-instruct | 0.333 | -0.037 | 0.00174 | -0.384 |
| variant_variant4_credibility | 0.111 | 0.006 | 0.00123 | 0.827 |
| variant_variant5_key_conditions | 0.111 | -0.008 | 0.00029 | -0.719 |
| variant_variant1_predicted_event | 0.111 | -0.004 | 0.00019 | -0.323 |
| variant_variant8_temporal_anchors | 0.111 | 0.004 | 0.00017 | 0.430 |

Top SHAP feature: `completeness` (mean |SHAP| = `0.06029`).

## kimi-k2.5__judge_only  (judge attributes only)

- Rows: `42653` across `1580` questions
- Positive rate: `0.754`
- CV ROC-AUC (grouped): `0.694`
- CV Accuracy (grouped): `0.623`

| Feature | Mean | Correct − Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.637 | 0.168 | 0.08059 | 0.895 |
| plausibility | 0.721 | 0.140 | 0.05083 | 0.869 |
| informativeness | 0.639 | 0.129 | 0.02314 | 0.783 |
| conciseness | 0.835 | 0.040 | 0.02012 | 0.595 |
| non_hallucination | 0.632 | 0.150 | 0.01533 | -0.724 |
| source_consistency | 0.635 | 0.164 | 0.00975 | -0.314 |

Top SHAP feature: `completeness` (mean |SHAP| = `0.08059`).

