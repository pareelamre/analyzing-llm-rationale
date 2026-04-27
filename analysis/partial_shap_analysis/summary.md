# Partial SHAP Analysis

This analysis uses the currently available LLM-judge outputs and predicts `forecast_correct` from the judged rationale attributes.

## combined_mean

- Rows: `13149`
- Positive rate: `0.723`
- CV ROC-AUC: `0.735`
- CV Accuracy: `0.654`

| Feature | Mean | Correct - Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| plausibility | 0.739 | 0.152 | 0.07216 | 0.904 |
| completeness | 0.675 | 0.147 | 0.07138 | 0.812 |
| non_hallucination | 0.658 | 0.126 | 0.02129 | -0.856 |
| informativeness | 0.675 | 0.124 | 0.01990 | 0.327 |
| source_consistency | 0.672 | 0.147 | 0.01598 | 0.520 |
| conciseness | 0.908 | 0.024 | 0.01205 | 0.751 |

Top SHAP feature: `plausibility` with mean |SHAP| `0.07216`.

## gemma-4-31b-it

- Rows: `17027`
- Positive rate: `0.734`
- CV ROC-AUC: `0.713`
- CV Accuracy: `0.659`

| Feature | Mean | Correct - Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| plausibility | 0.819 | 0.158 | 0.09238 | 0.790 |
| completeness | 0.794 | 0.141 | 0.05623 | 0.692 |
| source_consistency | 0.780 | 0.131 | 0.02311 | 0.769 |
| informativeness | 0.784 | 0.138 | 0.01899 | 0.470 |
| non_hallucination | 0.732 | 0.113 | 0.01236 | -0.342 |
| conciseness | 0.981 | 0.008 | 0.01026 | -0.852 |

Top SHAP feature: `plausibility` with mean |SHAP| `0.09238`.

## kimi-k2.5

- Rows: `13149`
- Positive rate: `0.723`
- CV ROC-AUC: `0.718`
- CV Accuracy: `0.624`

| Feature | Mean | Correct - Incorrect | Mean |SHAP| | Value-SHAP Corr |
| --- | ---: | ---: | ---: | ---: |
| completeness | 0.573 | 0.157 | 0.07958 | 0.876 |
| plausibility | 0.669 | 0.151 | 0.05244 | 0.889 |
| informativeness | 0.583 | 0.118 | 0.02095 | 0.304 |
| non_hallucination | 0.581 | 0.145 | 0.01868 | -0.726 |
| conciseness | 0.837 | 0.039 | 0.01866 | 0.793 |
| source_consistency | 0.571 | 0.169 | 0.01632 | 0.278 |

Top SHAP feature: `completeness` with mean |SHAP| `0.07958`.

