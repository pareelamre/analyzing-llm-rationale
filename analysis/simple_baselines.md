# Simple Forecasting Baselines

Dataset Yes base rate: `0.352`.

## Non-LLM Baselines

| Baseline | Accuracy | Brier score | Confidence ECE | Probability ECE | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 50% probability | n/a | 0.250 | 0.000 | 0.148 | Uninformative probabilistic baseline; hard accuracy is undefined at exactly 50%. |
| Global base rate | 0.648 | 0.228 | 0.000 | 0.000 | Always predicts the benchmark Yes base rate. |
| Category base rate (leave-one-out) | 0.647 | 0.225 | 0.041 | 0.041 | Uses category-specific resolved frequencies excluding the current question. |
| Category base rate (smoothed leave-one-out) | 0.648 | 0.224 | 0.024 | 0.024 | Shrinks leave-one-out category rates toward the global base rate. |

## Neutral LLM Baselines

| Model | Temperature | Accuracy | Brier score | ECE |
| --- | --- | ---: | ---: | ---: |
| GPT-OSS-120B | temperature_075 | 0.826 | 0.156 | 0.108 |
| Qwen2.5-7b-instruct | temperature_000 | 0.754 | 0.187 | 0.098 |
| Qwen3-32B | temperature_0 | 0.723 | 0.200 | 0.104 |

## Best Observed LLM Runs By Brier Score

| Model | Temperature | Variant | Accuracy | Brier score | ECE |
| --- | --- | --- | ---: | ---: | ---: |
| GPT-OSS-120B | temperature_025 | variant15_neutral_no_rationale | 0.824 | 0.140 | 0.042 |
| Qwen2.5-7b-instruct | temperature_000 | variant15_neutral_no_rationale | 0.787 | 0.179 | 0.135 |
| Qwen3-32B | temperature_0 | variant0_neutral_baseline | 0.723 | 0.200 | 0.104 |

Metaculus community prediction is the preferred external crowd baseline,
but the current API token does not expose aggregation histories for these
benchmark questions. Add it when Metaculus grants the required data tier.
