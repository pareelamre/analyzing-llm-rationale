# Paired Probabilistic Forecast Analysis

Comparisons are paired by question ID against `variant0_neutral_baseline`
within the same forecast model and temperature. Confidence intervals use
a paired bootstrap over question IDs. P-values use a paired sign-flip
permutation test over question-level metric differences, with
Benjamini-Hochberg correction across all variant-vs-baseline comparisons
within each metric.

Metric priority follows probabilistic forecasting practice: Brier score is
the primary metric, log loss is a secondary proper scoring rule, ECE is
reported elsewhere only as a calibration diagnostic, accuracy is auxiliary,
and output coverage is part of system reliability.

- Run-level estimates: `162` model/temperature/variant runs
- Paired comparisons: `144`
- Brier comparisons with BH q < 0.05: `78`
- Robust Brier differences (q < 0.05 and CI excludes 0): `78`
- Robust Brier improvements: `1`
- Robust Brier degradations: `77`
- Log-loss comparisons with BH q < 0.05: `58`
- Robust log-loss differences (q < 0.05 and CI excludes 0): `58`
- Accuracy comparisons with BH q < 0.05: `54`

Small metric gaps should not be interpreted unless the paired interval
excludes zero and the multiple-comparison-adjusted q-value remains small.
A lower ECE should not be described as an improvement when Brier/log loss
or output coverage worsens.

## Robust Brier Differences

| Model | Temp | Variant | Δ Brier | 95% CI | q | dz | Δ log loss | Coverage Δ | Direction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-7b-instruct | temperature_000 | variant5_key_conditions | 0.032 | [0.020, 0.044] | 0.000 | 0.132 | 0.078 | 0.000 | worsens Brier |
| Qwen2.5-7b-instruct | temperature_175 | variant5_key_conditions | 0.031 | [0.018, 0.043] | 0.000 | 0.121 | 0.066 | 0.000 | worsens Brier |
| Qwen2.5-7b-instruct | temperature_075 | variant5_key_conditions | 0.027 | [0.017, 0.038] | 0.000 | 0.126 | 0.062 | 0.000 | worsens Brier |
| Qwen2.5-7b-instruct | temperature_025 | variant5_key_conditions | 0.027 | [0.016, 0.038] | 0.000 | 0.121 | 0.059 | 0.000 | worsens Brier |
| Qwen3-32B | temperature_175 | variant6_step_by_step_reasoning | 0.027 | [0.018, 0.036] | 0.000 | 0.142 | 0.042 | 0.000 | worsens Brier |
| Qwen3-32B | temperature_125 | variant7_uncertainty_language | 0.027 | [0.018, 0.035] | 0.000 | 0.156 | 0.053 | 0.000 | worsens Brier |
| Qwen3-32B | temperature_075 | variant7_uncertainty_language | 0.026 | [0.018, 0.034] | 0.000 | 0.163 | 0.053 | 0.000 | worsens Brier |
| Qwen3-32B | temperature_200 | variant6_step_by_step_reasoning | 0.026 | [0.016, 0.037] | 0.000 | 0.125 | 0.038 | 0.001 | worsens Brier |
| GPT-OSS-120B | temperature_075 | variant3_reasoning_type | 0.025 | [0.017, 0.034] | 0.000 | 0.140 | 0.056 | 0.000 | worsens Brier |
| Qwen2.5-7b-instruct | temperature_125 | variant5_key_conditions | 0.024 | [0.013, 0.036] | 0.000 | 0.104 | 0.053 | 0.000 | worsens Brier |
| Qwen3-32B | temperature_0 | variant6_step_by_step_reasoning | 0.024 | [0.018, 0.030] | 0.000 | 0.203 | 0.052 | 0.000 | worsens Brier |
| Qwen3-32B | temperature_175 | variant7_uncertainty_language | 0.024 | [0.015, 0.033] | 0.000 | 0.130 | 0.029 | -0.006 | worsens Brier |
| Qwen3-32B | temperature_0 | variant7_uncertainty_language | 0.024 | [0.016, 0.031] | 0.000 | 0.164 | 0.045 | 0.000 | worsens Brier |
| GPT-OSS-120B | temperature_00 | variant3_reasoning_type | 0.024 | [0.015, 0.032] | 0.000 | 0.137 | 0.053 | 0.000 | worsens Brier |
| GPT-OSS-120B | temperature_025 | variant3_reasoning_type | 0.023 | [0.014, 0.032] | 0.000 | 0.127 | 0.048 | 0.000 | worsens Brier |
| Qwen3-32B | temperature_175 | variant1_predicted_event | 0.023 | [0.014, 0.032] | 0.000 | 0.125 | 0.056 | 0.000 | worsens Brier |
| Qwen3-32B | temperature_200 | variant5_key_conditions | 0.022 | [0.012, 0.033] | 0.000 | 0.105 | 0.011 | 0.002 | worsens Brier |
| Qwen3-32B | temperature_025 | variant7_uncertainty_language | 0.022 | [0.014, 0.029] | 0.000 | 0.147 | 0.043 | 0.000 | worsens Brier |
| Qwen3-32B | temperature_200 | variant7_uncertainty_language | 0.021 | [0.011, 0.032] | 0.001 | 0.104 | 0.003 | -0.012 | worsens Brier |
| Qwen3-32B | temperature_175 | variant8_temporal_anchors | 0.020 | [0.010, 0.029] | 0.000 | 0.100 | 0.067 | 0.000 | worsens Brier |

## Largest Accuracy Differences (Auxiliary)

| Model | Temp | Variant | Δ accuracy | 95% CI | q | dz | Δ Brier | Brier q | Coverage Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7b-instruct | temperature_175 | variant5_key_conditions | -0.054 | [-0.074, -0.035] | 0.002 | -0.136 | 0.031 | 0.000 | 0.000 |
| Qwen2.5-7b-instruct | temperature_000 | variant5_key_conditions | -0.049 | [-0.068, -0.032] | 0.002 | -0.132 | 0.032 | 0.000 | 0.000 |
| Qwen2.5-7b-instruct | temperature_075 | variant5_key_conditions | -0.049 | [-0.066, -0.033] | 0.002 | -0.145 | 0.027 | 0.000 | 0.000 |
| Qwen2.5-7b-instruct | temperature_025 | variant5_key_conditions | -0.047 | [-0.065, -0.030] | 0.002 | -0.135 | 0.027 | 0.000 | 0.000 |
| Qwen2.5-7b-instruct | temperature_125 | variant5_key_conditions | -0.046 | [-0.065, -0.028] | 0.002 | -0.124 | 0.024 | 0.000 | 0.000 |
| Qwen3-32B | temperature_200 | variant5_key_conditions | -0.044 | [-0.065, -0.022] | 0.003 | -0.100 | 0.022 | 0.000 | 0.002 |
| Qwen2.5-7b-instruct | temperature_200 | variant5_key_conditions | -0.043 | [-0.063, -0.023] | 0.002 | -0.105 | 0.018 | 0.015 | 0.000 |
| Qwen3-32B | temperature_200 | variant6_step_by_step_reasoning | -0.039 | [-0.060, -0.016] | 0.006 | -0.089 | 0.026 | 0.000 | 0.001 |
| Qwen3-32B | temperature_200 | variant2_key_attribute | -0.039 | [-0.060, -0.018] | 0.003 | -0.091 | 0.018 | 0.001 | 0.002 |
| GPT-OSS-120B | temperature_200 | variant1_predicted_event | -0.035 | [-0.053, -0.017] | 0.003 | -0.094 | 0.015 | 0.028 | 0.000 |
| Qwen3-32B | temperature_175 | variant6_step_by_step_reasoning | -0.034 | [-0.054, -0.013] | 0.006 | -0.082 | 0.027 | 0.000 | 0.000 |
| Qwen2.5-7b-instruct | temperature_200 | variant2_key_attribute | -0.033 | [-0.052, -0.014] | 0.006 | -0.086 | 0.016 | 0.018 | -0.001 |
| Qwen3-32B | temperature_200 | variant3_reasoning_type | -0.032 | [-0.053, -0.012] | 0.017 | -0.078 | 0.009 | 0.113 | 0.002 |
| Qwen2.5-7b-instruct | temperature_200 | variant7_uncertainty_language | -0.031 | [-0.051, -0.012] | 0.015 | -0.077 | 0.010 | 0.177 | -0.001 |
| Qwen2.5-7b-instruct | temperature_175 | variant2_key_attribute | -0.030 | [-0.049, -0.012] | 0.009 | -0.081 | 0.016 | 0.019 | -0.001 |
