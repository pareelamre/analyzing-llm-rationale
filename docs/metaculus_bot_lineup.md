# FutureEval bot lineup

Foresea's `forecast-metaculus` command now selects Qwen by default. Each named
profile uses one Metaculus bot account, one expected username, and one token
environment variable. Create the accounts under your existing human profile in
[My Forecasting Bots](https://www.metaculus.com/notebooks/38928/bot-tournament-resources-page/). Mark
Qwen as the prize-eligible primary and the other three as secondaries in
Metaculus settings; the local profile names do not set that status on the site.

This is a provisional selection from the [2026-09-26 Foresea market track
record](../static/track_record_live.json): Qwen's market-relative skill was
`+0.0034` over 105 resolved markets, Gemma's `+0.0017` over 152, GLM Flash's
`-0.0003` over 102, DeepSeek Flash's `-0.0037` over 109, and MiniMax's
`-0.0058` over 177. These are different market samples, not a matched
Metaculus tournament evaluation. Keep MiniMax available as a custom comparison
until a common held-out evaluation can establish a reliable ranking. The small
market-relative differences do not establish that Qwen is better than Gemma;
the prize-eligible primary assignment is a provisional operator choice.

| `--bot-profile` | Forecaster | Retryable-error forecaster backup | Token variable | Username variable |
| --- | --- | --- | --- | --- |
| `qwen-primary` (default) | `qwen3-8-27b` | `gemma-4-26b-a4b-it` | `METACULUS_QWEN_TOKEN` | `METACULUS_QWEN_USERNAME` |
| `gemma-secondary` | `gemma-4-26b-a4b-it` | none | `METACULUS_GEMMA_TOKEN` | `METACULUS_GEMMA_USERNAME` |
| `glm-secondary` | `glm-5-3-flash` | none | `METACULUS_GLM_TOKEN` | `METACULUS_GLM_USERNAME` |
| `deepseek-secondary` | `deepseek-v4-flash` | none | `METACULUS_DEEPSEEK_TOKEN` | `METACULUS_DEEPSEEK_USERNAME` |

Keep token values in your local environment or secret store, never in this
repository. The command checks `/users/me/` against the selected profile's
expected username before calling a model or submitting a forecast. Each profile
gets a separate JSONL audit log in `results/`. All four forecasters and the
primary backup use SCADS, so a SCADS-wide outage affects the lineup.
The command rejects shared usernames or tokens across configured profiles.

Preview one question after setting the selected profile's two variables:

```powershell
analyze-llm-rationale forecast-metaculus --bot-profile qwen-primary --tournament bot-testing-area --max-questions 1
analyze-llm-rationale forecast-metaculus --bot-profile gemma-secondary --tournament bot-testing-area --max-questions 1
analyze-llm-rationale forecast-metaculus --bot-profile glm-secondary --tournament bot-testing-area --max-questions 1
analyze-llm-rationale forecast-metaculus --bot-profile deepseek-secondary --tournament bot-testing-area --max-questions 1
```

Preview still makes model calls and consumes provider quota. Submission requires
both `--submit` and `--confirm-submit 'SUBMIT METACULUS FORECASTS'`.
For an experimental run outside the named lineup, use `--bot-profile custom`,
`--model`, and `--expected-bot-username` with the generic `METACULUS_TOKEN`.
MiniMax remains available this way for comparison.
Pass `--fallback-forecaster-model qwen3-8-27b` explicitly if a custom MiniMax
comparison should retain its old Qwen backup.

## Calibration gate

No post-hoc probability adjustment is enabled for these four profiles. A
successful unscored practice submission verifies the API path, not predictive
calibration. The current local Metaculus history has answers but no matched
forecasts from these four exact model/pipeline versions; its collected news can
postdate question resolution. Fitting a correction on it would leak future
information or transfer a different model's bias.

Before changing live probabilities, collect frozen forecasts with only
information available at forecast time, then score each model and question type
on resolved, chronologically held-out questions. Compare the unadjusted
baseline with candidate calibration using the tournament's log-based score,
plus Brier score and reliability diagnostics. Keep a correction only if the
held-out gain is stable across questions and the raw and adjusted forecasts are
both retained for audit. Do not fit on open practice predictions or on current
tournament outcomes and then report that fit as held-out performance.

The current forecast path is `forecast_question` followed by
`validate_forecast_payload` in `src/analyzing_llm_rationale/metaculus_bot.py`.
The CLI writes per-profile submission evidence to
`results/metaculus_<profile>_audit.jsonl`; these records are not a resolved,
leakage-safe calibration dataset.
