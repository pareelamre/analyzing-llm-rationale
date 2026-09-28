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

## Question context and private reasoning

The forecast prompt includes the full Metaculus question description (including
links in that text), resolution criteria, fine print, options and numeric
scaling, open/close/resolve times, current UTC time, available platform counts,
and a community aggregate only when a parsed reveal time confirms it is visible. It also
includes up to eight recent root-level staff clarifications from the comments
API and up to twelve ranked news summaries from Foresea's research pipeline.
Question, staff, and news text are reference data, not instructions. The outbound
rationale gate rejects URLs, markup, and common instruction-leakage patterns;
this is not a guarantee of factual grounding, so review private comments during
the tournament. A linked
article's full body is not guaranteed to be fetched merely because its URL
appears in the question description.

On `--submit`, the selected forecaster generates a concise rationale from the
*final validated forecast* and the same context before any API write. The bot
uses one additional model request for this note, capped at the lesser of 45 seconds
and `--max-model-time-s`, and 700 output tokens. `--max-model-calls` governs the
forecast and parser requests separately. It then posts the forecast, verifies its
readback, posts the rationale as a private
Metaculus comment with `included_forecast=true`, and verifies the private
comment's author, post, text, and flags. The comment text is not written to the
audit JSONL; its hash and the operation outcomes are. If a comment write or
readback is uncertain, the cycle halts and quarantines the question from an
automatic reforecast. Metaculus may later publish private tournament comments
under its tournament rules. The live comments readback represents
`included_forecast` as a forecast snapshot object even though the OpenAPI
schema calls it a boolean; verification accepts both nonempty snapshots and
`true` while still requiring the private flag and exact note text.

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
