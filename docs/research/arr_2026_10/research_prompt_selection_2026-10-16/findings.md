# Held-out prompt selection: first research result

New offline analysis completed16October2026 from28hash-verified inputs: one benchmark and27originalT=0result files,42,660stored forecasts. All conditions have1,580valid questions and no exclusions. No retrieval, enrichment, inference or model calls. The code lives on isolated branch `codex/research-prompt-selection` based on `a8c421669a0e61fa3057476ed05df11546ebc605`. Shared Foresea and the submitted paper are untouched.

The pre-run protocol fixes five question folds, seed42, all nine prompts, all three models and both conditional readings. Prompt selection uses training-question Brier only; per-question predictions are saved for independent reanalysis. The earlier uniform-ensemble results are reproduced as an existing comparator, not claimed as new.

| Model | Conditional reading | V0 Brier | Held-out selection Brier | Mean ensemble Brier | Selected in all five folds |
|---|---|---:|---:|---:|---|
| GPT-OSS-120B | selected | 0.159419 | 0.151498 | 0.153319 | variant5_key_conditions (5/5) |
| GPT-OSS-120B | direct | 0.365824 | 0.337201 | 0.345000 | variant7_uncertainty_language (5/5) |
| Qwen2.5-7b-instruct | selected | 0.187383 | 0.187383 | 0.187226 | variant0_neutral_baseline (5/5) |
| Qwen2.5-7b-instruct | direct | 0.476731 | 0.457675 | 0.470172 | variant3_reasoning_type (5/5) |
| Qwen3-32B | selected | 0.199729 | 0.199729 | 0.208103 | variant0_neutral_baseline (5/5) |
| Qwen3-32B | direct | 0.401109 | 0.332935 | 0.368319 | variant7_uncertainty_language (5/5) |

Under selected-answer confidence, GPT-OSS consistently selects V5 key conditions and its held-out Brier decreases by0.007921. Qwen2.5 and Qwen3 retain V0 in all folds. Under the direct-PYES interpretation different prompts are selected, so this analysis does not resolve the original confidence ambiguity. Fold-stable selection is descriptive, not a significance test.

Compared with the existing averaging comparator, selected-answer GPT selection has lower Brier while using one prompt on a future question after selection. Training selection still uses the existing nine-prompt corpus. No token, latency or dollar savings were measured; a nine-prompt ensemble requires nine responses.

Limits: retrospective evidence and archived forecasts; random question folds may share related events and temporal regimes; no intervals or multiplicity-adjusted tests; one generation per cell; model training information and historical availability remain unverified. Do not insert these estimates into the in-review paper without a separately reviewed design and implementation.

Code verification:10focused regressions pass, including held-out label perturbation, deterministic folds/ties, exact common-cohort accounting, duplicate/unknown ID rejection, input mutation between reading and parsing, all27synthetic cells, and no overwrite. Scoped Ruff and git diff checks pass. One in-session peer finding about verifying and parsing the same bytes was fixed with a regression. Broad scripts/tests Ruff reports80findings in unchanged existing files; no bulk cleanup. No commit, push, merge, release or deployment was performed; the audit-before-commit gate remains pending for this new logical change.

Next candidate directions (not yet run): event/time-separated selection robustness; V0-trained calibration transfer to other prompts; prompt-disagreement risk/coverage diagnostics. Prefer these existing-output analyses before funding further enrichment.

Files: `report.json` preserves the initial run; `report-verified.json` binds the final script; `PROTOCOL.md` preserves the plan fixed before running the data.
