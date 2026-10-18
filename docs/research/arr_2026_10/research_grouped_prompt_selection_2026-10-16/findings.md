# Exact-title-grouped prompt selection: bounded robustness result

Completed offline on 16 October 2026 under the protocol frozen before grouped
scores were inspected. All 28 original input hashes match: one benchmark and
27 T=0 forecast files, 42,660 forecasts, all 1,580 IDs in every condition.
No exclusions, retrieval, model/HPC/network calls, or modifications to the frozen
study, shared checkout, original tools, existing analyses or submitted paper.

The 1,580 IDs contain 1,254 exact raw titles. The 322 repeated-title groups cover
648 rows. Each title is assigned wholly to one of five outcome-free, deterministic
seed-42 folds. All five folds contain 316 IDs; title counts are 250/251/251/251/251.
Training selection uses the other four folds' 1,264 question outcomes only.

| Model | Conditional reading | Selected prompt in folds 0/1/2/3/4 | V0 Brier | Selected Brier | Paired delta | Exploratory title-cluster 95% interval |
|---|---|---|---:|---:|---:|---|
| GPT-OSS-120B | selected answer | V5 / V5 / V5 / V5 / V5 | 0.159419 | 0.151498 | -0.007921 | [-0.014814, -0.001070] |
| GPT-OSS-120B | direct PYES | V7 / V7 / V7 / V7 / V7 | 0.365824 | 0.337201 | -0.028623 | [-0.037924, -0.019973] |
| Qwen2.5-7b-instruct | selected answer | V0 / V0 / V0 / V0 / V0 | 0.187383 | 0.187383 | 0.000000 | [0, 0] |
| Qwen2.5-7b-instruct | direct PYES | V3 / V3 / V3 / V3 / V3 | 0.476731 | 0.457675 | -0.019056 | [-0.022496, -0.015792] |
| Qwen3-32B | selected answer | V0 / V0 / V0 / V0 / V0 | 0.199729 | 0.199729 | 0.000000 | [0, 0] |
| Qwen3-32B | direct PYES | V7 / V7 / V7 / V7 / V7 | 0.401109 | 0.332935 | -0.068174 | [-0.075262, -0.060748] |

V0 is neutral baseline; V3 reasoning type; V5 key conditions; V7 uncertainty
language. Lower Brier is better. Primary scores weight each original question ID
equally. Every fold selection and full training-score family are saved.

Under selected-answer confidence, the GPT-OSS V5 gain survives this exact-title
separation check. Both Qwens again choose V0 and obtain no selection gain.
All six question-weighted point estimates exactly equal the prior random-fold
estimates: the selected prompt remains the same across all five new training
subsets and all five earlier subsets. This is a robustness result from one fixed
new grouping, not an additional independent forecast sample.

Equal-title sensitivity averages losses within each title and then across titles,
using the same held-out selected predictions without changing the selection rule:

| Model | Conditional reading | Title-weighted V0 Brier | Title-weighted selected Brier | Paired delta | Exploratory title-cluster 95% interval |
|---|---|---:|---:|---:|---|
| GPT-OSS-120B | selected answer | 0.161793 | 0.154214 | -0.007579 | [-0.014348, -0.000638] |
| GPT-OSS-120B | direct PYES | 0.358539 | 0.330789 | -0.027750 | [-0.037020, -0.019102] |
| Qwen2.5-7b-instruct | selected answer | 0.190739 | 0.190739 | 0.000000 | [0, 0] |
| Qwen2.5-7b-instruct | direct PYES | 0.465159 | 0.446410 | -0.018749 | [-0.022208, -0.015351] |
| Qwen3-32B | selected answer | 0.193908 | 0.193908 | 0.000000 | [0, 0] |
| Qwen3-32B | direct PYES | 0.397735 | 0.330326 | -0.067409 | [-0.075022, -0.059675] |

Intervals use 2,000 seed-42 title-cluster resamples and linear percentile
endpoints. They condition on fixed out-of-fold selected predictions; selection
is not refitted within resamples, so shared-training uncertainty is incompletely
quantified. The six conditions are exploratory; intervals are pointwise, without
multiplicity adjustment or confirmatory p-values. The Qwen [0,0] intervals arise
because selected forecasts equal V0 for every ID; they are not a claim of zero
uncertainty about future prompt selection.

This experiment separates identical titles only. Different titles can still
describe related events, and temporal regimes overlap. Retrospective evidence,
model training knowledge, historical availability and one-generation variability
remain uncontrolled. Both archived-confidence readings remain conditional;
these scores do not identify the intended historical probability referent.
No prospective, causal, semantic-resolution, significance or general
cross-model-benefit claim follows. Averaging was not investigated as a new idea.

Seven synthetic tests pass, including grouped separation, raw-title exactness,
input-order-invariant folds, complete-cohort rejection and held-out-label
perturbation. Scoped Ruff and in-memory compilation pass. An internal read-only
source check found no concrete defect. `verify_outputs.py` independently checks
all 28 input hashes, output/source hashes, exact grouping and deterministic
folds, all training-only prompt minima, 18,960 held-out probabilities, both
weighted scores and all 12 bootstrap intervals. `verification.json` records pass.
`comparison_with_prior.json` records exact identity of the six point estimates.

Artifacts: `protocol.md`, `outputs/report.json`, `outputs/fold_map.csv`,
`outputs/manifest.json`, the exact helper snapshot and reference manifest,
`grouped_selection.py`, synthetic tests and independent verifier. The report
SHA256 is `eb7cd50b8c1ce4d9d3d787d1c34fcbc9be2550e082aa841736a951d511a995d2`.
The protocol SHA256 is
`f17bf5eee3a952785043df316131a284f885621a4a32379de9fc8b8ccd306d95`.

No external audit or commit was requested or performed. The AQG construction
checker was attempted but its Git-diff step cannot run because this private
analysis directory is not a Git repository; this is not recorded as a checker
pass. Implementation/audit-before-commit review remains separate from this
completed local research analysis.
