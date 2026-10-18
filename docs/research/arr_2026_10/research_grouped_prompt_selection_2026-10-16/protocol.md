# Exact-title-grouped prompt-selection robustness protocol

Frozen before inspecting any grouped performance outcomes, 2026-10-16 17:05 UTC.
This is one bounded exploratory reanalysis of existing forecasts. It makes zero
model, retrieval, HPC, or network calls and does not modify the frozen study,
submitted paper, original tools, shared checkout, or existing analyses.

## Information already known

The earlier random-question-fold analysis found selected-answer GPT-OSS V5
selection in 5/5 folds, Brier 0.159419 -> 0.151498 (delta -0.007921).
The two Qwens retained V0; direct-PYES selected different prompts. Uniform prompt
averaging was already analyzed and is not a new research finding here.
The pinned benchmark has 1,580 IDs and 1,254 exact raw question titles: 322
repeated-title groups cover 648 rows, with no conflicting outcome labels.
256 repeated-title groups span the original prompt-selection folds; 260 span
the separate original calibration folds. These counts were known before this
protocol. No grouped scores have been inspected.

## Fixed question and estimands

Primary question: does the earlier selected-answer GPT-OSS V5 selection gain
survive holding repeated exact question titles together?

Report the complete six-condition family: GPT-OSS-120B, Qwen2.5-7b-instruct,
Qwen3-32B, each under selected-answer and direct-PYES interpretations, T=0,
V0 through V8. Preserve all 1,580 common question IDs per condition; every
invalid, unknown, duplicate, or missing forecast MUST stop the analysis rather
than silently drop rows. Use the earlier tool's pinned parsing and scoring.

The primary score is question-weighted held-out Brier: selected minus V0.
Training selection minimizes question-weighted training Brier across V0--V8,
with variant-name tie break, using only the other four folds' outcomes.
Also report baseline/selected Brier with equal weight per unique title as a
sensitivity, averaging question losses within title and then across titles.
The unique-title sensitivity uses the same selected forecasts, without refitting
or optimizing a new selection rule. Retain all fold selections and training
scores. No best-seed, alternative-grouping, variant-family, or method search.

## Fixed grouping and folds

Grouping key is the exact raw `question` string, including capitalization and
whitespace. Do not normalize or broaden groups after viewing results. Sort
titles by Python Unicode order, shuffle with `random.Random(42)`, then stable
sort groups by descending number of IDs. Greedily assign each whole group to
the fold with the fewest question IDs, breaking load ties by lowest fold number.
Use five folds. This outcome-free rule is deterministic and approximately
question-balanced. Save exact titles/IDs/folds and the question-level fold map.
Assert no title spans folds and each fold is nonempty.

## Exploratory interval

Use 2,000 seed-42 title-cluster bootstrap draws. Draw 1,254 titles with
replacement per draw and include every question in each sampled title. For
the primary question-weighted delta, divide the sampled sum of paired losses
by the sampled number of question IDs. For the unique-title sensitivity, average
the sampled within-title mean paired losses. Report 2.5/97.5 percentiles with
linear interpolation. Reset the generator to seed 42 for each condition so the
same title-resampling pattern applies across all six conditions.
Intervals condition on fixed out-of-fold selected predictions: selection is not
refitted during resampling. These exploratory pointwise intervals do not include
full selection-training uncertainty, broader-event dependence, or multiplicity
adjustment; no confirmatory p-values or significance claims.

## Pinned inputs and implementation

- Data root: `C:/Users/paree/.codex/worktrees/research-prompt-selection/Foresea`.
- Original helper: `scripts/prompt_selection_analysis.py`, SHA256
  `471352c73fd85c565201ca866549f529daadd579d29d9df8349a3e8083d9a531`.
  Execute only the exact verified source buffer in a private module namespace;
  do not write import caches in the original worktree.
- Reference manifest: `../comment_followup_manifest.json`, SHA256
  `fb507013fd330a9d6c05a406b5026646ecb053935a30cb40d08dae004e1ce908`.
- Benchmark SHA256:
  `0ab34e631c26d9446dc26931819f4f0ebfd89b663aa7aa1144a0a2cd4aa05cde`.
  Verify the benchmark and all 27 original T=0 result files against the reference.
- Earlier report `../research_prompt_selection_2026-10-16/report-verified.json`
  SHA256 `a59b6b030d2515156c0abc430d6243654be90a03a43b463c751ceab2fd225748`.
- Save protocol, source/test/input/reference/output hashes, runtime, fold maps,
  per-question selected predictions, per-fold selections, and all six scores.
  Refuse existing output destinations; preserve failure rather than overwrite.

## Verification and interpretation

Before the data run: focused synthetic checks cover exact-title group
disjointness, deterministic/input-order-invariant balancing, held-out-label
perturbation, complete-cohort rejection, and hand-calculated paired scores.
Run scoped Ruff and in-memory compile checks. Independently recompute saved
scores, fold disjointness, chosen training minima, and hashes after the run.
No external panel or commit is part of this analysis.

Passing this robustness check does not establish prospective or causal validity,
resolve historical confidence semantics, remove broader related-event leakage,
repair retrospective evidence timing, measure run variability, or establish
significance. Chronological evaluation and broader family grouping remain
separate work; this protocol must not change in response to observed outcomes.
