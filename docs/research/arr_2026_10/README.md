# ARR research results, 16–18 October 2026

This archive records completed retrospective analyses and a **not-run** next experiment. It is separate from the submitted manuscript. The original confidence semantics remain unresolved: selected-answer and direct-P(YES) readings are reported separately, not chosen by score. No prospective forecasting or semantic/historical validation claim follows.

## Results

- **Prompt selection:** under the selected-answer reading, GPT-OSS consistently selects V5 and Brier changes 0.159419→0.151498. Both Qwens select the neutral V0 baseline. Identical-title grouping preserves these point estimates; this is robustness on the same forecasts, not an independent sample.
- **Chronological calibration transfer (selected-answer reading, question-weighted):** fit earlier 1,265 questions and evaluate later 315. Mean V1–V8 losses change 0.164503→0.138799 (GPT-OSS),0.214658→0.166583 (Qwen2.5),0.245669→0.152365 (Qwen3). These average eight separate prompt losses, not ensemble probabilities. Calibrated V0 has lower point loss than each transferred family mean; this does not establish that every individual prompt loses. Intervals are exploratory and conditional on fixed fits.
- **Enrichment:**1,580 completed questions,1,008 with 4,327 full-text article-question records and 572 empty.Saved collection labels are 4,122 schema/source-valid FRS records,129 are partial and 76 invalid. Raw bodies/cache/logs are retained privately, not in this archive. Coverage is not forecast performance.
- **Timing diagnostics:** fixed literal indicators flag 1,276 of 4,327 article-question records. These are not semantic error or timing-loss rates: reporting weekdays, durations and event deadlines can differ, and a rationale may retain timing absent from its dedicated field.
- **Criterion anchors:** both known development pools (1079 and 4882, 8 articles each) have zero fixed-anchor matches; original ordering and all 16 records remain. No reranking improvement was observed. Lexical absence alone does not prove irrelevance.

Human review was skipped by explicit user instruction. Automated and AI-assisted diagnostics are labeled as such. Historical bodies/question versions, outcome availability at an origin and broader event-family independence remain unverified.

## Files and provenance

Each named analysis directory contains its original protocol/findings and selected completed numerical reports. Statistical reports, derived prediction tables and frozen source/test/verifier snapshots are copied byte-for-byte; publication-manifest.json records their exact SHA256 values. Diagnostic summaries are explicit aggregate exports, not mislabeled original reports. Raw article HTML/body text, FRS cache dumps, private reviewer comments, annotation packets and assistant case-input exports are excluded.

Historical source snapshots use `.py.txt` names and preserve their original bytes/layout. They are source-only records, not importable/default executable validators. They may contain machine-specific paths or expect frozen inputs outside this repository. They are archival provenance, not installed production code or a claim of turnkey reruns. Do not overwrite historical outputs. Original source/input hashes, including unresolved inputs, remain authoritative. The portable offline entry tools are:

- [Prompt selection](../../../scripts/prompt_selection_analysis.py) — hash-verified archived forecast selection.
- [Calibration transfer](../../../scripts/calibration_transfer_analysis.py) — random-question-fold V0 isotonic transfer; the chronological slice is a separate archived source snapshot.
- [Criterion anchors](../../../scripts/criteria_anchor_ranking.py) — pure advisory lexical coverage, no model or retrieval calls.

Their tests and API documents are in the repository's tests/ and docs/research/ directories. Optional analysis dependencies are declared in pyproject.toml; this publication installs nothing and changes no live caller, routing or study data. Numerical results are saved artifacts, not newly rerun model outputs.

Pre-commit review identified limitations in the **historical source-only snapshots**: the original grouped verifier can miss nonfinite comparisons; original calibration loaders hash and parse via separate reads and permit boolean confidence coercion; the original census copies embedded body hashes/status labels and can export private case bodies; and the original calibration script can overwrite its own output directory. These snapshots are not shipped as executable entry points. Do not restore/execute them as current validators. In particular, never execute the original calibration script without a new external `--output-dir`, or the diagnostic exporter inside a public checkout. Private generated corpus/case outputs are ignored in this archive as an additional guard.

All published numerical JSON reports were separately checked for finite numeric values before this commit; the count labels above come from the preserved allocated collector and final saved-artifact verification, not an independent derivation by the lexical census. Source/quotation validity remains distinct from semantic/historical validity. This publication does not retroactively prove that a race could never have affected an original run. Historical hashes/limitations remain unchanged, and a future scientific rerun requires a separately reviewed corrected implementation.

The three portable scripts remain byte-identical to their original tested sources and are explicitly listed in publication-manifest.json. The prompt-selection race test fixture has one publication-only portability fix, tested with absolute and relative temporary paths; its original bytes are retained under original_test_fixtures/.

## Next experiment — NOT RUN

criteria_timing_experiment_2026-10-18/PROTOCOL.md specifies 96 questions × four arms (neutral/checklist×fulltext/FRS),32 earlier training and 64 later heldout questions, at most384 pinned local Qwen generations. **No cohort was selected, no job submitted, no new generation produced, and no results exist.** Capella access was reported lost. Implementation workers stopped before writing experiment code. This is a planned retrospective evidence-use comparison, not a completed forecasting experiment. No hosted substitution or automatic retry was performed.

No manuscript, live production behavior or pre-existing worktree content is changed by this archival publication.
