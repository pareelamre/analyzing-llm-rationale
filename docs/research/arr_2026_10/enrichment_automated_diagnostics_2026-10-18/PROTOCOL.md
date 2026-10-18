# Automated temporal-information-placement census

18 October 2026. Freeze this protocol and its hash before implementing the detector or inspecting its corpus-wide results. The user has skipped human review and authorized proceeding with automated checks and clearly labeled AI-assisted findings. This is one bounded existing-data diagnostic slice, not a new forecasting experiment or independent semantic validation.

## Prior knowledge and input

The earlier eight-case AI screen motivated these rules: a Romanian weekday survives in rationale despite an unspecified horizon; a TikTok horizon contains a 90-day reprieve duration while the rationale contains Monday; a hostage-release representation may omit an explicit Monday endpoint/sequence from its claim fields. These examples were known before this protocol. This is hypothesis-driven exploratory work, not a preregistered discovery or a new test on untouched examples.

Use only ../article_enrichment_2026-10-14/v9/retrieved-full-4424300/collected/enriched_questions.json, SHA256 `47fad5b213e4fd87e9ac0625a35c875a578dae30d0958a49dd8a1b22bff1607f`. Hash and parse the same bytes. Require 1,580 unique integer question IDs, 4,327 article-question records, 572 empty questions, the two saved windows and three saved FRS statuses; fail rather than silently exclude unexpected fields/types. Preserve all statuses, empty FRS and original article indices. This input binding does not rerun prior raw/cache/model/date/schema-source verification.

New artifacts are create-only in this folder. Do not alter the source, frozen packet/protocol/manifests, flags, cached outputs, submitted paper, shared repository or old worktree. No network, installation, retrieval, model API, HPC job, HTML/date/body re-extraction, response repair/retry, external panel or commit. The existing enrichment automation remains deleted.

## Fixed cue vocabulary and string matching

Calendar/relative cues are full English weekdays Monday through Sunday and these exact phrases: tomorrow, tonight, next week, next month, next year, this week, this month, this year. Match case-insensitively with word boundaries; the spaces in the six two-word phrases allow one or more whitespace characters. Canonicalize each matched cue by lowercasing/casefolding and collapsing whitespace. Retain exact original matched text and start/end offsets within its source field. Do not add months, numeric years, date parsers, synonyms, abbreviated weekdays or new phrases after seeing outcomes. Different phrases are not assumed equivalent (Monday versus next week can refer to the same time).

Claim-bearing text is each statement's predicted_future_event, rationale, conditions[*], modality.primary_expression and modality.expressions[*]. The dedicated time_horizon is a separate text field. Support quotations are only support_quotes[*].quote; they are not claim-bearing fields. Do not use the title, question, full body, model metadata, gold outcome, scores or original forecasts to trigger a flag. Record all matches and their field paths. This work does not re-establish that a saved support quotation is valid against its source.

## Three fixed lexical indicators, per saved statement

1. **unspecified_horizon_with_claim_cue**: stripped casefolded time_horizon equals exactly `unspecified`, and at least one fixed cue occurs in another claim-bearing field. It identifies information placed outside the dedicated horizon, not global loss or necessarily an inconsistency; a cue can describe background events.
2. **duration_horizon_with_claim_cue**: the entire stripped time_horizon matches a non-negative integer or decimal number, optional whitespace and optional ASCII/en dash, then hour(s), day(s), week(s), month(s) or year(s), case-insensitively; at least one fixed cue occurs elsewhere in the claim. Use fullmatch, not substring matching; do not include `in 90 days` or spelled-out numbers. A bare duration can legitimately express a horizon. This indicator shows a possible duration/event-timing ambiguity, not a proven error.
3. **support_cue_absent_from_claim**: a canonical cue occurring in a saved support quotation is absent from the union of all claim-bearing text AND time_horizon for that statement. Preserve the missing cue list, source quote index and match offsets. This is literal absence, not semantic omission: the source cue may be irrelevant or past, or equivalent timing may appear in different words. Count repeated occurrences separately in the evidence, but only one hit per statement and indicator.

Indicators can overlap. Do not sum their marginal counts as distinct records. Treat empty FRS as no statements, not as hallucinations or a passed semantic check. Keep null/unsupported schema types visible as failures before producing a final report; do not coerce them to strings or invent fields.

## Outputs and denominators

Produce report.json with source/protocol/code hashes, fixed detector specification, every original question's article count, all 4,327 record identities, preserved saved status/window, statement counts and per-statement indicator evidence. Summarize distinct statement, article-record and question hits with explicit total denominators. Also summarize by the six window/status cells. Preserve all zero-hit and empty records. A question with no articles remains in the 1,580-question coverage table. Do not call a source-valid status a semantic label.

This is a census of lexical indicators among retained article-question records. It is not an article-uniqueness estimate, population semantic error rate, sensitivity/specificity result, human agreement measure, significance test, validated forecasting cohort, historical availability judgment or forecasting-performance gain. Compute no confidence interval or semantic quality score. No frozen record is promoted or excluded.

## Bounded diagnostic reading

After the detector/results freeze, select at most the first two distinct article-records per indicator in ascending (integer question ID, original article index), restricted to bodies of at most8,000 characters and combined question/description/criteria at most10,000 characters. If a record meets multiple indicators, keep its memberships and deduplicate the full-text input. Do not replace difficult cases, choose more interesting examples or change the cue vocabulary after reading them. Save complete text/FRS/question/metadata without gold outcomes in diagnostic-case-inputs.json, at most six distinct records. Root may record provisional AI-assisted interpretations, exact source quotes and uncertainty; no human labels or semantic prevalence follow. Previously seen cases must be marked as such, not called fresh validation.

## Verification and completion

Use focused synthetic behavior tests before implementation: exact words and boundaries/whitespace/case; longer texts not substring-duration matches; temporal cue retained in rationale versus support-only; one supported claim not erasing another statement's hit; overlapping indicators and distinct-record summaries; empty FRS/zero-article questions; unexpected-input failure; same-buffer source binding; create-only outputs; no gold/predictions in diagnostic inputs. Use standard library only. Run scoped lint and in-memory compilation, then read-only independent reconstruction of input/record denominators and every hit's field/cue/offset binding without importing detector code. Root reviews output semantics and a bounded set of complete saved cases. Do not repeat old scientific tests or archive sweeps.

Done means the report, exact deterministic counts, focused checks, source-bound reconstruction, bounded AI notes and limitations are saved and the mutable handoff/status updated. Human review remains skipped, no forecasting experiment is launched and no paper result is silently edited.
