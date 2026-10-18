# Offline criterion-anchor ranking prototype

18 October 2026. Freeze this protocol and contracts.json before implementation or evaluation of the complete saved candidate pools. The user has instructed continued code work, skipped human review and prioritized conservative use of credits. This is a local research prototype and development-case check, not a retrieval-quality benchmark, live change or forecasting experiment.

## Motivation and prior knowledge

Previously read examples reveal two target ambiguities: question1079 means popularity by Google Trends rather than a legal appeal; question4882 restricts eligible Reddit alternatives to an explicit list that excludes Digg. Both are development examples and were seen before this design. No held-out, prospective or independent-human validation claim is permitted.

The existing NewsPipeline accepts one query string and uses the first512characters of joined candidate title/summary/text for ranking. One CLI caller already concatenates description/criteria/fine-print; do not claim criteria are universally omitted. Existing topical lexical/BM25 helpers do not preserve criterion roles and discard some operators. This prototype leaves all current pipeline signatures/callers, routing, source policies and frozen outputs unchanged.

## Isolated implementation

Use managed C:/Users/paree/.codex/worktrees/criteria-aware-retrieval/Foresea, branch codex/research-criteria-aware-retrieval, pinned base a8c421669a0e61fa3057476ed05df11546ebc605. It started clean after the pre-existing research worktree's dirty state was preserved. Add only scripts/criteria_anchor_ranking.py, tests/test_criteria_anchor_ranking.py, docs/research/criteria_anchor_ranking.md and a brief own AGENTS coordination entry. Keep source APIs and live inference/retrieval untouched. No dependency, installation, model/embedding/provider, network/HPC, commit/push/PR or deployment action. Use AQG construction before edits and focused behavior tests, with root review before closeout.

## API and fixed behavior

Implement a pure standard-library function over caller-supplied criteria text, explicit named anchor groups, and saved candidate mappings. The prototype does not infer a semantic contract automatically. Each group holds one or more alternative anchor phrases; alternatives are OR within a group. Every anchor must be present as a contiguous casefolded alphanumeric-token sequence in the supplied criteria text. Return the source criterion span for that binding. Reject an empty/malformed group, non-text anchor, duplicate normalized group name or duplicate normalized alternative within a group. Do not silently repair contracts or invent aliases. Empty/unavailable contracts preserve input order with unassessed coverage, rather than a semantic-failure score.

Tokenization uses Unicode alphanumeric words excluding underscores (`[^\W_]+`), casefolded for comparison, retaining original string offsets. Punctuation/whitespace between tokens is not meaningful; phrases must be contiguous within one field and cannot bridge candidate fields. Use full saved title, summary and text independently; absent or None fields are empty, other non-text values fail. There is no512-character truncation, stemming, synonym expansion or learned score. Raw markup is ordinary input text; this prototype neither renders HTML nor interprets markup semantics. Caller-supplied anchor roles are not automatically verified as the correct interpretation of criteria.

A candidate matches a group if any alternative occurs in an allowed field. Record the first match of each alternative in each field, including group, phrase, field and original character span/quote. Coverage is the fraction of distinct groups matched, with equal group weight; matching multiple alternatives cannot inflate a group's weight. This is lexical coverage only, not relevance probability, criterion satisfaction, historical availability or outcome evidence.

Return a new advisory assessment/ranking over original candidate indices. Stable-sort by descending group coverage; preserve input order for ties. If no candidate has any anchor match, preserve the original order and flag the pool as lacking anchor evidence. Preserve every candidate and do not apply thresholds, exclusions, retries or automatic requerying. Do not mutate inputs or existing relevance scores. All output remains explicitly advisory and all semantic/history flags remain unverified. A negated, misleading or irrelevant mention can still earn full lexical coverage; retain that negative control in tests and documentation.

## Fixed development contracts

contracts.json names two known question IDs and groups. Question1079 has measurement_source=[Google Trends] and measurement_quantity=[total interest]. Question4882 has eligible_alternative=[minds,voat,steemit,papaly,saidit,notabug,snapzu,ruqqus] and comparison_metric=[Alexa Traffic Rank]. All phrases derive from the stored criteria. Do not add aliases/weights/thresholds or change groups after candidate-pool results. Source binding confirms phrase presence, not semantic correctness.

## Saved-data check

Hash and parse the same bytes of ../article_enrichment_2026-10-14/v9/retrieved-full-4424300/collected/enriched_questions.json, expectedSHA25647fad5b213e4fd87e9ac0625a35c875a578dae30d0958a49dd8a1b22bff1607f. Select exactly those two development questions by ID, use every saved article in original order, and preserve identity/bodyhash/window/status. No new article or synthetic control may be inserted into those pools. Baseline is original saved order, not a reproduction of historical NewsPipeline ranking. Record original/new order, all group matches/misses, exact offsets and no-anchor-pool flags. Report descriptive changes only; no relevance labels, accuracy/recall score, statistical interval or forecast loss is available. A reranker cannot recover a useful article absent from the candidate pool.

Synthetic fixtures belong only in tests: legal versus metric senses of an ambiguous title; OR membership of a listed candidate versus an unlisted rival; qualifying text after character512; punctuation/case; no cross-field phrase synthesis; stable ties/duplicates; no-contract/all-zero fallback; invalid contracts/types; no mutation; and a high-coverage negated statement demonstrating lack of semantic validation. Freeze code and evaluation receipts, then independently reconstruct all candidate/group matches and summaries without importing the prototype, using the saved source and fixed contracts.

## Done and claim boundary

Done means the small API/tests/docs, fixed two-question development results, independent reconstruction, scoped lint/compile/diff checks, AST Graphify update and evidence-backed handoff exist locally. No commit/public code landing is requested; applicable audit-before-commit gates remain separate. Human review stays skipped. Historical body/question versions and unknown outcomes remain unverified. Successful tests establish API behavior, not a measured improvement in retrieval quality or the submitted paper's forecasts. Stop after this prototype; no automatic live integration or new study.
