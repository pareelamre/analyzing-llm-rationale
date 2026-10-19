# Collected forecast signal screen — 2026-10-19

**Result: no cohort qualifies for trading from this export.** This is an offline,
exploratory screen of the public, bundled paper entries. It changes no runtime
filter, sizing, mandate or live setting. Positive average returns alone do not
establish a repeatable signal. The BYOM twin must earn its own prospective record.

## Source and exclusions

Source: `static/track_record_live.json`, generated
`2026-10-18T14:42:08.686865+00:00`, SHA-256
`3ca985f934e0262b3111414223180b565fb3f846618b9894400a084e2b9c5ff5`.
Only `models_comparison[*].paper_pnl.bets` is used; the duplicated top-level
primary portfolio is not added again. No private user data or cloud store was read.

| Item | Count |
| --- | ---: |
| Exported paper entries across nine models | 2,740 |
| Excluded unknown/mismatched model identity | 12 |
| Excluded records from eight markets with conflicting outcome or normalized question identity | 47 |
| Retained model/market entries | 2,681 |
| Distinct venue/market identities | 669 |
| Nonempty model/domain and model/overall groups examined | 86 |
| Groups with at least 30 markets in both earlier and later periods | 8 |
| Groups passing the complete historical screen | 0 |

Identity conflicts are quarantined for review; a changed title does not by
itself prove a resolution was wrong. Repeated model/venue/market entries are
collapsed to the earliest valid entry after quarantine. There were no additional
duplicate exclusions in this cleaned export. Probabilities, binary outcomes,
identity, timestamp order and nonnegative fees are validated. Recomputed flat
paper P&L must match the exported value within $0.002 per $1 requested entry.
Domain labels are recomputed using Foresea's existing `tag_question` helper.

## Noise controls and decision rule

The primary published summary contains 7,374 daily snapshots across 479 markets.
The publisher version that produced this source artifact treated per-snapshot
Brier differences as independent. Repeated forecasts on the same market can
therefore make those intervals too narrow. This screen uses one entry per model/market and reuses
`market_clustered_brier_skill_interval`; it does not itself repair the published
artifact. The follow-up publisher correction is described below.

The two periods are separated at `2026-09-01T00:00:00+00:00`. Earlier entries
must both be forecast and resolve before the split; later entries must be
forecast after it. Entries that cross the boundary remain in the full-history
description but do not enter either period's qualification check. This split
was selected during retrospective analysis, so it is **not** a frozen forward
holdout or an unbiased strategy backtest.

Each group needs at least 30 markets in each period, a positive lower Brier
skill bound in each, and a positive lower stressed paper-return bound in each.
The normal-interval multiplier is widened for 86 cohort comparisons using
`NormalDist().inv_cdf(1 - 0.05 / (2 * 86)) = 3.4401483795642767`.
These are approximate screening intervals, not exact finite-sample guarantees;
paper returns can be heavy-tailed. Correlation across related markets remains.

The cost scenario worsens the selected outcome's entry price by two probability
points and doubles the published fee. If that price reaches the $1 payout,
the scenario skips the entry with zero return. Requested entries use $1 flat
notional. This is a hypothetical stress scenario, not measured execution cost.

## Examples that must remain research leads

Later-period rows below use all domains for the named model. Brier skill is
market Brier minus model Brier; its units are percentage points of Brier score.
Returns are percentage return per $1 requested entry, not portfolio returns.

| Model | Markets | Mean Brier skill | Adjusted lower skill bound | Mean paper return | Mean stressed return | Adjusted lower stressed return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| glm-5-3-flash | 203 | +0.370pp | -0.630pp | +10.453% | +6.169% | -7.152% |
| qwen3-8-27b | 204 | +0.179pp | -0.666pp | +9.989% | +5.651% | -7.864% |
| council | 210 | -0.341pp | -2.065pp | +6.485% | +2.130% | -12.014% |
| gpt-oss-120b | 200 | -1.960pp | -4.589pp | +5.019% | +0.733% | -13.696% |

The newer models also lack earlier-period observations for replication. None
of the 86 groups has a positive adjusted lower Brier bound in the later period
with at least 30 markets. This does not prove that every model has zero edge.
It establishes that this screen supplies no qualifying evidence.

## What can be acted on now

1. Quarantine ambiguous identity/outcome records instead of treating them as
   clean training labels. Retain a reason and source hash so corrections can be
   reconciled; do not delete the original data.
2. Treat repeated daily snapshots as correlated observations when estimating
   confidence. Preserve model identity and lead-time cohorts. The first-entry
   estimand here differs from the published all-snapshot sports/horizon scores;
   this report does not establish whether their matched-horizon effects survive.
3. Keep apparent sports, 7–14 day and 14–30 day effects as hypotheses until
   market/event dependence and costs are checked on compatible data. Large
   model/market disagreement alone supplies no qualification.
4. Before producing a live shortlist, require causal ingestion and close times,
   model/prompt/config versions, event-cluster identity, contemporaneous bid/ask,
   depth, venue fees and fill/settlement records. The paper-entry export lacks
   sufficient fields for those checks. Existing twin capture, replay, risk and
   execution code should supply them; no new paid research source is needed.

The original sports research prior in PR #687 is a research-allocation hint.
Follow-up implementation on `codex/twin-clustered-signal-gate` corrects publisher
confidence to market means with within-family comparison adjustment and30-market
minimum, and rejects legacy confidence in research-priority/horizon consumers.
Existing bundled reports lacking that metadata qualify no research prior until
recomputed from raw snapshots. This report's first-entry screen is not the
publisher's all-snapshot matched-horizon analysis; neither proves event-disjoint
performance or transferable BYOM returns. No signal here authorizes a live order.

## Verification and delivery

The local analysis reused `ResolvedForecast`, the market-clustered interval
helper and the domain tagger. Two runs produced identical aggregate JSON.
Checks confirmed exclusion/retention totals reconcile, group keys are unique,
domain labels are valid, and every group's `live_eligible` flag is false.
Analysis script SHA-256:
`785740872e875a1e750c222bf2280d83adf038e66871563b381da329cfa36946`.
The local script and detailed aggregates remain in this worktree's ignored
`.aqg/screen_collected_forecasts.py` and `.aqg/signal_screen.json`; they are
exploratory analysis, not a shipped or audited runtime filter.

Implementation base: `e2ebd8753`, draft PR #688. That base passed 2,813 tests
(12 skipped), Ruff, native worker/auth/concurrency/restart checks and one deep
deployment audit. This document changes no executable behavior. No deployment,
model call, venue request, collection schedule or R2/GCS operation was performed
for the screen.
