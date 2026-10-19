# Local raw-forecast evidence check — 2026-10-19

**Result: this local historical source supplies no qualifying research cohort
under the screen below, and no compatible BYOM calibration observations.**
Larger snapshot counts do not substitute for independent markets or provenance.
The machine-readable summary is `RAW_SIGNAL_SCREEN.json`.

This supplements `SIGNAL_SCREEN.md`; it uses a different, older source. The local
database ends on September 27, whereas that paper-entry export was generated on
October 18. These are not interchangeable populations, and this check does not
establish whether the October sports/horizon results survive recomputation.
No published report, runtime filter or live setting was changed.

## Source and integrity

The existing local `data/track_record_store.duckdb` in the owner's main checkout
contains only the five market/forecast tables inspected here. It was opened with
DuckDB `read_only=True`; the analysis adapter rejects store writes and immutable
appends, and a Python audit hook denies socket connect/DNS. This guard is not an
OS sandbox. The original file hash was checked before and after each run.

- Bytes: `408170496`.
- SHA-256: `7e7ffa476f97bd123fcf968d94b34455e8a90227082a8e6846929b7c9b745a86`.
- Snapshot range: `2026-06-03T11:11:14.332922+00:00` through
  `2026-09-27T21:06:37.868385+00:00`.
- Last ledger ingestion: `2026-09-27T21:06:43.764866+00:00`.
- Analysis used code baseline `0ce775d89` and the existing `ForecastLedger`
  reader plus corrected publisher `_skill_ci` helper. Repeating the analysis
  produced byte-identical results with unchanged source hash.
- Full local analysis artifact SHA-256:
  `4723bd6e8dfe40140bd1141a6dc7e73134ac4131c28589968399f2d7ae8e61b0`.
  The preserved local probe is `.aqg/analyze_raw_forecasts.py`, SHA-256
  `9e8b273da1da10270982bb1cdc607a65d94f2faa108dcc3c370d6eac57af6f42`.
  These ignored local files are not distributed by this documentation PR.

## What the data contains

| Check | Count |
|---|---:|
| Stored forecast snapshots, all models/states | 90,546 |
| Resolved snapshot rows with binary outcomes | 49,993 |
| Distinct resolved venue/market identities | 651 |
| Markets with conflicting binary labels | 0 |
| Markets with varying nonempty stored question text | 16 |
| Resolved snapshot rows without a model identity | 37 |
| Missing, nonfinite or out-of-range forecast/market probabilities | 0 |
| Missing/nonfinite Brier values or recomputation mismatches | 0 |
| Resolved snapshots without a denormalized close time | 30,338 |
| Resolved snapshots at/after their stored close time | 1,236 |
| Forecast ledger events, including resolution events | 92,366 |
| Resolved forecasts returned by the immutable ledger reader | 51,132 |
| Those passing the existing ledger timing/audit-grade checks | 42,094 |
| Those carrying model hash, prompt hash, cluster ID or category family | 0 each |
| Those sourced from `twin_research_v1` | 0 |

The snapshot and ledger populations differ: the append-only ledger retains
forecasts beyond the currently stored snapshot rows. Its reader rejects
post-resolution forecasts. Do not subtract their totals to infer a loss rate.
Question-text variation alone does not prove an incorrect outcome. Missing
denormalized close times also do not prove all corresponding evidence is absent:
the normalized market table or frozen ledger event may retain it.

All resolved ledger rows identify their source as `track_record_snapshot`.
The runtime's `_calibration_observations` requires `twin_research_v1`, timing
audit grade, and nonempty instrument/cluster/model/prompt/category identities.
**Zero observations in this local source can satisfy that contract.** Do not
retrofit current hashes onto old forecasts or assume a model name identifies
the historical model weights, prompt and strategy.

## Historical research screen

The selected model set is the nine named models plus `crowd-follow` baseline
from the October 18 public export. Archived routes, aliases and absent model
identities are excluded rather than guessed into current identities.

Two populations were inspected: 31,353 resolved selected-model snapshot rows
after backfill exclusion, and 25,697 selected-model ledger forecasts passing the
existing timing checks. Snapshot metadata is read through the publisher's existing normalized-table
join, using the market table in this same frozen database. Ledger metadata stays
frozen to its event: no current metadata is inserted into historical events.

The existing `market_means_normal_bonferroni_v1` helper averages paired Brier
differences within each venue/market and requires 30 markets. The planned search
family is `2 populations × 10 models × (1 overall + 10 domains + 6 horizons)`
= **340 comparisons**, including empty planned buckets. Of 300 nonempty buckets,
107 have at least 30 markets; **none has a positive adjusted lower skill bound**.
Every bucket produced available numerical inference; this is not a malformed
data fallback masquerading as a negative result.

Selected `council` rows illustrate why snapshot counts alone are misleading.
Values below are market-mean Brier skill and bounds in percentage points of
Brier score, not return percentages.

| Population | Bucket | Snapshots | Markets | Mean skill | Adjusted interval |
|---|---|---:|---:|---:|---:|
| Snapshots | sports | 1,170 | 20 | -0.79pp | [-4.12pp, +2.53pp] |
| Timing-checked ledger | sports | 880 | 9 | -1.32pp | [-8.95pp, +6.30pp] |
| Snapshots | 14–30d | 667 | 22 | +0.77pp | [-3.11pp, +4.65pp] |
| Timing-checked ledger | 14–30d | 453 | 13 | +1.52pp | [-4.98pp, +8.02pp] |
| Snapshots | 7–14d | 940 | 87 | -0.78pp | [-3.32pp, +1.76pp] |
| Timing-checked ledger | 7–14d | 862 | 54 | -0.87pp | [-3.06pp, +1.32pp] |

This exploratory screen is not a frozen out-of-sample or event-disjoint test.
Normal intervals are approximate; different contracts can share an event.
There is no executable spread/depth/fee/return test here. A different model,
period, estimand or preregistered hypothesis could produce different results.
This check supplies no trading shortlist and does not prove universal zero edge.

## Action from this check

Keep the historical data for research and preserve its original provenance.
Use the existing BYOM capture/ledger path to accumulate actual model, prompt,
strategy, cluster and causal-ingestion identities for future observations.
Keep the current calibration and live gates closed. Recompute the newer report
only from its matching raw source; this older database must not overwrite it.
No additional paid data source is required to resolve the provenance problem.
