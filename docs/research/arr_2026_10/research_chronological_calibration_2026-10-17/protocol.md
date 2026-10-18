# Frozen chronological baseline-calibration transfer protocol

Written before any new chronological scores, 17 October 2026 (Europe/Berlin;
initial metadata inspection 2026-10-16 23:20 UTC). One bounded offline split;
no model, network, retrieval, HPC, install, external-panel or commit operation.
No edits to the paper, shared checkout, old tools, old analyses or enrichment.

## Prior knowledge and fixed question

Prior random-question-fold calibration transfer improved structured-prompt mean
Brier in all six model/reading conditions and nearly matched separately trained
prompt calibrators. The completed exact-title-grouped prompt-selection analysis
is a different experiment; it is not repeated here. Those results and historical
confidence ambiguity were known before this protocol. No chronological scores
have been inspected and the split will not be tuned after outcomes.

Question: does an earlier-question V0-trained increasing isotonic reliability
correction transfer to V1--V8 on later exact-title-disjoint questions?
Primary conditional interpretation is selected-answer confidence. Direct-PYES
is reported as a sensitivity, never selected because of a better score.

## One predetermined metadata partition

Use all 1,580 original common IDs. Parse `resolve_time` as timezone-aware UTC,
sort by UTC timestamp then integer ID, and choose the timestamp at rank 1,264
(zero-based index 1,263). Train candidates have timestamp <= cutoff and heldout
IDs timestamp > cutoff, keeping all timestamp ties together. Purge from TRAIN
every exact raw `question` title present in heldout; retain every heldout ID.
No title normalization, broader family grouping or alternative cutoff.

Metadata-only inspection finds cutoff **2026-01-05T23:45:00Z**, with two IDs
at that timestamp: 1,265 train candidates and 315 heldout IDs. There are zero
train-title purges. Final training has 1,265 IDs, 1,048 titles and both labels
(478 YES / 787 NO); heldout has 315 IDs and 206 exact titles.
Training resolves 2025-02-01T01:00:00Z through the cutoff; heldout resolves
2026-01-06T01:15:00Z through 2026-02-16T23:44:00Z.
Fail as infeasible if training <500, heldout <100, or training lacks both labels.
Do not substitute another split. Save every one of the 1,580 rows' ID, exact raw
title, original timestamp, UTC timestamp and partition, plus train/test/purge
lists and every purge reason. Freeze this artifact and its hash before fitting.
All six model/conditional-reading conditions share the identical partition.

## Fixed inputs and source binding

Use the exact three archived artifacts used by the 16 October transfer analysis:

| Artifact under `../calibration_followup` | SHA256 |
|---|---|
| manifest.json | 4b4533bcf16586a6b92731de54f3728ec66ffc0c777df4afa230a49140ec7f90 |
| oof_predictions.csv.gz | 8eb9e2d891f92a6f89ef18e6e2a8e95397822c617eba066c7e92e8f365fd6fad |
| question_folds.csv | faf1a78295cf287214281b09a0e2edbd8c5dcd7e61b3144c00b121a630a7fd80 |

The old fold assignment and calibrated predictions are checked as archive
provenance only; neither governs this chronological fit. Use the saved raw
probabilities, including their original 16-significant-digit export precision.
Verify all 54 model/reading/prompt groups have exactly the full 1,580 IDs,
matching outcomes, finite probabilities and no duplicates/unknown IDs.
No missing or invalid forecast may be silently excluded.

Source snapshots are copied after checking exact hashes and executed from the
same verified buffers in private module namespaces, with no bytecode writes:

- Original `scripts/calibration_transfer_analysis.py` SHA256
  `bcf07ebb0cc7677fac0bbace1bc47387ce15b445c27b00d67e75cc5c57776dc4`:
  reuse its validated archived-artifact reader.
- Original `calibration_followup/calibration_analysis.py` SHA256
  `887d6bd93d823694881fc66ab2f6bdf344eb050669e1f7f7c64cb14e6fb3d5cf`:
  reuse its confidence-reading helper and imported IsotonicRegression class.
- Benchmark SHA256
  `0ab34e631c26d9446dc26931819f4f0ebfd89b663aa7aa1144a0a2cd4aa05cde`.
  Metadata and all 27 original T=0 forecast files come from the unchanged
  `research-prompt-selection/Foresea` worktree and retain original hashes.
  Reconstruct all archived raw probabilities using the source-bound confidence
  helper, requiring agreement within 1e-14 for the 16-digit export. The pinned
  archived raw values remain the fitting/scoring inputs; no reinterpretation.

## Fixed fitting, estimands and output

Use the original increasing isotonic fit: `increasing=True`, `y_min=0`,
`y_max=1`, `out_of_bounds='clip'`, sklearn PAVA with its linear interpolation
between fitted thresholds and endpoint clipping. No method or parameter search.
For each of three models and both readings, fit V0 using only final training
IDs/outcomes and apply that map to every heldout prompt. Separately fit one map
per V1--V8 using the same training IDs only as a diagnostic comparator. Reuse
V0's fit for its separate comparator. Save all 54 trained threshold arrays.

For each heldout ID, first average the eight individual V1--V8 Brier LOSSES.
Then average those per-question losses across heldout IDs. This is a mean of
eight separate prompt losses, **not** the Brier score of averaged probabilities.
Primary difference is transferred minus raw mean loss. Diagnostic difference
is transferred minus separately calibrated mean loss. Also report raw/calibrated
V0 and all 54 per-prompt rows; no prompt selection or probability ensemble.

Sensitivity gives each exact heldout title equal weight, first averaging the
same losses within title. Use 2,000 seed-42 exact-title-cluster draws, drawing
206 titles with replacement and including all IDs per selected title. Primary
resampled delta divides the sum of sampled paired losses by sampled ID count;
equal-title sensitivity averages sampled within-title mean paired losses.
Report linear 2.5/97.5 percentiles for primary and diagnostic differences under
both weightings. Reset seed 42 per condition to share resampling patterns.
Calibrators stay fixed: no refitting, split search, interpretation selection,
multiple cutoff testing or confirmatory p-values.

## Verification, stopping rule and limits

Synthetic tests FIRST cover UTC ordering/naive-time rejection/timestamp ties,
exact-title training purge, feasibility rejection without fallback, invalid or
missing IDs failing closed, heldout-label changes leaving training fits unchanged,
increasing interpolation/endpoints, and mean-loss rather than probability-mean
aggregation. Run only scoped tests, Ruff and in-memory compile checks.
Independently reconstruct partition, archived confidence conversions, PAVA
calibrator application, heldout probabilities, per-prompt/within-question losses,
both weighting schemes and intervals without calling the analysis scoring path.
Pin input, source, protocol, partition and output hashes. Refuse overwrites.

Stop after this one split regardless of results. This is retrospective
distribution-shift evidence only. Exact titles are a lower bound on related-event
dependence, not a validated event-family split. Chronological sorting does not
establish ex-ante article versions, unknown outcomes, model-training safety,
prospective forecasting, causal improvement or historical probability semantics.
Intervals condition on the training sample and fixed fitted maps; they omit
calibration-training uncertainty and broader-event dependence, are pointwise
across six exploratory conditions, and receive no multiplicity correction.
Do not claim significance, statistical equivalence or future gains.
