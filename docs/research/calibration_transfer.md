# Baseline calibration transfer protocol

Fixed before the new transfer run on16October2026: use all three original
models, both conditional archived-confidence readings, all V0--V8 prompts,
and the original five folds from the completed calibration follow-up. Use
increasing isotonic regression with clipped out-of-bounds predictions and
output bounds0..1, matching that follow-up exactly. No method, fold or seed search.

For each model/reading/fold, fit only V0 probabilities and outcomes from the
other four folds. Apply that function to each prompt's probabilities on the
held-out fold. Compare raw, transferred and the already-stored independently
calibrated probabilities on identical question IDs. No refitting on test labels.
The V0 transferred probabilities MUST reproduce the stored V0 isotonic values
within1e-12; otherwise stop, preserve failure, and investigate original inputs
without loosening tolerance. The exported CSV used16significant-digit floats.

Verify the gzip OOF table and original fold CSV against the supplied calibration
manifest before parsing those exact buffers. Require complete54condition groups,
globally consistent question folds/targets, unique IDs and complete identical
cohorts. Reject malformed probabilities or incomplete groups, and refuse existing
report paths. Preserve all54per-condition scores, per-fold fit provenance and
transferred probabilities. Aggregate V1--V8 with equal variant weights only.

Question: does a V0-trained reliability correction transfer, or do prompt-specific
confidence shifts require their own calibration? This is exploratory retrospective
analysis, with no p-values, intervals, prospective claims or automatic promotion.
Random question folds may share event families/time regimes. Both confidence
readings remain conditional. There are zero new LLM/retrieval calls; isotonic
fitting is ordinary local statistical computation. No submitted-paper edits.
