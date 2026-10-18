# Chronological V0 calibration transfer: one bounded robustness result

Completed 17 October 2026, offline, with a protocol and full cohort frozen
before any new scores. A single predetermined cutoff at rank 1,264 is
**2026-01-05T23:45:00Z**. Keeping its two timestamp-tied IDs together leaves
**1,265 training IDs / 315 heldout IDs**, covering 1,048 / 206 exact raw titles.
Zero training IDs require the prescribed heldout-title purge. Every original
1,580 ID remains recorded with its partition and original/UTC timestamps.
All six model/conditional-reading conditions use this same split.

Training resolves 2025-02-01 through the cutoff; heldout resolves
2026-01-06T01:15:00Z through 2026-02-16T23:44:00Z. Training has 478 YES and
787 NO outcomes. No alternative cutoff, model, method or reading was selected.
The existing input hashes match: three archived calibration artifacts, the
benchmark and all 27 original T=0 forecast files. No missing/invalid exclusions,
new model, network, retrieval, HPC or install calls occurred.

## Result

Under the primary conditional selected-answer interpretation, V0-trained
increasing isotonic calibration reduces later structured-prompt mean Brier for
all three models. Some remaining error is prompt-specific: separately trained
maps have lower point estimates, with transfer penalties of 0.002179 / 0.005796 /
0.004379 for GPT-OSS / Qwen2.5 / Qwen3. This supports a transferable reliability
correction within this retrospective chronological split, not improved reasoning.

Each table entry averages the eight **individual V1--V8 losses within question**,
then averages heldout questions. It does not score an average probability or
ensemble. V0 training uses earlier outcomes only; all separate prompt maps use
the identical earlier training IDs. Lower Brier and negative transfer-minus-raw
are better. All six conditions are disclosed below.

| Model | Conditional reading | Raw V1--V8 mean Brier | V0-trained transfer | Separate prompt fits | Transfer minus raw [exploratory 95% interval] | Transfer minus separate |
|---|---|---:|---:|---:|---|---:|
| GPT-OSS-120B | selected answer | 0.164503 | 0.138799 | 0.136619 | -0.025705 [-0.033247, -0.018107] | +0.002179 |
| Qwen2.5-7b-instruct | selected answer | 0.214658 | 0.166583 | 0.160788 | -0.048074 [-0.068695, -0.030598] | +0.005796 |
| Qwen3-32B | selected answer | 0.245669 | 0.152365 | 0.147986 | -0.093305 [-0.117461, -0.069489] | +0.004379 |
| GPT-OSS-120B | direct PYES sensitivity | 0.398075 | 0.195417 | 0.196878 | -0.202657 [-0.243333, -0.164353] | -0.001461 |
| Qwen2.5-7b-instruct | direct PYES sensitivity | 0.531991 | 0.197794 | 0.198270 | -0.334197 [-0.391795, -0.280490] | -0.000477 |
| Qwen3-32B | direct PYES sensitivity | 0.397638 | 0.178209 | 0.176839 | -0.219429 [-0.264106, -0.175680] | +0.001369 |

V0 itself changes from raw to calibrated Brier as follows: selected-answer
GPT-OSS 0.158428 -> 0.134427; Qwen2.5 0.183780 -> 0.153425; Qwen3 0.232894 ->
0.151704. Direct-PYES V0 values are 0.406428 -> 0.197347, 0.539272 -> 0.198793,
and 0.429338 -> 0.183288. Calibration therefore also improves the unstructured
baseline; these results do not establish an advantage of structured rationales.

Equal-title sensitivity first averages the same losses within each heldout title,
then gives each of 206 titles equal weight. It preserves the direction of the
transfer-minus-raw effect in all six conditions:

| Model | Conditional reading | Title-weighted transfer minus raw [exploratory 95% interval] | Title-weighted transfer minus separate |
|---|---|---|---:|
| GPT-OSS-120B | selected answer | -0.020300 [-0.028506, -0.012238] | +0.000981 |
| Qwen2.5-7b-instruct | selected answer | -0.041520 [-0.062377, -0.024522] | +0.004105 |
| Qwen3-32B | selected answer | -0.080965 [-0.104163, -0.059100] | +0.003908 |
| GPT-OSS-120B | direct PYES sensitivity | -0.176265 [-0.220125, -0.136300] | -0.001296 |
| Qwen2.5-7b-instruct | direct PYES sensitivity | -0.298182 [-0.357092, -0.243673] | -0.000065 |
| Qwen3-32B | direct PYES sensitivity | -0.199288 [-0.246770, -0.154835] | +0.001741 |

The complete 54 per-prompt scores are in `outputs/per_prompt.csv`; all heldout
probabilities are in `outputs/predictions.csv.gz`. `outputs/report.json` preserves
both weighting schemes, raw/calibrated V0, primary and diagnostic intervals,
all 54 trained threshold arrays, versions, exact train/test/purge IDs and hashes.

## Interpretation limits

The 2,000 seed-42 heldout exact-title-cluster intervals condition on this training
sample and fixed fitted calibrators. No refitting occurs in bootstrap draws.
They omit calibration-training uncertainty and broader-event dependence, are
pointwise across six exploratory conditions, and receive no multiplicity
correction. There are no confirmatory significance, statistical equivalence or
future-gain claims; diagnostic interval overlap with zero is not equivalence.

This is retrospective distribution-shift evidence only. Exact-title separation
is a lower bound on related-event dependence, not validated event-family
independence. Time sorting does not establish that article versions were available
at a forecast origin, outcomes were unknown, or events were absent from model
training. Historical confidence semantics remain ambiguous. Direct-PYES results
are a disclosed sensitivity, not a score-based choice of historical meaning.
No prompt selection, probability ensemble, prospective claim or manuscript
integration belongs to this slice. No further split was tried.

## Verification and preserved failure

Six analysis contracts and four independent-verifier tests pass, including UTC
ordering/ties, exact-title purge, infeasible split rejection, complete-ID failure,
heldout-label perturbation, average-of-loss aggregation, weighted PAVA, endpoint
clipping and near-equal input scores. Scoped Ruff and in-memory compilation pass.

Independent verification reconstructs all 85,320 archived raw probabilities,
54 training calibrators, 68,310 training curve values, 17,010 heldout rows,
34,020 calibrated probabilities, 54 per-prompt scores and 24 intervals. Maximum
training-curve, heldout-probability and raw-export differences are each
5.551115123125783e-17. The strict comparison tolerance remains 1e-12; archived
raw-export agreement uses the predeclared 1e-14 tolerance.

The first independent verifier failed because it grouped only exactly equal raw
scores. Installed sklearn 1.7.1 `_isotonic.pyx` lines 92/96 coalesces scores less
than NumPy 2.4.1 float64 decimal resolution (1e-15) apart, retaining the first
anchor. Across 54 training fits, 303 training-score entries differ from their
near-equal group anchor and require this coalescing; this counts entries across
fits, not globally unique probabilities. The initial maximum curve discrepancy
was 0.10690703280578562.
A focused regression failed first, then passed after correcting only the verifier
to the source-defined rule. Comparison tolerances were not relaxed. The initial
verifier, failure, local native-rule probe, source snapshot/hashes and numerical
diagnostic are preserved. Protocol, partition, main source, fitted outputs and
their manifests remained byte-identical; the study was not rerun.

No external panel or commit was performed. The private folder is not a Git
repository, so the attempted AQG Git-diff construction checker cannot pass;
this limitation and the construction evidence are recorded separately.
