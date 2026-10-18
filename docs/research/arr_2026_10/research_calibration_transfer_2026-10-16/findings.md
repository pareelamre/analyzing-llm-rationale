# Does baseline calibration transfer to structured prompts?

Exploratory existing-data result,16October2026. Protocol fixed increasing isotonic calibration, original five question folds, all three models, both conditional confidence readings and V0--V8. Input is the existing hash-verified OOF table; each condition has1,580questions. No exclusions, retrieval or new LLM calls.30local isotonic fits use V0 training questions only.

| Model | Conditional reading | Raw V1--V8 mean Brier | V0-trained transfer | Separate per-prompt calibration | Transfer minus separate |
|---|---|---:|---:|---:|---:|
| GPT-OSS-120B | selected | 0.167200 | 0.153198 | 0.153109 | +0.000089 |
| GPT-OSS-120B | direct | 0.355732 | 0.220191 | 0.219329 | +0.000862 |
| Qwen2.5-7b-instruct | selected | 0.201117 | 0.173666 | 0.173160 | +0.000506 |
| Qwen2.5-7b-instruct | direct | 0.470194 | 0.225681 | 0.223912 | +0.001769 |
| Qwen3-32B | selected | 0.215266 | 0.165136 | 0.162468 | +0.002668 |
| Qwen3-32B | direct | 0.368235 | 0.214664 | 0.212635 | +0.002029 |

Under selected-answer confidence, V0-trained calibration improves mean structured-prompt Brier for every model and nearly matches prompt-specific fits. The remaining mean penalties relative to separate calibration are0.000089(GPT-OSS),0.000506(Qwen2.5) and0.002668(Qwen3). This is consistent with a largely transferable probability-scale correction within these archived conditions, with a larger remaining prompt-specific component for Qwen3. It does not establish that rationale quality, evidence use or reasoning improved.

The direct-PYES interpretation also shows transfer but remains a conditional reinterpretation of ambiguous archived confidence. The experiment cannot establish the intended historical meaning of confidence. Report all54per-prompt conditions, not just the six family means.

All transferred V0 probabilities reproduce the stored V0 isotonic forecasts within3.61e-16, below the predeclared1e-12 tolerance despite16significant-digit OOF export. Independent verification reconstructs85,320transferred probabilities from saved piecewise-linear calibrator thresholds, reconciles all54raw/per-prompt scores with the original calibration score table, and checks the30disjoint training/test fold traces.

Limits: descriptive point estimates without p-values, uncertainty intervals, statistical equivalence or prospective claims. Random question folds can share related events and time regimes. Calibration training cost and future data-shift robustness were not measured. Old calibration artifacts, the submitted paper, shared checkout and all enrichment outputs remain untouched. New script/tests/docs are isolated and uncommitted; review before committing or integrating paper claims.

New code: scripts/calibration_transfer_analysis.py and tests/test_calibration_transfer_analysis.py on codex/research-prompt-selection. Six focused tests and scoped Ruff pass. A read-only peer check found no concrete defect.

Next question: does the correction still transfer when related events and time regimes are separated? Avoid another random-fold variant or method search without a prior design.
