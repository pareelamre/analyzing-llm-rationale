# Held-out prompt selection from archived forecasts

This offline diagnostic asks whether choosing one prompt using other questions
improves Brier score on held-out questions. It compares a fixed V0 baseline,
training-fold prompt selection, and the uniform average of all nine prompts.
Prompt averaging was already reported in the ARR interaction follow-up; it is
an existing comparator, not a new finding here.

The initial protocol is fixed before running the archived data: all three main
models, T=0, V0 through V8, five question folds, seed42, both selected-answer and
direct-PYES readings reported separately. Fold assignment uses question IDs only.
The selected prompt minimizes training-question Brier, with a deterministic
variant-name tie break. Every strategy uses the same complete nine-variant
question intersection within a model. Missing/invalid records reduce coverage;
duplicate, unknown or noninteger IDs reject the input. Missing cells, changed
input hashes and existing output paths reject the run.

```powershell
py -B scripts/prompt_selection_analysis.py `
  --data-root C:\path\to\archived-data `
  --reference-manifest C:\path\to\comment_followup_manifest.json `
  --output C:\path\to\new-report.json
```

The supplied manifest must pin the benchmark and all27T0 forecast files under
`input_sha256`. Input bytes are verified before scoring. The report preserves
input/script/reference hashes, Python version, coverage, exact folds, per-fold
selections and every held-out prediction. It does not alter archived inputs or
historical metric defaults and requires only the Python standard library.

Report point estimates for the full predeclared family. There are no hypothesis
tests, bootstrap intervals or automatic promotion decisions. A lower score here
does not establish prospective forecasting improvement: evidence is retrospective,
questions are randomly partitioned without event-family/time separation, archived
confidence semantics remain conditional, and one generation per cell does not
measure run variability. Evaluating prompt selection also depends on an existing
nine-prompt training corpus. On a future question the baseline or selected prompt
needs one response, while the nine-prompt ensemble needs nine responses; this
script makes zero model calls.

Before paper integration, review the implementation and freeze a separate
time/event-separated evaluation design. No submitted manuscript changes are
part of this code change.
