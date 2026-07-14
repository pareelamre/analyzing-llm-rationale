# Human annotation rubric — LLM-judge validation

Annotate each rationale on five criteria. Use the same three-point scale for every criterion:

- **Yes = 1.0** — clearly satisfies the criterion
- **Partial = 0.5** — partially / arguably
- **No = 0.0** — does not satisfy it

You are scoring the **rationale**, given the question, its resolution criteria, the evidence the model saw, and the model's answer + probability. You are *not* scoring whether the final answer was right — the true outcome is deliberately hidden so your judgement of the reasoning stays independent.

## Criteria

- **Criterion alignment** (`criterion_alignment`): Does the rationale address the question's actual resolution criteria (the specific event/threshold/date), rather than a vaguely related topic?
- **Evidence consistency** (`evidence_consistency`): Are the rationale's factual claims consistent with the provided evidence (no contradiction of what the sources say)?
- **Free of hallucination** (`hallucination`): Does the rationale avoid asserting facts, numbers, or events that are NOT in the provided evidence? Yes = clean, No = it invents unsupported facts.
- **Usefulness** (`usefulness`): Would this rationale genuinely help a human forecaster understand and sanity-check the forecast?
- **Supports the probability** (`probability_support`): Does the rationale justify the *specific* probability shown (e.g. 85% vs 60%), not merely the direction (Yes/No)?

## Workflow

1. Open `human_annotation.html` in a browser.
2. Score all items (progress autosaves to the browser).
3. Click **Download CSV** and save as `human_annotations.csv` in this folder.
4. Run `python scripts/score_human_annotation.py`.

Prefer a spreadsheet? Fill in `human_annotation_blank.csv` instead (same columns, values 0 / 0.5 / 1).
