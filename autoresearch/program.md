# Foresea Autoresearch Program

This is the Foresea version of Karpathy-style autoresearch: edit one research
surface, run one fixed-budget experiment, keep only measured improvements.

## Research Surface

Only edit:

- `autoresearch/candidate_prompt.txt`

Do not edit production prompts, server code, track-record files, or benchmark
data during an autoresearch loop. Promotion is handled by the CLI only when the
candidate beats the selected baseline.

## Experiment Command

Use the SCADS-hosted model configs in `configs/models.yaml`. The default
`gpt-oss-120b` runs through `https://llm.scads.ai/v1/chat/completions` and reads
`SCADS_AI_API_KEY` or `SCADS_AI_API_KEY.txt`. Other SCADS-hosted choices include
`llama-3.3-70b-instruct`, `llama-4-scout-17b-16e-instruct`, `deepseek-v3`,
`gemma-4-31b-it`, and `kimi-k2.5`.

Run one candidate experiment:

```bash
PYTHONPATH=src python -m analyzing_llm_rationale autoresearch \
  --model gpt-oss-120b \
  --candidate-prompt-path autoresearch/candidate_prompt.txt \
  --max-records 50 \
  --metric brier_score
```

Compare against an existing baseline and promote only if Brier improves:

```bash
PYTHONPATH=src python -m analyzing_llm_rationale autoresearch \
  --model gpt-oss-120b \
  --candidate-prompt-path autoresearch/candidate_prompt.txt \
  --baseline-results-path results/GPT-OSS-120B/temperature_00/results_variant0_neutral_baseline.json \
  --promote-to prompts/variant0_neutral_baseline.txt \
  --max-records 50 \
  --metric brier_score \
  --min-delta 0.001
```

Every run writes:

- `analysis/autoresearch/runs/<run_id>/candidate_prompt.txt`
- `analysis/autoresearch/runs/<run_id>/results_autoresearch_candidate.json`
- `analysis/autoresearch/runs/<run_id>/score.json`
- `analysis/autoresearch/experiments.jsonl`

## Objective

Optimize live-usable forecasting quality, not verbosity. Primary metric is
Brier score on resolved binary benchmark questions. Lower is better.

Secondary checks:

- Accuracy should not collapse.
- ECE should not materially worsen.
- Null predictions should stay near zero.
- Rationale should be short enough for the web app and track record.

## Candidate Prompt Rules

The candidate prompt must:

- contain `[question]`
- request strict JSON with `predicted_answer`, `confidence`, and `rationale`
- keep confidence calibrated as probability assigned to the predicted answer
- treat `Current Time` as the temporal anchor
- avoid using evidence published after the question's resolve time
- avoid long chain-of-thought; concise rationale only

## Loop

1. Read the latest `score.json` and `experiments.jsonl`.
2. Form one hypothesis about why the candidate failed or could improve.
3. Edit only `autoresearch/candidate_prompt.txt`.
4. Run the experiment command.
5. Keep iterating only if the score is meaningful and the output remains valid.

Do not tune to one example. Prefer changes that improve calibration across the
whole fixed subset.
