# Autonomous twin causal replay

T12 evaluates `foresea_edge_v1` without network access or the current clock. A
dataset records when each forecast occurred, when Foresea observed it, when the
market resolved, and when Foresea observed the outcome. The replay excludes
facts unavailable at the frozen evaluation time and never places the same event
cluster in calibration and test cohorts.

Reproduce the checked-in baseline from the repository root:

```powershell
$env:PYTHONPATH = "src;."
py scripts/twin_replay.py `
  --dataset tests/fixtures/twin/replay_dataset_v1.json `
  --config configs/twin.yaml `
  --output docs/autonomous-twin/REPLAY_BASELINE.json
```

The output binds the dataset, replay configuration, and relevant source files
with SHA-256 hashes. It reports Foresea and market Brier scores, a fixed-policy
baseline, P&L after modeled fees and slippage, drawdown, turnover, abstention,
and stresses for higher costs, missing quotes, and correlated losses.

The fixture deliberately has too few independent events, so its status is
`insufficient_evidence`. It proves replay mechanics and reproducibility; it is
not performance evidence. A fresh report must preserve a frozen split and add
new event-disjoint observations. T12 artifacts always set `live_eligible` to
false. Live eligibility requires the later forward-shadow and owner-approval
gates.
