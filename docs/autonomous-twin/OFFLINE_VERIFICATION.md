# Offline twin verification under the cost constraint

Verified 2026-10-13 from source baseline
`0b10028878b9ae48b5c789a4385c76c74547b752` in the isolated
`codex/twin-offline-cost-guard` worktree.

No cloud resource, storage operation, deployed worker, live order or paid model
was used. These results are local regression evidence, not a new forward trial
or production readback. The previous September worktree and its unfinished
source changes remain preserved.

## Focused regressions

```powershell
$env:PYTHONPATH = 'src'
py -m unittest tests.test_twin_trial tests.test_twin_budget tests.test_twin_recovery tests.test_twin_simulator tests.test_twin_worker tests.test_twin_research_gateway tests.test_twin_store tests.test_twin_cycle_runtime
```

Exit 0: 113 tests passed in 6.323 seconds. No tests skipped.

The existing behavior cases cover once-only request dispatch, uncertain budget
retention, model outage while maintenance continues, duplicate delivery returning
the saved result, stale-worker fencing, ambiguous-submit recovery, cancel/fill
races, captured-depth simulation, research-result persistence and trial thresholds.
Restart/fencing cases use in-memory test stores. This run does not satisfy a
Datastore emulator, real queue/IAM, or production restart gate.

## Reproducible historical report

```powershell
py scripts/twin_trial_report.py --evidence docs/autonomous-twin/trial/2026-09-12-runtime-audit.json --as-of 2026-10-13T00:00:00+00:00 --json-output .aqg/historical-trial-report.json --markdown-output .aqg/historical-trial-report.md
```

Exit 0. Report hash:
`0e224e1b3af03a4f0c49f3fe955877071e1735063cb7e835e5a040da358a88be`.
Evidence hash:
`bd6fc0a307caf18b6ccece865c7234e489481de8a70e3214191d809bd7d299de`.

This command re-evaluates checked-in September evidence; it does not fetch
October account/worker activity. Its release image is the historical
`b1b818bebfbf5b8ae69b0d165979e99c6675c1d35ae94398b77e2ea9fcc4d6ec`,
not the current source baseline. G1 is `blocked` with
`forward_collection_not_operational`; G2 is `collecting` with
`strategy_evidence_missing`. The historical input records zero collection days
and pending drills. The 113 local regression successes do not rewrite those
production evidence fields.

## Shared checks

`py -m ruff check src tests`: exit 0, all checks passed.
`git diff --check`: exit 0.
`PYTHONPATH=src py -m unittest discover -s tests`: exit 0, 2,755 tests
run in 367.873 seconds, with 12 skipped. Live provider and storage credentials
were empty in this process. Skipped integration gates are not counted as
completed; this run does not refresh production G0/G1/G2 evidence.
The AQG construction checker passed for the documentation-only staged diff.
An external audit is not required for this documentation-only increment.

No source changed in this increment, so the source-only Graphify update rule
does not apply. No generated graph or frontend assets are staged.

## Remaining work

Use [NO_NEW_CLOUD_SPEND.md](NO_NEW_CLOUD_SPEND.md) for the remaining GCS audit
consumer and the local-first next slices. T21 needs fresh, compatible forward
evidence; T22 still lacks owner capital authority. External evidence acquisition
must respect the owner's billing constraint before either gate can advance.
