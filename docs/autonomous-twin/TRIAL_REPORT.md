# Autonomous twin forward trial

Generated at `2026-09-12T19:25:01.605674+00:00` from immutable evidence
`bd6fc0a307caf18b6ccece865c7234e489481de8a70e3214191d809bd7d299de` for image `b1b818bebfbf5b8ae69b0d165979e99c6675c1d35ae94398b77e2ea9fcc4d6ec`.

| Gate | Status | Reason |
| --- | --- | --- |
| G1 mechanics | **blocked** | `forward_collection_not_operational` |
| G2 strategy | **collecting** | `strategy_evidence_missing` |

G1 has `0` of 7 required consecutive UTC days,
`0` complete market snapshots,
`0` decisions, and
`0` simulated commands. Unexplained ledger
divergences: `0`. Duplicate simulated
commands: `0`. Attempts to add exposure from
stale data: `0`.

| Required fault drill | Status | Evidence |
| --- | --- | --- |
| `provider_outage` | pending | pending |
| `duplicate_task` | pending | pending |
| `cancel_fill_race` | pending | pending |
| `kill_restart` | pending | pending |

## Blockers

- The five-minute scheduler dispatches existing jobs but no component creates forward strategy-cycle jobs.
- The deployed maintenance operation still degrades account reconciliation and exit jobs because its account adapter is not wired.

This artifact never authorizes live trading. `live_eligible` is fixed to
`false`; G3 still requires the owner's explicit, expiring capital mandate.
