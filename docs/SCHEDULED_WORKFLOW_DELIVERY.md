# What the scheduled workflows actually run at

GitHub does not deliver every `schedule` slot, and how much it drops depends
on how often you ask. Measured over the most recent 30 runs of each workflow
on 2026-09-07:

| workflow | declared cron | interval | `schedule` runs | delivered |
| --- | --- | ---: | ---: | ---: |
| `track-record-tick.yml` | `*/5 * * * *` | 5 min | 2 / 30 | ~7% |
| `agent-trading-board-publish.yml` | `27,57 * * * *` | 30 min | 9 / 30 | ~30% |
| `track-record-resolved.yml` | `7 * * * *` | 60 min | 29 / 30 | ~97% |

The relationship is monotonic: **an hourly cron is reliable here, a
half-hourly one is not, and a 5-minute one barely fires at all.** This matches
GitHub's documented behaviour -- scheduled workflows are best-effort and get
deprioritised under load -- but the size of the effect is worth knowing
concretely before relying on a frequent schedule.

## What is actually keeping things running

The balance of every run above is `workflow_dispatch`, on a clean cadence
(the MTM tick fires every 15 minutes almost without exception). Something
outside GitHub -- Cloud Scheduler -- is dispatching these, and it is doing
essentially all of the work for the sub-hourly workflows.

**This is a single point of failure that looks like redundancy.** The cron
line in `track-record-tick.yml` reads as a fallback for the dispatcher. It is
not: at 7% delivery it would turn a 15-minute cadence into a multi-hour one.
If the dispatcher stops, assume the sub-hourly workloads stop with it.

## Consequences already visible

- `static/mark_to_market_live.json` republishes roughly 96x/day, not the
  288x/day its cron implies. The repository grew more slowly than the
  schedule suggests -- still enough for ~5,400 revisions of a 2.4MB file.
- The agent-trading board goes stale for part of most hours against its
  3600s threshold, because its effective publish gap is ~1 hour and the
  even-hour dispatch runs a full agent tick (20-40 min) before publishing.
  See the comment on `_AGENT_TRADING_BOARD_STALE_AFTER_S` in `server.py`.

## If you want a sub-hourly workload to be reliable

Dispatch it externally and treat the cron as documentation of intent, or
raise the interval to hourly and accept the coarser cadence. Declaring
`*/5` and assuming it runs is the option that does not work.

## Re-measuring

```bash
gh run list --repo pareelamre/analyzing-llm-rationale \
  --workflow track-record-tick.yml --limit 30 \
  --json event -q '[.[].event] | group_by(.) | map("\(.[0])=\(length)") | join(" ")'
```
