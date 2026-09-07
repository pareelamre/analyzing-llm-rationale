# What the scheduled workflows actually run at

GitHub does not deliver every `schedule` slot, and in this repository it
delivers a small minority of them. Measured 2026-09-07 by counting runs whose
`event` is `schedule` against the number the cron declares over the same span:

| workflow | declared cron | interval | schedule runs | expected | delivered |
| --- | --- | ---: | ---: | ---: | ---: |
| `track-record-tick.yml` | `*/5 * * * *` | 5 min | 4 over 8.1h | ~97 | **~4%** |
| `agent-trading-board-publish.yml` | `27,57 * * * *` | 30 min | 18 over 40.5h | ~81 | **~22%** |
| `track-record-resolved.yml` | `7 * * * *` | 60 min | 59 over 269h | ~269 | **~22%** |

Two things to take from this:

**A 5-minute cron is close to decorative** -- roughly one slot in 25.

**Raising the interval past 30 minutes does not help.** The half-hourly and
hourly workflows both land near 22%. There is no monotonic "longer interval,
better delivery" effect to tune against; declaring a slower cron buys a
coarser cadence at the same delivery rate.

## Correcting an earlier version of this file

The first version reported ~97% for the hourly workflow and described the
relationship as monotonic. That was a measurement error: it counted the
proportion of *observed runs* whose event was `schedule` (29 of 30) rather
than the proportion of *declared slots* that fired. Almost all of that
workflow's runs are indeed scheduled ones -- there is little else triggering
it -- but they still arrive every two to four hours, not hourly.

## What is actually keeping things running

The balance is `workflow_dispatch` from Cloud Scheduler, on a clean cadence:
the MTM tick fires every 15 minutes almost without exception. That external
dispatcher is doing nearly all the work for the sub-hourly workflows.

**This is a single point of failure that looks like redundancy.** The `cron:`
line in `track-record-tick.yml` reads as a fallback for the dispatcher. At ~4%
delivery it is not one. If the dispatcher stops, assume the sub-hourly
workloads stop with it.

## Consequences already visible

- `static/mark_to_market_live.json` republishes on the dispatcher's 15-minute
  cadence rather than the cron's 5-minute one -- still enough for ~5,400
  revisions of a 2.4MB file.
- The agent-trading board is stale for part of most hours against its 3600s
  threshold; see the comment on `_AGENT_TRADING_BOARD_STALE_AFTER_S` in
  `server.py`.
- `static/track_record_live.json` declares `stale_after_seconds: 1800` and is
  routinely hours old -- it was 20,059s old when this was written, because
  its workflow last ran 5.6 hours earlier.

## Re-measuring

Count schedule-triggered runs against the span they cover, not against the
other events in the list:

```bash
gh run list --repo pareelamre/analyzing-llm-rationale \
  --workflow track-record-resolved.yml --limit 60 \
  --json createdAt,event -q '.[] | select(.event=="schedule") | .createdAt'
```

Divide the count by (span in minutes / cron interval in minutes).
