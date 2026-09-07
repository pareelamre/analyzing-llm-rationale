# Research budget integrity audit

Audited 2026-09-07 against source baseline e3d54fcb. T07 is in progress:
the former complete status overstated integration and recovery coverage.

The pre-call boundary now persists an atomic claim after reserving capacity.
Duplicate deliveries and resumed claimed requests cannot dispatch again. A crash
between claim and dispatch conservatively retains capacity until an authoritative
usage reconciliation; a lease or timeout cannot establish that billing was zero.
Reservation reuse requires the same day/account key and estimates.

Unknown cash and tokens count against subsequent requests. Invalid provider
usage retains uncertainty, including valid above-estimate facts in the other
usage dimension. Late authoritative usage replaces uncertainty exactly once;
conflicting changes after final reconciliation require an audited correction.
Neither malformed policy/prices nor negative/fractional persisted counters can
create spending capacity.

Existing nonempty Datastore budget aggregates lacking `uncertain_tokens` are
refused. Operators must audit their associated reservations and provider usage
before migrating those aggregates. Do not zero or delete old usage to resume
research. A new UTC day naturally uses a separate budget key; it does not clear
the previous day's obligations.

Verification uses fake providers and a real local Datastore emulator, including
two independent processes delivering the same request and competing for the
last token allowance. No paid provider, account credential or venue order is
used. Exact commands and exit codes are recorded in progress.json.

Remaining T07 work includes runtime configuration and approved pricing wiring,
time/rate limits, safe cancellation and stale-call recovery, and evidence-cache
integration. These fixes do not establish the entire card's completion or enable
live trading.
