# Durable private worker contract

T16 runs bounded, one-shot jobs. Scheduling and request delivery may repeat, but a durable Datastore claim decides whether work can run. No worker owns an in-process scheduler or resets account, mandate, capital, or research-budget state at startup.

## Roles and priority

The maintenance role processes recovery, reconciliation, exits, then research dispatch. It owns the trading and budget stores and must finish startup recovery and account reconciliation before reporting execution readiness. It continues recovery, reconciliation, and exits when research is unavailable or over budget.

The research role receives only a `ResearchJobGateway`. It cannot receive a trading store, venue submission callback, credentials, capital limits, or authorization fields. A research assignment consists of these stable identifiers:

- `research_assignment_id`
- `budget_reservation_id`
- `market_snapshot_id`
- `evidence_set_id`
- `model_config_id`
- `budget_key_id`

It returns only a typed durable research-result ID and usage-record ID, or a stable degradation reason. Maintenance validates and stores that response through the narrow gateway.

## Durable job record

`TwinWorkerJob` records are versioned and integrity-bound to the immutable job ID, account scope, kind, payload, deadline, and schema version. Payload values are stable IDs; task bodies contain only `job_id`. Results must be finite JSON objects no larger than 64 KiB.

Claims use a transaction, a bounded lease, and an incrementing fence. A duplicate delivery returns the stored result. An active lease returns `in_progress`. An expired lease can be claimed with a higher fence, and the former worker cannot complete it. A deadline-expired job cannot be claimed. Queue enumeration may be eventually consistent because the key lookup and claim transaction are authoritative.

Hard order ambiguity raises `WorkerPaused` and persists a `paused` result. It is not retried as a fresh side-effecting loop. Model, provider, data, or research-budget failures raise `WorkerDegraded`; the bounded result is persisted while maintenance remains available. Safe reads retry at most five times with capped exponential delay.

## Dispatch and restart behavior

Cloud Tasks names are the SHA-256 hash of the stable job ID. Creating the same task twice is treated as successful duplicate dispatch. Maintenance and research use separate queues, URLs, OIDC audiences, and service identities. Task creation has a bounded timeout.

On startup, maintenance runs recovery and reconciliation first. `execution_ready` remains false if that step is incomplete, although the process may still accept recovery work. Shutdown stops new claims and lets the current one-shot request finish within its platform deadline. `stale(now)` exposes deadline-expired jobs and running jobs with expired leases for operations and recovery scans.

The worker never interprets queue delivery as trading authority. Submission still requires the mandate, account-generation, risk, reservation, command-fence, and venue checks implemented by T13 through T15.
