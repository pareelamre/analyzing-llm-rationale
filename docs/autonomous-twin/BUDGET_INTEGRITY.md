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

The private research worker now loads one named provider from `models.yaml` and
requires an explicit, unexpired price declaration, input/output token ceilings,
a request timeout, and the fixed 3-candidate/8-tool/1-repair fanout before it can
report ready. SCADS institutional capacity is declared explicitly at zero
per-token cost through 2027-01-01; omission or expiry closes readiness. The
existing daily token and request ceilings remain active even when USD cost is
zero.

Expired research leases now retain their claimed capacity as uncertain during
maintenance recovery. Recovery is idempotent and does not infer zero usage from
a timeout, cancellation, or lost acknowledgement. Public evidence can be stored
under a content-, market-, and as-of-bound identity in memory or Datastore, and
the research gateway verifies a cache round trip before it calls a model.

OpenAI-compatible calls now expose a bounded token receipt when the upstream
response reports internally consistent prompt, completion, and total token
counts. The gateway computes cost from the approved price table and reconciles
that receipt atomically. Missing or inconsistent receipts remain uncertain;
provider-reported prices are not trusted over the configured price authority.

Frozen public research captures now have a strict versioned representation and
create-once in-memory and Datastore stores. The maintenance service exposes them
only to the service account that owns the current fenced claim. Both services
verify the assignment, model, market snapshot, evidence content, market identity,
and as-of time before the isolated research worker accepts the capture.

The same authenticated fenced channel now accepts a strict research result and
an optional complete usage receipt. Maintenance recomputes the capture/config
request hash, verifies the result's snapshot and instrument identities, persists
the decision create-once, reconciles the original claimed reservation, and
derives the public result and usage IDs. Large transport fields are never copied
into the durable worker-job status. A lost response can therefore retry without
duplicating spend or replacing a prior decision.

The isolated worker now invokes one bounded provider request against the exact
frozen capture and approved model configuration. It returns a strict forecast or
PASS plus measured usage through the fenced channel. Maintenance persists the
result, records an accepted forecast through the existing prospective-ledger
contract, and reconciles the original claim before marking the job complete.

Schema repair now uses a separate authenticated, fence-bound maintenance
authorization. Maintenance reconciles the primary receipt before it atomically
reserves and claims `:repair`; denial stops the worker before a second provider
call. Primary and repair usage travel separately, and missing usage retains the
corresponding claim as uncertain. Duplicate authorization cannot dispatch or
charge a second repair, while stale-job recovery also retains a possibly claimed
repair. Bounded OTel signals distinguish primary and repair tokens and repair
authorization outcomes without using account or reservation IDs as metric
labels. T07 is complete; these changes do not enable live trading.
