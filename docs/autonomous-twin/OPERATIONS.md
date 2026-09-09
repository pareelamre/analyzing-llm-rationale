# Autonomous twin operations

This runbook covers the private shadow runtime and any later owner-authorized
live pilot. Live execution remains disabled unless the exact approved mandate,
readiness artifact, account epoch, strategy version, and runtime flag agree.
Telemetry is provider-neutral OpenTelemetry plus ordinary application logs; no
Superlog account, token, connector, or paid subscription is required.

## Operator references and data handling

Metrics use bounded labels only: operation, outcome, venue, worker role, job
kind, and stable decision reason. They never contain account IDs, order IDs,
wallet addresses, request text, or credentials. Traces may carry
`twin.account_ref` and `twin.command_ref`; ordinary logs contain only the first
16 hexadecimal characters of their SHA-256 hashes. Use the owner-authenticated
`/twin/commands` and `/twin/portfolio` views to resolve an affected record, then
use restricted Cloud Logging and Datastore access for incident work. Never paste
tokens, signed requests, worker payloads, or venue responses into a ticket.

## Signals and initial alert thresholds

| Signal | Meaning | Initial alert |
| --- | --- | --- |
| `twin.decisions` | Strategy result by bounded decision and reason | page if a live-capable account has no decisions for two cycle intervals |
| `twin.duplicate_suppressions` | Duplicate reserve, enqueue, delivery, or submit prevented | warn on a sustained increase; page only with queue lag or ambiguity |
| `twin.account.drift` | Venue/local cash, holdings, or external-activity divergence | page on any live-capable account |
| `twin.submissions.ambiguous` | A send, persistence, cancellation, or recovery result is uncertain | page immediately and pause the account |
| `twin.data.stale` | Account, market, or fee input exceeded its freshness bound | warn after two cycles; page if it persists for ten minutes |
| `twin.retries.exhausted` | A bounded safe read or Datastore reservation retry ended | warn immediately; page after three in five minutes |
| `twin.queue.lag` | Seconds from durable job creation to dispatch or claim | warn above 60 seconds; page above 300 seconds for maintenance |
| `twin.research.budget.usd` | Reserved and provider-reported USD per request | warn at 80% of the daily mandate budget; stop at 100% |
| `twin.research.budget.tokens` | Reserved and provider-reported tokens | warn at 80% of the daily token budget; stop at 100% |
| `twin.research.budget.operations` | Reconciled, uncertain, or invalid usage outcomes | page on repeated uncertain or invalid usage |

Keep alerts scoped to the private maintenance service and bounded deployment
labels. Validate thresholds during the seven-day shadow trial before enabling a
live mandate. Exporter failures may reduce telemetry, but must never change a
risk decision, release a reservation, retry a venue write, or stop settlement
reconciliation. Cloud Run logs remain the fallback signal.

## Universal containment procedure

1. Use the owner desk to set the global or account pause. If the desk is
   unavailable, set the runtime live flag false and stop new task dispatch.
2. Do not delete jobs, commands, reservations, fills, or settlement records.
3. Record the deployment revision and the hashed account and command references.
4. Reconcile acknowledged, partially filled, cancellation-requested, and
   submission-unknown commands read-only before releasing capacity.
5. Resume only after the original invariant is restored and a fresh readiness
   result is bound to the same code/config hashes.

## Kill or pause

**Trigger:** unexpected exposure, ambiguous submission, drawdown breach, venue
halt, stale authority, operator request, or unexplained account drift.

Set the narrowest effective pause in `/trade`: command, market, account, or
global. Revoke the mandate when authority itself is in doubt. Confirm the status
view reports the pause and that maintenance work still runs. Cancel only through
`POST /twin/commands/{id}/cancel`; it enqueues a deduplicated maintenance job and
does not accept raw venue parameters. Reconcile late fills before resuming.

## Lost research provider

**Signals:** research jobs become `degraded`, safe-read retries exhaust, or no
new research results arrive while maintenance stays healthy.

Leave trading paused for candidates without fresh research. Check provider
status and the configured model identifier without logging credentials. Keep
claimed budget as uncertain when billing facts are missing. Restore the provider
or select a separately configured and priced model, then enqueue a new research
assignment; never replay the claimed assignment.

## Lost durable store

**Signals:** Datastore operations fail, startup reconciliation fails, or the
maintenance readiness endpoint is false.

Disable new dispatch and live execution. Do not fall back to the in-memory store
for live-capable scopes. Confirm Datastore service health, IAM, namespace, and
the deployed project. Restore access, restart maintenance, and allow startup
recovery to classify every nonterminal command. Compare the rebuilt projections
with complete venue account pages before clearing the pause.

## Stalled queue

**Signals:** `twin.queue.lag` exceeds 60 seconds, maintenance jobs remain queued,
or Cloud Tasks delivery errors rise.

Pause new exposure if maintenance lag exceeds 300 seconds. Inspect queue depth,
rate limits, OIDC audience, target revision health, and the deterministic task
name. Preserve the existing job; dispatching it again is safe because the durable
claim fence suppresses duplicate work. Scale or repair the affected queue while
keeping research capacity isolated from maintenance capacity.

## Uncertain submission or cancellation

**Signals:** a command is `submission_unknown`, a cancellation result is
unconfirmed, or `twin.submissions.ambiguous` increments.

Pause the account immediately. Resolve the command from its hashed log reference
in the owner-only command list. Query the venue using the persisted client order
identity and request fingerprint. Do not submit again. Use the venue-specific
separated absence observations before marking an order absent; otherwise retain
the reservation and operator-attention state. Continue late-fill and settlement
reconciliation even after cancellation.

## Broken credentials or worker identity

**Signals:** venue 401/403, worker OIDC rejection, KMS denial, or research gateway
authorization failure.

Pause the affected account or role. Verify Secret Manager version, service
account binding, exact OIDC audience, clock, and deployed revision. Never print
the secret or authorization header. Rotate a credential only through the normal
secret-management process, deploy a new revision, then prove read-only account
access before allowing shadow work. A live mandate requires a new readiness
artifact when the bound deployment identity changes.

## Missing or incomplete portfolio page

**Signals:** account snapshot is incomplete, a cursor is missing, generation is
stale, settlements are not final, or `stale_account_snapshot` blocks risk.

Keep the last complete snapshot for display but block new exposure. Fetch every
page for balances, positions, orders, fills, and settlements under one captured
generation. Treat unsupported cash or settlement authority as unavailable. Once
a complete snapshot is stored, compare venue cash, reservations, holdings, fees,
and external activity before clearing the pause.

## Venue halt or stale market mechanics

**Signals:** venue status is closed/halted, market snapshots age out, fee version
changes, tick/lot rules fail validation, or order endpoints reject valid reads.

Pause the venue or affected markets. Continue read-only account and settlement
work. Refresh market metadata, fee schedule, eligibility, order status, and
venue time. Re-run contract fixtures and shadow previews against the new version.
Require a fresh readiness artifact before allowing new exposure.

## Deployment rollback

Pause new exposure first. Reconcile all unknown and cancellation-requested
commands. Deploy the last known-good immutable image revision; do not roll back
or delete Datastore entities. Confirm `/health`, private readiness, worker OIDC,
queue delivery, projection rebuild, and shadow-only runtime state. Re-enable
scheduled shadow work only after code/config hashes match the readiness record.

## Settlement correction

Keep fills immutable and append the higher-version venue settlement observation.
Verify it matches account scope, client order identity, venue order identity, and
instrument. Rebuild the lifecycle and account projections. Release reservation
capacity only from an explicitly final settlement; the settlement reference
makes repeated processing idempotent. If cash still diverges, retain the account
pause and escalate with the hashed account and command references.

## Live-blocking reason map

| Reason family | Required action |
| --- | --- |
| inactive, expired, revoked, paused, stale, mismatched, or missing mandate/readiness | pause; renew owner authority or generate fresh bound readiness evidence |
| live runtime disabled or nondurable store | keep live disabled; restore the protected runtime/store and redeploy |
| incomplete/stale market, fee, account, portfolio, or settlement data | refresh complete immutable inputs and reconcile drift |
| insufficient cash/loss, drawdown, daily loss, market, cluster, or total-loss limit | do not override; reduce exposure or wait for authorized capacity |
| stale claim/fence, command state mismatch, reservation mismatch, or contention exhaustion | stop dispatch; recover from durable state and re-claim only after lease rules permit |
| unknown submission/cancel result or incomplete absence proof | pause account; search by persisted identity; never resubmit |
| provider, credential, store, queue, or venue unavailable | follow the matching dependency runbook and retain conservative reservations/budgets |
| external activity or account drift | pause; attribute activity and rebuild projections from complete venue pages |

