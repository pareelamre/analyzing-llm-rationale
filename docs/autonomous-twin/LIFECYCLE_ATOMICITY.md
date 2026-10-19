# Fenced lifecycle commit

Lifecycle reconciliation groups evidence, command state and reservation/cash in
one account transaction. Datastore/SQLite command and lifecycle stores must share
the same client. Existing transition and settlement rules are reused. Nested
account mutations join the transaction; memory development stores lock both
ledgers and restore immutable records on failure. SQLite restores its original
transaction depth after interrupted entry, body failure or failed commit.

Canonical stored scope, intent, client ID, reservation and fingerprint are bound
to the entire current claim. Processing time is separate from observation time;
old evidence cannot backdate a lease. In-transaction reads use a consistent
snapshot; they do not discover another transaction's new fence. Database commit
validation/serialization prevents a conflicting claimant from being overwritten.
Clock checks also repeat before commit. No venue/model request runs under the
account transaction. Submission recovery consumes bounded reads first, then
commits under fresh authority. Cancellation checks again after transport and
does not accept an expired/refenced response; the actual cancel may remain unknown.

Account snapshots now carry optional explicit epoch provenance through capture
and persistence. Legacy missing-epoch payloads retain their original fingerprint
format and remain readable, but cannot certify recovery presence/absence. They
are never stamped with the current epoch on load. Capture helpers must receive
the actual bound account epoch; VenueOrderLookup requires a matching positive
epoch. Do not assume older binaries can read new epoch-bearing fingerprints.

A raw filled/partially-filled status cannot create inventory or settle an
incompletely observed filled order. Rejection releases cash only with complete
zero-fill terminal evidence and does not trigger a second settlement decrement.
Older status updates cannot overwrite newer evidence, and final terminal authority
cannot be unsealed by a replay. Duplicate receive times do not duplicate fills.
Versioned settlement corrections remain supported; contradictory terminal fills
and reductions of already-filled quantity require explicit reconciliation.

After the financial transaction rolls back on a semantic contradiction, a
separate current-fenced transaction retains private TwinRecoveryAttention evidence
and sets recovery_blocked on the account root. Both new reservations and dispatch
of already-reserved commands refuse that root. Rejected/released commands remain
read-reconcilable for late fills; begin_dispatch cannot send them. Attention JSON
is excluded from indexes and bounded to 128 KiB; oversized evidence retains its
hash/size with an explicit refetch requirement. Metrics count committed states
only after commit; logs/spans exclude raw backend error text and evidence payloads.

## Operator recovery

1. Keep new entries paused. Inspect the account's private attention records with
   TwinStore.recovery_attention(scope_id); this diagnostic read is bounded to 200
   records. Use an authenticated ancestor query with cursors for complete history.
   These records contain private account evidence and must never become public
   GitHub artifacts, issues or dashboard payloads.
2. Reconcile the exact wallet, current epoch, original native/client order ID,
   full paginated order/fill/settlement history, actual balances, fees and inventory.
   Retain the existing command, reservation and evidence; do not generate another
   order identity, re-sign, or assume a missing response proves no acceptance.
3. A failed/unconfirmed database commit requires a durable re-read before retry.
   A timed-out cancellation requires another fresh bound lookup. Do not resubmit
   the order. A newer worker may obtain a read-reconciliation claim for a terminal
   command without gaining dispatch authority.
4. This patch exposes no unpause bypass. Keep the root paused until an
   owner-reviewed correction/reconciliation path is available. Do not clear the
   marker or discard epoch provenance to make a job or older binary pass. Quiesce
   old writers before any deployment/rollback that cannot honor these fields.

## Local evidence

The initial regressions failed on the old implementation. Post-review focused
validation passes 91 tests, including 38 real local emulator cases. Both venues'
commit-conflict drills retain the old clock inside its lease, prove the replacement
claim committed before releasing the old transaction, and observe the old commit
abort with no evidence/cash mutation. Expiry rollback is tested separately.

Full emulator-enabled unittest passed 2,939 tests with no skips in 260.933s; four
additional test-only cases added afterward are covered by the current91-case
focused run. Isolated SDK1.2 matrix 51 tests, Ruff, release contracts and the real
local OTel debug export pass. Graphify and final CI are recorded before merge.
One deep review aud_Y7dTd66krE completed, with four usable structured voices,
one unusable output and one model fallback. Fifteen usable findings were normalized
to 13 accepted dispositions, adjudicated as adj-0bBxfPBuYRlcQ4QQ. Fixes were verified
locally, not panel re-reviewed. See LIFECYCLE_ATOMICITY_REVIEW.md. Semgrep was
unavailable locally; no SAST pass is claimed.
Local emulator checks are not production IAM, venue execution or forward-performance
evidence. Native order/fill/settlement readers, maintenance worker selection and
release reconciliation remain T14/T15 work; this change does not activate trading,
provision resources or replace the private R2 object-storage destination.
