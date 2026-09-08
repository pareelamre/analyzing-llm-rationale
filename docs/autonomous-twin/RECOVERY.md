# Trading recovery runbook

Foresea never retries a complete trading loop after a venue write might have happened. Recovery uses the client order ID and request fingerprint persisted before dispatch and keeps the original reservation until evidence proves the exposure state.

| Durable state | Automatic action | Operator action when evidence is incomplete |
|---|---|---|
| `reserved` | A worker may acquire the first lease and run current authorization checks. | Confirm the outbox and account projection agree before dispatch. |
| `submitting` or `submission_unknown` | Query complete order and fill collections for the exact account, instrument and client order identity. | Pause the account scope; do not submit another identity. Inspect venue order history and fills. |
| `acknowledged` or `partially_filled` | Re-fence an expired worker, apply each immutable fill ID/version once and retain the reservation. | Reconcile account drift before allowing new exposure. |
| `cancel_requested` | Reconcile first. Retry cancellation only when a complete identity-matching lookup is at most five seconds old and still shows the order open. | Keep the reservation when the cancel response is missing or does not explicitly confirm cancellation. A late fill remains authoritative after a cancel acknowledgement. |
| `cancelled` or `filled` | Continue durable read-only fill and settlement reconciliation; these states remain claimable for that purpose. | Correct discrepancies through new immutable venue observations; never edit stored evidence. |
| `rejected` | Release the reservation atomically with the durable rejection transition. | Investigate any account projection mismatch before new exposure. |

Foresea's confirmed-absence policy requires complete, identity-matching queries at distinct times. Kalshi requires at least two observations separated by one second. Polymarket requires at least three observations separated by five seconds because its order and data surfaces can converge independently. These are conservative Foresea controls, not venue consistency guarantees. Any incomplete page, mismatched identity, future timestamp or insufficient spacing leaves the reservation held and marks operator attention.

Settlements are bound to the exact venue order, client identity and instrument. They remain provisional until every latest settlement revision is explicitly final. A higher version replaces the prior value while preserving its stable settlement identity; corrections update the lifecycle projection and reservation reconciliation reference without replaying fills or releasing capital twice. Live recovery refuses process-local lifecycle storage.

After an account reconnect, increment the account epoch and reconcile a complete generation before granting new authority. Any unknown manual order or fill sets account divergence and blocks autonomous new exposure until the owner explains it.
