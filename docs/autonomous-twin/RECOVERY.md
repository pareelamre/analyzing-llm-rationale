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

Foresea's confirmed-absence policy applies only to explicitly unstarted commands and requires complete, identity-matching queries at distinct times, each no more than 30 seconds old. Kalshi requires at least two observations separated by one second. Polymarket requires at least three observations separated by five seconds. These are conservative Foresea controls, not venue consistency guarantees. A started or legacy-unknown dispatch, incomplete page, mismatched identity, future timestamp or insufficient spacing retains the reservation and marks operator attention. The native Polymarket lookup supplies positive presence only; missing/404/error never supplies absence evidence.

Settlements are bound to the exact venue order, client identity and instrument. They release capacity only after the order is terminal, the trusted caller explicitly supplies complete order/fill/settlement evidence (`terminal_evidence_complete=True`), and every latest settlement revision is final. The default is incomplete, including older stored lifecycles. A complete zero-fill cancellation may finalize a zero-exposure hold without settlement rows; a cancellation acknowledgement alone cannot. A higher settlement version replaces the prior value while preserving its stable identity; corrections update the reference without releasing capital twice. Live recovery/cancellation requires durable stores. Persistence mutation methods are private trusted-server primitives, never client proof APIs.

After an account reconnect, increment the account epoch and reconcile a complete generation before granting new authority. Any unknown manual order or fill sets account divergence and blocks autonomous new exposure until the owner explains it.
