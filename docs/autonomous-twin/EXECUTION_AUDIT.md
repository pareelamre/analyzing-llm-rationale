# Venue-write audit

T14 reviewed the repository's order submission boundaries after introducing the shared fenced dispatcher.

- `trading.place_order` is the only new-order adapter. It validates confirmation and runtime switches, then calls the private Kalshi or Polymarket adapter exactly once.
- Both `POST /trading/orders` and saved-run execution inject `trading.place_order` into `ConfirmedManualOrderService`, reserve a stable client identity, and call `submit_claimed_command`. They do not call the venue adapter directly.
- Autonomous workers must enter through `submit_claimed_command`; it rechecks the immutable intent, current lease fence, reservation, account epoch, mandate, pause state, readiness hash, environment and mandate budgets immediately before its single injected venue call.
- `benchmark_tools.place_trade` remains shadow-only, and `scripts/live_trader_bridge.py` remains simulation-only.
- Reconciliation and cancellation calls are read/reduce operations over an existing venue order identity. They cannot create new exposure through this boundary.
- Generic venue-extension operations do not grant autonomous authority and remain behind their existing authenticated, allowlisted route policy.

A transport exception or unrecognized acknowledgement moves the command to `submission_unknown`. No submission path retries a trading loop; recovery must search using the persisted client order identity and request fingerprint.
