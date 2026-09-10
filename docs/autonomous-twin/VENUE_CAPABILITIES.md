# Venue capability matrix

Retrieved 2026-09-10. This is a release-control document, not evidence that an
account may trade. The autonomous twin remains in `shadow` mode until every
required live capability is explicitly verified for its dedicated account and
execution environment.

| Venue | Operation | Current contract | Evidence | Status | Autonomous-live decision |
| --- | --- | --- | --- | --- | --- |
| Kalshi | Create event-market order | V2 `POST /portfolio/events/orders`; payload is always the YES book. `bid` buys YES and `ask` sells YES, so selected-NO price is complemented. | Official [Create Order V2](https://docs.kalshi.com/api-reference/orders/create-order-v2); `kalshi_create_order_v2.json`; `tests.test_trading` | fixture-verified | Blocked pending dedicated-account demo and production contract verification. |
| Kalshi | Request identity and auth | Client order ID plus RSA-PSS request headers; V2 acknowledgement contains order ID, matching client ID, fill and remaining quantities. | Official contract; signing and acknowledgement tests | fixture-verified | Shared manual path only. A twin must persist identity before dispatch in T14. |
| Kalshi | Cancel and reconcile | Existing code uses V2 single-order cancel and order lookup. Pagination, timeout recovery and terminal-state semantics remain unverified. | Source inspection; no contract fixture yet | source-only | Blocked until T06/T15 fixtures and demo verification. |
| Polymarket | Limit / immediate order preparation | Exact token ID, tick and market minimum are required. Buy market amount is USD; sell is shares. FAK/FOK need a bounded price. | Official [Place Orders](https://docs.polymarket.com/trading/place-orders); preview tests | fixture-verified | Blocked: installed environment does not contain `py-clob-client-v2`; signing signature/identity cannot be verified. |
| Polymarket | Submission receipt | An accepted receipt includes order ID and a status such as `matched` or `delayed`; a delayed receipt has no fill. Rejection has `success: false` or `ok: false`. | Official order docs; `polymarket_market_order_ack.json`; acknowledgement tests | fixture-verified | Blocked pending supported SDK install and demo account verification. |
| Polymarket | Signed identity / recovery lookup | The optional SDK dependency is declared as `py-clob-client-v2>=1.0.1`, but it is not installed in the checked environment. No client-order-ID support is assumed. | `pyproject.toml`; local package check on 2026-09-06 | blocked | Do not submit autonomous Polymarket orders until a version-pinned SDK can create a recoverable signed identity and lookup path. |
| Both | Account, balances, orders, fills and settlement | Foresea reads every cursor/offset page, fingerprints all five collections before and after normalization, persists only complete generations, and derives open-order reservations separately from venue cash. Kalshi merges live and historical order/fill partitions, uses fixed-point exposure as basis, and keys each account-scope settlement by market ticker so corrected payout fields replace the prior generation. Polymarket uses the atomic accounting ZIP for cash/positions and onchain `REDEEM` activity for settlement identity. Both liquidate positions through executable bid depth, valuing missing depth at zero. | Official [Kalshi balance](https://docs.kalshi.com/api-reference/portfolio/get-balance), [positions](https://docs.kalshi.com/api-reference/portfolio/get-positions), [settlements](https://docs.kalshi.com/api-reference/portfolio/get-settlements), and [historical data](https://docs.kalshi.com/getting_started/historical_data); official [Polymarket accounting snapshot](https://docs.polymarket.com/api-reference/misc/download-an-accounting-snapshot-zip-of-csvs) and [activity](https://docs.polymarket.com/api-reference/core/get-user-activity); T06 fixtures/tests | fixture-verified | Ready for shadow account synchronization. A Polymarket fee-enabled trade that omits its actual fee remains fail-closed; venue credentials and owner mandate still gate live execution. |
| Both | Bounded-price immediate execution | Manual controls permit market-like orders only behind `FORESEA_ALLOW_MARKET_ORDERS`; the twin must use verified bounded immediate behavior. | Current guardrails and venue docs | source-only | Blocked until T09/T14. |

## Contract rules locked in T01

- At selected outcome price `0.30`, Kalshi orders are `BUY YES -> bid/0.30`,
  `SELL YES -> ask/0.30`, `BUY NO -> ask/0.70`, and `SELL NO -> bid/0.70`.
  Foresea retains the selected price for its local risk calculation; only the
  exchange payload is expressed as the YES-leg price.
- A successful HTTP response is only an acknowledgement. It never represents a
  full fill. Missing order identity, mismatched Kalshi client identity, missing
  quantitative fields, malformed trade IDs or an unknown receipt status fail
  closed.
- This matrix deliberately does not claim full endpoint coverage. It identifies
  the capabilities the strategy needs and the still-blocking verification work.
