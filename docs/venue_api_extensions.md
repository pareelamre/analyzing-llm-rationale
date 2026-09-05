# Kalshi and Polymarket operations

Foresea exposes 48 additional or paginated REST operations and native exchange
WebSocket streams. The checked-in `venue_contracts.json` contains the exact
operation names, upstream paths, parameter locations and input schemas. Contracts
were checked against the exchanges' specifications on September 5, 2026:
[Kalshi](https://docs.kalshi.com/openapi.yaml),
[Polymarket CLOB](https://docs.polymarket.com/api-spec/clob-openapi.yaml),
[Polymarket Data API](https://docs.polymarket.com/api-spec/data-openapi.yaml).

## Routes and discovery

| Foresea route | Access | Purpose |
| --- | --- | --- |
| `GET /market/venue/catalog` | Public | Discover public operations and schemas |
| `POST /market/venue/{platform}/{operation}` | Public | Public reads, including upstream batch POST requests |
| `GET /trading/venue/catalog` | Signed-in user | Discover account reads and order actions |
| `POST /trading/venue/{platform}/{operation}` | Signed-in user with connected venue | Read one account page |
| `POST /trading/venue/{platform}/actions/{operation}` | Connected account and trading controls | Explicit order management |
| `WS /ws/venue/{platform}` | See stream section | Native market, order and fill events |

`platform` is `kalshi` or `polymarket`. Read requests accept
`{"parameters": {...}, "body": ...}`. Path identifiers and query parameters both
go in `parameters`; upstream POST arrays go in `body`. Unknown fields and
operations, invalid identifiers, excessive batch sizes and invalid time windows
are rejected before making a venue request. No route accepts an upstream URL or
raw exchange credentials.

Responses are `{"platform": "...", "operation": "...", "data": ...}`. `data`
preserves the upstream payload. Follow `next_cursor` or `next_offset` explicitly
to fetch subsequent pages. A null continuation marks the end of the result, or
an offset ceiling when `pagination_limit_reached` is true. These routes fetch
one bounded page; they do not claim to return an entire account history.

Account requests require the existing `Authorization: Bearer <Foresea session>`
header. Polymarket account addresses come from the connected account; callers
cannot override `user` or `maker_address`. Kalshi batch orderbooks also require a
connected account because the upstream endpoint requires signed requests.

## Operations

| Venue | Public operations |
| --- | --- |
| Kalshi | `historical_cutoff`, `historical_markets`, `historical_market`, `historical_candles`, `historical_trades`, `candlesticks`, `series_fee_changes`, `event_fee_changes`, `event_metadata`, `milestones`, `milestone`, `weather`, `weather_calibrations` |
| Polymarket | `books`, `midpoints`, `spreads`, `fee_rate`, `holders`, `oi`, `live_volume` |

| Venue | Account reads |
| --- | --- |
| Kalshi | `historical_fills`, `historical_orders`, `historical_positions`, `settlements`, `positions`, `orders`, `fills`, `orderbooks`, `order_groups`, `order_group` |
| Polymarket | `activity`, `closed_positions`, `value`, `positions`, `orders`, `fills` |

| Venue | Order actions |
| --- | --- |
| Kalshi | `amend_order`, `decrease_order`, `cancel_orders`, `cancel_all`, `create_order_group`, `delete_order_group`, `reset_order_group`, `trigger_order_group`, `limit_order_group` |
| Polymarket | `cancel_all`, `cancel_market_orders`, `heartbeat` |

Examples:

```http
POST /market/venue/kalshi/historical_candles
Content-Type: application/json

{"parameters":{"ticker":"KXEXAMPLE-26JAN01","start_ts":1767225600,"end_ts":1767312000,"period_interval":60}}
```

```http
POST /market/venue/polymarket/books
Content-Type: application/json

{"body":[{"token_id":"123"},{"token_id":"456"}]}
```

```http
POST /trading/venue/kalshi/settlements
Authorization: Bearer <Foresea session>
Content-Type: application/json

{"parameters":{"limit":100,"cursor":"<previous next_cursor>"}}
```

Existing Kalshi market lookup and settlement resolution automatically try the
historical market endpoint on a live HTTP 404. Candle history also checks the
archive when live candles are empty. Provider failures such as 503 do not become
successful empty responses. Cutoff and historical list operations allow callers
to explicitly partition larger backfills by the venue's retention boundary.

MCP exposes `foresea_venue_data` and the ReAct loop exposes `venue_data` for public
operations. Omit `operation` to discover the public catalog. Private account
reads and writes are available only through authenticated Foresea routes.

## Order management

Every action requires `execute: true`,
`confirmation: "MANAGE REAL ORDERS"`, and the existing live-trading enablement
gate. Amendments require an owned `audit_order_id`, preserve its ticker and
trade direction, and pass Foresea's existing fresh-quote, portfolio, exposure,
pause and notional checks. The original audit record supplies the venue order ID
and shard/subaccount routing. Only that order is excluded from the duplicate
check; other recent orders still block duplicate exposure.

```http
POST /trading/venue/kalshi/actions/decrease_order
Authorization: Bearer <Foresea session>
Content-Type: application/json

{"audit_order_id":"<Foresea audit id>","body":{"reduce_by":"2.00"},"execute":true,"confirmation":"MANAGE REAL ORDERS"}
```

Cancel-all and market/batch cancellation operate on the connected venue account,
including orders placed outside Foresea. Group reset/limit/create also honor the
platform kill switch and the user's pause setting. All actions record an audit
intent and outcome. No write is retried automatically. After uncertain outcomes,
reconcile with the existing order or portfolio reconciliation routes before
retrying; uncertain amendment/decrease audit records block further mutations.

The Polymarket heartbeat action sends a single authenticated heartbeat. Start
with an empty `heartbeat_id`, then pass the returned ID on subsequent calls.
It does not start a background heartbeat service. The exchange may cancel open
orders when an activated heartbeat expires; see the
[heartbeat contract](https://docs.polymarket.com/api-spec/clob-openapi.yaml).

## Native streams

Connect to `/ws/venue/polymarket` or `/ws/venue/kalshi` and send one subscription
frame within ten seconds:

```json
{"scope":"market","identifiers":["123","456"]}
```

Polymarket market identifiers are CLOB token IDs; Kalshi identifiers are market
tickers. Market subscriptions require 1–100 identifiers. To receive authenticated
order/fill updates, use `scope: "user"`; identifiers are optional filters (Kalshi
tickers or Polymarket condition IDs).

Polymarket public market streams need no account. Kalshi streams and both user
streams require a Foresea session header, or `session_token` in the initial frame
for browser clients, plus a connected venue account. Never put a session token
in the URL. Exchange keys remain server-side. Polymarket API-key owner fields
are removed from forwarded events.

The bridge signs Kalshi handshakes, sends native subscription messages, maintains
Polymarket's application heartbeat and reconnects up to three times with backoff.
It preserves upstream event payloads and emits `stream_reset` before each new
connection's data. Discard cached books on reset and wait for new snapshots;
reconcile account state after reconnect because streams do not replay missing
fills. Kalshi sequence gaps trigger reconnection. `stream_reconnecting` indicates
that data is temporarily stale. Connections close after 15 minutes for session
refresh, and downstream disconnects cancel upstream work.

These are on-demand streams. `/ws/radar` continues to serve Foresea's existing
periodic radar snapshots; the REST health probe does not establish exchange
stream connections. Protocol references:
[Kalshi](https://docs.kalshi.com/getting_started/quick_start_websockets),
[Polymarket market](https://docs.polymarket.com/market-data/realtime-data),
[Polymarket user](https://docs.polymarket.com/trading/realtime-order-updates).

## Observability and validation

Operations produce native OpenTelemetry spans, `venue.operations` and
`venue.stream.events` counters, and completion/failure logs without credentials
or account payloads. Tests cover endpoint contracts, bounded pagination, account
isolation, archive fallback, guarded writes, native subscriptions, disconnect
cleanup and sequence-gap handling. Live private trading is not part of tests.
