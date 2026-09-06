# Autonomous twin implementation baseline

Captured: 2026-09-06. This report completes T00 only; it does not enable trading, change a venue connection, or submit an order.

## Checkout and workspace

| Item | Result |
| --- | --- |
| Implementation worktree | `C:\Users\paree\OneDrive\Documents\Foresea\.codex-worktrees\autonomous-twin` |
| Branch | `codex/autonomous-twin`, tracking `origin/main` |
| Effective source commit | `0944bdc83b8c71d639fd09f2b5ad3cfa6d06e15e` (`track record: mtm tick [skip ci]`) |
| Original checkout at investigation | `d6b7a162ffca05649ba8246932a7c7f6ccc9d9a7`, behind current `origin/main` |
| Original untracked files preserved | `tmp_glm.sqlite` and the `docs/autonomous-twin/` plan pack |
| Implementation worktree status | clean before this task's documentation is added |

The plan pack was created in the original checkout before the clean worktree existed. This report records the branch that must receive it before T01 is committed; no source behavior was copied or modified in T00.

## Verification baseline

| Command | Result | Notes |
| --- | --- | --- |
| `PYTHONPATH=src py -m unittest discover -s tests` | pass: 1,257 tests in 152.289s | Test fixtures deliberately log provider, venue, storage and cancellation failures. The suite also emitted existing OTLP `402 Payment Required`, Starlette/httpx deprecation, and one SQLite resource warning. None failed the suite. |
| `py -m ruff check src tests` | pass | `All checks passed!` |
| `git diff --check` | pass | Clean implementation worktree before documentation is added. |
| `py -m graphify query "Where are all real prediction-market order submission callers, trade guardrails, durable trade runs, trading connection storage, and automated agent trading connected?" --budget 5000` | pass | It connects `trading.py`, `server.py`, `venue_routes.py`, `agent_trading_tick.py`, `benchmark_tools.py`, accounting, venue streams and the durable run path. |

`observability.py` initializes FastAPI, requests, traces, metrics and log instrumentation. Future T01+ business operations must use the existing module-scope tracer/meter style, record exceptions and use bounded metric dimensions. The current external OTLP exporter’s 402 response is not allowed to alter execution or reconciliation behavior.

## Real-order submission map

| Layer | Current responsibility | Entry points / symbols | Twin implication |
| --- | --- | --- | --- |
| Low-level venue submit | Signs and sends one order to Kalshi or Polymarket | `trading._place_kalshi_order`, `trading._place_polymarket_order` | T01 verifies contracts; T02 must prevent new callers from bypassing the shared boundary. |
| Generic submit gate | Requires `execute=true`, exact `PLACE REAL ORDER`, preview and venue enablement | `trading.place_order` | Remains mandatory for manual routes. The twin needs separately verified mandate authority, never a fabricated phrase. |
| Manual direct submit | Session, encrypted connection, fresh quote/balance/exposure/duplicate guards and audit | `POST /trading/orders` -> `server.trading_order` -> `_validate_live_trade_guardrails` -> `trading.place_order` | Shares the future account reservation boundary. |
| Durable manual run | Saves an order plan, atomically changes `awaiting_approval` to `submitting`, then submits/reconciles | `POST /trading/runs`, `POST /trading/runs/{id}/execute`, `_claim_trading_run_for_execution` | Closest existing primitive to reuse for durable autonomous commands. It is currently scoped to a saved run, not all account spending. |
| Connection custody | Session-owned, KMS envelope encrypted venue credentials; decrypt only for a requested venue operation | `PUT /trading/connections/{platform}`, `_stored_trading_credentials` | Reuse only after enforcing a dedicated account/connection scope. No secret crosses worker tasks. |
| Read/reconciliation | Reads one order or bounded portfolio data and merges known audit records | `trading.reconcile_order`, `trading.reconcile_portfolio`; `/trading/portfolio`, `/trading/orders/{id}/reconcile`, `/trading/runs/{id}/reconcile` | T06 must prove completeness and distinguish an empty successful result from an invalid financial response. |
| Cancel | Explicit `CANCEL OPEN ORDER`, then venue cancel and audit merge | `DELETE /trading/orders/{audit_order_id}`, `trading.cancel_order` | T15 must handle cancellation/fill races rather than treating cancel acknowledgement as no fill. |
| Generic venue writes | Authenticated venue action catalog supports amend/decrease/order-group actions; amend routes through trading guardrails | `POST /trading/venue/{platform}/actions/{operation}` in `venue_routes.py` | T02 must include these write operations in the shared audit/reservation review; they are not public data routes. |
| Scheduled reconciliation | Token-gated, manual-only workflow calls bounded internal reconciliation | `.github/workflows/trading-reconcile.yml`, `POST /internal/trading/reconcile` | It is intentionally inactive without a secret and is not a live agent scheduler. T16/T17 replace this runtime responsibility without converting GitHub Actions into an order runner. |

### Current low-level semantics to test next

- `trading.preview_order` normalizes Kalshi/Polymarket orders; `trading.place_order` treats a successful venue HTTP/SDK return as `submitted`, then routes callers to reconciliation. It must never be elevated to a fill assertion.
- Kalshi signing uses RSA-PSS over timestamp/method/path. The V2 create endpoint returns an `order_id`, `client_order_id`, `fill_count` and `remaining_count`; zero initial fill is a valid acknowledgement. The official V2 request uses a single YES-book `bid`/`ask`, fixed-point price/count and a self-trade-prevention field. T01 is responsible for selected-outcome mapping and contract fixtures. [Kalshi Create Order V2](https://docs.kalshi.com/api-reference/orders/create-order-v2)
- Polymarket submission currently relies on `py-clob-client-v2` methods after building `OrderArgs` or `MarketOrderArgs`. T01 must pin the installed SDK contract, outcome token mapping, fee/precision rules, fill acknowledgement and stable recovery identity. [Polymarket place orders](https://docs.polymarket.com/trading/place-orders)
- The relevant official Kalshi and Polymarket order/cancel documentation was retrieved on 2026-09-06. No credentials or live venue calls were made for T00.

## Autonomous and legacy paths

| Path | Current safety boundary | Required treatment |
| --- | --- | --- |
| `benchmark_tools.place_trade` | Rejects every mode except `shadow`; it never calls `trading.place_order`. | Preserve as benchmark-only paper trading. |
| `scripts/agent_trading_tick.py` | `_assert_shadow_mode()` independently rejects non-shadow configuration; reusable workflow hard-pins `FORESEA_AGENT_PLACE_TRADE_MODE=shadow`. | Keep scheduled agent work shadow-only. T08 builds a separate proposal-only research path. |
| Agent analysis | Produces a bounded live-trade *intent* for a human terminal, never a submitted order. | Treat as an input to a future typed proposal, not autonomous authority. |
| `scripts/live_trader_bridge.py` | Simulation-only compatibility tool; `--live` and `dry_run=False` return a migration block and contain no venue write methods. | It is not a source of authority, capital allocation or live-fill truth. |
| `venue_mcp.py` and public venue routes | Documented/read-only discovery interfaces. | Keep all public/MCP tooling unable to submit orders or access private twin state. |

## T02 execution-write audit

The 2026-09-06 source audit found these remaining venue-write capabilities:

| Capability | Code path | Current control | Autonomous status |
| --- | --- | --- | --- |
| Create an ordinary Kalshi or Polymarket order | `server.trading_order` or `server.execute_trading_run` -> `trading_control.ConfirmedManualOrderService` -> `trading.place_order` -> private low-level adapter | Session-bound encrypted connection, exact `PLACE REAL ORDER` confirmation, live enablement, fresh quote/balance/exposure/duplicate guards, audit and saved-run claim where applicable | Manual only. T14 replaces the confirmation phrase with a server-validated authority record for autonomous execution. |
| Amend, decrease, cancel or manage venue order groups | authenticated `/trading/venue/{platform}/actions/{operation}` -> `venue_api.action` | Exact `MANAGE REAL ORDERS` confirmation, live enablement, request audit; amendments also use the normal live guardrails and stored audit identity | Manual only. It does not create a new trade proposal and is excluded from worker capabilities until T14/T15 converge its reservation and recovery controls. |
| Reconcile and cancel an audited submitted order | private trading reconciliation/cancel routes -> `trading.reconcile_*` / `trading.cancel_order` | Session-bound encrypted connection, stored audit identity and reconciliation merge | Manual maintenance only until T15 provides durable worker recovery. |

`trading._place_kalshi_order` and `trading._place_polymarket_order` remain private functions invoked only by `trading.place_order`. The only ordinary create-order adapters are injected into `ConfirmedManualOrderService`; the two direct `trading.place_order` calls in server routes occur solely on the disabled-live gate and cannot reach a venue write. Benchmark tooling remains shadow-only, and the retired bridge cannot submit an order.

## Existing focused coverage

The most relevant existing tests are `tests/test_trading.py`, `tests/test_trading_byo.py`, `tests/test_server.py`, `tests/test_live_trader_bridge.py`, `tests/test_benchmark_tools.py`, `tests/test_agent_trading_tick.py`, `tests/test_venue_api_contracts.py`, `tests/test_venue_extensions.py` and `tests/test_venue_mcp.py`.

Important current assertions already protect: enablement and exact confirmation; KMS connection confidentiality; per-user guardrails; duplicate run claim; kill switch before exchange submission; audit/reconcile/cancel flow; scheduled reconciliation token gating; benchmark shadow-only operation; marketable shadow prices and order-book depth; and that an invalid benchmark live mode never calls the real submitter.

T03 can now define typed twin records and state transitions without importing FastAPI or the server's Datastore globals.

