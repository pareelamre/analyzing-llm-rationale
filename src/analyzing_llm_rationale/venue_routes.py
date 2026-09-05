"""Foresea routes for the explicit venue operation catalog and native streams."""
from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from . import market_data, trading, venue_api, venue_streams

router = APIRouter(tags=["Venue APIs"])


class VenueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parameters: dict[str, Any] = Field(default_factory=dict)
    body: dict[str, Any] | list[dict[str, Any]] | None = None


class VenueActionRequest(VenueRequest):
    execute: bool = False
    confirmation: str = ""
    audit_order_id: str | None = None


def _runtime():
    from . import server
    return server


def _credentials(runtime, user_id: str, platform: str) -> dict:
    credentials = runtime._stored_trading_credentials(user_id, platform)
    if not credentials:
        raise HTTPException(409, "Connect this venue account first")
    return credentials


@router.get("/market/venue/catalog")
async def venue_catalog() -> dict:
    return venue_api.catalog()


@router.post("/market/venue/{platform}/{operation}")
async def public_venue_data(platform: str, operation: str, req: VenueRequest, request: Request) -> dict:
    _runtime()._check_rate_limit(request)
    return await asyncio.to_thread(venue_api.read, platform, operation, req.parameters, req.body)


@router.get("/trading/venue/catalog")
async def account_venue_catalog(request: Request) -> dict:
    _runtime()._require_session(request)
    return {"account": venue_api.catalog("account"), "actions": venue_api.catalog("write")}


@router.post("/trading/venue/{platform}/{operation}")
async def account_venue_data(platform: str, operation: str, req: VenueRequest, request: Request) -> dict:
    runtime = _runtime()
    runtime._check_rate_limit(request)
    claims = runtime._require_session(request)
    venue_api.contract(platform, operation, "account")
    credentials = _credentials(runtime, claims["sub"], platform)
    try:
        return await asyncio.to_thread(venue_api.read, platform, operation, req.parameters, req.body,
                                       access="account", creds=credentials)
    except trading.TradingError as exc:
        raise runtime._trading_http_exception(exc) from exc


@router.post("/trading/venue/{platform}/actions/{operation}")
async def venue_action(platform: str, operation: str, req: VenueActionRequest, request: Request) -> dict:
    runtime = _runtime()
    runtime._check_rate_limit(request)
    claims = runtime._require_session(request)
    venue_api.contract(platform, operation, "write")
    user_id = claims["sub"]
    credentials = _credentials(runtime, user_id, platform)
    parameters, body = dict(req.parameters), req.body
    record = None
    dispatched = False
    try:
        if operation in ("amend_order", "decrease_order"):
            if not req.audit_order_id:
                raise HTTPException(422, "audit_order_id is required")
            record = runtime._read_trading_order(user_id, req.audit_order_id)
            if not record or record.get("platform") != platform or not record.get("venue_order_id"):
                raise HTTPException(404, "Submitted order was not found")
            if record.get("status") in ("filled", "canceled", "rejected", "submission_unknown", "reconciliation_required"):
                raise HTTPException(409, "Order is already terminal")
            if "order_id" in parameters or "subaccount" in parameters:
                raise HTTPException(422, "Order routing comes from the stored audit record")
            parameters["order_id"] = record["venue_order_id"]
            if record.get("subaccount") is not None:
                parameters["subaccount"] = record["subaccount"]
            if not isinstance(body, dict):
                raise market_data.MarketDataInputError("Action body must be an object")
            body = dict(body)
            body["exchange_index"] = int(record.get("exchange_index") or 0)
            if operation == "decrease_order":
                body["market_ticker"] = record["ticker"]
        venue_api.validate_action(platform, operation, parameters, body, execute=req.execute,
                                  confirmation=req.confirmation, creds=credentials)
        if operation in ("create_order_group", "reset_order_group", "limit_order_group"):
            policy = runtime._effective_trading_guardrails(user_id)
            if policy["paused"] or runtime._trading_guardrail_env_bool("FORESEA_TRADING_KILL_SWITCH", False):
                raise HTTPException(409, "Trading is paused in risk controls")
        preview = None
        if operation == "amend_order":
            # The original trade direction cannot be changed by an amendment.
            side, _ = trading._kalshi_side(record["action"], record["outcome"])
            if body["ticker"] != record["ticker"] or body["side"] != side:
                raise HTTPException(422, "Amendment must preserve the original ticker and side")
            payload = {"platform": platform, "ticker": record["ticker"], "action": record["action"],
                       "outcome": record["outcome"], "order_type": "limit", "price": body["price"],
                       "quantity": body["count"], "execute": True, "confirmation": trading.CONFIRMATION_PHRASE}
            preview = await asyncio.to_thread(trading.preview_order, payload, credentials)
            await runtime._validate_live_trade_guardrails(user_id, payload=payload, preview=preview,
                                                         credentials=credentials,
                                                         exclude_audit_order_id=req.audit_order_id)
        # Persist intent before a request that could be accepted even if its response is lost.
        runtime._record_trading_risk_event(user_id, venue=platform, event=operation,
                                          outcome="requested", audit_order_id=req.audit_order_id)
        dispatched = True
        result = await asyncio.to_thread(venue_api.action, platform, operation, parameters, body,
                                         execute=req.execute, confirmation=req.confirmation, creds=credentials)
        if record:
            source = result.get("data", {}).get("order")
            if isinstance(source, dict):
                state = trading._normalise_reconciled_order(platform, source)
                record = runtime._merge_order_reconciliation(record, state)
                for field in ("quantity", "price"):
                    if state.get(field) is not None:
                        record[field] = state[field]
            else:
                record["status"] = "reconciliation_required"
            if preview:
                record["quantity"] = preview["normalized_order"]["quantity"]
                record["price"] = preview["normalized_order"]["price"]
                record["estimated_notional"] = preview["estimated_notional"]
            runtime._put_trading_order(user_id, record)
            runtime._sync_trade_run_from_order(user_id, record)
        runtime._record_trading_risk_event(user_id, venue=platform, event=operation,
                                          outcome="success", audit_order_id=req.audit_order_id)
        return result
    except Exception as exc:
        if dispatched and record:
            record["status"] = "submission_unknown"
            runtime._put_trading_order(user_id, record)
        runtime._record_trading_risk_event(user_id, venue=platform, event=operation,
                                          outcome="error", audit_order_id=req.audit_order_id)
        if isinstance(exc, market_data.MarketDataInputError):
            raise
        raise runtime._trading_http_exception(exc) from exc


class StreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: Literal["market", "user"] = "market"
    identifiers: list[str] = Field(default_factory=list, max_length=100)
    session_token: str = ""


@router.websocket("/ws/venue/{platform}")
async def venue_stream(websocket: WebSocket, platform: str) -> None:
    runtime = _runtime()
    runtime._check_rate_limit(websocket)
    await websocket.accept()
    tasks = []
    try:
        req = StreamRequest.model_validate(await asyncio.wait_for(websocket.receive_json(), 10))
        credentials = None
        if platform == "kalshi" or req.scope == "user":
            claims = runtime._decode_session(req.session_token) if req.session_token else runtime._require_session(websocket)
            credentials = _credentials(runtime, claims["sub"], platform)
        # Validate before starting either task, so no invalid subscription reaches the exchange.
        venue_streams.subscription(platform, req.scope, req.identifiers, credentials)

        async def forward():
            upstream = venue_streams.stream(platform, req.scope, req.identifiers, credentials)
            try:
                async for event in upstream:
                    await asyncio.wait_for(websocket.send_json(event), 10)
            finally:
                await upstream.aclose()

        async def disconnected():
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return

        tasks = [asyncio.create_task(forward()), asyncio.create_task(disconnected())]
        done, _ = await asyncio.wait(tasks, timeout=900, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except WebSocketDisconnect:
        return
    except Exception:
        # Stream errors must never echo the initial frame, auth headers or keys.
        await websocket.close(code=1008, reason="Venue stream failed or subscription was invalid")
        return
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    if websocket.client_state.name != "DISCONNECTED":
        await websocket.close(code=1000, reason="Reconnect to refresh the session")
