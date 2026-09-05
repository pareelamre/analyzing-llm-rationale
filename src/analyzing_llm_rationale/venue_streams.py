"""Native exchange streams with bounded reconnects and explicit snapshot resets."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import suppress
from urllib.parse import urlparse

from opentelemetry import metrics, trace

from . import market_data, trading

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
events = metrics.get_meter(__name__).create_counter("venue.stream.events", unit="1")


def subscription(platform: str, scope: str, identifiers: list[str], creds: dict | None) -> tuple:
    if scope not in ("market", "user") or platform not in ("kalshi", "polymarket"):
        raise market_data.MarketDataInputError("Invalid venue or stream scope")
    if not isinstance(identifiers, list) or len(identifiers) > 100 or (scope == "market" and not identifiers):
        raise market_data.MarketDataInputError("Provide 1-100 market identifiers (user filters may be empty)")
    pattern = r"[A-Za-z0-9_.-]{1,200}" if platform == "kalshi" else (r"[0-9]{1,100}" if scope == "market" else r"0x[0-9a-fA-F]{64}")
    if any(not isinstance(item, str) or not re.fullmatch(pattern, item) for item in identifiers):
        raise market_data.MarketDataInputError("Invalid stream identifier")
    if (platform == "kalshi" or scope == "user") and not creds:
        raise trading.TradingNotConfiguredError("Connect a venue account to use this stream")
    if platform == "kalshi":
        base = trading._cv(creds, "kalshi_base_url", "https://external-api.kalshi.com/trade-api/v2")
        host = urlparse(base).hostname
        if host not in ("external-api.kalshi.com", "api.elections.kalshi.com", "demo-api.kalshi.co", "external-api.demo.kalshi.co"):
            raise trading.TradingValidationError("Unsupported Kalshi stream environment")
        demo = "demo" in host
        url = "wss://external-api-ws." + ("demo.kalshi.co" if demo else "kalshi.com") + "/trade-api/ws/v2"
        headers = trading._kalshi_auth_headers("GET", "/trade-api/ws/v2", creds=creds)
        params = {"channels": ["orderbook_delta", "ticker", "trade"] if scope == "market" else ["fill", "user_orders"]}
        if identifiers:
            params["market_tickers"] = identifiers
        return url, headers, {"id": 1, "cmd": "subscribe", "params": params}
    frame = {"type": scope}
    if scope == "market":
        frame["assets_ids"] = identifiers
    else:
        frame["auth"] = {name: trading._cv(creds, key) for name, key in (
            ("apiKey", "polymarket_api_key"), ("secret", "polymarket_api_secret"),
            ("passphrase", "polymarket_api_passphrase"))}
        if not all(frame["auth"].values()):
            raise trading.TradingNotConfiguredError("Polymarket stream credentials are incomplete")
        if identifiers:
            frame["markets"] = identifiers
    return f"wss://ws-subscriptions-clob.polymarket.com/ws/{scope}", {}, frame


def _redact(data):
    if isinstance(data, list):
        return [_redact(item) for item in data]
    if isinstance(data, dict):
        return {key: _redact(value) for key, value in data.items()
                if key not in {"auth", "apiKey", "secret", "passphrase", "owner", "order_owner", "trade_owner"}}
    return data


async def _heartbeat(upstream) -> None:
    while True:
        await asyncio.sleep(10)
        await upstream.send("PING")


async def stream(platform: str, scope: str, identifiers: list[str], creds: dict | None = None):
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed

    attrs = {"venue": platform, "scope": scope}
    # Regenerate signed headers on every connection attempt.
    for generation in range(4):
        url, headers, frame = subscription(platform, scope, identifiers, creds)
        try:
            with tracer.start_as_current_span("venue.stream.connect", attributes=attrs):
                upstream = await connect(url, additional_headers=headers, open_timeout=10,
                                         close_timeout=3, max_size=1048576, max_queue=32,
                                         ping_interval=20, ping_timeout=20)
            async with upstream:
                await upstream.send(json.dumps(frame))
                yield {"type": "stream_reset", "platform": platform, "generation": generation,
                       "message": "Discard cached books; apply new snapshots. Reconcile account state after reconnect."}
                heartbeat = asyncio.create_task(_heartbeat(upstream)) if platform == "polymarket" else None
                sequence = {}
                try:
                    async for raw in upstream:
                        if raw == "PONG":
                            continue
                        data = json.loads(raw)
                        if isinstance(data, dict) and data.get("type") == "error":
                            raise market_data.MarketDataError("Exchange rejected stream subscription")
                        if platform == "kalshi" and isinstance(data, dict) and "seq" in data:
                            sid, seq = data.get("sid"), data["seq"]
                            if sid in sequence and seq != sequence[sid] + 1:
                                raise market_data.MarketDataError("Exchange stream sequence gap")
                            sequence[sid] = seq
                        events.add(1, {**attrs, "outcome": "received"})
                        yield {"platform": platform, "generation": generation, "data": _redact(data)}
                finally:
                    if heartbeat:
                        heartbeat.cancel()
                        with suppress(asyncio.CancelledError, ConnectionClosed):
                            await heartbeat
        except (OSError, TimeoutError, ConnectionClosed, ValueError, market_data.MarketDataError):
            events.add(1, {**attrs, "outcome": "disconnected"})
            logger.warning("Venue stream disconnected: %s/%s", platform, scope)
        if generation == 3:
            raise market_data.MarketDataError("Exchange stream unavailable after reconnect attempts")
        yield {"type": "stream_reconnecting", "platform": platform, "generation": generation}
        await asyncio.sleep(min(2 ** generation, 8))
