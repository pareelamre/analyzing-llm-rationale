"""Foresea as an MCP *client* of prediction-market venues (Polymarket / Kalshi).

When ``POLYMARKET_MCP_URL`` / ``KALSHI_MCP_URL`` are set, Foresea's agent loop can
call those venues' own MCP servers as extra tools — for richer live data
(orderbook, depth, recent trades) than the direct `market_data` quotes.

This is a read-only context source, not an execution path — Foresea's real order
placement always goes through trading.py's guardrail chain
(_validate_live_trade_guardrails, the kill switch, CONFIRMATION_PHRASE), never
through a discovered venue tool. A configured venue server is untrusted
third-party code we don't control, and several public Kalshi/Polymarket MCP
servers bundle trading tools alongside read-only ones, so discover_tools() drops
anything whose name suggests a write/trading action before it can reach the
agent loop at all (see _is_write_tool). A clear read prefix ("get_orderbook",
"list_trades") is always allowed even though "order"/"trade" are otherwise
ambiguous; an unprefixed name containing an exact write-verb token (e.g. a bare
"close_position") is excluded even if it might be a read in a given server's
vocabulary — deliberately biased toward over-excluding an ambiguous unprefixed
name over ever letting a real trading tool through.

Strictly additive and best-effort:
- Unset or unreachable venues are silently skipped (the loop is unchanged).
- Venue tool output is **untrusted context**: truncated, and only ever fed back to
  the model as text — never executed. Foresea's own forecast stays the source of
  truth (guards against prompt-injection via venue responses).

The ``mcp`` SDK (Python 3.10+) is imported lazily so this module imports anywhere;
if the SDK is missing, discovery just yields nothing.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

_VENUE_ENV = {"polymarket": "POLYMARKET_MCP_URL", "kalshi": "KALSHI_MCP_URL"}
_TIMEOUT_S = float(os.environ.get("VENUE_MCP_TIMEOUT_S", "20"))
_MAX_OUTPUT = int(os.environ.get("VENUE_MCP_MAX_OUTPUT", "4000"))
_MAX_TOOLS_PER_VENUE = int(os.environ.get("VENUE_MCP_MAX_TOOLS", "8"))

# A configured venue server is untrusted third-party code we don't control, and
# several public Kalshi/Polymarket MCP servers bundle trading tools alongside
# read-only ones. This is read-only-context augmentation, not an execution path
# -- any tool whose name suggests a write/trading action is dropped before it
# can ever reach the agent loop, regardless of what a given server advertises.
#
# A tool clearly named as a read (starts with one of _READ_PREFIXES, e.g.
# "get_orderbook") is always allowed, even though "order" would otherwise be
# ambiguous -- order book depth and recent trades are exactly the data this
# module exists to surface. Anything else is excluded if any name token is
# exactly a write verb. This is deliberately biased toward over-excluding an
# unprefixed, ambiguous name over ever letting a real trading tool through.
_READ_PREFIXES = ("get", "fetch", "list", "read", "view", "check", "query", "show", "describe", "search")
_WRITE_VERBS = frozenset((
    "order", "trade", "buy", "sell", "cancel", "withdraw", "transfer",
    "deposit", "close", "place", "execute", "submit", "create", "modify", "amend",
))
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(name: str) -> List[str]:
    # Splits on non-alphanumeric separators (_, -, space) and camelCase boundaries.
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return _TOKEN_RE.findall(spaced.lower())


def _is_write_tool(name: str) -> bool:
    tokens = _tokens(name)
    if tokens and tokens[0] in _READ_PREFIXES:
        return False
    return any(tok in _WRITE_VERBS for tok in tokens)


def configured_venues() -> List[Tuple[str, str]]:
    """[(prefix, url)] for venues with an MCP URL configured."""
    out: List[Tuple[str, str]] = []
    for prefix, env in _VENUE_ENV.items():
        url = os.environ.get(env)
        if url:
            out.append((prefix, url.rstrip("/")))
    return out


# -- SDK-touching primitives (mocked in tests) --

async def _list_tools(url: str) -> List[Tuple[str, str]]:
    from mcp import ClientSession  # noqa: PLC0415
    from mcp.client.streamable_http import streamablehttp_client  # noqa: PLC0415
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.list_tools()
            return [(t.name, (t.description or "")) for t in res.tools]


async def _call_tool(url: str, name: str, args: Dict[str, Any]) -> str:
    from mcp import ClientSession  # noqa: PLC0415
    from mcp.client.streamable_http import streamablehttp_client  # noqa: PLC0415
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(name, args or {})
            parts = [getattr(c, "text", "") or "" for c in (getattr(res, "content", None) or [])]
            return "\n".join(p for p in parts if p)


# -- orchestration --

async def discover_tools() -> Dict[str, Dict[str, str]]:
    """{namespaced_name: {url, name, description}} across all configured venues.
    Best-effort: a venue that fails to list is skipped. Tools whose name suggests
    a write/trading action are dropped -- see _is_write_tool."""
    out: Dict[str, Dict[str, str]] = {}
    for prefix, url in configured_venues():
        try:
            tools = await asyncio.wait_for(_list_tools(url), _TIMEOUT_S)
        except Exception:
            continue
        read_only = []
        for name, desc in tools:
            if _is_write_tool(name):
                logger.warning(
                    "venue MCP tool excluded (write/trading name): venue=%s tool=%s", prefix, name
                )
                continue
            read_only.append((name, desc))
        for name, desc in read_only[:_MAX_TOOLS_PER_VENUE]:
            out[f"{prefix}_{name}"] = {"url": url, "name": name,
                                       "description": f"[{prefix}] {desc}".strip()}
    return out


async def call_tool_safe(url: str, name: str, args: Dict[str, Any]) -> str:
    """Call a venue tool with a timeout; never raise — return text or an error note,
    truncated to keep untrusted venue output bounded."""
    try:
        text = await asyncio.wait_for(_call_tool(url, name, args), _TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001
        return f"(venue tool {name} unavailable: {type(exc).__name__})"
    return (text or "(no content)")[:_MAX_OUTPUT]


def make_tool_fn(url: str, name: str) -> Callable[[Dict[str, Any]], Awaitable[str]]:
    """Build an agent-loop tool fn (takes an args dict) bound to one venue tool."""
    async def _fn(args: Dict[str, Any]) -> str:
        return await call_tool_safe(url, name, args if isinstance(args, dict) else {})
    return _fn
