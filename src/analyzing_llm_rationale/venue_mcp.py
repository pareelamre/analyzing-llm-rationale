"""Foresea as an MCP *client* of prediction-market venues (Polymarket / Kalshi).

When ``POLYMARKET_MCP_URL`` / ``KALSHI_MCP_URL`` are set, Foresea's agent loop can
call those venues' own MCP servers as extra tools — for richer live data
(orderbook, depth, recent trades) than the direct `market_data` quotes, and an
execution path later.

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
import os
from typing import Any, Awaitable, Callable, Dict, List, Tuple

_VENUE_ENV = {"polymarket": "POLYMARKET_MCP_URL", "kalshi": "KALSHI_MCP_URL"}
_TIMEOUT_S = float(os.environ.get("VENUE_MCP_TIMEOUT_S", "20"))
_MAX_OUTPUT = int(os.environ.get("VENUE_MCP_MAX_OUTPUT", "4000"))
_MAX_TOOLS_PER_VENUE = int(os.environ.get("VENUE_MCP_MAX_TOOLS", "8"))


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
    Best-effort: a venue that fails to list is skipped."""
    out: Dict[str, Dict[str, str]] = {}
    for prefix, url in configured_venues():
        try:
            tools = await asyncio.wait_for(_list_tools(url), _TIMEOUT_S)
        except Exception:
            continue
        for name, desc in tools[:_MAX_TOOLS_PER_VENUE]:
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
