#!/usr/bin/env python3
"""Minimal agent demo: talk to Foresea's public MCP server end to end.

Scans Polymarket for the biggest model-vs-market disagreement, then forecasts
that market and prints the edge — the same loop an autonomous agent would run.

No API key (Foresea's MCP server is anonymous). Requires the MCP SDK (Python 3.10+):

    pip install "mcp>=1.27.1"
    python examples/foresea_agent_demo.py
"""
from __future__ import annotations

import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

FORESEA_MCP = "https://foresea.ink/mcp/"


def _text(result) -> str:
    return "\n".join(getattr(c, "text", "") or "" for c in (result.content or []))


async def main() -> None:
    async with streamablehttp_client(FORESEA_MCP) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = [t.name for t in (await session.list_tools()).tools]
            print("Foresea tools:", tools)

            # 1) Find the most mispriced live market.
            scan = json.loads(_text(await session.call_tool(
                "foresea_scan_markets", {"platform": "polymarket", "limit": 4, "min_edge": 0.05})))
            opps = scan.get("opportunities") or []
            if not opps:
                print("No edges found right now.")
                return
            top = max(opps, key=lambda o: abs(o.get("edge") or 0))
            print(f"\nTop edge: {top['question']!r}\n  model={top['model_probability']} "
                  f"market={top['market_probability']} edge={top['edge']:+}")

            # 2) Re-forecast it explicitly and show the structured result.
            fc = json.loads(_text(await session.call_tool("foresea_forecast", {
                "question": top["question"],
                "market_url": top.get("market_url"),
                "market_probability": top.get("market_probability"),
            })))
            ma = fc.get("market_analysis") or {}
            print(f"\nForecast: {fc.get('predicted_answer')} @ {fc.get('confidence')}")
            print(f"  edge vs market: {ma.get('edge')} ({ma.get('stance')})")
            print(f"  rationale: {(fc.get('model_rationale') or '')[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
