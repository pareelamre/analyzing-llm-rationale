#!/usr/bin/env python3
"""A prediction-market trading agent that uses Foresea as its fair-value oracle.

The pattern any Polymarket/Kalshi trading bot wants: before placing a bet, get a
calibrated fair value + the edge vs the current price — and size the position by
the edge, trusting it more where Foresea's *resolved* track record says
disagreements that size have actually paid.

Foresea's `foresea_edge_board` already returns live disagreements with the trade
direction, entry price, payout odds, and a `track_record.skill_significant` flag,
so the agent is mostly sizing + filtering. **Paper only** — no real orders, no
fees/slippage; this is a fair-value + sizing demo, not financial advice.

    pip install "mcp>=1.27.1"
    python examples/polymarket_trading_agent.py
"""
from __future__ import annotations

import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

FORESEA_MCP = "https://foresea.ink/mcp/"
BANKROLL = 1000.0      # paper units
MAX_STAKE_FRAC = 0.10  # never risk more than 10% of bankroll on one bet
MIN_EDGE = 0.05        # ignore disagreements smaller than 5 percentage points


def _text(result) -> str:
    return "\n".join(getattr(c, "text", "") or "" for c in (result.content or []))


def _size(edge: float) -> float:
    """Edge-proportional stake, capped. (A real agent would use fractional Kelly.)"""
    return round(min(abs(edge), MAX_STAKE_FRAC) * BANKROLL, 2)


async def main() -> None:
    async with streamablehttp_client(FORESEA_MCP) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            board = json.loads(_text(await session.call_tool("foresea_edge_board", {})))

    rows = [r for r in board.get("edge_board", []) if abs(r.get("edge") or 0) >= MIN_EDGE]
    if not rows:
        print("No actionable edges right now.")
        return

    print(f"Foresea fair-value oracle — paper book (bankroll {BANKROLL:.0f}):\n")
    staked = 0.0
    for r in rows:
        tr = r.get("track_record") or {}
        proven = tr.get("skill_significant")
        tag = "PROVEN" if proven else "speculative"   # only the proven bucket is high-conviction
        stake = _size(r["edge"]) * (1.0 if proven else 0.5)  # half-size unproven edges
        staked += stake
        odds = f"~{r['payout_odds']}:1" if r.get("payout_odds") else ""
        print(f"  Buy {r['side']:3} @ {round((r.get('entry_price') or 0)*100):>2}¢ {odds:>7} "
              f"| stake {stake:7.2f} | edge {r['edge']:+.3f} [{tag}] "
              f"| {(r.get('question') or '')[:50]}")
    print(f"\nTotal paper exposure: {staked:.2f} / {BANKROLL:.0f}  "
          f"({len(rows)} positions). Paper only — not financial advice.")


if __name__ == "__main__":
    asyncio.run(main())
