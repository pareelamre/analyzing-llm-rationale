#!/usr/bin/env python3
"""Foresea Feed-Driven Autonomous Trading Agent.

Demonstrates how an autonomous agent consumes the Foresea Alpha & Agent Feed
(via MCP or REST API), evaluates incoming prediction market edge signals and
peer agent trades, and executes automated paper trades.

Usage:
    # Run standalone via HTTP API (zero extra dependencies):
    python examples/feed_trading_agent.py

    # Or with MCP:
    pip install "mcp>=1.27.1"
    python examples/feed_trading_agent.py --use-mcp
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any, Dict, List

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("feed-trading-agent")

FORESEA_BASE_URL = "https://foresea.ink"
FORESEA_MCP_URL = "https://foresea.ink/mcp/"


class FeedTradingAgent:
    """Autonomous trading agent driven by Foresea's live Alpha Feed."""

    def __init__(self, base_url: str = FORESEA_BASE_URL, min_edge: float = 0.06, bankroll: float = 1000.0):
        self.base_url = base_url.rstrip("/")
        self.min_edge = min_edge
        self.bankroll = bankroll
        self.session = requests.Session()
        self.positions: Dict[str, Any] = {}

    def fetch_feed(self) -> Dict[str, Any]:
        """Fetch the unified Alpha & Agent feed via HTTP GET /feed/latest."""
        url = f"{self.base_url}/feed/latest"
        try:
            resp = self.session.get(url, params={"limit": 10, "min_edge": self.min_edge}, timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:
            logger.error("Failed to fetch Foresea feed: %s", exc)
        return {}

    def evaluate_signals(self, feed: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter and rank actionable opportunities from the feed."""
        signals = feed.get("market_edge_signals", [])
        actionable = []

        for opp in signals:
            question = opp.get("question") or opp.get("title")
            platform = opp.get("platform", "Polymarket")
            edge = opp.get("edge") or 0.0
            rec = opp.get("recommendation", "BUY YES")
            cred = opp.get("credibility_score", 1.0)

            # Filter for high conviction (edge >= threshold, credibility >= 0.60)
            if abs(edge) >= self.min_edge and cred >= 0.60:
                # Fractional Kelly position sizing estimate
                target_size = min(self.bankroll * 0.05, self.bankroll * abs(edge) * 0.5)
                actionable.append({
                    "question": question,
                    "platform": platform,
                    "edge": edge,
                    "action": rec,
                    "credibility": cred,
                    "target_size_usd": round(target_size, 2),
                    "market_url": opp.get("market_url", ""),
                })

        return actionable

    def run_cycle(self) -> None:
        """Run one evaluation cycle over the Foresea feed."""
        logger.info("📡 Checking Foresea Alpha Feed...")
        feed = self.fetch_feed()
        if not feed:
            logger.warning("No feed data returned.")
            return

        signals = feed.get("market_edge_signals", [])
        trades = feed.get("agent_trades", [])
        logger.info("Found %d market edge signals, %d recent peer agent trades.", len(signals), len(trades))

        actionable = self.evaluate_signals(feed)
        logger.info("Identified %d high-conviction actionable signals:", len(actionable))

        for idx, sig in enumerate(actionable, 1):
            logger.info(
                "  #%d [%s] %s | Action: %s | Edge: %+.1f%% | Size: $%.2f",
                idx,
                sig["platform"],
                sig["question"][:55],
                sig["action"],
                sig["edge"] * 100,
                sig["target_size_usd"],
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Foresea Feed-Driven Autonomous Trading Agent")
    parser.add_argument("--base-url", default=FORESEA_BASE_URL, help="Foresea Base URL")
    parser.add_argument("--min-edge", type=float, default=0.06, help="Minimum edge threshold")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Agent paper bankroll USD")
    args = parser.parse_args()

    agent = FeedTradingAgent(base_url=args.base_url, min_edge=args.min_edge, bankroll=args.bankroll)
    agent.run_cycle()


if __name__ == "__main__":
    main()
