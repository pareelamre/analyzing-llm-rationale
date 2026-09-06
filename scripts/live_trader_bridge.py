"""Foresea edge-board simulation bridge.

This script intentionally never submits an exchange order.  It is retained for
local scenario simulation while execution is consolidated behind Foresea's
durable saved-run and guardrail flow.  It has no venue credentials, raw order
HTTP calls, or authority to turn an HTTP acknowledgement into a fill.

Usage:
    # Dry-run scan & simulate order generation:
    python scripts/live_trader_bridge.py --dry-run --min-edge 0.08

    # The old --live flag is intentionally rejected.  Use Foresea's authenticated
    # /trading/runs workflow for a human-confirmed manual order.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("foresea-live-bridge")

DEFAULT_FORESEA_URL = "https://foresea.ink"


class RiskLimits:
    """Local simulation limits; they are not capital authority."""

    def __init__(
        self,
        max_position_usd: float = 50.0,
        min_edge_threshold: float = 0.08,
        max_total_allocation_usd: float = 500.0,
        max_slippage_cents: float = 0.02,
    ):
        self.max_position_usd = max_position_usd
        self.min_edge_threshold = min_edge_threshold
        self.max_total_allocation_usd = max_total_allocation_usd
        self.max_slippage_cents = max_slippage_cents
        self.allocated_usd: float = 0.0

    def can_allocate(self, amount_usd: float) -> bool:
        return (self.allocated_usd + amount_usd) <= self.max_total_allocation_usd

    def record_allocation(self, amount_usd: float) -> None:
        self.allocated_usd += amount_usd


class LiveTraderBridge:
    """Simulation bridge routing Foresea edge signals through no venue writes."""

    def __init__(
        self,
        risk: RiskLimits,
        foresea_url: str = DEFAULT_FORESEA_URL,
        dry_run: bool = True,
        session: Optional[requests.Session] = None,
    ):
        self.risk = risk
        self.foresea_url = foresea_url.rstrip("/")
        self.dry_run = dry_run
        self.session = session or requests.Session()

    def fetch_edge_opportunities(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch ranked mispriced opportunities from Foresea GET /edge-board."""
        url = f"{self.foresea_url}/edge-board"
        try:
            resp = self.session.get(url, params={"limit": limit}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    return data.get("opportunities") or data.get("edge_board") or []
                elif isinstance(data, list):
                    return data
        except Exception as exc:
            logger.error("Failed to fetch edge board from %s: %s", url, exc)
        return []

    def evaluate_opportunity(self, opp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter opportunity against risk guards and construct order intent."""
        model_p = opp.get("model_probability")
        mkt_p = opp.get("market_probability")
        if model_p is None or mkt_p is None:
            return None

        edge = abs(model_p - mkt_p)
        if edge < self.risk.min_edge_threshold:
            return None

        platform = (opp.get("platform") or "kalshi").lower()
        side = "yes" if model_p > mkt_p else "no"
        market_price = mkt_p if side == "yes" else (1.0 - mkt_p)

        # Price sanity check
        if market_price <= 0.02 or market_price >= 0.98:
            return None

        # Position sizing: allocate up to max_position_usd
        alloc_usd = min(self.risk.max_position_usd, 50.0)
        if not self.risk.can_allocate(alloc_usd):
            logger.warning("Allocation cap reached ($%.2f). Skipping trade.", self.risk.allocated_usd)
            return None

        contracts = int(alloc_usd / market_price)
        if contracts < 1:
            return None

        ticker = opp.get("ticker") or opp.get("slug") or opp.get("question", "")

        return {
            "platform": platform,
            "ticker": ticker,
            "side": side,
            "target_price": round(market_price, 2),
            "contracts": contracts,
            "cost_usd": round(contracts * market_price, 2),
            "edge": round(edge, 4),
            "question": opp.get("question", ""),
        }

    def execute_order(self, order_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate an order intent; live execution is permanently disabled here."""
        platform = order_intent["platform"]
        cost = order_intent["cost_usd"]

        if self.dry_run:
            logger.info(
                "[DRY RUN SIMULATION] %s: BUY %s x %d @ $%.2f ($%.2f total) on '%s' (Edge: +%.1f%%)",
                platform.upper(),
                order_intent["side"].upper(),
                order_intent["contracts"],
                order_intent["target_price"],
                cost,
                order_intent["ticker"][:40],
                order_intent["edge"] * 100,
            )
            self.risk.record_allocation(cost)
            return {"status": "simulated", "order": order_intent}

        logger.error("Live execution is disabled in live_trader_bridge; use a saved Foresea trade run.")
        return {
            "status": "blocked",
            "reason": "live_execution_removed_use_trading_runs",
            "order": order_intent,
        }

    def run_cycle(self) -> List[Dict[str, Any]]:
        """Run a single scan, evaluation, and execution cycle."""
        logger.info("Running Foresea Quant Bridge cycle (dry_run=%s)...", self.dry_run)
        opps = self.fetch_edge_opportunities()
        logger.info("Fetched %d opportunities from Foresea Edge Board.", len(opps))
        executed = []
        for opp in opps:
            intent = self.evaluate_opportunity(opp)
            if intent:
                result = self.execute_order(intent)
                executed.append(result)
        return executed


def main() -> None:
    parser = argparse.ArgumentParser(description="Foresea Edge Board Simulation Bridge")
    parser.add_argument("--foresea-url", default=os.getenv("FORESEA_BASE_URL", DEFAULT_FORESEA_URL))
    parser.add_argument("--dry-run", action="store_true", default=True, help="Run local simulation mode (default)")
    parser.add_argument("--live", action="store_true", help="Removed: live execution must use Foresea trade runs")
    parser.add_argument("--min-edge", type=float, default=0.08, help="Minimum model-vs-market edge (default: 0.08)")
    parser.add_argument("--max-position-usd", type=float, default=50.0, help="Max position USD per market")
    parser.add_argument("--max-total-allocation", type=float, default=500.0, help="Total allocation USD cap")
    args = parser.parse_args()
    if args.live:
        parser.error("--live was removed; use Foresea's authenticated /trading/runs workflow instead.")

    risk = RiskLimits(
        max_position_usd=args.max_position_usd,
        min_edge_threshold=args.min_edge,
        max_total_allocation_usd=args.max_total_allocation,
    )
    bridge = LiveTraderBridge(risk=risk, foresea_url=args.foresea_url, dry_run=args.dry_run)
    results = bridge.run_cycle()
    logger.info("Cycle completed with %d orders processed.", len(results))


if __name__ == "__main__":
    main()
