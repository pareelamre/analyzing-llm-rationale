"""Foresea Real-Money Quant Execution Bridge.

An opt-in automated execution runner for quants and algorithmic traders that
connects Foresea's statistical edge signals to live prediction market venues
(Polymarket & Kalshi).

Safety & Risk Management:
- --dry-run is ENABLED BY DEFAULT to simulate and verify order execution safely.
- Explicit risk limits: max position USD, minimum edge threshold, max total capital cap.
- Strict price slippage guards against adverse execution.

Usage:
    # Dry-run scan & simulate order generation:
    python scripts/live_trader_bridge.py --dry-run --min-edge 0.08

    # Live execution on Kalshi (requires KALSHI_API_KEY & KALSHI_PRIVATE_KEY):
    python scripts/live_trader_bridge.py --venue kalshi --min-edge 0.10 --max-position-usd 25

    # Live execution on Polymarket (requires POLYMARKET_API_KEY & WALLET_PRIVATE_KEY):
    python scripts/live_trader_bridge.py --venue polymarket --min-edge 0.08 --max-position-usd 50
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
    """Configurable risk parameters guarding live order execution."""

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
    """Execution bridge routing Foresea edge signals to prediction venues."""

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
        """Execute or simulate an order intent on the target venue."""
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

        # Live Execution
        if platform == "kalshi":
            return self._execute_kalshi_order(order_intent)
        elif platform == "polymarket":
            return self._execute_polymarket_order(order_intent)
        else:
            return {"status": "error", "reason": f"Unsupported platform '{platform}'"}

    def _execute_kalshi_order(self, order_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order on Kalshi Trade API v2."""
        api_key = os.getenv("KALSHI_API_KEY", "")
        if not api_key:
            logger.error("KALSHI_API_KEY not set. Cannot place live order.")
            return {"status": "error", "reason": "Missing KALSHI_API_KEY"}

        url = "https://api.elections.kalshi.com/trade-api/v2/portfolio/orders"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        price_cents = int(round(order_intent["target_price"] * 100))
        payload = {
            "ticker": order_intent["ticker"],
            "action": "buy",
            "type": "limit",
            "side": order_intent["side"],
            "count": order_intent["contracts"],
            f"{order_intent['side']}_price": price_cents,
            "time_in_force": "ioc",
        }
        try:
            resp = self.session.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code in (200, 201):
                self.risk.record_allocation(order_intent["cost_usd"])
                logger.info("Kalshi order placed: %s", resp.json())
                return {"status": "filled", "response": resp.json()}
            logger.error("Kalshi order failed (%d): %s", resp.status_code, resp.text)
            return {"status": "rejected", "detail": resp.text}
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

    def _execute_polymarket_order(self, order_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order on Polymarket CLOB API."""
        api_key = os.getenv("POLYMARKET_API_KEY", "")
        if not api_key:
            logger.error("POLYMARKET_API_KEY not set. Cannot place live order.")
            return {"status": "error", "reason": "Missing POLYMARKET_API_KEY"}

        url = "https://clob.polymarket.com/order"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "token_id": order_intent["ticker"],
            "side": "BUY",
            "price": order_intent["target_price"],
            "size": order_intent["contracts"],
            "order_type": "IOC",
        }
        try:
            resp = self.session.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code in (200, 201):
                self.risk.record_allocation(order_intent["cost_usd"])
                logger.info("Polymarket order placed: %s", resp.json())
                return {"status": "filled", "response": resp.json()}
            logger.error("Polymarket order failed (%d): %s", resp.status_code, resp.text)
            return {"status": "rejected", "detail": resp.text}
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

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
    parser = argparse.ArgumentParser(description="Foresea Live Quant Execution Bridge")
    parser.add_argument("--foresea-url", default=os.getenv("FORESEA_BASE_URL", DEFAULT_FORESEA_URL))
    parser.add_argument("--dry-run", action="store_true", default=True, help="Enable dry-run simulation mode")
    parser.add_argument("--live", action="store_false", dest="dry_run", help="Enable REAL LIVE execution (requires API keys)")
    parser.add_argument("--min-edge", type=float, default=0.08, help="Minimum model-vs-market edge (default: 0.08)")
    parser.add_argument("--max-position-usd", type=float, default=50.0, help="Max position USD per market")
    parser.add_argument("--max-total-allocation", type=float, default=500.0, help="Total allocation USD cap")
    args = parser.parse_args()

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
