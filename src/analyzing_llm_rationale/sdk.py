"""Foresea Official Python SDK.

Modern, typed client for integrating Foresea forecasting intelligence,
cross-venue prediction market data, adversarial debates, and quant execution.

Usage:
    from analyzing_llm_rationale.sdk import Foresea

    client = Foresea(base_url="https://foresea.ink")

    # 1. Calibrated Forecast
    fc = client.forecast("Will the Fed cut interest rates in September?")
    print(fc["predicted_answer"], fc["confidence"])

    # 2. Multi-Agent Bull vs Bear Debate
    debate = client.debate("Will Ethereum reach $5000 in 2026?")
    print("Judge Verdict:", debate["chief_risk_judge"]["verdict"])

    # 3. Live Radar Desk Opportunities
    radar = client.radar(limit=5)

    # 4. Cross-Venue Arbitrage Scanner
    arbs = client.arbitrage(min_spread=0.04)

    # 5. Optimal Kelly Portfolio Allocation
    alloc = client.optimize_portfolio(bankroll_usd=5000.0, kelly_fraction=0.25)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterator, List, Optional

import requests

logger = logging.getLogger("foresea-sdk")


class Foresea:
    """Synchronous Foresea API Client."""

    def __init__(
        self,
        base_url: str = "https://foresea.ink",
        api_key: Optional[str] = None,
        timeout: float = 20.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Foresea-Python-SDK/1.0",
            "Accept": "application/json",
        })
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    def forecast(
        self,
        question: str,
        platform: Optional[str] = None,
        market_probability: Optional[float] = None,
        market_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a calibrated forecast with live news evidence retrieval."""
        url = f"{self.base_url}/predict"
        payload = {
            "question": question,
            "market_platform": platform,
            "market_probability": market_probability,
            "market_url": market_url,
            "model": model,
        }
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def debate(
        self,
        question: str,
        platform: str = "Market",
        market_probability: Optional[float] = None,
        resolution_criteria: str = "",
    ) -> Dict[str, Any]:
        """Conduct an adversarial Bull vs. Bear debate and blind-spot audit."""
        url = f"{self.base_url}/agent/debate"
        payload = {
            "question": question,
            "platform": platform,
            "market_probability": market_probability,
            "resolution_criteria": resolution_criteria,
        }
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def radar(self, limit: int = 10) -> Dict[str, Any]:
        """Fetch live mispriced opportunities from the Foresea Radar Desk."""
        url = f"{self.base_url}/radar"
        resp = self.session.get(url, params={"limit": limit}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def arbitrage(self, min_spread: float = 0.03) -> List[Dict[str, Any]]:
        """Scan for synthetic arbitrage and price divergence between Polymarket & Kalshi."""
        url = f"{self.base_url}/v1/arbitrage/cross-venue"
        resp = self.session.get(url, params={"min_spread": min_spread}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("opportunities", [])

    def optimize_portfolio(
        self,
        bankroll_usd: float = 1000.0,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.05,
    ) -> Dict[str, Any]:
        """Calculate mathematically optimal Fractional Kelly position sizing."""
        url = f"{self.base_url}/portfolio/optimal-allocation"
        params = {
            "bankroll": bankroll_usd,
            "kelly_fraction": kelly_fraction,
            "min_edge": min_edge,
        }
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def orderbook(self, platform: str = "kalshi", ident: str = "") -> Dict[str, Any]:
        """Fetch live orderbook bids and asks for a market."""
        url = f"{self.base_url}/market/orderbook"
        params = {"platform": platform, "ident": ident}
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def trades(self, platform: str = "kalshi", ident: str = "", limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch recent executed public trades / trade tape."""
        url = f"{self.base_url}/market/trades"
        params = {"platform": platform, "ident": ident, "limit": limit}
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("trades", [])

    def whale_flow(self, min_notional_usd: float = 250.0, limit: int = 30) -> Dict[str, Any]:
        """Track large block trades and net smart-money flow."""
        url = f"{self.base_url}/v1/market/whale-flow"
        params = {"min_notional": min_notional_usd, "limit": limit}
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def system_health(self) -> Dict[str, Any]:
        """Fetch venue and system latency health metrics."""
        url = f"{self.base_url}/v1/system/health"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def stream_radar(self) -> Iterator[Dict[str, Any]]:
        """Stream real-time market ticks via Server-Sent Events (SSE)."""
        url = f"{self.base_url}/stream/radar"
        resp = self.session.get(url, stream=True, timeout=60.0)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data:"):
                    raw_data = decoded[5:].strip()
                    try:
                        yield json.loads(raw_data)
                    except Exception:
                        pass


class AsyncForesea:
    """Asynchronous Foresea API Client."""

    def __init__(
        self,
        base_url: str = "https://foresea.ink",
        api_key: Optional[str] = None,
        timeout: float = 20.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def forecast(self, question: str, **kwargs) -> Dict[str, Any]:
        import asyncio
        loop = asyncio.get_running_loop()
        sync_client = Foresea(self.base_url, self.api_key, self.timeout)
        return await loop.run_in_executor(None, lambda: sync_client.forecast(question, **kwargs))

    async def debate(self, question: str, **kwargs) -> Dict[str, Any]:
        import asyncio
        loop = asyncio.get_running_loop()
        sync_client = Foresea(self.base_url, self.api_key, self.timeout)
        return await loop.run_in_executor(None, lambda: sync_client.debate(question, **kwargs))

    async def radar(self, limit: int = 10) -> Dict[str, Any]:
        import asyncio
        loop = asyncio.get_running_loop()
        sync_client = Foresea(self.base_url, self.api_key, self.timeout)
        return await loop.run_in_executor(None, lambda: sync_client.radar(limit))

    async def arbitrage(self, min_spread: float = 0.03) -> List[Dict[str, Any]]:
        import asyncio
        loop = asyncio.get_running_loop()
        sync_client = Foresea(self.base_url, self.api_key, self.timeout)
        return await loop.run_in_executor(None, lambda: sync_client.arbitrage(min_spread))

    async def optimize_portfolio(self, bankroll_usd: float = 1000.0, **kwargs) -> Dict[str, Any]:
        import asyncio
        loop = asyncio.get_running_loop()
        sync_client = Foresea(self.base_url, self.api_key, self.timeout)
        return await loop.run_in_executor(None, lambda: sync_client.optimize_portfolio(bankroll_usd, **kwargs))
