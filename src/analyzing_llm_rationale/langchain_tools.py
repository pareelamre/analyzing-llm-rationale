"""LangChain & LangGraph Tool integrations for Foresea.

Provides ready-to-use BaseTool instances that can be passed directly to
LangChain ChatModels and ReAct agents for prediction market intelligence,
probability forecasting, and Polymarket/Kalshi edge detection.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import requests
from pydantic import BaseModel, Field

DEFAULT_FORESEA_URL = "https://foresea.ink"


class ForecastInput(BaseModel):
    question: str = Field(
        ...,
        description="The event question to forecast (e.g. 'Will SpaceX launch Starship orbital flight before December 2026?').",
    )
    market_url: Optional[str] = Field(
        None,
        description="Optional Polymarket or Kalshi URL to compare model predictions against live orderbooks.",
    )
    market_price: Optional[float] = Field(
        None,
        description="Optional current market probability between 0.0 and 1.0 (e.g. 0.35 for 35%).",
    )


class ScanMarketsInput(BaseModel):
    query: Optional[str] = Field(
        None,
        description="Optional keyword filter (e.g. 'fed', 'bitcoin', 'election', 'spacex').",
    )
    platform: Optional[str] = Field(
        None,
        description="Filter by venue: 'polymarket' or 'kalshi'. Leave empty for all.",
    )
    limit: int = Field(
        5,
        description="Max number of markets to return (default 5, max 25).",
    )


class EdgeBoardInput(BaseModel):
    min_edge: float = Field(
        0.05,
        description="Minimum absolute difference between model probability and market price (e.g. 0.06 for 6%).",
    )
    limit: int = Field(
        5,
        description="Max number of mispriced opportunities to return (default 5, max 20).",
    )


class FeedInput(BaseModel):
    limit: int = Field(
        5,
        description="Max items to retrieve from latest alpha stream.",
    )
    min_edge: float = Field(
        0.04,
        description="Minimum edge filter for market signals.",
    )


class ForeseaClient:
    """Lightweight HTTP client for Foresea API endpoints."""

    def __init__(self, base_url: str = DEFAULT_FORESEA_URL, timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def forecast(
        self,
        question: str,
        market_url: Optional[str] = None,
        market_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"question": question}
        if market_url:
            payload["market_url"] = market_url
        if market_price is not None:
            payload["market_price"] = market_price

        resp = requests.post(f"{self.base_url}/predict", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def scan_markets(
        self,
        query: Optional[str] = None,
        platform: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit}
        if query:
            params["q"] = query
        if platform:
            params["platform"] = platform

        resp = requests.get(f"{self.base_url}/api/markets/scan", params=params, timeout=self.timeout)
        if resp.status_code == 404:
            # Fallback to general market scan or edge board
            resp = requests.get(f"{self.base_url}/edge-board", params={"limit": limit}, timeout=self.timeout)
        resp.raise_for_status()
        d = resp.json()
        return d.get("markets") or d.get("opportunities") or []

    def get_edge_board(self, min_edge: float = 0.05, limit: int = 5) -> List[Dict[str, Any]]:
        params = {"min_edge": min_edge, "limit": limit}
        resp = requests.get(f"{self.base_url}/edge-board", params=params, timeout=self.timeout)
        resp.raise_for_status()
        d = resp.json()
        return d.get("edge_board") or d.get("opportunities") or []

    def get_feed_latest(self, limit: int = 5, min_edge: float = 0.04) -> Dict[str, Any]:
        params = {"limit": limit, "min_edge": min_edge}
        resp = requests.get(f"{self.base_url}/feed/latest", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


# LangChain Structured Tools (graceful fallback if langchain is not installed)
try:
    from langchain_core.tools import BaseTool  # type: ignore

    class ForeseaForecastTool(BaseTool):
        name: str = "foresea_forecast"
        description: str = (
            "Generate a calibrated probability forecast for any future event. "
            "Returns fair probability, written rationale, evidence sources, and model-vs-market edge."
        )
        args_schema: Type[BaseModel] = ForecastInput
        client: Any = Field(default_factory=ForeseaClient)

        def _run(

            self,
            question: str,
            market_url: Optional[str] = None,
            market_price: Optional[float] = None,
        ) -> str:
            try:
                data = self.client.forecast(question, market_url, market_price)
                prob = data.get("predicted_probability")
                prob_str = f"{prob*100:.1f}%" if prob is not None else "N/A"
                ans = data.get("predicted_answer", "UNKNOWN")
                rationale = data.get("rationale", "")
                edge = data.get("model_vs_market_edge")
                edge_str = f" | Model Edge: {edge*100:+.1f}%" if edge is not None else ""

                return (
                    f"Foresea Calibrated Forecast:\n"
                    f"• Probability: {prob_str} ({ans.upper()}){edge_str}\n"
                    f"• Rationale: {rationale}"
                )
            except Exception as e:
                return f"Error running Foresea forecast: {e}"

    class ForeseaEdgeBoardTool(BaseTool):
        name: str = "foresea_edge_board"
        description: str = (
            "Scan Polymarket and Kalshi for mispriced opportunities where Foresea's "
            "calibrated model disagrees significantly with the current market price."
        )
        args_schema: Type[BaseModel] = EdgeBoardInput
        client: Any = Field(default_factory=ForeseaClient)

        def _run(self, min_edge: float = 0.05, limit: int = 5) -> str:
            try:
                opps = self.client.get_edge_board(min_edge=min_edge, limit=limit)
                if not opps:
                    return f"No prediction market mispricings found with >= {min_edge*100:.0f}% edge."

                out = [f"Found {len(opps)} prediction market opportunities:"]
                for i, o in enumerate(opps, 1):
                    q = o.get("question", "Unknown")
                    platform = o.get("platform", "Venue")
                    edge = o.get("edge", 0)
                    mkt = o.get("market_probability", 0)
                    model = o.get("model_probability", 0)
                    rec = o.get("recommendation", "BUY")
                    out.append(
                        f"{i}. [{platform}] {q}\n"
                        f"   Action: {rec} | Market: {mkt*100:.0f}% vs Foresea: {model*100:.0f}% ({edge*100:+.1f}% Edge)"
                    )
                return "\n".join(out)
            except Exception as e:
                return f"Error scanning edge board: {e}"

    class ForeseaFeedTool(BaseTool):
        name: str = "foresea_alpha_feed"
        description: str = (
            "Stream real-time prediction market alpha signals, live autonomous agent trade executions, "
            "and active community broadcast updates from Foresea."
        )
        args_schema: Type[BaseModel] = FeedInput
        client: Any = Field(default_factory=ForeseaClient)


        def _run(self, limit: int = 5, min_edge: float = 0.04) -> str:
            try:
                data = self.client.get_feed_latest(limit=limit, min_edge=min_edge)
                signals = data.get("market_edge_signals", [])
                trades = data.get("agent_trades", [])

                lines = ["Foresea Live Alpha Stream:"]
                if signals:
                    lines.append("\nTop Mispricings:")
                    for s in signals[:limit]:
                        edge = s.get("edge", 0)
                        lines.append(f"• [{s.get('platform')}] {s.get('question')} -> {s.get('recommendation')} ({edge*100:+.1f}% Edge)")
                if trades:
                    lines.append("\nRecent Autonomous Agent Trades:")
                    for t in trades[:limit]:
                        lines.append(f"• [{t.get('model')}] {t.get('action')} {t.get('side')} on '{t.get('question')}' (Confidence: {t.get('confidence', 0)*100:.0f}%)")

                return "\n".join(lines)
            except Exception as e:
                return f"Error retrieving alpha feed: {e}"

    def get_foresea_langchain_tools(base_url: str = DEFAULT_FORESEA_URL) -> List[BaseTool]:
        """Convenience function returning all Foresea tools for LangChain agent binding."""
        client = ForeseaClient(base_url=base_url)
        return [
            ForeseaForecastTool(client=client),
            ForeseaEdgeBoardTool(client=client),
            ForeseaFeedTool(client=client),
        ]

except ImportError:
    # Graceful degradation if langchain_core is not in environment
    ForeseaForecastTool = None  # type: ignore
    ForeseaEdgeBoardTool = None  # type: ignore
    ForeseaFeedTool = None  # type: ignore

    def get_foresea_langchain_tools(base_url: str = DEFAULT_FORESEA_URL) -> List[Any]:  # type: ignore
        raise ImportError("Please install langchain-core (`pip install langchain-core`) to use Foresea LangChain tools.")
