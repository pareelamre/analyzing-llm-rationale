"""Langflow Custom Component for Foresea Prediction Markets.

Drag-and-drop this component into Langflow to connect any visual agent flow
to Foresea calibrated probability forecasts and Polymarket/Kalshi edge detection.
"""
from typing import Any, Dict, Optional

# Langflow CustomComponent signature
try:
    from langflow.custom import Component  # type: ignore
    from langflow.io import DropdownInput, FloatInput, IntInput, Output, StrInput  # type: ignore
    from langflow.schema import Data  # type: ignore
except ImportError:
    Component = object  # type: ignore
    DropdownInput = StrInput = FloatInput = IntInput = Output = Data = None  # type: ignore

from analyzing_llm_rationale.langchain_tools import ForeseaClient


class ForeseaPredictionMarketComponent(Component):
    display_name = "Foresea Prediction Markets"
    description = "Calibrated probability forecasting, Polymarket & Kalshi edge detection, and live alpha feed."
    icon = "trending-up"

    inputs = [
        DropdownInput(
            name="action",
            display_name="Action",
            options=["Forecast Question", "Scan Edge Board", "Get Alpha Feed"],
            value="Forecast Question",
            info="Select the prediction market operation.",
        ),
        StrInput(
            name="question",
            display_name="Question / Topic",
            placeholder="e.g. Will SpaceX complete a Starship orbital flight in 2026?",
            info="The prediction question to evaluate (for 'Forecast Question').",
        ),
        FloatInput(
            name="min_edge",
            display_name="Minimum Edge Filter",
            value=0.05,
            info="Minimum probability discrepancy for market mispricings.",
        ),
        IntInput(
            name="limit",
            display_name="Limit",
            value=5,
            info="Max number of items to return.",
        ),
    ]

    outputs = [
        Output(display_name="Result Text", name="result_text", method="process_text"),
        Output(display_name="Structured Data", name="result_data", method="process_data"),
    ]

    def process_text(self) -> str:
        client = ForeseaClient()
        action = getattr(self, "action", "Forecast Question")
        question = getattr(self, "question", "")
        min_edge = getattr(self, "min_edge", 0.05)
        limit = getattr(self, "limit", 5)

        if action == "Forecast Question":
            if not question:
                return "Please provide a question to forecast."
            res = client.forecast(question=question)
            prob = res.get("predicted_probability", 0)
            ans = res.get("predicted_answer", "UNKNOWN")
            return f"Probability: {prob*100:.1f}% ({ans.upper()})\nRationale: {res.get('rationale', '')}"

        elif action == "Scan Edge Board":
            opps = client.get_edge_board(min_edge=min_edge, limit=limit)
            if not opps:
                return "No mispricings found meeting edge criteria."
            lines = [f"Found {len(opps)} opportunities:"]
            for o in opps:
                lines.append(f"• [{o.get('platform')}] {o.get('question')} (Edge: {o.get('edge', 0)*100:+.1f}%)")
            return "\n".join(lines)

        else:
            feed = client.get_feed_latest(limit=limit, min_edge=min_edge)
            signals = feed.get("market_edge_signals", [])
            lines = [f"Alpha Feed ({len(signals)} items):"]
            for s in signals:
                lines.append(f"• {s.get('question')} -> {s.get('recommendation')}")
            return "\n".join(lines)

    def process_data(self) -> Any:
        client = ForeseaClient()
        action = getattr(self, "action", "Forecast Question")
        question = getattr(self, "question", "")
        min_edge = getattr(self, "min_edge", 0.05)
        limit = getattr(self, "limit", 5)

        if action == "Forecast Question":
            return client.forecast(question=question) if question else {}
        elif action == "Scan Edge Board":
            return client.get_edge_board(min_edge=min_edge, limit=limit)
        else:
            return client.get_feed_latest(limit=limit, min_edge=min_edge)
