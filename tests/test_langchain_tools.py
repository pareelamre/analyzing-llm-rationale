"""Unit tests for Foresea LangChain integration tools."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from analyzing_llm_rationale.langchain_tools import (
    EdgeBoardInput,
    FeedInput,
    ForecastInput,
    ForeseaClient,
    ForeseaForecastTool,
    get_foresea_langchain_tools,
)


class TestLangChainTools(unittest.TestCase):

    def test_input_schemas(self):
        f_in = ForecastInput(question="Will X happen?", market_price=0.45)
        self.assertEqual(f_in.question, "Will X happen?")
        self.assertEqual(f_in.market_price, 0.45)

        e_in = EdgeBoardInput(min_edge=0.06, limit=10)
        self.assertEqual(e_in.min_edge, 0.06)
        self.assertEqual(e_in.limit, 10)

        feed_in = FeedInput(limit=3, min_edge=0.02)
        self.assertEqual(feed_in.limit, 3)

    @patch("requests.post")
    def test_forecast_client(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "predicted_probability": 0.65,
            "predicted_answer": "yes",
            "rationale": "High launch cadence.",
            "model_vs_market_edge": 0.15,
        }

        client = ForeseaClient(base_url="https://foresea.test")
        res = client.forecast("Will Starship reach orbit?")
        self.assertEqual(res["predicted_probability"], 0.65)
        self.assertEqual(res["predicted_answer"], "yes")

    @patch("requests.get")
    def test_edge_board_client(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "edge_board": [
                {
                    "question": "Sample question",
                    "platform": "polymarket",
                    "edge": 0.12,
                    "market_probability": 0.20,
                    "model_probability": 0.32,
                    "recommendation": "BUY YES",
                }
            ]
        }

        client = ForeseaClient(base_url="https://foresea.test")
        opps = client.get_edge_board(min_edge=0.05)
        self.assertEqual(len(opps), 1)
        self.assertEqual(opps[0]["platform"], "polymarket")

    @patch("requests.get")
    def test_feed_client(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "market_edge_signals": [{"question": "Q1", "platform": "kalshi", "edge": 0.08}],
            "agent_trades": [],
        }

        client = ForeseaClient(base_url="https://foresea.test")
        feed = client.get_feed_latest()
        self.assertEqual(len(feed["market_edge_signals"]), 1)

    def test_langchain_tools_instantiation(self):
        if ForeseaForecastTool is None:
            self.skipTest("langchain_core not installed")

        client = MagicMock()
        client.forecast.return_value = {
            "predicted_probability": 0.70,
            "predicted_answer": "yes",
            "rationale": "Solid base rate.",
            "model_vs_market_edge": 0.20,
        }

        tool = ForeseaForecastTool(client=client)
        output = tool.run({"question": "Test question"})
        self.assertIn("70.0%", output)
        self.assertIn("Solid base rate", output)

        tools = get_foresea_langchain_tools()
        self.assertEqual(len(tools), 3)


if __name__ == "__main__":
    unittest.main()
