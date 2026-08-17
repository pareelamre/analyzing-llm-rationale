"""Unit tests for tool alias normalization and unknown tool error feedback."""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import agent_capabilities as ac  # noqa: E402


class ToolAliasTests(unittest.TestCase):
    def test_normalize_action_place_order_alias(self):
        obj = ac.parse_action('{"thought": "buy yes", "action": "place_order", "args": {"ticker": "XYZ"}}')
        self.assertEqual(obj["action"], "place_trade")
        self.assertEqual(obj["args"], {"ticker": "XYZ"})

    def test_normalize_action_buy_alias(self):
        obj = ac.parse_action('{"thought": "buy yes", "action": "buy", "args": {"ticker": "XYZ"}}')
        self.assertEqual(obj["action"], "place_trade")

    def test_normalize_action_search_alias(self):
        obj = ac.parse_action('{"thought": "search news", "action": "search", "args": {"query": "elections"}}')
        self.assertEqual(obj["action"], "web_search")

    def test_normalize_openai_native_alias(self):
        obj = ac.parse_action('{"name": "google_search", "parameters": {"query": "fed interest rate"}}')
        self.assertEqual(obj["action"], "web_search")

    def test_run_tool_loop_executes_alias_tool(self):
        called = []

        async def fake_place_trade(args):
            called.append(args)
            return "trade placed successfully"

        async def fake_chat(messages):
            if len(called) == 0:
                return '{"thought": "trading", "action": "place_order", "args": {"ticker": "KX"}}'
            return '{"final": "done"}'

        tools = {"place_trade": fake_place_trade}
        specs = [{"name": "place_trade", "args": "ticker", "description": "buy"}]

        res = asyncio.run(ac.run_tool_loop("question", tools, specs, fake_chat, max_steps=3))
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0], {"ticker": "KX"})
        self.assertEqual(res["answer"], "done")

    def test_run_tool_loop_unknown_tool_message(self):
        async def fake_chat(messages):
            return '{"thought": "unknown", "action": "nonexistent_tool", "args": {}}'

        tools = {"place_trade": lambda a: None, "web_search": lambda a: None}
        specs = []

        res = asyncio.run(ac.run_tool_loop("question", tools, specs, fake_chat, max_steps=1))
        obs = res["transcript"][0]["observation"]
        self.assertIn("unknown tool 'nonexistent_tool'", obs)
        self.assertIn("available tools: 'place_trade', 'web_search'", obs)
    def test_normalize_api_aliases(self):
        obj1 = ac.parse_action('{"action": "http_get", "args": {"url": "https://api.com/v1"}}')
        self.assertEqual(obj1["action"], "fetch_api")

        obj2 = ac.parse_action('{"action": "call_api", "args": {"url": "/edge-board"}}')
        self.assertEqual(obj2["action"], "fetch_api")

        obj3 = ac.parse_action('{"action": "foresea_edge_board", "args": {}}')
        self.assertEqual(obj3["action"], "edge_board")

        obj4 = ac.parse_action('{"action": "foresea_batch_quotes", "args": {"refs": ["KX1"]}}')
        self.assertEqual(obj4["action"], "batch_quotes")
        obj5 = ac.parse_action('{"action": "get_exchange_status", "args": {}}')
        self.assertEqual(obj5["action"], "exchange_status")

        obj6 = ac.parse_action('{"action": "get_orderbook", "args": {"ticker": "KX1"}}')
        self.assertEqual(obj6["action"], "orderbook")

        obj7 = ac.parse_action('{"action": "polymarket_tags", "args": {}}')
        self.assertEqual(obj7["action"], "market_tags")
