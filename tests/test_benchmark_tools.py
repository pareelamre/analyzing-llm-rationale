from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import benchmark_tools  # noqa: E402


class BenchmarkToolTests(unittest.TestCase):
    def test_manage_notes_add_search_edit_delete_with_limits(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "notes.json"
            ctx = benchmark_tools.ToolContext(agent_id="model-a")

            added = benchmark_tools.manage_notes(
                {"action": "add", "text": "Watch the FOMC date.", "tags": ["rates"]},
                ctx,
                path=path,
            )
            self.assertTrue(added["ok"])
            note_id = added["note"]["id"]

            found = benchmark_tools.manage_notes(
                {"action": "search", "query": "fomc"},
                ctx,
                path=path,
            )
            self.assertEqual([n["id"] for n in found["notes"]], [note_id])

            edited = benchmark_tools.manage_notes(
                {"action": "edit", "id": note_id, "text": "Watch CPI before FOMC."},
                ctx,
                path=path,
            )
            self.assertEqual(edited["note"]["text"], "Watch CPI before FOMC.")

            deleted = benchmark_tools.manage_notes(
                {"action": "delete", "id": note_id},
                ctx,
                path=path,
            )
            self.assertEqual(deleted["deleted"], 1)

            too_long = benchmark_tools.manage_notes(
                {"action": "add", "text": "x" * (benchmark_tools.MAX_NOTE_CHARS + 1)},
                ctx,
                path=path,
            )
            self.assertFalse(too_long["ok"])

    def test_web_search_filters_blacklisted_citations(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "output": [{
                "content": [{
                    "text": "Search summary.",
                    "annotations": [
                        {"title": "Bad", "url": "https://coinmarketcap.com/currencies/x"},
                        {"title": "Good", "url": "https://example.com/story"},
                    ],
                }]
            }]
        }

        with (
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False),
            mock.patch.object(benchmark_tools.requests, "post", return_value=response) as post,
        ):
            result = benchmark_tools.web_search({"query": "test market"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"], "Search summary.")
        self.assertEqual(result["sources"], [{"title": "Good", "url": "https://example.com/story"}])
        self.assertEqual(result["blocked_results"], 1)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["tools"][0]["type"], "web_search")
        self.assertIn("coinmarketcap.com", payload["instructions"])

    def test_place_trade_defaults_to_shadow_kalshi_buy(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {"FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl")}
            with mock.patch.dict(os.environ, env, clear=False):
                result = benchmark_tools.place_trade(
                    {"ticker": "KXTEST", "side": "yes", "price": 0.42, "quantity": 2},
                    ctx,
                )

        self.assertTrue(result["ok"])
        self.assertFalse(result["submitted"])
        self.assertEqual(result["mode"], "shadow")
        self.assertEqual(result["normalized_order"]["platform"], "kalshi")
        self.assertEqual(result["normalized_order"]["action"], "buy")
        self.assertEqual(result["normalized_order"]["outcome"], "yes")
        self.assertTrue(result["risk_guard"]["allowed"])

    def test_place_trade_rejects_single_market_concentration_over_15_percent(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "100",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "0.15",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT": "1000",
                "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                first = benchmark_tools.place_trade(
                    {"ticker": "KXCONC", "side": "yes", "price": 0.10, "quantity": 100},
                    ctx,
                )
                second = benchmark_tools.place_trade(
                    {"ticker": "KXCONC", "side": "yes", "price": 0.10, "quantity": 60},
                    ctx,
                )

        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        self.assertTrue(second["rejected"])
        self.assertEqual(second["reason"], "concentration_limit")
        self.assertEqual(second["risk_guard"]["concentration_cap"], 15.0)
        self.assertEqual(second["risk_guard"]["market_cost_basis_after"], 16.0)

    def test_place_trade_rejects_orders_that_are_insolvent_after_fees(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "10",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT": "1000",
                "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                result = benchmark_tools.place_trade(
                    {"ticker": "KXSOLV", "side": "yes", "price": 0.95, "quantity": 10.5},
                    ctx,
                )

        self.assertFalse(result["ok"])
        self.assertTrue(result["rejected"])
        self.assertEqual(result["reason"], "solvency")
        self.assertGreater(result["risk_guard"]["cash_required"], result["risk_guard"]["cash_before"])

    def test_place_trade_netting_payout_counts_for_solvency(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "10",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT": "1000",
                "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                opened = benchmark_tools.place_trade(
                    {"ticker": "KXNET", "side": "yes", "price": 0.40, "quantity": 10},
                    ctx,
                )
                closed = benchmark_tools.place_trade(
                    {"ticker": "KXNET", "side": "no", "price": 0.59, "quantity": 10},
                    ctx,
                )

        self.assertTrue(opened["ok"])
        self.assertTrue(closed["ok"])
        self.assertEqual(closed["risk_guard"]["netting_payout"], 10.0)
        self.assertEqual(closed["risk_guard"]["cash_required"], 0.0)
        self.assertEqual(closed["risk_guard"]["market_cost_basis_after"], 0.0)

    def test_place_trade_rejects_per_cycle_spend_over_limit(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "100",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT": "0.9",
                "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                first = benchmark_tools.place_trade(
                    {"ticker": "KXCYCLE1", "side": "yes", "price": 0.40, "quantity": 2},
                    ctx,
                )
                second = benchmark_tools.place_trade(
                    {"ticker": "KXCYCLE2", "side": "yes", "price": 0.10, "quantity": 1},
                    ctx,
                )

        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        self.assertTrue(second["rejected"])
        self.assertEqual(second["reason"], "per_cycle_spend")
        self.assertGreater(
            second["risk_guard"]["cycle_spend_after"],
            second["risk_guard"]["per_cycle_spend_limit"],
        )


if __name__ == "__main__":
    unittest.main()
