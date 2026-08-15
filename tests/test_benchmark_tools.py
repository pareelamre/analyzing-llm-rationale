from __future__ import annotations

import os
import sqlite3
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
        from analyzing_llm_rationale import news_pipeline

        articles = [
            {"title": "Bad", "url": "https://coinmarketcap.com/currencies/x", "summary": "Bad summary."},
            {"title": "Good", "url": "https://example.com/story", "summary": "Good summary."},
        ]
        mock_pipeline = mock.Mock()
        mock_pipeline.fetch_summarize_rank.return_value = articles

        with mock.patch.object(news_pipeline, "NewsPipeline", return_value=mock_pipeline) as pipeline_cls:
            result = benchmark_tools.web_search({"query": "test market"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["sources"], [{"title": "Good", "url": "https://example.com/story"}])
        self.assertEqual(result["blocked_results"], 1)
        self.assertIn("Good summary.", result["summary"])
        self.assertNotIn("Bad summary.", result["summary"])
        pipeline_cls.assert_called_once_with(fetch_sources=benchmark_tools.WEB_SEARCH_SOURCES)
        mock_pipeline.fetch_summarize_rank.assert_called_once_with(
            "test market", top_k=benchmark_tools.WEB_SEARCH_TOP_K
        )

    def test_web_search_does_not_require_an_openai_key(self):
        # Regression: web_search used to hit api.openai.com and required
        # OPENAI_API_KEY, which was never configured as a repo secret, so
        # every single call failed. It now runs through the keyless
        # multi-source news pipeline instead (SCADS_AI_API_KEY, already
        # required elsewhere, covers query planning/summarization).
        from analyzing_llm_rationale import news_pipeline

        mock_pipeline = mock.Mock()
        mock_pipeline.fetch_summarize_rank.return_value = []

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(news_pipeline, "NewsPipeline", return_value=mock_pipeline),
        ):
            result = benchmark_tools.web_search({"query": "test market"})

        self.assertTrue(result["ok"])

    def test_web_search_requires_a_query(self):
        result = benchmark_tools.web_search({"query": "  "})
        self.assertFalse(result["ok"])
        self.assertIn("query is required", result["error"])

    def test_place_trade_defaults_to_shadow_kalshi_buy(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
            }
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
        self.assertEqual(
            result["normalized_order"]["exchange_order"]["time_in_force"],
            "immediate_or_cancel",
        )
        self.assertFalse(result["normalized_order"]["exchange_order"]["post_only"])
        self.assertEqual(result["execution"]["filled_quantity"], 2.0)
        self.assertTrue(result["risk_guard"]["allowed"])

    def test_place_trade_forces_ioc_and_rejects_resting_order_options(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                result = benchmark_tools.place_trade(
                    {
                        "ticker": "KXIOC",
                        "side": "yes",
                        "price": 0.42,
                        "quantity": 2,
                        "time_in_force": "good_till_canceled",
                        "post_only": True,
                        "order_type": "market",
                    },
                    ctx,
                )

        self.assertTrue(result["ok"])
        order = result["normalized_order"]["exchange_order"]
        self.assertEqual(order["time_in_force"], "immediate_or_cancel")
        self.assertFalse(order["post_only"])
        self.assertTrue(any("time_in_force" in w for w in result["warnings"]))
        self.assertTrue(any("post_only" in w for w in result["warnings"]))
        self.assertTrue(any("order_type" in w for w in result["warnings"]))

    def test_place_trade_rejects_live_mode_and_never_calls_trading_place_order(self):
        # place_trade is called from an autonomous LLM tool loop, which cannot
        # supply real human confirmation and does not route through
        # create_trading_run/execute_trading_run/the guardrail chain/the kill
        # switch. FORESEA_AGENT_PLACE_TRADE_MODE=live used to let it call
        # trading.place_order directly with a self-supplied confirmation
        # phrase and shared server credentials -- it must now be rejected
        # outright, before trading.place_order is ever reached.
        from analyzing_llm_rationale import trading

        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_PLACE_TRADE_MODE": "live",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(trading, "place_order") as fake_place_order,
            ):
                result = benchmark_tools.place_trade(
                    {"ticker": "KXPARTIAL", "side": "yes", "price": 0.40, "quantity": 10},
                    ctx,
                )
                fake_place_order.assert_not_called()

        self.assertFalse(result["ok"])
        self.assertIn("must be 'shadow'", result["error"])

    def test_place_trade_rejects_single_market_concentration_over_15_percent(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "100",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "0.15",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
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

    def test_rejected_trade_stores_zero_cash_delta_not_the_hypothetical_amount(self):
        # A rejected order never reaches the exchange -- no cash moves, so
        # its agent_actions row must record cash_delta=0, not the delta the
        # risk-guard preview computed before rejecting it. Storing that
        # hypothetical value here previously corrupted every downstream
        # cash-delta-based reconstruction (agent_trading_stats.agent_equity_curve).
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "accounts.sqlite"
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(db_path),
                "FORESEA_AGENT_ACCOUNT_VALUE": "100",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "0.15",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
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
            conn = sqlite3.connect(db_path)
            try:
                cash_after = conn.execute(
                    "SELECT cash FROM agent_accounts WHERE agent_id = 'model-a'"
                ).fetchone()[0]
                rejected_row = conn.execute(
                    "SELECT cash_delta, cash_required FROM agent_actions "
                    "WHERE agent_id = 'model-a' AND action_type = 'rejected_trade'"
                ).fetchone()
            finally:
                conn.close()

        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        # The would-be cost stays visible in risk_guard/cash_required (the
        # tool's own response and the audit row) -- only cash_delta, which
        # readers trust as "cash that actually moved", must be zero.
        self.assertGreater(second["risk_guard"]["cash_required"], 0.0)
        self.assertIsNotNone(rejected_row)
        self.assertEqual(rejected_row[0], 0.0)
        self.assertGreater(rejected_row[1], 0.0)
        # Confirms the rejection genuinely never touched cash.
        self.assertAlmostEqual(cash_after, 100.0 - first["risk_guard"]["cash_required"])

    def test_place_trade_rejects_orders_that_are_insolvent_after_fees(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "10",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
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
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "10",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
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
        self.assertEqual(closed["account"]["n_open_positions"], 0)

    def test_place_trade_rejects_per_cycle_spend_over_limit(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "100",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "0.009",  # $0.90 at $100 account value
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

    def test_per_cycle_spend_limit_scales_with_account_value(self):
        # Regression: this used to be a flat dollar amount (DEFAULT was
        # $500/cycle regardless of account size) -- now it's a percentage, so
        # the same FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT must yield a
        # bigger dollar limit for a bigger account, not a fixed number.
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        def _cap_at(account_value: str) -> float:
            with tempfile.TemporaryDirectory() as td:
                env = {
                    "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                    "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                    "FORESEA_AGENT_ACCOUNT_VALUE": account_value,
                    "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                    "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "0.2",
                    "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                }
                with mock.patch.dict(os.environ, env, clear=False):
                    result = benchmark_tools.place_trade(
                        {"ticker": "KXSCALE", "side": "yes", "price": 0.10, "quantity": 1},
                        ctx,
                    )
            return result["risk_guard"]["per_cycle_spend_limit"]

        self.assertAlmostEqual(_cap_at("100"), 20.0)
        self.assertAlmostEqual(_cap_at("10000"), 2000.0)

    def test_place_trade_updates_weighted_average_entry_in_positions_table(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "accounts.sqlite"
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(db_path),
                "FORESEA_AGENT_ACCOUNT_VALUE": "100",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
                "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                first = benchmark_tools.place_trade(
                    {"ticker": "KXAVG", "side": "yes", "price": 0.60, "quantity": 10},
                    ctx,
                )
                second = benchmark_tools.place_trade(
                    {"ticker": "KXAVG", "side": "yes", "price": 0.70, "quantity": 5},
                    ctx,
                )
            conn = sqlite3.connect(db_path)
            try:
                pos = conn.execute(
                    """
                    SELECT quantity, cost_basis, avg_entry_price
                    FROM agent_positions
                    WHERE agent_id = 'model-a' AND ticker = 'KXAVG' AND side = 'yes'
                    """
                ).fetchone()
                actions = conn.execute("SELECT COUNT(*) FROM agent_actions").fetchone()[0]
            finally:
                conn.close()

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos[0], 15.0)
        self.assertAlmostEqual(pos[1], 9.5)
        self.assertAlmostEqual(pos[2], 9.5 / 15.0)
        self.assertEqual(actions, 2)

    def test_place_trade_settles_open_positions_before_new_cycle(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "accounts.sqlite"
            base_env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(db_path),
                "FORESEA_AGENT_ACCOUNT_VALUE": "100",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
                "FORESEA_AGENT_SETTLEMENT_FEE_RATE": "0.014",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }

            def resolve(ticker):
                return 1 if ticker == "KXSETTLE" else None

            with (
                mock.patch.dict(os.environ, {**base_env, "FORESEA_AGENT_CYCLE_ID": "cycle-1"}, clear=False),
                mock.patch("analyzing_llm_rationale.market_data.resolve_kalshi", side_effect=resolve),
            ):
                opened = benchmark_tools.place_trade(
                    {"ticker": "KXSETTLE", "side": "yes", "price": 0.40, "quantity": 10},
                    ctx,
                )

            with (
                mock.patch.dict(os.environ, {**base_env, "FORESEA_AGENT_CYCLE_ID": "cycle-2"}, clear=False),
                mock.patch("analyzing_llm_rationale.market_data.resolve_kalshi", side_effect=resolve),
            ):
                after_settlement = benchmark_tools.place_trade(
                    {"ticker": "KXOTHER", "side": "yes", "price": 0.10, "quantity": 1},
                    ctx,
                )

            conn = sqlite3.connect(db_path)
            try:
                settlement = conn.execute(
                    """
                    SELECT payout, settlement_fee, realized_pnl
                    FROM agent_actions
                    WHERE action_type = 'settlement' AND ticker = 'KXSETTLE'
                    """
                ).fetchone()
                remaining = conn.execute(
                    """
                    SELECT COUNT(*) FROM agent_positions
                    WHERE ticker = 'KXSETTLE'
                    """
                ).fetchone()[0]
            finally:
                conn.close()

        self.assertTrue(opened["ok"])
        self.assertTrue(after_settlement["ok"])
        self.assertEqual(after_settlement["risk_guard"]["settlements_before_trade"][0]["ticker"], "KXSETTLE")
        self.assertIsNotNone(settlement)
        self.assertAlmostEqual(settlement[0], 10.0)
        self.assertAlmostEqual(settlement[1], 0.14)
        self.assertAlmostEqual(settlement[2], 5.86)
        self.assertEqual(remaining, 0)

    def test_agent_cycles_table_round_trips(self):
        # Stores per-cycle thesis/transcript/candidates -- the source for the
        # agentic-trading transparency feed. Not written by any existing
        # function yet, so this just locks in the schema itself.
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "accounts.sqlite"
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(db_path)}, clear=False
            ):
                conn = benchmark_tools._account_conn()
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO agent_cycles
                        (agent_id, cycle_id, ts, thesis, transcript_json, steps, truncated)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "gpt-oss-120b", "2026-08-11T00:00", "2026-08-11T00:00:05+00:00",
                            "Held flat this cycle.", '{"candidates_offered": ["KXTEST"]}',
                            2, 0,
                        ),
                    )
                    conn.commit()
                    row = conn.execute(
                        "SELECT agent_id, cycle_id, thesis, steps, truncated FROM agent_cycles"
                        " WHERE agent_id = ? AND cycle_id = ?",
                        ("gpt-oss-120b", "2026-08-11T00:00"),
                    ).fetchone()
                finally:
                    conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row["thesis"], "Held flat this cycle.")
        self.assertEqual(row["steps"], 2)
        self.assertEqual(row["truncated"], 0)


if __name__ == "__main__":
    unittest.main()
