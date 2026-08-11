from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "agent_trading_tick.py"
_SPEC = importlib.util.spec_from_file_location("agent_trading_tick_test_module", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
agent_trading_tick = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(agent_trading_tick)

from analyzing_llm_rationale import benchmark_tools, market_data  # noqa: E402


def _quote(ident, question="Q?", bid=0.4, ask=0.45, close="2026-09-01T00:00:00Z"):
    return {
        "platform": "Kalshi", "ident": ident, "question": question,
        "probability": (bid + ask) / 2, "yes_bid": bid, "yes_ask": ask,
        "close_time": close,
    }


class ShadowModeAssertionTests(unittest.TestCase):
    def test_default_is_shadow_and_passes(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORESEA_AGENT_PLACE_TRADE_MODE", None)
            agent_trading_tick._assert_shadow_mode()  # must not raise

    def test_explicit_shadow_passes(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_PLACE_TRADE_MODE": "shadow"}, clear=False):
            agent_trading_tick._assert_shadow_mode()  # must not raise

    def test_live_mode_raises(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_PLACE_TRADE_MODE": "live"}, clear=False):
            with self.assertRaises(RuntimeError):
                agent_trading_tick._assert_shadow_mode()


class CandidateSelectionTests(unittest.TestCase):
    def test_discover_candidates_excludes_known_tickers_and_caps_count(self):
        listed = [_quote("KXA"), _quote("KXB"), _quote("KXC"), _quote("KXD")]
        with (
            mock.patch.object(market_data, "list_kalshi", return_value=listed),
            mock.patch.object(agent_trading_tick, "CANDIDATE_COUNT", 2),
        ):
            found = agent_trading_tick._discover_candidates({"KXA"})
        self.assertEqual([q["ident"] for q in found], ["KXB", "KXC"])

    def test_discover_candidates_skips_unpriced_markets(self):
        unpriced = dict(_quote("KXE"))
        unpriced["probability"] = None
        with mock.patch.object(market_data, "list_kalshi", return_value=[unpriced, _quote("KXF")]):
            found = agent_trading_tick._discover_candidates(set())
        self.assertEqual([q["ident"] for q in found], ["KXF"])

    def test_discover_candidates_survives_market_data_error(self):
        with mock.patch.object(
            market_data, "list_kalshi", side_effect=market_data.MarketDataError("boom")
        ):
            found = agent_trading_tick._discover_candidates(set())
        self.assertEqual(found, [])

    def test_requote_held_skips_failed_lookups(self):
        def fake_fetch(ticker):
            if ticker == "KXBAD":
                raise market_data.MarketDataError("gone")
            return _quote(ticker)

        with mock.patch.object(market_data, "fetch_kalshi", side_effect=fake_fetch):
            quotes = agent_trading_tick._requote_held(["KXGOOD", "KXBAD"])
        self.assertEqual([q["ident"] for q in quotes], ["KXGOOD"])


class PortfolioBlockTests(unittest.TestCase):
    def test_portfolio_block_reports_cash_and_positions(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite")}, clear=False
            ):
                ctx = benchmark_tools.ToolContext(agent_id="model-a")
                benchmark_tools.place_trade(
                    {"ticker": "KXTEST", "side": "yes", "price": 0.42, "quantity": 2}, ctx,
                )
                with benchmark_tools._account_transaction() as conn:
                    block = agent_trading_tick._build_portfolio_block(conn, "model-a", "Held flat last time.")

        self.assertIn("KXTEST", block)
        self.assertIn("Held flat last time.", block)
        self.assertIn("Open positions:", block)

    def test_portfolio_block_reports_no_positions(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite")}, clear=False
            ):
                with benchmark_tools._account_transaction() as conn:
                    block = agent_trading_tick._build_portfolio_block(conn, "model-b", None)

        self.assertIn("Open positions: none.", block)
        self.assertNotIn("previous cycle", block)


class RunCycleTests(unittest.TestCase):
    def _fake_report(self, thesis="Passed this cycle.", transcript=None):
        return SimpleNamespace(thesis=thesis, tool_transcript=transcript or [])

    def test_run_cycle_persists_thesis_and_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_NOTES_PATH": str(Path(td) / "notes.json"),
                "FORESEA_AGENT_CYCLE_ID": "test-cycle-1",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(agent_trading_tick, "_init_local_agent"),
                mock.patch.object(market_data, "list_kalshi", return_value=[_quote("KXNEW")]),
                mock.patch.object(
                    agent_trading_tick, "_call_agent_analyze",
                    return_value=self._fake_report(thesis="Bought KXNEW on a genuine edge.",
                                                     transcript=[{"action": "place_trade"}]),
                ) as call_mock,
            ):
                agent_trading_tick.run_cycle("model-c")

                with benchmark_tools._account_transaction() as conn:
                    row = conn.execute(
                        "SELECT * FROM agent_cycles WHERE agent_id = ? AND cycle_id = ?",
                        ("model-c", "test-cycle-1"),
                    ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row["thesis"], "Bought KXNEW on a genuine edge.")
        self.assertEqual(row["steps"], 1)
        self.assertIn("KXNEW", row["transcript_json"])
        # The question passed to the tool loop must actually mention the candidate.
        question_arg = call_mock.call_args[0][0]
        self.assertIn("KXNEW", question_arg)
        self.assertIn("Your portfolio", question_arg)

    def test_run_cycle_offers_held_positions_for_exit(self):
        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_NOTES_PATH": str(Path(td) / "notes.json"),
                "FORESEA_AGENT_CYCLE_ID": "test-cycle-2",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                ctx = benchmark_tools.ToolContext(agent_id="model-d")
                benchmark_tools.place_trade(
                    {"ticker": "KXHELD", "side": "yes", "price": 0.5, "quantity": 3}, ctx,
                )

                with (
                    mock.patch.object(agent_trading_tick, "_init_local_agent"),
                    mock.patch.object(market_data, "list_kalshi", return_value=[]),
                    mock.patch.object(
                        market_data, "fetch_kalshi", return_value=_quote("KXHELD", question="Held market"),
                    ),
                    mock.patch.object(
                        agent_trading_tick, "_call_agent_analyze",
                        return_value=self._fake_report(),
                    ) as call_mock,
                ):
                    agent_trading_tick.run_cycle("model-d")

        question_arg = call_mock.call_args[0][0]
        self.assertIn("KXHELD", question_arg)
        self.assertIn("currently hold", question_arg)

    def test_run_cycle_refuses_when_not_shadow(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_PLACE_TRADE_MODE": "live"}, clear=False):
            with self.assertRaises(RuntimeError):
                agent_trading_tick.run_cycle("model-e")


class AgentAnalyzeRetryTests(unittest.TestCase):
    def test_retries_then_raises_after_exhausting_attempts(self):
        import asyncio

        calls = []

        async def _always_fails(req, request=None):
            calls.append(1)
            raise RuntimeError("upstream down")

        with (
            mock.patch.object(agent_trading_tick, "AGENT_ANALYZE_RETRIES", 2),
            mock.patch.object(agent_trading_tick, "AGENT_ANALYZE_RETRY_BACKOFF_S", 0.0),
        ):
            with mock.patch(
                "analyzing_llm_rationale.server.agent_analyze", side_effect=_always_fails,
            ):
                with self.assertRaises(RuntimeError):
                    asyncio.run(agent_trading_tick._call_agent_analyze("question text"))
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
