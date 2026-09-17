from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import agent_trading_tick  # noqa: E402

from analyzing_llm_rationale import benchmark_tools  # noqa: E402


class LlamaTradingRemediationTests(unittest.TestCase):
    def test_portfolio_block_warns_against_phantom_close_when_flat(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "accounts.sqlite")
            with mock.patch.dict(os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": db_path}, clear=False):
                with benchmark_tools._account_transaction() as conn:
                    block = agent_trading_tick._build_portfolio_block(conn, "llama-3.3-70b-instruct", None)

        self.assertIn("Open positions: none.", block)
        self.assertIn("Do NOT call sizing_mode='close'", block)
        self.assertIn("probe_kelly", block)

    def test_learning_block_triggers_critical_calibration_warning_on_high_bias(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "accounts.sqlite")
            with mock.patch.dict(os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": db_path}, clear=False):
                with benchmark_tools._account_transaction() as conn:
                    # Seed resolved thesis forecast with severe positive bias (+30%)
                    conn.execute(
                        """
                        INSERT INTO agent_thesis_forecasts (
                            agent_id, cycle_id, forecast_ts, platform, ticker,
                            model_probability, resolved_outcome, brier_score, market_brier_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        ("llama-3.3-70b-instruct", "c1", "2026-09-01T00:00:00Z", "kalshi", "KXTEST", 0.70, 0, 0.21, 0.16),
                    )
                    block = agent_trading_tick._build_learning_block(conn, "llama-3.3-70b-instruct")

        self.assertIn("CRITICAL CALIBRATION WARNING", block)
        self.assertIn("Bayesian shrinkage", block)
        self.assertIn("P_calibrated = 0.5 * P_model + 0.5 * P_market", block)
        self.assertIn("low-probability milestones (<30c)", block)

    def test_place_trade_rejection_message_explains_close_exceeds_open_position(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "accounts.sqlite")
            with mock.patch.dict(
                os.environ,
                {
                    "FORESEA_AGENT_ACCOUNT_DB_PATH": db_path,
                    "FORESEA_AGENT_PLACE_TRADE_MODE": "shadow",
                },
                clear=False,
            ):
                ctx = benchmark_tools.ToolContext(
                    agent_id="llama-3.3-70b-instruct",
                    user_id="test-user",
                    model="llama-3.3-70b-instruct",
                    require_kelly_sizing=False,
                )
                res = benchmark_tools.place_trade(
                    {
                        "ticker": "KXTEST-CLEMSON",
                        "side": "no",
                        "price": 0.60,
                        "quantity": 1.0,
                        "sizing_mode": "close",
                    },
                    ctx,
                )

        self.assertFalse(res["ok"])
        self.assertTrue(res["rejected"])
        self.assertEqual(res["reason"], "close_exceeds_open_position")
        self.assertIn("close_exceeds_open_position", res["message"])
        self.assertIn("You currently hold 0 open contracts", res["message"])
        self.assertIn("probe_kelly", res["message"])

    def test_trading_instruction_includes_ioc_discipline_and_longshot_guard(self):
        instruction = agent_trading_tick._TRADING_INSTRUCTION
        self.assertIn("IMMEDIATE-OR-CANCEL (IOC) PRICING DISCIPLINE", instruction)
        self.assertIn("pricing at the ask guarantees execution", instruction)
        self.assertIn("LONGSHOT & MILESTONE SPECULATION GUARD", instruction)
        self.assertIn("Buying YES on low-probability milestones is the single largest source of model losses", instruction)


if __name__ == "__main__":
    unittest.main()
