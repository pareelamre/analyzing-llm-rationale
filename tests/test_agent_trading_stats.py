from __future__ import annotations

import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import agent_trading_stats, benchmark_tools  # noqa: E402


@contextmanager
def _fixture_conn():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "accounts.sqlite"
        with mock.patch.dict(
            os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(db_path)}, clear=False
        ):
            conn = benchmark_tools._account_conn()
            try:
                yield conn
            finally:
                conn.close()


def _insert_account(conn, agent_id, *, starting_cash=10_000.0, cash=9_000.0,
                     realized_pnl=0.0, fees_paid=0.0, settlement_fees_paid=0.0,
                     updated_at="2026-08-11T00:00:00+00:00"):
    conn.execute(
        """
        INSERT OR REPLACE INTO agent_accounts
        (agent_id, starting_cash, cash, realized_pnl, fees_paid, settlement_fees_paid, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (agent_id, starting_cash, cash, realized_pnl, fees_paid, settlement_fees_paid, updated_at),
    )


def _insert_position(conn, agent_id, *, platform="kalshi", ticker="KXFOO-26",
                      side="yes", quantity=100.0, cost_basis=40.0, avg_entry_price=0.40,
                      updated_at="2026-08-11T00:00:00+00:00"):
    conn.execute(
        """
        INSERT OR REPLACE INTO agent_positions
        (agent_id, platform, ticker, side, quantity, cost_basis, avg_entry_price, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (agent_id, platform, ticker, side, quantity, cost_basis, avg_entry_price, updated_at),
    )


def _insert_action(conn, agent_id, *, action_type="trade", ts="2026-08-11T00:00:00+00:00",
                    cash_delta=0.0, realized_pnl=0.0, mode="shadow", platform="kalshi",
                    ticker="KXFOO-26", side="yes", quantity=None, price=None, outcome=None,
                    cycle_id="15m:1"):
    conn.execute(
        """
        INSERT INTO agent_actions
        (id, ts, agent_id, action_type, mode, submitted, platform, ticker, side, price,
         quantity, notional, fee, settlement_fee, payout, netting_payout, cash_required,
         cash_delta, realized_pnl, realized_pairs, cycle_id, client_order_id, outcome,
         metadata_json)
        VALUES (lower(hex(randomblob(16))), :ts, :agent_id, :action_type, :mode, 0,
                :platform, :ticker, :side, :price, :quantity, 0, 0, 0, 0, 0, 0,
                :cash_delta, :realized_pnl, 0, :cycle_id, NULL, :outcome, '{}')
        """,
        {
            "ts": ts, "agent_id": agent_id, "action_type": action_type, "mode": mode,
            "platform": platform, "ticker": ticker, "side": side, "price": price,
            "quantity": quantity, "cash_delta": cash_delta, "realized_pnl": realized_pnl,
            "cycle_id": cycle_id, "outcome": outcome,
        },
    )


def _insert_cycle(conn, agent_id, *, cycle_id="15m:1", ts="2026-08-11T00:00:00+00:00",
                   thesis="Held flat this cycle."):
    conn.execute(
        """
        INSERT OR REPLACE INTO agent_cycles
        (agent_id, cycle_id, ts, thesis, transcript_json, steps, truncated)
        VALUES (?, ?, ?, ?, '{}', 0, 0)
        """,
        (agent_id, cycle_id, ts, thesis),
    )


class LeaderboardTests(unittest.TestCase):
    def test_marks_open_positions_to_market_using_quotes(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", cash=9_600.0)
            _insert_position(conn, "model-a", quantity=100.0, cost_basis=40.0)
            conn.commit()

            quotes = {("kalshi", "KXFOO-26"): {"yes_bid": 0.55}}
            rows = agent_trading_stats.compute_agent_leaderboard(conn, quotes)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["agent_id"], "model-a")
        self.assertAlmostEqual(row["cash"], 9_600.0)
        # 100 contracts @ 0.55 bid = 55.0 liquidation value
        self.assertAlmostEqual(row["account_value"], 9_600.0 + 55.0)
        self.assertAlmostEqual(row["unrealized_pnl"], 55.0 - 40.0)

    def test_quotes_are_keyed_by_lowercase_platform_matching_agent_positions(self):
        # agent_positions.platform is always stored lowercase ("kalshi"); a
        # raw quote's own "platform" field may say "Kalshi" -- the leaderboard
        # must key its quote lookup lowercase regardless, or every open
        # position silently prices as illiquid (bid=0).
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", cash=9_600.0)
            _insert_position(conn, "model-a", platform="kalshi", quantity=10.0, cost_basis=4.0)
            conn.commit()

            quotes = {("kalshi", "KXFOO-26"): {"platform": "Kalshi", "yes_bid": 0.5}}
            rows = agent_trading_stats.compute_agent_leaderboard(conn, quotes)

        self.assertEqual(rows[0]["illiquid_positions"], [])
        self.assertAlmostEqual(rows[0]["account_value"], 9_600.0 + 5.0)

    def test_win_rate_counts_settlements_only_not_rejected_trades(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", cash=10_000.0)
            _insert_action(conn, "model-a", action_type="trade", cash_delta=-40.0)
            _insert_action(conn, "model-a", action_type="rejected_trade", cash_delta=0.0)
            _insert_action(conn, "model-a", action_type="settlement", realized_pnl=20.0)
            _insert_action(conn, "model-a", action_type="settlement", realized_pnl=-10.0)
            conn.commit()

            rows = agent_trading_stats.compute_agent_leaderboard(conn, {})

        row = rows[0]
        self.assertEqual(row["trade_count"], 1)  # rejected_trade excluded
        self.assertEqual(row["settled_count"], 2)
        self.assertEqual(row["won_count"], 1)
        self.assertAlmostEqual(row["win_rate"], 0.5)

    def test_no_settlements_yields_none_win_rate_not_zero(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a")
            conn.commit()
            rows = agent_trading_stats.compute_agent_leaderboard(conn, {})
        self.assertIsNone(rows[0]["win_rate"])

    def test_multiple_agents_sorted_by_account_value_descending(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-low", cash=8_000.0)
            _insert_account(conn, "model-high", cash=12_000.0)
            conn.commit()
            rows = agent_trading_stats.compute_agent_leaderboard(conn, {})
        self.assertEqual([r["agent_id"] for r in rows], ["model-high", "model-low"])


class EquityCurveTests(unittest.TestCase):
    def test_curve_is_a_running_cash_total_starting_from_starting_cash(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", starting_cash=10_000.0)
            _insert_action(conn, "model-a", ts="2026-08-11T00:00:00+00:00", cash_delta=-50.0)
            _insert_action(conn, "model-a", ts="2026-08-11T00:15:00+00:00", cash_delta=60.0)
            conn.commit()

            curve = agent_trading_stats.agent_equity_curve(conn, "model-a")

        values = [p["account_value"] for p in curve["value_curve"]]
        self.assertEqual(values, [10_000.0, 9_950.0, 10_010.0])

    def test_unknown_agent_yields_a_single_zero_starting_point(self):
        with _fixture_conn() as conn:
            curve = agent_trading_stats.agent_equity_curve(conn, "ghost")
        self.assertEqual(len(curve["value_curve"]), 1)
        self.assertEqual(curve["value_curve"][0]["account_value"], 0.0)

    def test_includes_sharpe_and_max_drawdown_keys(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", starting_cash=10_000.0)
            _insert_action(conn, "model-a", ts="2026-08-11T00:00:00+00:00", cash_delta=-500.0)
            _insert_action(conn, "model-a", ts="2026-08-11T00:15:00+00:00", cash_delta=100.0)
            conn.commit()
            curve = agent_trading_stats.agent_equity_curve(conn, "model-a")
        self.assertIn("sharpe", curve)
        self.assertIn("max_drawdown", curve)
        self.assertGreater(curve["max_drawdown"], 0.0)


class RecentActivityTests(unittest.TestCase):
    def test_merges_trades_theses_and_notes_sorted_newest_first(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a")
            _insert_action(conn, "model-a", action_type="trade", ts="2026-08-11T00:00:00+00:00")
            _insert_cycle(conn, "model-a", ts="2026-08-11T00:05:00+00:00", thesis="Bought some yes.")
            conn.commit()

            notes = {"model-a": [{"text": "Watch the Fed date.", "updated_at": "2026-08-11T00:10:00+00:00"}]}
            items = agent_trading_stats.recent_activity(conn, notes, limit=10)

        self.assertEqual([i["type"] for i in items], ["note", "thesis", "trade"])

    def test_limit_is_respected_after_merge(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a")
            for i in range(5):
                _insert_action(conn, "model-a", action_type="trade", ts=f"2026-08-11T00:0{i}:00+00:00")
            conn.commit()
            items = agent_trading_stats.recent_activity(conn, {}, limit=3)
        self.assertEqual(len(items), 3)

    def test_empty_thesis_is_excluded(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a")
            _insert_cycle(conn, "model-a", thesis="")
            conn.commit()
            items = agent_trading_stats.recent_activity(conn, {}, limit=10)
        self.assertEqual(items, [])


if __name__ == "__main__":
    unittest.main()
