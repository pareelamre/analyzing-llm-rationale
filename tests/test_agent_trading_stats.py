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

    def test_admin_reset_scopes_trade_count_and_win_rate_to_since_the_reset(self):
        # Regression: agent_equity_curve() trims its chart to start at the
        # latest admin_reset (see EquityCurveTests), but this leaderboard row
        # is what the table shows next to that chart. Before this fix,
        # trade_count/settled_count/win_rate here kept counting every
        # pre-reset action forever, so the table and chart visibly disagreed
        # about how much history the account had.
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", cash=10_000.0)
            _insert_action(conn, "model-a", action_type="trade",
                            ts="2026-08-11T00:00:00+00:00", cash_delta=-40.0)
            _insert_action(conn, "model-a", action_type="settlement",
                            ts="2026-08-11T00:05:00+00:00", realized_pnl=-10.0)
            _insert_action(conn, "model-a", action_type="admin_reset",
                            ts="2026-08-12T00:00:00+00:00", cash_delta=40.0)
            _insert_action(conn, "model-a", action_type="trade",
                            ts="2026-08-13T00:00:00+00:00", cash_delta=-25.0)
            _insert_action(conn, "model-a", action_type="settlement",
                            ts="2026-08-13T00:05:00+00:00", realized_pnl=20.0)
            conn.commit()

            row = agent_trading_stats.compute_agent_leaderboard(conn, {})[0]

        self.assertEqual(row["trade_count"], 1)
        self.assertEqual(row["settled_count"], 1)
        self.assertEqual(row["won_count"], 1)
        self.assertAlmostEqual(row["win_rate"], 1.0)

    def test_no_reset_still_counts_all_time_activity(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", cash=10_000.0)
            _insert_action(conn, "model-a", action_type="trade",
                            ts="2026-08-11T00:00:00+00:00", cash_delta=-40.0)
            _insert_action(conn, "model-a", action_type="trade",
                            ts="2026-08-12T00:00:00+00:00", cash_delta=-25.0)
            conn.commit()

            row = agent_trading_stats.compute_agent_leaderboard(conn, {})[0]

        self.assertEqual(row["trade_count"], 2)


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

    def test_rejected_trade_never_moves_the_curve_even_if_its_stored_cash_delta_is_nonzero(self):
        # A rejected order never reaches the exchange, so no cash moves --
        # but rows written before this fix stored the guard's hypothetical
        # pre-rejection delta anyway. The curve must ignore it regardless of
        # what's on the row, both for that old data and as a defense against
        # any future write-side regression.
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", starting_cash=10_000.0)
            _insert_action(conn, "model-a", action_type="trade",
                            ts="2026-08-11T00:00:00+00:00", cash_delta=-50.0)
            _insert_action(conn, "model-a", action_type="rejected_trade",
                            ts="2026-08-11T00:15:00+00:00", cash_delta=-847.04)
            _insert_action(conn, "model-a", action_type="trade",
                            ts="2026-08-11T00:30:00+00:00", cash_delta=-30.0)
            conn.commit()

            curve = agent_trading_stats.agent_equity_curve(conn, "model-a")

        values = [p["account_value"] for p in curve["value_curve"]]
        event_types = [p["event_type"] for p in curve["value_curve"]]
        self.assertEqual(event_types, ["starting_cash", "trade", "rejected_trade", "trade"])
        self.assertEqual(values, [10_000.0, 9_950.0, 9_950.0, 9_920.0])

    def test_admin_correction_moves_the_curve(self):
        # Regression: an admin_correction row updates agent_accounts directly
        # (see scripts/reset_agent_trading_accounts.py) -- without including
        # it here, the curve would climb through the pre-correction trade and
        # never show the adjustment, silently diverging from the corrected
        # agent_accounts.cash it's supposed to be a running total of.
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", starting_cash=10_000.0)
            _insert_action(conn, "model-a", action_type="trade",
                            ts="2026-08-11T00:00:00+00:00", cash_delta=6_000.0)
            _insert_action(conn, "model-a", action_type="admin_correction",
                            ts="2026-08-11T00:15:00+00:00", cash_delta=-5_500.0)
            conn.commit()

            curve = agent_trading_stats.agent_equity_curve(conn, "model-a")

        values = [p["account_value"] for p in curve["value_curve"]]
        event_types = [p["event_type"] for p in curve["value_curve"]]
        self.assertEqual(event_types, ["starting_cash", "trade", "admin_correction"])
        self.assertEqual(values, [10_000.0, 16_000.0, 10_500.0])

    def test_admin_reset_trims_the_curve_to_the_reset_point(self):
        # A full admin_reset means "start this account over" -- the chart
        # (and the Sharpe/drawdown computed from it) should only cover what
        # happened since then, not the pre-reset trades that got wiped.
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", starting_cash=10_000.0)
            _insert_action(conn, "model-a", action_type="trade",
                            ts="2026-08-11T00:00:00+00:00", cash_delta=6_000.0)
            _insert_action(conn, "model-a", action_type="admin_correction",
                            ts="2026-08-11T00:15:00+00:00", cash_delta=-5_500.0)
            _insert_action(conn, "model-a", action_type="admin_reset",
                            ts="2026-08-11T00:30:00+00:00", cash_delta=-500.0)
            _insert_action(conn, "model-a", action_type="trade",
                            ts="2026-08-11T00:45:00+00:00", cash_delta=-25.0)
            conn.commit()

            curve = agent_trading_stats.agent_equity_curve(conn, "model-a")

        values = [p["account_value"] for p in curve["value_curve"]]
        event_types = [p["event_type"] for p in curve["value_curve"]]
        # Neither the pre-reset trade (16_000.0) nor the correction
        # (10_500.0) appear -- the curve starts at the admin_reset itself.
        self.assertEqual(event_types, ["admin_reset", "trade"])
        self.assertEqual(values, [10_000.0, 9_975.0])

    def test_curve_trims_to_the_latest_of_multiple_resets(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", starting_cash=10_000.0)
            _insert_action(conn, "model-a", action_type="trade",
                            ts="2026-08-11T00:00:00+00:00", cash_delta=-1_000.0)
            _insert_action(conn, "model-a", action_type="admin_reset",
                            ts="2026-08-12T00:00:00+00:00", cash_delta=1_000.0)
            _insert_action(conn, "model-a", action_type="trade",
                            ts="2026-08-13T00:00:00+00:00", cash_delta=-2_000.0)
            _insert_action(conn, "model-a", action_type="admin_reset",
                            ts="2026-08-14T00:00:00+00:00", cash_delta=2_000.0)
            conn.commit()

            curve = agent_trading_stats.agent_equity_curve(conn, "model-a")

        values = [p["account_value"] for p in curve["value_curve"]]
        event_types = [p["event_type"] for p in curve["value_curve"]]
        self.assertEqual(event_types, ["admin_reset"])
        self.assertEqual(values, [10_000.0])


class PromotionEligibilityTests(unittest.TestCase):
    def test_eligible_when_all_checks_pass(self):
        row = {"agent_id": "model-a", "settled_count": 40, "return_pct": 5.0}
        equity = {"sharpe": 0.8, "max_drawdown": 0.10}
        result = agent_trading_stats.compute_promotion_eligibility(row, equity)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["agent_id"], "model-a")
        self.assertTrue(all(result["checks"].values()))

    def test_insufficient_sample_blocks_eligibility_even_with_good_metrics(self):
        row = {"agent_id": "model-a", "settled_count": 5, "return_pct": 5.0}
        equity = {"sharpe": 2.0, "max_drawdown": 0.02}
        result = agent_trading_stats.compute_promotion_eligibility(row, equity)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["checks"]["sufficient_sample"])
        self.assertTrue(result["checks"]["sharpe_above_floor"])

    def test_negative_return_blocks_eligibility(self):
        row = {"agent_id": "model-a", "settled_count": 40, "return_pct": -2.0}
        equity = {"sharpe": 0.8, "max_drawdown": 0.10}
        result = agent_trading_stats.compute_promotion_eligibility(row, equity)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["checks"]["positive_return"])

    def test_sharpe_below_floor_blocks_eligibility(self):
        row = {"agent_id": "model-a", "settled_count": 40, "return_pct": 1.0}
        equity = {"sharpe": 0.1, "max_drawdown": 0.10}
        result = agent_trading_stats.compute_promotion_eligibility(row, equity)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["checks"]["sharpe_above_floor"])

    def test_drawdown_over_cap_blocks_eligibility(self):
        row = {"agent_id": "model-a", "settled_count": 40, "return_pct": 5.0}
        equity = {"sharpe": 0.8, "max_drawdown": 0.40}
        result = agent_trading_stats.compute_promotion_eligibility(row, equity)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["checks"]["drawdown_within_cap"])

    def test_missing_sharpe_or_drawdown_blocks_eligibility_rather_than_crashing(self):
        row = {"agent_id": "model-a", "settled_count": 40, "return_pct": 5.0}
        equity = {"sharpe": None, "max_drawdown": None}
        result = agent_trading_stats.compute_promotion_eligibility(row, equity)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["checks"]["sharpe_above_floor"])
        self.assertFalse(result["checks"]["drawdown_within_cap"])


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

    def test_admin_reset_appears_in_the_feed(self):
        # A balance adjustment shouldn't be invisible in the one feed meant
        # to show what happened to that balance.
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a")
            _insert_action(conn, "model-a", action_type="admin_reset",
                            ts="2026-08-11T00:00:00+00:00", outcome="reset")
            conn.commit()
            items = agent_trading_stats.recent_activity(conn, {}, limit=10)
        self.assertEqual([i["type"] for i in items], ["admin_reset"])
        self.assertEqual(items[0]["outcome"], "reset")

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
