from __future__ import annotations

import json
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
                    notional=0.0, payout=0.0, cycle_id="15m:1", metadata_json="{}"):
    conn.execute(
        """
        INSERT INTO agent_actions
        (id, ts, agent_id, action_type, mode, submitted, platform, ticker, side, price,
         quantity, notional, fee, settlement_fee, payout, netting_payout, cash_required,
         cash_delta, realized_pnl, realized_pairs, cycle_id, client_order_id, outcome,
         metadata_json)
        VALUES (lower(hex(randomblob(16))), :ts, :agent_id, :action_type, :mode, 0,
                :platform, :ticker, :side, :price, :quantity, :notional, 0, 0, :payout, 0, 0,
                :cash_delta, :realized_pnl, 0, :cycle_id, NULL, :outcome, :metadata_json)
        """,
        {
            "ts": ts, "agent_id": agent_id, "action_type": action_type, "mode": mode,
            "platform": platform, "ticker": ticker, "side": side, "price": price,
            "quantity": quantity, "cash_delta": cash_delta, "realized_pnl": realized_pnl,
            "notional": notional, "payout": payout, "cycle_id": cycle_id, "outcome": outcome,
            "metadata_json": metadata_json,
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


class ClassifyLedgerActionTests(unittest.TestCase):
    def test_only_a_zero_fill_trade_is_relabelled(self):
        cases = [
            (("trade", 0), "unfilled_order"),
            (("trade", 0.0), "unfilled_order"),
            (("trade", -1), "unfilled_order"),
            (("trade", 25), "trade"),
            (("trade", None), "trade"),
            (("trade", "not-a-number"), "trade"),
            (("settlement", 0), "settlement"),
            (("rejected_trade", 0), "rejected_trade"),
        ]
        for (action_type, quantity), expected in cases:
            with self.subTest(action_type=action_type, quantity=quantity):
                self.assertEqual(
                    agent_trading_stats.classify_ledger_action(action_type, quantity),
                    expected,
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
        self.assertAlmostEqual(row["total_pnl"], (9_600.0 + 55.0) - 10_000.0)
        self.assertAlmostEqual(row["open_positions"][0]["current_price"], 0.55)
        self.assertAlmostEqual(row["open_positions"][0]["unrealized_pnl"], 15.0)

    def test_total_pnl_matches_account_value_minus_starting_cash_accounting_for_fees(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", cash=9_800.0, starting_cash=10_000.0, fees_paid=20.0, realized_pnl=50.0)
            _insert_position(conn, "model-a", quantity=100.0, cost_basis=100.0)
            conn.commit()

            quotes = {("kalshi", "KXFOO-26"): {"yes_bid": 0.90}}
            rows = agent_trading_stats.compute_agent_leaderboard(conn, quotes)

        row = rows[0]
        # Account value = cash (9800) + liq_val (90) = 9890.0
        self.assertAlmostEqual(row["account_value"], 9890.0)
        # Total PnL = 9890.0 - 10000.0 = -110.0
        self.assertAlmostEqual(row["total_pnl"], -110.0)
        self.assertAlmostEqual(row["unrealized_pnl"], -10.0)
        self.assertAlmostEqual(row["realized_pnl"], 50.0)
        self.assertAlmostEqual(row["fees_paid"], 20.0)
        # Net PnL is exactly Realized (50) + Unrealized (-10) - Fees (20) = -110 (which also accounts for initial cash difference)
        self.assertAlmostEqual(row["total_pnl"], row["account_value"] - row["starting_cash"])

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

    def test_win_rate_counts_realized_exits_and_settlements_not_rejected_trades(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", cash=10_000.0)
            _insert_action(conn, "model-a", action_type="trade", cash_delta=-40.0)
            _insert_action(conn, "model-a", action_type="trade", outcome="realized", realized_pnl=30.0)
            _insert_action(conn, "model-a", action_type="rejected_trade", cash_delta=0.0)
            _insert_action(conn, "model-a", action_type="settlement", realized_pnl=20.0)
            _insert_action(conn, "model-a", action_type="settlement", realized_pnl=-10.0)
            conn.commit()

            rows = agent_trading_stats.compute_agent_leaderboard(conn, {})

        row = rows[0]
        self.assertEqual(row["trade_count"], 2)  # rejected_trade excluded
        self.assertEqual(row["settled_count"], 2)
        self.assertEqual(row["realized_count"], 3)
        self.assertEqual(row["won_count"], 2)
        self.assertAlmostEqual(row["win_rate"], round(2 / 3, 4))

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

    def test_closing_a_position_does_not_double_count_its_notional(self):
        # Regression, found live on kimi-k3 2026-08-18: exiting a position
        # (there is no sell tool -- closing means buying the opposite side,
        # recorded as an ordinary action_type='trade' row with outcome=
        # 'realized' and a positive notional, the closing order's own
        # dollar size) was treated exactly like an OPENING trade: notional
        # was added to open_positions_basis instead of removed. Every close
        # inflated the reported account value by its own notional, on top
        # of the basis already counted when the position was opened. Live
        # impact: one agent's chart showed $14,600 against a real $9,774 --
        # 5 closing trades in a row each adding ~$1000 of phantom equity.
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", starting_cash=10_000.0)
            # Open: buy 1200 YES @ 0.30 = $360 notional, cash -360.
            _insert_action(
                conn, "model-a", action_type="trade", outcome="open",
                ts="2026-08-18T05:17:06+00:00", cash_delta=-360.0, notional=360.0,
            )
            # Close: sell (net) all 1200 @ 0.32, $384 in, +$24 realized gain
            # (384 proceeds - 360 original cost basis). cash_delta is what
            # the account actually nets from closing -- cost basis removed
            # is cash_delta - realized_pnl = 384 - 24 = 360, exactly the
            # original open's notional, since this fully closes it.
            _insert_action(
                conn, "model-a", action_type="trade", outcome="realized",
                ts="2026-08-18T09:23:36+00:00", cash_delta=384.0, realized_pnl=24.0,
                notional=384.0,
            )
            conn.commit()

            curve = agent_trading_stats.agent_equity_curve(conn, "model-a")

        values = [p["account_value"] for p in curve["value_curve"]]
        event_types = [p["event_type"] for p in curve["value_curve"]]
        self.assertEqual(event_types, ["starting_cash", "trade", "trade"])
        # Open: cash 10000-360=9640, basis 0+360=360, account_val=10000.
        # Close: cash 9640+384=10024, basis max(0, 360-(384-24))=0,
        # account_val=10024 -- exactly starting cash + the 24 realized gain,
        # never the old buggy 10024+384(=basis 360+384=744)=10408.
        self.assertEqual(values, [10_000.0, 10_000.0, 10_024.0])

    def test_repeated_partial_closes_never_inflate_the_curve(self):
        # Same bug, exercised the way it actually happened live: several
        # partial closes of the same position in a row, each with its own
        # positive notional and outcome='realized'.
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a", starting_cash=10_000.0)
            _insert_action(
                conn, "model-a", action_type="trade", outcome="open",
                ts="2026-08-18T05:17:06+00:00", cash_delta=-1000.0, notional=1000.0,
            )
            running_expected = 10_000.0
            for i in range(4):
                _insert_action(
                    conn, "model-a", action_type="trade", outcome="realized",
                    ts=f"2026-08-18T09:2{i}:00+00:00", cash_delta=260.0,
                    realized_pnl=10.0, notional=270.0,
                )
                running_expected += 10.0  # each close only ever adds its realized gain
            conn.commit()

            curve = agent_trading_stats.agent_equity_curve(conn, "model-a")

        values = [p["account_value"] for p in curve["value_curve"]]
        # Never climbs by anywhere near a closing trade's own notional (270)
        # on top of the realized gain -- the old bug would have landed at
        # 10000 + 4*270 = 11080 by the last point instead of 10040.
        self.assertAlmostEqual(values[-1], running_expected)
        self.assertLess(max(values), 10_100.0)


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


class ForecastLearningTests(unittest.TestCase):
    def test_keeps_forecast_scoring_separate_from_pnl_and_compares_the_market(self):
        with _fixture_conn() as conn:
            conn.execute(
                """
                INSERT INTO agent_thesis_forecasts
                (agent_id, cycle_id, forecast_ts, platform, ticker, model_probability,
                 market_probability, resolved_outcome, resolved_at, brier_score, market_brier_score)
                VALUES ('model-a', 'cycle-1', '2026-08-11T00:00:00+00:00', 'kalshi', 'KXFORECAST',
                        0.70, 0.60, 1, '2026-08-12T00:00:00+00:00', 0.09, 0.16)
                """
            )
            conn.commit()
            learning = agent_trading_stats.compute_forecast_learning(conn, "model-a")

        self.assertEqual(learning["recorded_forecasts"], 1)
        self.assertEqual(learning["resolved_forecasts"], 1)
        self.assertEqual(learning["status"], "small_sample")
        self.assertAlmostEqual(learning["brier_score"], 0.09)
        self.assertAlmostEqual(learning["market_brier_score"], 0.16)
        self.assertAlmostEqual(learning["probability_bias"], -0.30)
        self.assertEqual(learning["recent_reviews"][0]["ticker"], "KXFORECAST")

    def test_keeps_weather_calibration_separate_by_contract_type_and_source(self):
        with _fixture_conn() as conn:
            conn.executemany(
                """
                INSERT INTO agent_thesis_forecasts
                (agent_id, cycle_id, forecast_ts, platform, ticker, model_probability,
                 resolved_outcome, resolved_at, brier_score, market_brier_score,
                 weather_market_type, weather_settlement_source)
                VALUES (?, ?, '2026-08-11T00:00:00+00:00', 'kalshi', ?, ?, ?,
                        '2026-08-12T00:00:00+00:00', ?, ?, ?, ?)
                """,
                [
                    ("model-a", "cycle-high-1", "KXHIGH1", 0.8, 1, 0.04, 0.09,
                     "daily_high_temperature", "nws_daily_climate_report"),
                    ("model-a", "cycle-high-2", "KXHIGH2", 0.6, 0, 0.36, 0.49,
                     "daily_high_temperature", "nws_daily_climate_report"),
                    ("model-a", "cycle-rain", "KXRAIN", 0.3, 1, 0.49, 0.36,
                     "precipitation", "weather_company"),
                ],
            )
            conn.commit()
            learning = agent_trading_stats.compute_forecast_learning(conn, "model-a")

        self.assertEqual(len(learning["weather_calibration"]), 2)
        high = learning["weather_calibration"][0]
        self.assertEqual(high["market_type"], "daily_high_temperature")
        self.assertEqual(high["settlement_source"], "nws_daily_climate_report")
        self.assertEqual(high["resolved_forecasts"], 2)
        self.assertAlmostEqual(high["brier_score"], 0.2)


class RecentActivityTests(unittest.TestCase):
    def test_clean_thesis_display_omits_raw_react_tool_transcript(self):
        thesis = (
            'Planning research. {"thought":"Check the market",'
            '"action":"web_search","args":{"query":"market news"}}'
            '{"thought":"Check the market again",'
            '"action":"web_search","args":{"query":"market news"}}'
        )
        display = agent_trading_stats.clean_thesis_display(thesis)
        self.assertEqual(display, "")

    def test_clean_thesis_display_omits_truncated_json_envelope(self):
        self.assertEqual(
            agent_trading_stats.clean_thesis_display('{"thought": "unfinished tool response'),
            "",
        )

    def test_clean_thesis_display_salvages_final_copy_from_a_truncated_envelope(self):
        # A provider cut off mid-serialisation after it had already written its
        # final copy. Dropping the whole payload left a blank card for a cycle
        # in which the model really did produce a thesis.
        truncated = (
            '{"thought": "All three markets evaluated.", "final": '
            '"### 0. Research Delta\\n- **Action**: PASS\\n- No edge clears the gate.'
        )
        display = agent_trading_stats.clean_thesis_display(truncated)
        self.assertIn("### 0. Research Delta", display)
        self.assertIn("**Action**: PASS", display)
        self.assertNotIn('"thought"', display)

    def test_clean_thesis_display_still_drops_a_bare_tool_envelope(self):
        # No final copy anywhere -> nothing reader-facing to salvage.
        self.assertEqual(
            agent_trading_stats.clean_thesis_display(
                '{"thought": "I need to research these markets."," "action": "web_search", '
                '"args": {"query": "x"}}'
            ),
            "",
        )

    def test_clean_thesis_display_uses_only_explicit_reader_ready_json_field(self):
        display = agent_trading_stats.clean_thesis_display(json.dumps({
            "thought": "Private chain of thought",
            "action": "place_trade",
            "final": "### 1. Decision & Execution\n- **Action**: HOLD",
        }))
        self.assertEqual(display, "### 1. Decision & Execution\n- **Action**: HOLD")

    def test_clean_thesis_display_keeps_only_first_complete_template(self):
        thesis = (
            "### 0. Research Delta\n- **Strategy**: EVIDENCE_EDGE\n"
            "### 1. Decision & Execution\n- **Action**: HOLD\n\n"
            "### 0. Research Delta\n- **Strategy**: CATALYST_EDGE\n"
            "### 1. Decision & Execution\n- **Action**: BUY YES"
        )
        display = agent_trading_stats.clean_thesis_display(thesis)
        self.assertIn("EVIDENCE_EDGE", display)
        self.assertNotIn("CATALYST_EDGE", display)

    def test_clean_thesis_display_does_not_repeat_matching_json_fields(self):
        thesis = json.dumps({
            "final": "### 1. Decision & Execution\n- **Action**: HOLD",
            "thought": "### 1. Decision & Execution\n- **Action**: HOLD",
        })
        display = agent_trading_stats.clean_thesis_display(thesis)
        self.assertEqual(display, "### 1. Decision & Execution\n- **Action**: HOLD")
        self.assertNotIn("Detailed Analysis", display)

    def test_clean_thesis_display_strips_preamble_before_template(self):
        thesis = (
            "Here is my step-by-step thinking about today's market conditions:\n"
            "I will evaluate Kalshi and Polymarket.\n\n"
            "### 0. Research Delta\n- **Strategy**: PASS\n"
            "### 1. Decision & Execution\n- **Action**: HOLD"
        )
        display = agent_trading_stats.clean_thesis_display(thesis)
        self.assertTrue(display.startswith("### 0. Research Delta"))
        self.assertNotIn("Here is my step-by-step thinking", display)

    def test_clean_thesis_display_handles_unstructured_deliberation(self):
        raw_deliberation = (
            "Let me reconsider my analysis. I have three candidate markets:\n"
            "1. Market A: YES at 0.08\n2. Market B: NO at 0.86\n"
            + "Repeating internal analysis thoughts. " * 30
        )
        display = agent_trading_stats.clean_thesis_display(raw_deliberation)
        self.assertIn("Research cycle completed without standard thesis template", display)
        self.assertNotIn("Repeating internal analysis thoughts", display)

    def test_clean_thesis_display_strips_unmodified_answer_dump(self):
        thesis = (
            "### 0. Research Delta\n- **Strategy**: PASS\n\n"
            "### 1. Decision & Execution\n- **Action**: HOLD\n\n"
            "---\n"
            "The model's own answer, unmodified:\n\n"
            "Let me evaluate my candidate markets. Step 1: scratchpad internal monologue..."
        )
        display = agent_trading_stats.clean_thesis_display(thesis)
        self.assertIn("### 1. Decision & Execution", display)
        self.assertNotIn("The model's own answer, unmodified", display)
        self.assertNotIn("scratchpad internal monologue", display)

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

    def test_zero_fill_is_labeled_unfilled_not_trade(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a")
            _insert_action(conn, "model-a", action_type="trade", quantity=0.0, outcome="open")
            conn.commit()
            items = agent_trading_stats.recent_activity(conn, {}, limit=10)
            rows = agent_trading_stats.compute_agent_leaderboard(conn, {})

        self.assertEqual(items[0]["type"], "unfilled_order")
        self.assertEqual(rows[0]["trade_count"], 0)

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

    def test_repeated_same_agent_theses_are_collapsed_to_the_latest(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-a")
            _insert_cycle(conn, "model-a", cycle_id="15m:1", ts="2026-08-11T00:00:00+00:00",
                          thesis="No material change. Hold the position.")
            _insert_cycle(conn, "model-a", cycle_id="15m:2", ts="2026-08-11T00:05:00+00:00",
                          thesis="No material change. Hold the position.")
            _insert_cycle(conn, "model-b", cycle_id="15m:3", ts="2026-08-11T00:10:00+00:00",
                          thesis="No material change. Hold the position.")
            conn.commit()
            items = agent_trading_stats.recent_activity(conn, {}, limit=10)

        theses = [item for item in items if item["type"] == "thesis"]
        self.assertEqual(len(theses), 2)
        self.assertEqual(theses[0]["agent_id"], "model-b")
        self.assertEqual(theses[1]["agent_id"], "model-a")
        self.assertEqual(theses[1]["cycle_id"], "15m:2")


if __name__ == "__main__":
    unittest.main()


class RejectionContextTests(unittest.TestCase):
    """A rejection on the feed used to be a price, a quantity and no cause."""

    def test_versioned_metadata_yields_reasons(self):
        meta = json.dumps({
            "audit": {"version": 1, "risk": {"reasons": ["drawdown_limit"]}},
        })
        self.assertEqual(
            agent_trading_stats.rejection_context(meta),
            {"reasons": ["drawdown_limit"]},
        )

    def test_a_pre_sizing_refusal_says_its_quantity_is_the_models_ask(self):
        """This is the one that misleads.

        llama's 00:12 refusal recorded quantity 7000.9 -- what the model
        asked for, refused before _sizing_plan ever ran -- next to a filled
        order of 7000.88862548003, which Kelly produced. Identical-looking
        numbers, entirely different provenance.
        """
        meta = json.dumps({
            "audit": {
                "version": 1,
                "status": "rejected_before_sizing",
                "risk": {
                    "reasons": ["no_executable_price"],
                    "rejected_before_sizing": True,
                },
            },
        })
        self.assertEqual(
            agent_trading_stats.rejection_context(meta),
            {"reasons": ["no_executable_price"], "quantity_is_pre_sizing": True},
        )

    def test_legacy_rows_fall_back_to_risk_guard(self):
        meta = json.dumps({"risk_guard": {"reasons": ["concentration_limit"]}})
        self.assertEqual(
            agent_trading_stats.rejection_context(meta),
            {"reasons": ["concentration_limit"]},
        )

    def test_unreadable_or_empty_metadata_adds_nothing(self):
        for meta in (None, "", "not json", json.dumps([1, 2]), json.dumps({})):
            with self.subTest(meta=meta):
                self.assertEqual(agent_trading_stats.rejection_context(meta), {})


class FillContextTests(unittest.TestCase):
    """A fill of 100 reads as a chosen size unless the target is beside it."""

    def _meta(self, *, target=None, status=None):
        audit = {"version": 1}
        if target is not None:
            audit["sizing"] = {"target_quantity": target}
        if status is not None:
            audit["execution"] = {"fill_status": status}
        return json.dumps({"audit": audit})

    def test_a_partial_fill_reports_the_target_and_the_fraction(self):
        """llama's real numbers from the 04:18 tick."""
        ctx = agent_trading_stats.fill_context(
            self._meta(target=6533.986210065798, status="shadow_filled_partial"),
            100.0,
        )
        self.assertEqual(ctx["fill_status"], "shadow_filled_partial")
        self.assertEqual(ctx["target_quantity"], 6533.98621)
        self.assertAlmostEqual(ctx["filled_fraction"], 0.015304, places=5)

    def test_a_full_fill_adds_nothing(self):
        self.assertEqual(
            agent_trading_stats.fill_context(self._meta(target=50.0), 50.0), {},
        )

    def test_a_rounding_shortfall_is_not_a_liquidity_story(self):
        # 99.5% filled: rounding, not the book refusing size.
        self.assertEqual(
            agent_trading_stats.fill_context(self._meta(target=100.0), 99.5), {},
        )

    def test_missing_or_unreadable_metadata_adds_nothing(self):
        for meta in (None, "", "not json", json.dumps({}), json.dumps({"audit": 5})):
            with self.subTest(meta=meta):
                self.assertEqual(agent_trading_stats.fill_context(meta, 10.0), {})

    def test_a_non_numeric_quantity_still_yields_the_status(self):
        ctx = agent_trading_stats.fill_context(
            self._meta(target=10.0, status="shadow_filled_partial"), None,
        )
        self.assertEqual(ctx, {"fill_status": "shadow_filled_partial"})


class FillEfficiencyTests(unittest.TestCase):
    """Rationing is only legible in aggregate."""

    def _row(self, target, filled):
        meta = json.dumps({"audit": {"version": 1, "sizing": {"target_quantity": target}}})
        return (meta, filled)

    def test_median_and_partial_count_across_trades(self):
        """The three real fills from the 04:17-04:20 tick."""
        rows = [
            self._row(6533.986210065798, 100.0),   # 1.5%
            self._row(475.5713145290175, 22.43),   # 4.7%
            self._row(402.201399882214, 22.43),    # 5.6%
        ]
        out = agent_trading_stats.fill_efficiency(rows)
        self.assertAlmostEqual(out["median_fill_fraction"], 0.047166, places=5)
        self.assertEqual(out["partially_filled_count"], 3)
        self.assertEqual(out["sized_trade_count"], 3)

    def test_an_even_count_averages_the_middle_two(self):
        rows = [self._row(100.0, 10.0), self._row(100.0, 30.0)]
        self.assertAlmostEqual(
            agent_trading_stats.fill_efficiency(rows)["median_fill_fraction"], 0.2,
        )

    def test_full_fills_report_one_and_no_partials(self):
        rows = [self._row(50.0, 50.0), self._row(20.0, 20.0)]
        out = agent_trading_stats.fill_efficiency(rows)
        self.assertEqual(out["median_fill_fraction"], 1.0)
        self.assertEqual(out["partially_filled_count"], 0)

    def test_an_overfill_is_capped_at_one(self):
        out = agent_trading_stats.fill_efficiency([self._row(10.0, 12.0)])
        self.assertEqual(out["median_fill_fraction"], 1.0)

    def test_nothing_reported_when_no_trade_carried_a_target(self):
        rows = [
            (None, 10.0),
            ("not json", 10.0),
            (json.dumps({"audit": {"version": 1}}), 10.0),
            (json.dumps({"audit": {"version": 1, "sizing": {"target_quantity": 0}}}), 10.0),
        ]
        self.assertEqual(agent_trading_stats.fill_efficiency(rows), {})


class FillEfficiencyResetTests(unittest.TestCase):
    """Fill stats must start where trade_count starts.

    Every other leaderboard query filters on the latest admin_reset. The
    fill query did not, so a reset account reported fill stats over trades
    trade_count had already dropped -- and sized_trade_count could come out
    larger than trade_count, which cannot be true.
    """

    def _sized(self, target):
        return json.dumps(
            {"audit": {"version": 1, "sizing": {"target_quantity": target}}}
        )

    def test_trades_before_the_reset_are_excluded(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-r")
            # Before the reset: a badly rationed fill that must not count.
            _insert_action(
                conn, "model-r", ts="2026-08-01T00:00:00+00:00",
                quantity=1.0, metadata_json=self._sized(1000.0),
            )
            _insert_action(
                conn, "model-r", action_type="admin_reset",
                ts="2026-08-05T00:00:00+00:00",
            )
            # After the reset: a full fill.
            _insert_action(
                conn, "model-r", ts="2026-08-10T00:00:00+00:00",
                quantity=50.0, metadata_json=self._sized(50.0),
            )
            conn.commit()

            row = next(
                r for r in agent_trading_stats.compute_agent_leaderboard(conn, {})
                if r["agent_id"] == "model-r"
            )

        self.assertEqual(row["sized_trade_count"], 1)
        self.assertEqual(row["median_fill_fraction"], 1.0)
        self.assertEqual(row["partially_filled_count"], 0)
        self.assertLessEqual(row["sized_trade_count"], row["trade_count"])


class FillEfficiencyZeroFillTests(unittest.TestCase):
    """A zero fill is usually a pricing choice, not a liquidity fact.

    Of llama-3.3-70b-instruct's 17 zero-fill trades, six are
    shadow_unfilled_below_market and none are shadow_unfilled_no_depth, so
    counting them would fold "priced away from the market" into a number
    that is supposed to mean "the book was not there".
    """

    def test_a_zero_fill_is_not_counted_as_total_rationing(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-z")
            meta = json.dumps(
                {"audit": {"version": 1, "sizing": {"target_quantity": 500.0}}}
            )
            _insert_action(conn, "model-z", quantity=0.0, metadata_json=meta)
            _insert_action(conn, "model-z", quantity=500.0, metadata_json=meta)
            conn.commit()

            row = next(
                r for r in agent_trading_stats.compute_agent_leaderboard(conn, {})
                if r["agent_id"] == "model-z"
            )

        # Only the filled trade counts; the zero-fill would otherwise drag
        # the median to 0.5 and assert rationing that did not happen.
        self.assertEqual(row["sized_trade_count"], 1)
        self.assertEqual(row["median_fill_fraction"], 1.0)
        self.assertEqual(row["partially_filled_count"], 0)


class FillStatusNamesTests(unittest.TestCase):
    """The exclusion set has to name statuses that actually exist.

    The first version excluded "shadow_filled_full" and "filled", neither of
    which _extract_filled_quantity can emit -- so it excluded nothing and
    every ordinary fill carried a fill_status. deepseek-v4-flash's routine
    full fill was published with fill_status=shadow_assumed_full.
    """

    def _meta(self, status):
        return json.dumps(
            {"audit": {"version": 1, "execution": {"fill_status": status}}}
        )

    def test_every_excluded_status_is_one_the_code_can_emit(self):
        from pathlib import Path as _P

        source = (
            _P(__file__).resolve().parents[1]
            / "src" / "analyzing_llm_rationale" / "benchmark_tools.py"
        ).read_text(encoding="utf-8")
        for status in agent_trading_stats._COMPLETE_FILL_STATUSES:
            with self.subTest(status=status):
                self.assertIn(f'"{status}"', source)

    def test_complete_fills_carry_no_status(self):
        for status in ("shadow_assumed_full", "venue_status_assumed_full",
                       "venue_unknown_assumed_full"):
            with self.subTest(status=status):
                self.assertEqual(
                    agent_trading_stats.fill_context(self._meta(status), 10.0), {},
                )

    def test_shortfall_statuses_are_reported(self):
        for status in ("shadow_filled_partial", "shadow_unfilled_below_market",
                       "shadow_unfilled_no_depth", "venue_status_assumed_zero",
                       "venue_reported_remaining"):
            with self.subTest(status=status):
                self.assertEqual(
                    agent_trading_stats.fill_context(self._meta(status), 10.0),
                    {"fill_status": status},
                )


class LeaderboardAccountingInvariantTests(unittest.TestCase):
    """The identities the public leaderboard implicitly claims.

    Reconciled against the live board on 2026-09-07 and exact to the cent
    for all eight agents. Pinned here because these are the numbers people
    read as a performance claim, and a drift in any of them would look like
    a trading result rather than an accounting error.
    """

    def _rows(self, conn, quotes):
        return agent_trading_stats.compute_agent_leaderboard(conn, quotes)

    def test_account_value_is_cash_plus_basis_plus_unrealized(self):
        quotes = {("kalshi", "KXFOO-26"): {"yes_bid": 0.55}}
        with _fixture_conn() as conn:
            _insert_account(conn, "flat", cash=10_400.0)
            _insert_account(conn, "holder", cash=9_600.0)
            _insert_position(conn, "holder", quantity=100.0, cost_basis=40.0)
            conn.commit()
            rows = self._rows(conn, quotes)

        self.assertTrue(rows)
        for row in rows:
            with self.subTest(agent=row["agent_id"]):
                basis = sum(
                    float(p.get("cost_basis", 0.0))
                    for p in (row.get("open_positions") or [])
                )
                self.assertAlmostEqual(
                    row["account_value"],
                    row["cash"] + basis + row["unrealized_pnl"],
                    places=6,
                    msg="account_value must equal cash + open basis + unrealized",
                )

    def test_total_pnl_and_return_pct_follow_from_account_value(self):
        with _fixture_conn() as conn:
            _insert_account(conn, "model-p", starting_cash=10_000.0, cash=8_500.0)
            conn.commit()
            rows = self._rows(conn, {})

        for row in rows:
            with self.subTest(agent=row["agent_id"]):
                self.assertAlmostEqual(
                    row["total_pnl"],
                    row["account_value"] - row["starting_cash"],
                    places=6,
                )
                if row["starting_cash"]:
                    self.assertAlmostEqual(
                        row["return_pct"],
                        row["total_pnl"] / row["starting_cash"] * 100.0,
                        places=3,
                    )
