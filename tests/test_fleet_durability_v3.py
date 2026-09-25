"""Unit tests for Fleet Durability v3 upgrades:
1. Cross-venue parity: Polymarket pre-expiry loss exits and early profit-harvesting.
2. Broken-thesis / severe adverse drift stop-loss exit rule.
3. Candidate pre-filtering by risk / cluster capacity.
4. Fallback indicative pricing in accounting snapshot for unpriced / illiquid positions.
"""

from __future__ import annotations

import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import agent_trading_tick  # noqa: E402

from analyzing_llm_rationale import accounting, market_data  # noqa: E402
from tests.test_agent_trading_tick import _quote  # noqa: E402
from tests.test_pre_expiry_exit_rule import NOW, held, iso  # noqa: E402


@contextmanager
def _dummy_transaction():
    yield None


class CrossVenueParityTests(unittest.TestCase):
    def test_polymarket_pre_expiry_exit_default_enabled(self):
        position = held(platform="polymarket", ticker="poly-loss-mkt", avg=0.50)
        quote = {
            "platform": "Polymarket",
            "ident": "poly-loss-mkt",
            "question": "Will event occur?",
            "yes_bid": 0.10,
            "yes_ask": 0.12,
            "close_time": iso(4),
        }
        # Down 80% (> 30%) with 4 hours left
        candidates = agent_trading_tick._pre_expiry_exit_candidates([position], [quote], now=NOW)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["platform"], "polymarket")
        self.assertEqual(candidates[0]["ticker"], "poly-loss-mkt")
        self.assertAlmostEqual(candidates[0]["bid"], 0.10)

    def test_polymarket_pre_expiry_exit_disabled_when_flag_off(self):
        position = held(platform="polymarket", ticker="poly-loss-mkt", avg=0.50)
        quote = {
            "platform": "Polymarket",
            "ident": "poly-loss-mkt",
            "question": "Will event occur?",
            "yes_bid": 0.10,
            "yes_ask": 0.12,
            "close_time": iso(4),
        }
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_POLYMARKET_PRE_EXPIRY_EXIT": "off"}):
            candidates = agent_trading_tick._pre_expiry_exit_candidates([position], [quote], now=NOW)
            self.assertEqual(candidates, [])

    def test_polymarket_early_profit_harvest_default_enabled(self):
        position = held(platform="polymarket", ticker="poly-win-mkt", avg=0.10)
        quote = {
            "platform": "Polymarket",
            "ident": "poly-win-mkt",
            "question": "Will event occur?",
            "yes_bid": 0.88,
            "yes_ask": 0.90,
            "close_time": iso(48),
        }
        # Bought at 0.10, current bid 0.88 -> profit 0.78 / max profit 0.90 = 86.7% >= 85%
        candidates = agent_trading_tick._early_profit_harvest_candidates([position], [quote], now=NOW)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["platform"], "polymarket")
        self.assertEqual(candidates[0]["ticker"], "poly-win-mkt")
        self.assertAlmostEqual(candidates[0]["bid"], 0.88)

    def test_polymarket_early_profit_harvest_disabled_when_flag_off(self):
        position = held(platform="polymarket", ticker="poly-win-mkt", avg=0.10)
        quote = {
            "platform": "Polymarket",
            "ident": "poly-win-mkt",
            "question": "Will event occur?",
            "yes_bid": 0.88,
            "yes_ask": 0.90,
            "close_time": iso(48),
        }
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_POLYMARKET_EARLY_HARVEST": "off"}):
            candidates = agent_trading_tick._early_profit_harvest_candidates([position], [quote], now=NOW)
            self.assertEqual(candidates, [])


class BrokenThesisExitTests(unittest.TestCase):
    def test_candidate_selected_when_loss_ge_60_and_bid_le_25c(self):
        position = held(platform="kalshi", ticker="KXADVERSE", avg=0.50)
        quote = _quote("KXADVERSE", bid=0.15, ask=0.18, close=iso(48))
        # Entry 0.50, bid 0.15 -> down 70% >= 60%, bid 0.15 <= 0.25
        candidates = agent_trading_tick._broken_thesis_exit_candidates([position], [quote], now=NOW)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["ticker"], "KXADVERSE")
        self.assertAlmostEqual(candidates[0]["bid"], 0.15)
        self.assertAlmostEqual(candidates[0]["change"], -0.70)

    def test_candidate_ignored_if_bid_exceeds_max_bid(self):
        # Entry 0.80, bid 0.30 -> down 62.5% >= 60%, but bid 0.30 > 0.25
        position = held(platform="kalshi", ticker="KXEXPENSIVE", avg=0.80)
        quote = _quote("KXEXPENSIVE", bid=0.30, ask=0.32, close=iso(48))
        candidates = agent_trading_tick._broken_thesis_exit_candidates([position], [quote], now=NOW)
        self.assertEqual(candidates, [])

    def test_candidate_ignored_if_loss_under_60_pct(self):
        # Entry 0.30, bid 0.15 -> down 50% < 60%, even though bid <= 0.25
        position = held(platform="kalshi", ticker="KXMODEST", avg=0.30)
        quote = _quote("KXMODEST", bid=0.15, ask=0.18, close=iso(48))
        candidates = agent_trading_tick._broken_thesis_exit_candidates([position], [quote], now=NOW)
        self.assertEqual(candidates, [])

    def test_broken_thesis_disabled_by_env(self):
        position = held(platform="kalshi", ticker="KXADVERSE", avg=0.50)
        quote = _quote("KXADVERSE", bid=0.15, ask=0.18, close=iso(48))
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_BROKEN_THESIS_EXIT": "off"}):
            candidates = agent_trading_tick._broken_thesis_exit_candidates([position], [quote], now=NOW)
            self.assertEqual(candidates, [])

    def test_learning_lesson_formats_correctly(self):
        lesson = agent_trading_tick._learning_lesson(
            action_type="close",
            realized_pnl=-35.0,
            initiated_by=agent_trading_tick.BROKEN_THESIS_EXIT_RULE,
        )
        self.assertIn("broken thesis exit rule closed this position, not you", lesson)
        self.assertIn("liquidated to salvage remaining principal", lesson)

    def test_broken_thesis_execution_calls_close(self):
        position = held(platform="polymarket", ticker="poly-adverse", side="yes", avg=0.60, quantity=50.0)
        quote = {
            "platform": "Polymarket",
            "ident": "poly-adverse",
            "question": "Q?",
            "yes_bid": 0.12,
            "yes_ask": 0.15,
            "close_time": iso(72),
        }
        with mock.patch("agent_trading_tick.benchmark_tools._account_transaction", side_effect=_dummy_transaction), \
             mock.patch("agent_trading_tick.benchmark_tools._account_summary", return_value={"open_positions": [position]}), \
             mock.patch("agent_trading_tick.benchmark_tools.place_trade") as mock_place:
            mock_place.return_value = {"ok": True, "execution": {"filled_quantity": 50.0, "realized_pnl": -24.0}}

            outcomes = agent_trading_tick._run_broken_thesis_exits(
                agent_id="minimax-01",
                held_quotes=[quote],
                now=NOW,
            )
            self.assertEqual(len(outcomes), 1)
            mock_place.assert_called_once()
            call_order = mock_place.call_args[0][0]
            call_ctx = mock_place.call_args[0][1]
            self.assertEqual(call_order["platform"], "polymarket")
            self.assertEqual(call_order["ticker"], "poly-adverse")
            self.assertEqual(call_order["side"], "no")  # opposite side for close
            self.assertEqual(call_order["sizing_mode"], "close")
            self.assertEqual(call_ctx.initiated_by, agent_trading_tick.BROKEN_THESIS_EXIT_RULE)


class CapacityExhaustedCandidateFilterTests(unittest.TestCase):
    def test_drops_candidates_when_capacity_exhausted(self):
        quotes = [
            {"platform": "kalshi", "ident": "IRAN-LEAD-2026", "ticker": "IRAN-LEAD-2026"},
            {"platform": "kalshi", "ident": "FED-RATE-CUT", "ticker": "FED-RATE-CUT"},
        ]
        # Mock policy with account value $1000, 10% market limit ($100), 15% cluster limit ($150)
        mock_policy = mock.MagicMock(
            account_value=1000.0,
            concentration_limit=0.10,  # $100 cap
            cluster_concentration_limit=0.15,  # $150 cap
        )
        mock_profile = mock.MagicMock(max_cluster_concentration_pct=None)
        # 145 >= 150 * 0.95 (142.5) -> cluster capacity is exhausted
        mock_open_positions = [
            {"platform": "kalshi", "ticker": "IRAN-OTHER-MKT", "cost_basis": 145.0},
        ]

        with mock.patch("agent_trading_tick.benchmark_tools._risk_guard_policy", return_value=mock_policy), \
             mock.patch("agent_trading_tick.benchmark_tools.get_agent_profile", return_value=mock_profile), \
             mock.patch("agent_trading_tick.benchmark_tools._account_transaction", side_effect=_dummy_transaction), \
             mock.patch("agent_trading_tick.benchmark_tools._account_summary", return_value={"open_positions": mock_open_positions}), \
             mock.patch("agent_trading_tick.benchmark_tools._extract_market_cluster") as mock_cluster:

            def fake_cluster(ticker, platform=None):
                if "IRAN" in ticker:
                    return "iran_succession"
                return "monetary_policy"

            mock_cluster.side_effect = fake_cluster

            kept = agent_trading_tick._drop_capacity_exhausted_candidates(quotes, agent_id="llama-3.3-70b")
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0]["ident"], "FED-RATE-CUT")

    def test_keeps_all_candidates_when_no_agent_id(self):
        quotes = [{"platform": "kalshi", "ident": "TICKER1"}]
        kept = agent_trading_tick._drop_capacity_exhausted_candidates(quotes, agent_id=None)
        self.assertEqual(kept, quotes)


class IndicativePricingSnapshotTests(unittest.TestCase):
    def test_snapshot_includes_indicative_price_when_bid_is_zero(self):
        acct = accounting.PredictionMarketAccount(starting_cash=1000.0)
        acct.buy(platform="kalshi", ident="KXTEST", side="yes", quantity=100.0, price=0.40)

        # Quote has 0 bid, 0 ask, but known probability of 0.35
        raw_quote = accounting.MarketQuote(
            platform="kalshi",
            ticker="KXTEST",
            yes_bid=0.0,
            yes_ask=0.0,
            yes_probability=0.35,
        )
        snap = acct.snapshot({("kalshi", "KXTEST"): raw_quote})
        open_pos = snap["open_positions"]
        self.assertEqual(len(open_pos), 1)
        self.assertIsNone(open_pos[0]["current_price"])
        self.assertEqual(open_pos[0]["indicative_price"], 0.35)
        self.assertEqual(open_pos[0]["valuation_status"], "unpriced_no_executable_bid")

    def test_snapshot_no_side_indicative_price(self):
        acct = accounting.PredictionMarketAccount(starting_cash=1000.0)
        acct.buy(platform="kalshi", ident="KXTEST", side="no", quantity=100.0, price=0.60)

        raw_quote = accounting.MarketQuote(
            platform="kalshi",
            ticker="KXTEST",
            yes_bid=0.0,
            yes_ask=0.0,
            yes_probability=0.30,
        )
        snap = acct.snapshot({("kalshi", "KXTEST"): raw_quote})
        open_pos = snap["open_positions"]
        self.assertEqual(len(open_pos), 1)
        self.assertIsNone(open_pos[0]["current_price"])
        # For NO holding, indicative_price should be 1.0 - 0.30 = 0.70
        self.assertAlmostEqual(open_pos[0]["indicative_price"], 0.70)


if __name__ == "__main__":
    unittest.main()
