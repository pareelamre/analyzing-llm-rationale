"""One market where model and crowd agreed erased every order recommendation.

build_edge_board keeps rows where the model and the market agree exactly
(min_abs_edge defaults to zero) and gives them side None and entry_price
None, because there is no directional trade. strategy_filter_edge_entry
then read the price with `entry.get("entry_price", 0.5)` -- which returns
None, not the default, because the key is present -- and under the
"smart" strategy compared None < 0.20 and raised TypeError.

The exception never surfaced. The chat path builds this context inside

    try:
        ...
        ctx = _edge_board_order_context(trl)
    except Exception:
        pass

so a single agreeing market anywhere on the board removed every
recommendation from a trading-intent reply, with nothing logged. "smart"
is the strategy the board ranks on, so it is the one most likely to be
picked. Under the other strategies the same row passed the filter and
rendered as "Bet ? @ 50%" at zero edge.

These drive the real board builder's output shape through the real
context builder, so the fixture carries None exactly where production
does rather than an idealised row.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.live_track_record import (  # noqa: E402
    edge_board_order_context,
    strategy_filter_edge_entry,
)


def tradeable(question, side="NO", market=0.40, model=0.25):
    return {
        "question": question, "platform": "Polymarket", "side": side,
        "market_probability": market, "model_probability": model,
        "abs_edge": round(abs(model - market), 3),
        "entry_price": market if side == "YES" else 1.0 - market,
        "payout_odds": 1.5, "market_url": "https://example.test/" + question,
        "track_record": {"skill_significant": True},
    }


def agreeing(question="Model and market agree"):
    """The shape build_edge_board emits when signed == 0."""
    return {
        "question": question, "platform": "Kalshi", "side": None,
        "market_probability": 0.5, "model_probability": 0.5,
        "abs_edge": 0.0, "entry_price": None, "payout_odds": None,
        "market_url": "https://example.test/agree", "track_record": {},
    }


PAPER_PNL = {
    "flat": {"roi": 0.05, "n_bets": 50},
    "smart": {"roi": 0.12, "n_bets": 50},
}


class FilterTests(unittest.TestCase):
    def test_a_side_less_entry_does_not_raise_under_smart(self):
        self.assertFalse(strategy_filter_edge_entry(agreeing(), "smart"))

    def test_a_side_less_entry_is_excluded_under_every_strategy(self):
        for strategy in ("smart", "flat", "half_kelly", "crowd_baseline"):
            with self.subTest(strategy=strategy):
                self.assertFalse(strategy_filter_edge_entry(agreeing(), strategy))

    def test_a_real_trade_still_passes(self):
        self.assertTrue(strategy_filter_edge_entry(tradeable("keep"), "smart"))
        self.assertTrue(strategy_filter_edge_entry(tradeable("keep"), "flat"))

    def test_the_smart_price_guard_still_applies(self):
        extreme = tradeable("extreme", side="YES", market=0.91, model=0.95)
        self.assertFalse(strategy_filter_edge_entry(extreme, "smart"))


class OrderContextTests(unittest.TestCase):
    def test_one_agreeing_market_does_not_erase_the_others(self):
        context = edge_board_order_context({
            "paper_pnl": PAPER_PNL,
            "edge_board": [tradeable("first"), agreeing(), tradeable("second")],
        })
        self.assertIn("Live order recommendations", context)
        self.assertIn("first", context)
        self.assertIn("second", context)

    def test_the_agreeing_market_is_not_rendered_as_a_bet(self):
        context = edge_board_order_context({
            "paper_pnl": {"flat": {"roi": 0.05, "n_bets": 50}},
            "edge_board": [tradeable("first"), agreeing()],
        })
        self.assertNotIn("Model and market agree", context)
        self.assertNotIn("Bet None", context)

    def test_a_board_of_only_agreement_yields_no_context_rather_than_an_error(self):
        self.assertEqual(
            edge_board_order_context(
                {"paper_pnl": PAPER_PNL, "edge_board": [agreeing()]}
            ),
            "",
        )


if __name__ == "__main__":
    unittest.main()
