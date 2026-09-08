"""Where whale flow turns numbers into a label, and how it gets its rows.

Three lines survived the whole 2,038-test suite, across both existing
whale test modules:

    if sentiment_index >= 65.0:   ->  > 65.0     SURVIVED
    elif sentiment_index <= 35.0: ->  < 35.0     SURVIVED
    fetch_recent_trades(...) or []  ->  and []   SURVIVED

The first two are the boundary of a published label: a book split
exactly 65/35 is called BULLISH ACCUMULATION, and the strict form would
call the same book NEUTRAL. Nothing sat on either boundary, so the
choice of >= over > was never expressed anywhere but the source.

The third is worse than a boundary. `or []` is there so a venue
returning None does not break the loop. Turned into `and []` it discards
every row from a venue that did return trades, and the analysis then
reports an empty tape -- 0 prints, 50.0 sentiment, NEUTRAL. That is the
same shape the except branch produces, so the endpoint keeps answering
and looks merely quiet rather than broken.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import market_data  # noqa: E402
from analyzing_llm_rationale.whale_flow import (  # noqa: E402
    analyze_whale_trades,
    fetch_live_whale_flow,
)


def print_of(notional, side, title="M"):
    """One tape row worth exactly `notional` dollars on `side`."""
    return {"price": 0.50, "size": notional / 0.50, "outcome": side, "title": title}


class SentimentBoundaryTests(unittest.TestCase):
    """The label is published, so its edges are part of the contract."""

    @staticmethod
    def _label(yes_usd, no_usd):
        result = analyze_whale_trades(
            [print_of(yes_usd, "Yes"), print_of(no_usd, "No")]
        )
        return result["whale_sentiment_index_pct"], result["whale_sentiment_label"]

    def test_exactly_sixty_five_is_already_bullish(self):
        index, label = self._label(650.0, 350.0)
        self.assertEqual(index, 65.0)
        self.assertEqual(label, "BULLISH ACCUMULATION")

    def test_exactly_thirty_five_is_already_bearish(self):
        index, label = self._label(350.0, 650.0)
        self.assertEqual(index, 35.0)
        self.assertEqual(label, "BEARISH DISTRIBUTION")

    def test_just_inside_either_boundary_is_neutral(self):
        self.assertEqual(self._label(649.0, 351.0), (64.9, "NEUTRAL / BALANCED"))
        self.assertEqual(self._label(351.0, 649.0), (35.1, "NEUTRAL / BALANCED"))

    def test_an_empty_tape_is_neutral_rather_than_a_division(self):
        result = analyze_whale_trades([])
        self.assertEqual(result["whale_sentiment_index_pct"], 50.0)
        self.assertEqual(result["whale_sentiment_label"], "NEUTRAL / BALANCED")
        self.assertEqual(result["total_whale_volume_usd"], 0.0)

    def test_a_one_sided_book_reaches_the_extremes(self):
        self.assertEqual(self._label(1000.0, 0.0)[0], 100.0)
        self.assertEqual(self._label(0.0, 1000.0)[0], 0.0)

    def test_prints_under_the_threshold_are_not_counted_at_all(self):
        """The filter is >=, so a print exactly at the minimum counts."""
        at_the_line = analyze_whale_trades([print_of(250.0, "Yes")])
        self.assertEqual(at_the_line["n_whale_prints"], 1)
        below = analyze_whale_trades([print_of(249.0, "Yes")])
        self.assertEqual(below["n_whale_prints"], 0)
        self.assertEqual(below["total_whale_volume_usd"], 0.0)


class LiveFetchTests(unittest.TestCase):
    """fetch_live_whale_flow must keep the rows a venue actually returned."""

    def _patch(self, fake):
        original = market_data.fetch_recent_trades
        market_data.fetch_recent_trades = fake
        self.addCleanup(
            setattr, market_data, "fetch_recent_trades", original
        )

    def test_rows_from_a_venue_reach_the_analysis(self):
        self._patch(lambda venue, limit=40: [print_of(400.0, "Yes")])
        result = fetch_live_whale_flow()
        self.assertEqual(
            result["n_whale_prints"], 2, "one print from each of the two venues"
        )
        self.assertEqual(result["total_whale_volume_usd"], 800.0)

    def test_a_venue_returning_nothing_does_not_take_the_other_down(self):
        def only_kalshi_answers(venue, limit=40):
            return [print_of(400.0, "Yes")] if venue == "kalshi" else None

        self._patch(only_kalshi_answers)
        result = fetch_live_whale_flow()
        self.assertEqual(result["n_whale_prints"], 1)
        self.assertEqual(result["total_whale_volume_usd"], 400.0)

    def test_both_venues_are_asked_and_each_print_is_tagged(self):
        asked = []

        def record(venue, limit=40):
            asked.append(venue)
            return [print_of(400.0, "Yes")]

        self._patch(record)
        result = fetch_live_whale_flow()
        self.assertEqual(sorted(asked), ["kalshi", "polymarket"])
        self.assertEqual(
            sorted(p["platform"] for p in result["top_prints"]),
            ["Kalshi", "Polymarket"],
        )

    def test_a_venue_that_raises_degrades_to_a_neutral_report(self):
        def explode(venue, limit=40):
            raise RuntimeError("venue down")

        self._patch(explode)
        result = fetch_live_whale_flow()
        self.assertEqual(result["n_whale_prints"], 0)
        self.assertEqual(result["whale_sentiment_label"], "NEUTRAL / BALANCED")


if __name__ == "__main__":
    unittest.main()
