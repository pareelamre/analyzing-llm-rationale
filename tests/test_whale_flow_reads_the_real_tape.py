"""Whale flow parsed a trade shape neither venue publishes.

fetch_recent_trades returns each venue's rows unchanged. Captured live on
2026-09-07:

    Kalshi      ticker, taker_side, yes_price_dollars, no_price_dollars,
                count_fp (all strings)
    Polymarket  title, slug, asset, conditionId, side (BUY/SELL),
                outcome (Yes/No), price, size

The parser looked for `price`/`size`/`ticker`/`market_title`. Three
consequences:

1. Kalshi has neither `price` nor `size`, so every row priced at the 0.50
   default with a size of 0. A notional of 0 clears no min_notional, so
   Kalshi trades could not appear at all -- on an endpoint whose summary is
   "across Polymarket and Kalshi". Measured: 40 Kalshi rows produced 0
   prints before and 30 after.

2. Polymarket has `title` and `slug` but not `market_title` or `ticker`, so
   every print read platform "Venue", ticker "", title "Market".

3. Polymarket's `side` is BUY/SELL and the position is in `outcome`, so a
   buy of NO was counted as YES, inverting sentiment for those trades.

The old unit test invented `{"platform", "price", "size", "side",
"question"}`, which matches neither venue -- the fixture was written to fit
the parser rather than the data, which is why none of this failed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.whale_flow import analyze_whale_trades  # noqa: E402

#: A Kalshi print, keys and string types exactly as the tape returns them.
KALSHI_ROW = {
    "count_fp": "467.90",
    "created_time": "2026-09-07T11:15:16.112063Z",
    "is_block_trade": False,
    "no_price_dollars": "0.9800",
    "taker_side": "yes",
    "ticker": "KXNCAAFGAME-26SEP07SMUFSU-FSU",
    "trade_id": "072195c2-8961-b280-ed40-dc20c8e28f2b",
    "yes_price_dollars": "0.0200",
}

#: A Polymarket print, likewise.
POLY_ROW = {
    "asset": "3333979837291603722078613613340647849111615062822044195175342668344",
    "conditionId": "0xefa17dee3af09f69f9ddf245b969aa4efbe7c71cdf06ee49d694408bc33e2ed2",
    "outcome": "Yes",
    "price": 0.40,
    "side": "BUY",
    "size": 2000,
    "slug": "will-shakhtar-donetsk-win-the-2026-27-ucl",
    "title": "Will Shakhtar Donetsk win the 2026-27 UEFA Champions League?",
}


class KalshiTradesAreVisibleTests(unittest.TestCase):
    def test_a_kalshi_print_is_priced_from_its_own_fields(self):
        big = dict(KALSHI_ROW, count_fp="20000", yes_price_dollars="0.0200")
        result = analyze_whale_trades([big], min_notional_usd=250.0)

        self.assertEqual(result["n_whale_prints"], 1)
        [print_] = result["top_prints"]
        self.assertEqual(print_["platform"], "Kalshi")
        self.assertEqual(print_["ticker"], KALSHI_ROW["ticker"])
        self.assertEqual(print_["side"], "YES")
        self.assertAlmostEqual(print_["notional_usd"], 400.0)

    def test_the_no_side_is_priced_off_the_no_column(self):
        row = dict(KALSHI_ROW, taker_side="no", count_fp="1000",
                   yes_price_dollars="0.0200", no_price_dollars="0.9800")
        [print_] = analyze_whale_trades([row], min_notional_usd=250.0)["top_prints"]
        self.assertEqual(print_["side"], "NO")
        self.assertAlmostEqual(print_["notional_usd"], 980.0)

    def test_the_old_parser_would_have_seen_nothing(self):
        """The defect, stated as arithmetic on the same row."""
        price = float(KALSHI_ROW.get("price") or KALSHI_ROW.get("yes_price") or 0.50)
        size = float(
            KALSHI_ROW.get("size") or KALSHI_ROW.get("quantity")
            or KALSHI_ROW.get("count") or 0
        )
        self.assertEqual(price * size, 0.0)

    def test_a_kalshi_row_names_its_market(self):
        [print_] = analyze_whale_trades(
            [dict(KALSHI_ROW, count_fp="20000")], min_notional_usd=250.0,
        )["top_prints"]
        self.assertNotEqual(print_["market_title"], "Market")
        self.assertEqual(print_["market_title"], KALSHI_ROW["ticker"])


class PolymarketTradesAreIdentifiedTests(unittest.TestCase):
    def test_a_polymarket_print_carries_its_title_and_slug(self):
        [print_] = analyze_whale_trades([POLY_ROW], min_notional_usd=250.0)["top_prints"]
        self.assertEqual(print_["platform"], "Polymarket")
        self.assertEqual(print_["ticker"], POLY_ROW["slug"])
        self.assertEqual(print_["market_title"], POLY_ROW["title"])
        self.assertNotIn(print_["market_title"], ("Market", ""))

    def test_buying_no_is_not_counted_as_bullish(self):
        """`side` is BUY/SELL; the position is in `outcome`."""
        buy_no = dict(POLY_ROW, outcome="No", side="BUY")
        result = analyze_whale_trades([buy_no], min_notional_usd=250.0)
        self.assertEqual(result["top_prints"][0]["side"], "NO")
        self.assertEqual(result["whale_yes_volume_usd"], 0.0)
        self.assertGreater(result["whale_no_volume_usd"], 0.0)

    def test_selling_yes_is_the_same_exposure_as_buying_no(self):
        sell_yes = dict(POLY_ROW, outcome="Yes", side="SELL")
        [print_] = analyze_whale_trades([sell_yes], min_notional_usd=250.0)["top_prints"]
        self.assertEqual(print_["side"], "NO")

    def test_buying_yes_is_still_bullish(self):
        [print_] = analyze_whale_trades([POLY_ROW], min_notional_usd=250.0)["top_prints"]
        self.assertEqual(print_["side"], "YES")


class BothVenuesTogetherTests(unittest.TestCase):
    def test_sentiment_counts_both_tapes(self):
        rows = [
            dict(KALSHI_ROW, count_fp="20000", taker_side="yes"),   # 400 YES
            dict(POLY_ROW, outcome="No", side="BUY"),               # 800 NO
        ]
        result = analyze_whale_trades(rows, min_notional_usd=250.0)
        self.assertEqual(result["n_whale_prints"], 2)
        self.assertAlmostEqual(result["whale_yes_volume_usd"], 400.0)
        self.assertAlmostEqual(result["whale_no_volume_usd"], 800.0)
        self.assertEqual({p["platform"] for p in result["top_prints"]},
                         {"Kalshi", "Polymarket"})

    def test_an_explicit_platform_is_respected_over_inference(self):
        row = dict(KALSHI_ROW, count_fp="20000", platform="Kalshi")
        [print_] = analyze_whale_trades([row], min_notional_usd=250.0)["top_prints"]
        self.assertEqual(print_["platform"], "Kalshi")


if __name__ == "__main__":
    unittest.main()
