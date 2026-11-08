from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import fund_universe as fu  # noqa: E402


class _Resp:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _candles(volume_by_key):
    """A fake Upstox route: every instrument closes at 100 each day with its own volume."""
    def http_get(url, **_):
        key = url.split("/historical-candle/", 1)[1].split("/", 1)[0].replace("%7C", "|")
        volume = volume_by_key.get(key, 0)
        candles = [[f"2026-11-0{d}T00:00:00+05:30", 100, 100, 100, 100.0, volume, 0] for d in (2, 3, 4)]
        return _Resp({"status": "success", "data": {"candles": candles}})
    return http_get


def _future(symbol, key, expiry=1_800_000_000_000, mult=1.0):
    return {"underlying_symbol": symbol, "underlying_type": "INDEX", "instrument_key": key, "expiry": expiry,
            "trading_symbol": f"{symbol} FUT", "qty_multiplier": mult, "lot_size": 1}


class FundUniverseTests(unittest.TestCase):
    def test_indices_rank_by_dollar_volume_across_venues(self):
        nse = [_future("NIFTY", "NSE_FO|1"), _future("FINNIFTY", "NSE_FO|2")]
        bse = [_future("SENSEX", "BSE_FO|3")]
        kalshi = [{"ticker": "KXUS500PERP", "asset_class": "Indices", "status": "active",
                   "volume_24h_notional_value_dollars": "500"},
                  {"ticker": "KXUS2000PERP", "asset_class": "Indices", "status": "inactive",
                   "volume_24h_notional_value_dollars": "999999"}]
        # Daily notional = volume x 100 rupees / 100 rupees per dollar = volume dollars.
        http_get = _candles({"NSE_FO|1": 1000, "NSE_FO|2": 10, "BSE_FO|3": 300})
        ranked = fu.rank_indices(nse, bse, kalshi, date(2026, 11, 8), 100.0, http_get=http_get, size=3)
        self.assertEqual([i.label for i in ranked], ["Nifty 50", "US 500 (Kalshi perp)", "Sensex"])
        self.assertAlmostEqual(ranked[0].volume_usd, 1000.0)
        self.assertEqual(ranked[1].underlying, "us_equity")

    def test_a_pair_ranks_by_its_thinner_leg_and_needs_both(self):
        def leg(underlying, venue, volume):
            return fu.Instrument(id=f"{venue}:{underlying}", label=underlying, venue=venue, asset_class="commodity",
                                 underlying=underlying, currency="INR", multiplier=1.0, key=underlying, volume_usd=volume)
        legs = {("gold", "mcx"): leg("gold", "mcx", 1000), ("silver", "mcx"): leg("silver", "mcx", 400),
                ("gold", "kalshi_perp"): leg("gold", "kalshi_perp", 3), ("crude", "mcx"): leg("crude", "mcx", 900)}
        ranked = fu.rank_pairs(legs)
        self.assertEqual([p["key"] for p in ranked], ["gold-silver", "gold-mcx-kalshi"])
        self.assertEqual(ranked[0]["volume_usd"], 400)

    def test_a_kalshi_side_without_orders_is_none(self):
        quote = fu.kalshi_quote({"bid": "6.1954", "ask": "922337203685477.5807", "reference_price": "6.2"})
        self.assertEqual(quote["bid"], 6.1954)
        self.assertIsNone(quote["ask"])
        self.assertIsNone(quote["mid"])
        self.assertEqual(fu.kalshi_quote({"bid": "0.0000", "ask": "4.0653"})["bid"], None)


if __name__ == "__main__":
    unittest.main()
