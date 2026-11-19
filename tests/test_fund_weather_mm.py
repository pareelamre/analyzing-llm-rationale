"""The weather market-making test's paper fills (scripts/fund_weather_mm_eval.py)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fund_weather_mm_eval as mm  # noqa: E402

T0 = 1_790_000_000 - (1_790_000_000 % 60)


def _iso(ts: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def trade(ts: int, price: float, taker: str, count: int = 5, block: bool = False) -> dict:
    return {"created_time": _iso(ts), "yes_price_dollars": f"{price:.4f}", "taker_side": taker,
            "count_fp": f"{count}.00", "is_block_trade": block}


class PaperFillTests(unittest.TestCase):
    def test_a_seller_into_the_bid_fills_the_improved_paper_bid_at_its_price(self):
        book = {T0: (0.40, 0.45)}
        fills = mm.simulate_market(book, [trade(T0 + 30, 0.40, "no")], result=1.0)
        self.assertEqual(len(fills), 1)
        self.assertEqual((fills[0]["side"], fills[0]["price"], fills[0]["qty"]), (1, 0.41, 5))
        self.assertAlmostEqual(fills[0]["profit"], 0.59)
        self.assertAlmostEqual(fills[0]["half_spread"], 0.425 - 0.41)

    def test_a_buyer_from_the_ask_fills_the_paper_ask_and_loses_when_yes_wins(self):
        fills = mm.simulate_market({T0: (0.40, 0.45)}, [trade(T0 + 10, 0.45, "yes", 3)], result=1.0)
        self.assertEqual((fills[0]["side"], fills[0]["price"], fills[0]["qty"]), (-1, 0.44, 3))
        self.assertAlmostEqual(fills[0]["profit"], -0.56)

    def test_trades_through_a_better_price_or_outside_the_minute_do_not_fill(self):
        book = {T0: (0.40, 0.45)}
        flow = [trade(T0 + 20, 0.42, "no"),      # someone bid 42 inside the minute: no fill at 41
                trade(T0 + 61, 0.40, "no"),      # after the quote expired
                trade(T0, 0.40, "no"),           # at the quote time itself, before it rested
                trade(T0 + 30, 0.40, "no", block=True)]
        self.assertEqual(mm.simulate_market(book, flow, result=0.0), [])

    def test_a_spread_under_three_cents_or_a_one_sided_book_is_not_quoted(self):
        flow = [trade(T0 + 30, 0.40, "no"), trade(T0 + 90, 0.40, "no")]
        self.assertEqual(mm.simulate_market({T0: (0.40, 0.42), T0 + 60: (0.40, None)}, flow, result=0.0), [])

    def test_size_per_minute_and_the_position_cap(self):
        minutes = {T0 + 60 * i: (0.40, 0.45) for i in range(8)}
        flow = [trade(T0 + 60 * i + 30, 0.40, "no", 25) for i in range(8)]
        fills = mm.simulate_market(minutes, flow, result=0.0)
        self.assertEqual([f["qty"] for f in fills], [10, 10, 10, 10, 10])  # 10 a minute, then capped at 50
        self.assertEqual(sum(f["qty"] for f in fills), mm.MAX_POSITION)

    def test_profit_and_verdict_pool_by_contract(self):
        rows = [{"qty": 10, "profit": 0.1, "maker_fee": 0.004}, {"qty": 30, "profit": -0.02, "maker_fee": 0.004}]
        self.assertAlmostEqual(mm.per_contract(rows), (1.0 - 0.6) / 40)
        self.assertAlmostEqual(mm.per_contract(rows, fee=True), (1.0 - 0.6) / 40 - 0.004)


if __name__ == "__main__":
    unittest.main()
