from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("fund_ndx_put_eval", _ROOT / "scripts" / "fund_ndx_put_eval.py")
ndx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ndx)
fo = ndx.fo


class ScaleTests(unittest.TestCase):
    def test_the_strike_grid_scales_with_the_index_as_nifty_does(self):
        self.assertEqual(ndx.nice_step(0.002 * 25_000), 50)  # Nifty's own grid
        self.assertEqual(ndx.nice_step(0.002 * 1_000), 2.5)
        self.assertEqual(ndx.nice_step(0.002 * 21_000), 50)
        strikes = {c["strike_price"] for c in ndx.chain(21_000)}
        self.assertTrue(all(k % 50 == 0 for k in strikes))

    def test_the_minimum_half_spread_scales_with_the_index(self):
        far = [{"instrument_key": "PE100", "instrument_type": "PE", "strike_price": 100.0, "lot_size": 1}]
        self.assertAlmostEqual(ndx.quotes(far, 20_000, 20.0)["PE100"]["ask"], 0.00002 * 20_000)


class RunTests(unittest.TestCase):
    def test_the_put_spread_is_sold_every_21_sessions_on_the_past_only(self):
        start = date(2001, 2, 2)
        days = [start + timedelta(days=i) for i in range(1500)]
        closes = [(d, 1_500 * math.exp(0.0003 * i + 0.01 * math.sin(i))) for i, d in enumerate(days)]
        vol = [(d, 25.0 + 4 * math.sin(i / 9)) for i, d in enumerate(days)]
        trades = ndx.run(closes, vol)
        puts = trades["put_spread"]
        self.assertTrue(puts)
        self.assertTrue(all(t["kind"] == "put_spread" for t in puts))
        gaps = {(date.fromisoformat(b["day"]) - date.fromisoformat(a["day"])).days for a, b in zip(puts, puts[1:])}
        self.assertTrue(all(g % ndx.nifty.H == 0 for g in gaps))
        first = date.fromisoformat(puts[0]["day"])
        self.assertGreaterEqual((first - start).days, ndx.nifty.ifc.WARMUP_SESSIONS + 260)


if __name__ == "__main__":
    unittest.main()
