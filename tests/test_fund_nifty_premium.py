from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("fund_nifty_premium_eval", _ROOT / "scripts" / "fund_nifty_premium_eval.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)
fo = ev.fo


class PricingTests(unittest.TestCase):
    def test_black_76_obeys_put_call_parity_and_its_limits(self):
        sd = 0.18 * math.sqrt(21 / 252)
        call, put = ev.black(25_000, 25_500, sd, "CE"), ev.black(25_000, 25_500, sd, "PE")
        self.assertAlmostEqual(call - put, 25_000 - 25_500, places=6)
        self.assertEqual(ev.black(25_000, 24_000, 0.0, "CE"), 1_000)

    def test_every_leg_pays_a_half_spread_of_at_least_half_a_point(self):
        contracts = ev.chain(25_000)
        self.assertTrue(all(c["strike_price"] % ev.GRID == 0 for c in contracts))
        q = ev.quotes(contracts, 25_000, 15.0)
        atm, far = q["CE25000"], q[f"CE{int(contracts[-1]['strike_price'])}"]
        self.assertAlmostEqual(atm["ask"] - atm["last"], 0.01 * atm["last"])
        self.assertAlmostEqual(far["ask"] - far["last"], 0.5)
        self.assertGreaterEqual(far["bid"], 0.0)


class DecisionTests(unittest.TestCase):
    def _dist(self, sd):
        levels = ev.LEVELS
        returns = [sd * fo.NormalDist().inv_cdf(p) for p in levels]
        return fo.Distribution(levels, returns, 25_000, 25_000)

    def test_a_market_dearer_than_the_model_sells_a_condor_and_a_fair_one_does_not(self):
        sd = 0.04
        dist = self._dist(sd)
        dear = ev.decide(dist, 25_000, sd / math.sqrt(21 / 252) * 100 * 1.3)
        self.assertIsNotNone(dear["a"])
        self.assertIn(dear["a"]["structure"].kind, ("condor", "put_spread", "call_spread"))
        self.assertIsNotNone(dear["b"])
        self.assertEqual(set(dear["legs"]), {"put_spread", "call_spread"})
        fair = ev.decide(dist, 25_000, sd / math.sqrt(21 / 252) * 100)
        self.assertIsNone(fair["a"])
        self.assertIsNotNone(fair["b"])  # the systematic sale trades regardless

    def test_a_structure_returns_its_payoff_less_cost_and_charges_per_rupee_at_risk(self):
        condor = fo.Structure("condor", "RANGE", [fo.Leg("p1", "PE", 24_300, -1), fo.Leg("p0", "PE", 24_000, 1),
                                                  fo.Leg("c1", "CE", 25_700, -1), fo.Leg("c0", "CE", 26_000, 1)],
                              const=300.0)
        entry = {"structure": condor, "cost": 200.0}
        self.assertAlmostEqual(ev.net_return(entry, 25_000), (300 - 200 - 4) / 200)
        self.assertAlmostEqual(ev.net_return(entry, 23_000), (0 - 200 - 4) / 200)

    def test_holds_never_overlap_and_the_model_sees_only_the_past(self):
        start = date(2008, 1, 1)
        days = [start + timedelta(days=i) for i in range(1800)]
        closes = [(d, 10_000 * math.exp(0.0002 * i + 0.01 * math.sin(i))) for i, d in enumerate(days)]
        vix = [(d, 18.0 + 3 * math.sin(i / 7)) for i, d in enumerate(days)]
        seen = []
        real = ev.model_distribution

        def spy(c, v):
            seen.append((c[-1][0], v[-1][0]))
            return real(c, v)
        ev.model_distribution = spy
        try:
            trades = ev.run(closes, vix)
        finally:
            ev.model_distribution = real
        self.assertTrue(seen and all(c == v for c, v in seen))
        b = trades["b"]
        gaps = {(date.fromisoformat(y["day"]) - date.fromisoformat(x["day"])).days for x, y in zip(b, b[1:])}
        self.assertTrue(gaps and all(g % ev.H == 0 and g >= ev.H for g in gaps))  # a skipped month, never an overlap


if __name__ == "__main__":
    unittest.main()
