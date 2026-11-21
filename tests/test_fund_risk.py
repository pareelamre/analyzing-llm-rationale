"""Risk limits: worst-case loss caps new positions, and held positions that lose are flagged."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import fund_orchestrator as orch  # noqa: E402
from analyzing_llm_rationale import fund_risk as risk  # noqa: E402

NAV = 10_000.0


def _snap(positions=(), cash=10_000.0):
    return {"nav_usd": NAV, "cash_usd": cash, "positions": list(positions), "by_asset_class": {},
            "net_exposure_by_underlying": {}}


def _binary(value, stake=None, pnl=0.0, iid="kalshi:X", side="NO", label="bet"):
    return {"instrument_id": iid, "side": side, "label": label, "kind": "binary", "asset_class": "event",
            "collateral_usd": stake if stake is not None else value, "value_usd": value, "pnl_usd": pnl}


EVENT = {"id": "kalshi:NEW", "label": "New bet", "venue": "kalshi", "asset_class": "event", "underlying": "kalshi:NEW",
         "currency": "USD", "multiplier": 1.0}


class EntryTests(unittest.TestCase):
    def test_a_near_certain_bet_is_held_to_three_percent_of_the_fund(self):
        # The Millennium Prize No: 94 cents, a model at 99.5%, Kelly wants a big position.
        s = orch.Signal(analyst="blend_model", instrument=EVENT, side="NO", price=0.94, edge=0.05, trust=1.0,
                        kelly=(0.995 - 0.94) / 0.06)
        [order] = orch.plan(_snap(), [s], {}, {})["orders"]
        self.assertAlmostEqual(order.quantity * order.price, risk.MAX_LOSS_PER_POSITION * NAV)
        facts = {f["label"]: f["value"] for f in order.facts}
        self.assertEqual(facts["worst case"], "$300 (3.0% of the fund)")

    def test_no_new_risk_once_everything_held_could_lose_forty_percent(self):
        held = [_binary(1000.0, iid=f"kalshi:H{i}") for i in range(4)]  # $4,000 = 40% at stake
        s = orch.Signal(analyst="blend_model", instrument=EVENT, side="YES", price=0.40, edge=0.1, trust=1.0, kelly=0.2)
        out = orch.plan(_snap(held, cash=6_000.0), [s], {}, {})
        self.assertEqual(out["orders"], [])
        self.assertIn("risk limits", out["passed"][0]["reason"])

    def test_room_left_under_the_total_trims_the_next_position(self):
        held = [_binary(1000.0, iid=f"kalshi:H{i}") for i in range(3)] + [_binary(900.0, iid="kalshi:H3")]
        s = orch.Signal(analyst="blend_model", instrument=EVENT, side="YES", price=0.40, edge=0.1, trust=1.0, kelly=0.2)
        [order] = orch.plan(_snap(held, cash=6_100.0), [s], {}, {})["orders"]
        self.assertAlmostEqual(order.quantity * order.price, 100.0)

    def test_a_futures_leg_counts_a_severe_move_not_its_whole_notional(self):
        self.assertEqual(risk.loss_per_dollar("linear", "commodity"), 0.15)
        room = risk.entry_room(NAV, 0.0, 0.15, legs=2)
        self.assertAlmostEqual(room, 2_000.0)  # 3% of the fund on a 15% move
        self.assertAlmostEqual(risk.entry_room(NAV, 3_970.0, 0.15, legs=2), 100.0)  # $30 left, two legs


class ReviewTests(unittest.TestCase):
    def test_losing_positions_are_flagged_and_nothing_is_sold(self):
        millennium = _binary(186.8, stake=585.0, pnl=-398.2, iid="kalshi:KXAIMILLENNIUM", label="Millennium")
        venezuela = _binary(1093.0, stake=1142.0, pnl=-49.0, iid="poly:delcy", label="Delcy")
        fine = _binary(200.0, stake=190.0, pnl=10.0, iid="poly:ok", label="Fine")
        previous = {"positions": [{"instrument_id": "kalshi:KXAIMILLENNIUM", "side": "NO", "value_usd": 547.0}]}
        out = risk.review(_snap([millennium, venezuela, fine]), previous=previous)
        alerts = {a["label"]: a["alerts"] for a in out["alerts"]}
        self.assertIn("losing 4.0% of the fund", alerts["Millennium"])
        self.assertIn("lost 68% of its stake", alerts["Millennium"])
        self.assertIn("fell $360 since the last update", alerts["Millennium"])
        self.assertNotIn("Delcy", alerts)  # not losing much: listed as over the limit instead
        self.assertEqual([r["label"] for r in out["over_limit"]], ["Delcy"])
        self.assertNotIn("Fine", alerts)
        self.assertEqual(out["positions"][0]["label"], "Delcy")  # largest worst case first
        self.assertAlmostEqual(out["worst_case_total_usd"], 186.8 + 1093.0 + 200.0)
        self.assertFalse(out["new_risk_paused"])

    def test_the_review_pauses_new_risk_above_the_total_limit(self):
        held = [_binary(1100.0, iid=f"kalshi:H{i}") for i in range(4)]
        self.assertTrue(risk.review(_snap(held))["new_risk_paused"])


if __name__ == "__main__":
    unittest.main()
