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


class RiskTrimTests(unittest.TestCase):
    """2026-11-25: the per-position limit binds held positions too (fund_orchestrator.plan risk_trims)."""

    def _inherited(self):
        big = {**_binary(1_081.0, iid="polymarket:delcy", label="Delcy NO"), "quantity": 1_200.0, "mark": 0.9008}
        small = {**_binary(279.0, iid="polymarket:hormuz", label="Hormuz NO"), "quantity": 300.0, "mark": 0.93}
        leg = {"instrument_id": "mcx:COPPER", "side": "LONG", "label": "Copper", "kind": "linear",
               "asset_class": "commodity", "quantity": 0.007, "mark": 1128.5, "collateral_usd": 900.0,
               "value_usd": 900.0, "pnl_usd": 0.0, "pair_key": "copper-aluminium", "multiplier": 1.0}
        return big, small, leg

    def test_an_over_limit_bet_is_sold_down_to_three_percent(self):
        big, small, leg = self._inherited()
        out = orch.plan(_snap([big, small, leg], cash=6_000.0), [], {}, {}, risk_trims=True)
        [order] = out["orders"]
        self.assertEqual((order.instrument["id"], order.action, order.purpose), ("polymarket:delcy", "sell", "trim"))
        part = 1 - 0.03 * NAV / 1_081.0
        self.assertAlmostEqual(order.quantity, 1_200.0 * part)
        self.assertIn("per-position risk limit", order.reason)

    def test_without_the_switch_nothing_is_trimmed(self):
        big, small, leg = self._inherited()
        self.assertEqual(orch.plan(_snap([big, small, leg], cash=6_000.0), [], {}, {})["orders"], [])

    def test_risk_freed_by_a_trim_lets_a_new_position_in_the_same_cycle(self):
        held = [{**_binary(1_400.0, iid=f"kalshi:H{i}", label=f"held {i}"), "quantity": 1_500.0, "mark": 0.93}
                for i in range(3)]  # 3 x 14% = 42%: over the 40% total
        s = orch.Signal(analyst="blend_model", instrument=EVENT, side="YES", price=0.40, edge=0.1, trust=1.0, kelly=0.2)
        blocked = orch.plan(_snap(held, cash=5_000.0), [s], {}, {})
        self.assertEqual(blocked["orders"], [])
        out = orch.plan(_snap(held, cash=5_000.0), [s], {}, {}, risk_trims=True)
        self.assertEqual(sorted(o.purpose for o in out["orders"]), ["trim", "trim", "trim", "view"])


if __name__ == "__main__":
    unittest.main()


class OwnerTrimTests(unittest.TestCase):
    SPREAD = "nse_fo:BANKNIFTY:put_spread:2026-12-29:59300/57000"

    def _spread(self, value, quantity=19.04):
        return {"instrument_id": self.SPREAD, "side": "BULL", "label": "Nifty Bank put credit spread",
                "kind": "option", "asset_class": "option", "quantity": quantity, "mark": 2196.435,
                "collateral_usd": 416.93, "value_usd": value, "pnl_usd": value - 416.93}

    def _trim(self):
        return [{"id": "trim-banknifty", "instrument_id": self.SPREAD, "side": "BULL", "max_worst_case_share": 0.03,
                 "reason": "Owner: bring it inside the 3% limit."}]

    def test_an_over_limit_spread_is_cut_back_to_three_percent(self):
        snap = {**_snap([self._spread(451.38)], cash=4_000.0), "nav_usd": 9_745.25}
        [order] = orch.plan(snap, [], {}, {}, owner_trims=self._trim())["orders"]
        self.assertEqual((order.action, order.purpose, order.side), ("sell", "trim", "BULL"))
        part = 1 - 0.03 * 9_745.25 / 451.38
        self.assertAlmostEqual(order.quantity, 19.04 * part)
        self.assertAlmostEqual(451.38 * (1 - part), 0.03 * 9_745.25)  # what is left risks 3%
        self.assertIn("Owner", order.reason)

    def test_a_spread_within_the_limit_is_left_alone(self):
        snap = {**_snap([self._spread(300.0)]), "nav_usd": 9_745.25}  # 3.1%, inside the 10% tolerance
        self.assertEqual(orch.plan(snap, [], {}, {}, owner_trims=self._trim())["orders"], [])

    def test_no_instruction_no_trim(self):
        snap = {**_snap([self._spread(451.38)]), "nav_usd": 9_745.25}
        self.assertEqual(orch.plan(snap, [], {}, {})["orders"], [])
