from __future__ import annotations

import dataclasses
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import fund_notes  # noqa: E402
from analyzing_llm_rationale import fund_orchestrator as orch  # noqa: E402

NOW = datetime(2026, 11, 9, 20, 0, tzinfo=timezone.utc)


def _row(ident, p, bid, ask, at="2026-11-09T17:25:00+00:00"):
    return {"platform": "Polymarket", "ident": ident, "question": f"Will {ident}?", "model_probability": p,
            "market_bid": bid, "market_ask": ask, "forecasted_at": at}


def _snapshot(nav=10_000.0, cash=6_000.0, positions=(), by_class=None, exposure=None):
    return {"nav_usd": nav, "cash_usd": cash, "positions": list(positions), "by_asset_class": by_class or {},
            "net_exposure_by_underlying": exposure or {}}


COPPER = {"id": "mcx:COPPER", "label": "Copper (MCX)", "venue": "mcx", "asset_class": "commodity",
          "underlying": "copper", "currency": "INR", "multiplier": 1.0, "key": "MCX_FO|9"}
ALUMINIUM = {"id": "mcx:ALUMINI", "label": "Aluminium (MCX)", "venue": "mcx", "asset_class": "commodity",
             "underlying": "aluminium", "currency": "INR", "multiplier": 1.0, "key": "MCX_FO|8"}
TRACKER = {
    "rule": {"entry_z": 2.0, "exit_z": 0.5},
    "backtest": {"window": "2011-2026", "pairs": {"copper-aluminium": {"avg_return": 0.003, "win_rate": 0.63},
                                                   "copper-gold": {"avg_return": -0.006, "win_rate": 0.56}}},
    "pairs": [{"key": "copper-aluminium", "z": -2.4, "stretched": True, "beta": 0.9, "date": "2026-11-06"},
              {"key": "copper-gold", "z": -2.58, "stretched": True, "beta": 0.16, "date": "2026-11-06"}],
}
UNIVERSE_PAIRS = [{"key": "copper-aluminium", "label": "Copper / Aluminium (MCX)", "a": COPPER, "b": ALUMINIUM},
                  {"key": "copper-gold", "label": "Copper / Gold (MCX)", "a": COPPER, "b": ALUMINIUM}]


class EventSignalTests(unittest.TestCase):
    def test_a_forecast_beating_the_ask_after_costs_is_a_signal(self):
        signals, _ = orch.event_signals([_row("hamas", 0.02, 0.20, 0.22)], NOW)
        self.assertEqual([s.side for s in signals], ["NO"])
        no = signals[0]
        self.assertAlmostEqual(no.price, 0.80)
        self.assertAlmostEqual(no.edge, (0.98 - 0.80) / 0.80 - orch.EVENT_COST)
        self.assertIn("98.0%", {f["value"] for f in no.facts})

    def test_stale_or_thin_forecasts_are_ignored(self):
        self.assertEqual(orch.event_signals([_row("x", 0.6, 0.55, 0.57, at="2026-11-01T00:00:00+00:00")], NOW)[0], [])
        self.assertEqual(orch.event_signals([_row("x", 0.58, 0.55, 0.57)], NOW)[0], [])

    def test_a_tiny_gap_on_a_cheap_contract_is_not_a_signal(self):
        # 8% against a 6c ask is a 31% edge per dollar but only 2 points: noise for an unproven model.
        self.assertEqual(orch.event_signals([_row("delcy", 0.08, 0.05, 0.06)], NOW)[0], [])

    def test_a_forecast_far_from_the_market_is_skipped_as_a_likely_model_error(self):
        # The inverted forecast of 2026-11-08: 98% against a 6c market.
        signals, skipped = orch.event_signals([_row("hamas", 0.98, 0.05, 0.06)], NOW)
        self.assertEqual(signals, [])
        self.assertIn("92 points from the market", skipped[0]["reason"])

    def test_a_market_closing_within_a_day_is_skipped(self):
        row = {**_row("match", 0.45, 0.30, 0.32), "lead_days": 0.2}
        signals, skipped = orch.event_signals([row], NOW)
        self.assertEqual((signals, skipped[0]["reason"]), ([], "closes within a day"))


class PairSignalTests(unittest.TestCase):
    def test_only_pairs_with_a_profitable_backtest_trade_and_skips_say_why(self):
        signals, skipped = orch.pair_signals(TRACKER, UNIVERSE_PAIRS)
        self.assertEqual({(s.instrument["id"], s.side) for s in signals},
                         {("mcx:COPPER", "LONG"), ("mcx:ALUMINI", "SHORT")})
        self.assertEqual(skipped[0]["pair"], "copper-gold")
        self.assertIn("-0.6% per signal", skipped[0]["reason"])


class PlanTests(unittest.TestCase):
    def test_entries_respect_the_position_cap_and_carry_attribution(self):
        signals, _ = orch.event_signals([_row("hamas", 0.02, 0.20, 0.22)], NOW)
        out = orch.plan(_snapshot(), signals, {}, {})
        order = out["orders"][0]
        self.assertEqual((order.action, order.side, order.purpose), ("buy", "NO", "view"))
        self.assertLessEqual(order.notional, orch.POSITION_CAP * 10_000 + 1e-6)
        self.assertEqual(order.attribution, {"event_model": 1.0})

    def test_the_cash_floor_stops_new_buys(self):
        signals, _ = orch.event_signals([_row("hamas", 0.02, 0.20, 0.22)], NOW)
        out = orch.plan(_snapshot(cash=2_000.0), signals, {}, {})
        self.assertEqual(out["orders"], [])
        self.assertEqual(out["passed"][0]["reason"], "no room under the fund's limits")

    def test_held_sides_are_kept_unless_a_fresh_forecast_values_them_below_the_bid(self):
        held = [{"instrument_id": "polymarket:iran", "label": "Iran", "side": "NO", "kind": "binary", "quantity": 100,
                 "mark": 0.94, "attribution": {"transfer": 1.0}},
                {"instrument_id": "kalshi:fed", "label": "Fed", "side": "YES", "kind": "binary", "quantity": 10,
                 "mark": 0.90, "attribution": {"transfer": 1.0}}]
        forecasts = {"polymarket:iran": {"model_probability": 0.25, "forecasted_at": "2026-11-09T17:00:00+00:00"}}
        out = orch.plan(_snapshot(positions=held), [], {}, {}, forecasts=forecasts)
        sells = [(o.instrument["id"], o.purpose) for o in out["orders"]]
        # Iran NO is worth 75% to the model against a 94c bid: sell. The Fed side has no fresh forecast: keep.
        self.assertEqual(sells, [("polymarket:iran", "exit")])

    def _held_no(self, mark=0.94):
        return [{"instrument_id": "polymarket:delcy", "label": "Delcy", "side": "NO", "kind": "binary",
                 "quantity": 1214.78, "mark": mark, "attribution": {"agent_transfer": 1.0}}]

    def test_a_one_or_two_point_disagreement_does_not_sell(self):
        # 2026-11-09: the model put Delcy NO at 92% against a 94c bid, and Mojtaba NO at 6%
        # against 6.9c. Without a margin the fund would have sold 17% of itself on that.
        forecasts = {"polymarket:delcy": {"model_probability": 0.08, "market_bid": 0.05, "market_ask": 0.06,
                                          "forecasted_at": "2026-11-09T17:00:00+00:00"}}
        self.assertEqual(orch.plan(_snapshot(positions=self._held_no()), [], {}, {}, forecasts=forecasts)["orders"], [])
        mojtaba = [{**self._held_no(0.069)[0], "instrument_id": "polymarket:mojtaba"}]
        forecasts = {"polymarket:mojtaba": {"model_probability": 0.94, "market_bid": 0.92, "market_ask": 0.93,
                                            "forecasted_at": "2026-11-09T17:00:00+00:00"}}
        self.assertEqual(orch.plan(_snapshot(positions=mojtaba), [], {}, {}, forecasts=forecasts)["orders"], [])

    def test_a_likely_model_error_never_triggers_an_exit(self):
        # NO is worth 30% to the model against a 94c bid, but the forecast is 64 points from the market.
        forecasts = {"polymarket:delcy": {"model_probability": 0.70, "market_bid": 0.05, "market_ask": 0.07,
                                          "forecasted_at": "2026-11-09T17:00:00+00:00"}}
        self.assertEqual(orch.plan(_snapshot(positions=self._held_no()), [], {}, {}, forecasts=forecasts)["orders"], [])

    def test_the_other_side_of_a_held_market_is_never_bought(self):
        row = {**_row("delcy", 0.30, 0.10, 0.12)}
        signals, _ = orch.event_signals([row], NOW)
        self.assertEqual([s.side for s in signals], ["YES"])
        out = orch.plan(_snapshot(positions=self._held_no(0.88)), signals, {}, {})
        self.assertEqual(out["orders"], [])
        self.assertEqual(out["passed"][0]["reason"], "the fund holds NO on this market")

    def test_a_pair_back_in_its_band_closes_both_legs(self):
        held = [{"instrument_id": "mcx:COPPER", "side": "LONG", "kind": "linear", "quantity": 1, "mark": 900.0,
                 "pair_key": "copper-aluminium", "attribution": {"pairs_model": 1.0}},
                {"instrument_id": "mcx:ALUMINI", "side": "SHORT", "kind": "linear", "quantity": 1, "mark": 250.0,
                 "pair_key": "copper-aluminium", "attribution": {"pairs_model": 1.0}}]
        out = orch.plan(_snapshot(positions=held), [], {}, {}, held_pairs={"copper-aluminium": {"sessions": 3}},
                        pair_rows={"copper-aluminium": {"z": 0.3}})
        self.assertEqual(sorted(o.instrument["id"] for o in out["orders"]), ["mcx:ALUMINI", "mcx:COPPER"])
        self.assertTrue(all(o.purpose == "exit" for o in out["orders"]))

    def test_concentration_beyond_the_underlying_limit_is_hedged_on_another_venue(self):
        perp = {"id": "kalshi_perp:KXGOLDPERP", "label": "Gold (Kalshi perp)", "multiplier": 1.0,
                "asset_class": "commodity", "underlying": "gold"}
        out = orch.plan(_snapshot(exposure={"gold": 2_000.0}), [], {perp["id"]: {"bid": 4.0, "ask": 4.01}},
                        {"gold": perp})
        hedge = out["orders"][0]
        self.assertEqual((hedge.side, hedge.purpose), ("SHORT", "hedge"))
        self.assertAlmostEqual(hedge.notional, 2_000.0 - orch.UNDERLYING_CAP * 10_000)
        self.assertEqual(hedge.attribution, {"risk_overlay": 1.0})


class BlendTests(unittest.TestCase):
    PARAMS = {"coef": [0.693864, 0.402811, -0.291052], "intercept": 0.173417}

    def test_the_blend_leans_on_the_market_and_nudges_toward_the_model(self):
        # The 2026-11-09 weights: about 0.99 x logit(market) + 0.11 x logit(model).
        artwork = orch.blend_probability(0.05, 0.93, self.PARAMS)
        self.assertTrue(0.88 < artwork < 0.93)
        self.assertAlmostEqual(orch.blend_probability(0.5, 0.5, self.PARAMS), 0.5433, places=3)

    def test_the_board_keeps_the_raw_forecast_beside_the_blend(self):
        rows = orch.blend_board([_row("x", 0.05, 0.92, 0.94), {"ident": "no-price", "model_probability": 0.3}],
                                self.PARAMS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["raw_model_probability"], 0.05)
        self.assertGreater(rows[0]["model_probability"], 0.85)


class NoteTests(unittest.TestCase):
    def _order(self):
        signals, _ = orch.event_signals([_row("hamas", 0.02, 0.20, 0.22)], NOW)
        order = orch.plan(_snapshot(), signals, {}, {})["orders"][0]
        return dataclasses.asdict(order)

    def test_a_note_with_an_invented_number_falls_back_to_the_fact_sheet(self):
        order = self._order()
        note = fund_notes.write_note(order, lambda _: "The model gives Hamas disarming a 40% chance, so buy NO.")
        self.assertEqual(note["note_source"], "facts")
        self.assertTrue(note["note"].startswith("Buy NO Will hamas?"))

    def test_a_grounded_note_is_kept(self):
        order = self._order()
        text = "The model puts NO at 98.0% while it costs 80¢ to buy, leaving about 20.5% expected profit per dollar."
        note = fund_notes.write_note(order, lambda _: f"<think>draft 7</think>{text}")
        self.assertEqual(note["note_source"], "model")
        self.assertEqual(note["note"], text)
        self.assertTrue(any("98.0%" in e["figure"] for e in note["evidence"]))

    def test_a_failing_model_still_leaves_a_note(self):
        def asleep(_):
            raise TimeoutError("mac asleep")
        self.assertEqual(fund_notes.write_note(self._order(), asleep)["note_source"], "facts")


if __name__ == "__main__":
    unittest.main()
