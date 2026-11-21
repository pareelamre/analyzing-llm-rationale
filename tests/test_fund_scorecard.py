from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from analyzing_llm_rationale import fund_ledger as ledger  # noqa: E402
from analyzing_llm_rationale import fund_orchestrator as orch  # noqa: E402
from analyzing_llm_rationale import fund_scorecard as sc  # noqa: E402

FUND = "test-fund"


def _fund() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ledger.ensure_schema(conn)
    ledger.open_fund(conn, FUND, 10_000.0, now="2026-11-09T00:00:00+00:00")
    return conn


def _event(ident: str) -> dict:
    return orch.event_instrument({"platform": "kalshi", "ident": ident, "question": ident})


def _leg(ident: str) -> dict:
    return {"id": ident, "label": ident, "venue": "mcx", "asset_class": "commodity", "underlying": ident,
            "currency": "USD", "multiplier": 1.0}


class PoolTests(unittest.TestCase):
    def test_a_test_on_another_market_counts_as_half_its_trades(self):
        e = sc.Evidence("Nasdaq test", 100, 0.10, 0.40, sc.PROXY_WEIGHT)
        score = sc.pool([e], [])
        self.assertEqual(score["counted"], 50.0)
        self.assertAlmostEqual(score["low"], 0.10 - 1.96 * 0.40 / 50 ** 0.5, places=4)
        self.assertAlmostEqual(score["high"], 0.10 + 1.96 * 0.40 / 50 ** 0.5, places=4)

    def test_live_positions_pool_with_the_test(self):
        e = sc.Evidence("test", 20, 0.10, 0.0)
        score = sc.pool([e], [-0.10] * 20)
        self.assertEqual(score["counted"], 40.0)
        self.assertAlmostEqual(score["mean"], 0.0)
        self.assertAlmostEqual(score["sd"], 0.10)

    def test_no_evidence_has_no_score(self):
        self.assertIsNone(sc.pool([], [])["mean"])
        self.assertIsNone(sc.pool([], [0.2])["low"])  # one position has no range


class AllocateTests(unittest.TestCase):
    def test_a_proven_strategy_gets_a_quarter_kelly_at_its_low_end_within_the_bounds(self):
        status, scale, kelly = sc.allocate({"counted": 141.5, "low": 0.0387, "high": 0.167, "sd": 0.39}, "option")
        self.assertEqual(status, "proven")
        self.assertAlmostEqual(kelly, 0.25 * 0.0387 / 0.39 ** 2, places=4)
        self.assertEqual(scale, sc.MAX_SCALE)  # 6.4% at risk is above 2.5x the 2% start
        status, scale, _ = sc.allocate({"counted": 60, "low": 0.005, "high": 0.2, "sd": 0.39}, "option")
        self.assertEqual((status, scale), ("proven", sc.MIN_PROVEN_SCALE))  # Kelly says less than trial: floor
        _, scale, kelly = sc.allocate({"counted": 60, "low": 0.02, "high": 0.2, "sd": 0.45}, "option")
        self.assertAlmostEqual(scale, kelly / orch.OPTION_RISK, places=3)

    def test_few_positions_or_a_range_across_zero_mean_trial_and_a_range_below_zero_means_failed(self):
        self.assertEqual(sc.allocate({"counted": 19, "low": 0.05, "high": 0.2, "sd": 0.1}, "pair")[:2],
                         ("trial", sc.TRIAL_SCALE))
        self.assertEqual(sc.allocate({"counted": 50, "low": -0.05, "high": 0.2, "sd": 0.3}, "pair")[:2],
                         ("trial", sc.TRIAL_SCALE))
        self.assertEqual(sc.allocate({"counted": 50, "low": -0.3, "high": -0.01, "sd": 0.3}, "event")[:2],
                         ("failed", 0.0))
        self.assertEqual(sc.allocate({"counted": 0, "low": None, "high": None, "sd": None}, "event")[:2],
                         ("trial", sc.TRIAL_SCALE))


class ClosedPositionTests(unittest.TestCase):
    def test_each_closed_position_counts_once_with_its_fees(self):
        conn = _fund()
        ledger.buy(conn, FUND, _event("A"), "YES", 100, 0.40, fee_usd=1.0, attribution={"blend_model": 1.0},
                   now="2026-11-10T00:00:00+00:00")
        ledger.sell(conn, FUND, "kalshi:A", "YES", 40, 0.50, action="reduce", now="2026-11-11T00:00:00+00:00")
        ledger.sell(conn, FUND, "kalshi:A", "YES", 60, 1.0, action="settle", now="2026-11-12T00:00:00+00:00")
        ledger.buy(conn, FUND, _event("B"), "NO", 50, 0.20, attribution={"blend_model": 1.0},
                   now="2026-11-10T00:00:00+00:00")  # still open
        [p] = sc.closed_positions(conn, FUND)
        self.assertEqual(p["analyst"], "blend_model")
        self.assertAlmostEqual(p["at_risk"], 40.0)
        self.assertAlmostEqual(p["realized"], 40 * 0.10 + 60 * 0.60 - 1.0)
        self.assertAlmostEqual(p["return"], 39.0 / 40.0)

    def test_both_legs_of_a_pair_are_one_position_and_hedges_do_not_count(self):
        conn = _fund()
        for leg, side in (("cu", "LONG"), ("al", "SHORT")):
            ledger.buy(conn, FUND, _leg(leg), side, 10, 50.0, attribution={"pairs_model": 1.0},
                       pair_key="copper-aluminium", now="2026-11-10T00:00:00+00:00")
        ledger.buy(conn, FUND, _leg("perp"), "SHORT", 1, 100.0, attribution={"risk_overlay": 1.0}, purpose="hedge",
                   now="2026-11-10T00:00:00+00:00")
        ledger.sell(conn, FUND, "cu", "LONG", 10, 53.0, pair_key="copper-aluminium", cycle_id="c2",
                    now="2026-11-12T00:00:00+00:00")
        ledger.sell(conn, FUND, "al", "SHORT", 10, 51.0, pair_key="copper-aluminium", cycle_id="c2",
                    now="2026-11-12T00:00:00+00:00")
        ledger.sell(conn, FUND, "perp", "SHORT", 1, 90.0, purpose="hedge", now="2026-11-12T00:00:00+00:00")
        [p] = sc.closed_positions(conn, FUND)
        self.assertEqual((p["analyst"], p["pair_key"]), ("pairs_model", "copper-aluminium"))
        self.assertAlmostEqual(p["at_risk"], 1000.0)
        self.assertAlmostEqual(p["return"], (30.0 - 10.0) / 1000.0)


class ScorecardTests(unittest.TestCase):
    def test_on_the_repos_evidence_put_spreads_are_proven_and_the_rest_on_trial(self):
        card = sc.scorecard(None, FUND, _ROOT)
        rows = {r["analyst"]: r for r in card["strategies"]}
        put = rows["put_premium"]
        # Nasdaq-100 and Bank Nifty at half weight; the two spreads share a quarter-Kelly by (1 + rho).
        self.assertEqual(put["status"], "proven")
        self.assertEqual([t["source"] for t in put["tests"]],
                         ["reports/fund_ndx_put_eval.json", "reports/fund_banknifty_put_eval.json"])
        self.assertTrue(all(t["counted"] == t["trades"] * sc.PROXY_WEIGHT for t in put["tests"]))
        self.assertAlmostEqual(put["together"], 0.7071, places=3)
        self.assertAlmostEqual(put["scale"] * sc.STARTING_SIZE["option"],
                               put["kelly_at_low"] / (1 + put["together"]), places=4)
        self.assertEqual(put["size"], "4.1% of the fund at risk per spread")
        self.assertIn("move together", put["why"])
        self.assertEqual((rows["pairs_model"]["status"], rows["pairs_model"]["size"]),
                         ("trial", "2.5% of the fund per leg"))
        self.assertEqual((rows["blend_model"]["status"], rows["blend_model"]["size"]),
                         ("trial", "1/32 of the Kelly bet per market"))
        self.assertIn("Brier", rows["blend_model"]["context"])
        self.assertTrue(all(t["idea"] and t["result"] and t["verdict"] for t in card["tested"]))

    def test_losses_in_the_fund_take_a_proven_strategys_money_away(self):
        conn = _fund()
        spread = {"id": "nse_fo:NIFTY:put_spread:x", "label": "spread", "venue": "nse_fo", "asset_class": "option",
                  "underlying": "NIFTY", "currency": "USD", "multiplier": 1.0, "legs": [{"strike": 1}]}
        def lose(n: int) -> dict:
            for _ in range(n):
                ledger.buy(conn, FUND, spread, "BULL", 1, 50.0, attribution={"put_premium": 1.0},
                           now="2026-11-10T00:00:00+00:00")
                ledger.sell(conn, FUND, spread["id"], "BULL", 1, 0.0, action="expire", now="2026-11-10T01:00:00+00:00")
            return {r["analyst"]: r for r in sc.scorecard(conn, FUND, _ROOT)["strategies"]}["put_premium"]
        put = lose(20)  # 20 total losses pull the low end below zero: back on trial
        self.assertEqual((put["live"]["positions"], put["live"]["mean"]), (20, -1.0))
        self.assertEqual((put["status"], put["scale"]), ("trial", sc.TRIAL_SCALE))
        put = lose(20)  # 20 more and even the high end is below zero
        self.assertEqual((put["status"], put["scale"]), ("failed", 0.0))

    def test_a_strategy_with_no_test_and_a_losing_record_fails(self):
        conn = _fund()
        for i in range(25):
            ledger.buy(conn, FUND, _event(f"M{i}"), "YES", 10, 0.5, attribution={"blend_model": 1.0},
                       now="2026-11-10T00:00:00+00:00")
            ledger.sell(conn, FUND, f"kalshi:M{i}", "YES", 10, 0.0 if i % 5 else 0.6, action="settle",
                        now="2026-11-11T00:00:00+00:00")
        blend = {r["analyst"]: r for r in sc.scorecard(conn, FUND, _ROOT)["strategies"]}["blend_model"]
        self.assertEqual((blend["status"], blend["scale"], blend["size"]), ("failed", 0.0, "no money"))

    def test_retired_analysts_records_are_kept_apart(self):
        conn = _fund()
        ledger.buy(conn, FUND, _event("OLD"), "NO", 10, 0.9, attribution={"agent_transfer": 1.0}, action="transfer",
                   now="2026-11-09T00:00:00+00:00")
        ledger.sell(conn, FUND, "kalshi:OLD", "NO", 10, 1.0, action="settle", now="2026-11-14T00:00:00+00:00")
        card = sc.scorecard(conn, FUND, _ROOT)
        self.assertEqual(card["retired_records"], [{"analyst": "agent_transfer", "positions": 1, "realized_usd": 1.0}])
        self.assertEqual({r["analyst"]: r["live"]["positions"] for r in card["strategies"]},
                         {"put_premium": 0, "pairs_model": 0, "blend_model": 0})

    def test_bank_nifty_counts_and_shares_the_size_only_while_its_test_says_go(self):
        import json
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "reports").mkdir()
            shutil.copy(_ROOT / "reports" / "fund_ndx_put_eval.json", Path(tmp) / "reports")
            alone = {r["analyst"]: r for r in sc.scorecard(None, FUND, Path(tmp))["strategies"]}["put_premium"]
            self.assertEqual((len(alone["tests"]), alone["together"], alone["scale"]), (1, None, sc.MAX_SCALE))
            report = json.loads((_ROOT / "reports" / sc.BANK_NIFTY_REPORT).read_text())
            report["put_spread"]["verdict"] = "NO-GO"
            (Path(tmp) / "reports" / sc.BANK_NIFTY_REPORT).write_text(json.dumps(report))
            off = {r["analyst"]: r for r in sc.scorecard(None, FUND, Path(tmp))["strategies"]}["put_premium"]
            self.assertEqual((len(off["tests"]), off["together"]), (1, None))

    def test_two_positions_that_move_together_split_the_kelly_bet(self):
        score = {"counted": 200, "mean": 0.08, "sd": 0.4, "low": 0.025, "high": 0.135}
        _status, alone, kelly = sc.allocate(score, "option")
        _status, each, _kelly = sc.allocate(score, "option", rho=0.6)
        self.assertAlmostEqual(kelly, 0.25 * 0.025 / 0.16, places=5)
        self.assertAlmostEqual(each, round(alone / 1.6, 4), places=3)
        _status, same, _kelly = sc.allocate(score, "option", rho=1.0)  # one bet in two halves
        self.assertAlmostEqual(same, round(max(sc.MIN_PROVEN_SCALE, alone / 2), 4), places=3)

    def test_a_missing_report_leaves_the_strategy_on_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = {r["analyst"]: r for r in sc.scorecard(None, FUND, Path(tmp))["strategies"]}
        self.assertEqual(rows["put_premium"]["status"], "trial")
        self.assertIn("missing", rows["put_premium"]["context"])


class ApplyTests(unittest.TestCase):
    def _signal(self, analyst: str) -> orch.Signal:
        return orch.Signal(analyst=analyst, instrument={"id": f"{analyst}:x", "label": "x"}, side="YES",
                           price=0.5, edge=0.1, trust=orch.DEFAULT_TRUST)

    def test_signals_take_their_analysts_trust_and_unfunded_ones_are_skipped_with_why(self):
        card = {"strategies": [{"analyst": "put_premium", "scale": 2.5, "trust": 0.625, "label": "Put", "why": "ok"},
                               {"analyst": "blend_model", "scale": 0.0, "trust": 0.0, "label": "Blend",
                                "why": "Failed: below zero."}]}
        out, skipped = sc.apply([self._signal("put_premium"), self._signal("blend_model"),
                                 self._signal("someone_new")], card)
        self.assertEqual([(s.analyst, s.trust) for s in out], [("put_premium", 0.625),
                                                               ("someone_new", orch.DEFAULT_TRUST)])
        self.assertEqual(skipped, [{"instrument": "blend_model:x", "label": "x",
                                    "reason": "the scorecard gives Blend no money: Failed: below zero."}])

    def test_a_proven_put_spread_at_the_cap_is_held_to_the_three_percent_risk_limit(self):
        # Trust at the 2.5x cap would put 5% at risk; fund_risk (2026-11-21) holds any one
        # position's worst case to 3% of the fund.
        inst = {"id": "nse_fo:NIFTY:put_spread:2026-12-08:x", "label": "spread", "asset_class": "option",
                "underlying": "NIFTY", "currency": "INR", "multiplier": 1.0}
        s = orch.Signal(analyst="put_premium", instrument=inst, side="BULL", price=50.0, edge=0.1, trust=0.625)
        snap = {"nav_usd": 10_000.0, "cash_usd": 10_000.0, "positions": [], "by_asset_class": {},
                "net_exposure_by_underlying": {}}
        [order] = orch.plan(snap, [s], {}, {}, rates={"INR": 88.0})["orders"]
        self.assertAlmostEqual(order.quantity * order.price / 88.0, 300.0)


if __name__ == "__main__":
    unittest.main()
