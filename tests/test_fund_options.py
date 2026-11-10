from __future__ import annotations

import math
import sqlite3
import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import fund_ledger as ledger  # noqa: E402
from analyzing_llm_rationale import fund_options as fo  # noqa: E402
from analyzing_llm_rationale import fund_orchestrator as orch  # noqa: E402

NOW = datetime(2026, 11, 9, 20, 0, tzinfo=timezone.utc)  # 01:30 IST on 10 Nov, an NSE holiday
EXPIRY = date(2026, 11, 17)
LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
Z = [NormalDist().inv_cdf(p) for p in LEVELS]
SPOT = 25_000.0


def _forecast(sigma_per_session=0.01, as_of="2026-11-09"):
    """An index model payload whose quantiles are lognormal with no drift."""
    horizons = [{"h": h, "quantiles": [SPOT * math.exp(sigma_per_session * math.sqrt(h) * z) for z in Z]}
                for h in (1, 5, 21)]
    return {"model": {"quantiles": list(LEVELS)}, "holidays": ["2026-11-10"],
            "indices": [{"name": "Nifty 50", "as_of": as_of, "close": SPOT, "horizons": horizons}]}


def _black(forward, strike, sd, kind):
    if sd <= 0:
        return max(forward - strike, 0.0) if kind == "CE" else max(strike - forward, 0.0)
    d1 = math.log(forward / strike) / sd + sd / 2
    n = NormalDist().cdf
    call = forward * n(d1) - strike * n(d1 - sd)
    return call if kind == "CE" else call - forward + strike


def _rows():
    ms = int(datetime(2026, 11, 17, 15, 30, tzinfo=fo.IST).timestamp() * 1000)
    return [{"segment": "NSE_FO", "underlying_symbol": "NIFTY", "instrument_type": kind, "expiry": ms,
             "strike_price": float(k), "instrument_key": f"NSE_FO|{kind}{k}", "trading_symbol": f"NIFTY {k} {kind}",
             "lot_size": 65} for k in range(23_000, 27_001, 100) for kind in ("CE", "PE")]


def _market(sigma_per_session, sessions=5.0, spread=0.5):
    """Live book quotes priced by Black-76 at the market's own volatility."""
    sd = sigma_per_session * math.sqrt(sessions)
    by_key = {r["instrument_key"]: r for r in _rows()}

    def quote_fn(keys):
        out = {}
        for k in keys:
            if k == "NSE_INDEX|Nifty 50":
                out[k] = {"last": SPOT, "bid": None, "ask": None, "at": NOW.isoformat(), "source": "book"}
            elif k in by_key:
                r = by_key[k]
                mid = _black(SPOT, r["strike_price"], sd, r["instrument_type"])
                out[k] = {"bid": max(mid - spread / 2, 0.05), "ask": mid + spread / 2, "last": mid,
                          "at": NOW.isoformat(), "source": "book"}
        return out
    return quote_fn


class DistributionTests(unittest.TestCase):
    def test_lognormal_quantiles_price_a_straddle_as_black_does(self):
        levels, returns = fo.return_quantiles(_forecast()["indices"][0]["horizons"], SPOT, 5.0, LEVELS)
        dist = fo.Distribution(levels, returns, SPOT, SPOT)
        straddle = dist.expect(lambda s: abs(s - SPOT))
        sd = 0.01 * math.sqrt(5)
        self.assertAlmostEqual(straddle / (2 * _black(SPOT, SPOT, sd, "CE")), 1.0, delta=0.01)

    def test_time_is_interpolated_by_its_square_root(self):
        horizons = _forecast()["indices"][0]["horizons"]
        for tau in (5.0, 3.0, 42.0):
            _levels, returns = fo.return_quantiles(horizons, SPOT, tau, LEVELS)
            self.assertAlmostEqual(returns[-1], 0.01 * math.sqrt(tau) * Z[-1], places=9)

    def test_the_models_own_tail_quantiles_join_the_knots_when_published(self):
        horizons = _forecast()["indices"][0]["horizons"]
        levels, _ = fo.return_quantiles(horizons, SPOT, 5.0, LEVELS)
        self.assertEqual(levels, list(LEVELS))
        fat = [{**h, "tail_quantiles": {"0.01": SPOT * math.exp(-0.04 * math.sqrt(h["h"])),
                                        "0.99": SPOT * math.exp(0.04 * math.sqrt(h["h"]))}} for h in horizons]
        levels, returns = fo.return_quantiles(fat, SPOT, 5.0, LEVELS)
        self.assertEqual((levels[0], levels[-1]), (0.01, 0.99))
        self.assertAlmostEqual(returns[-1], 0.04 * math.sqrt(5), places=9)
        thin = fo.Distribution(*fo.return_quantiles(horizons, SPOT, 5.0, LEVELS), SPOT, SPOT)
        wide = fo.Distribution(levels, returns, SPOT, SPOT)
        self.assertGreater(wide.expect(lambda s: abs(s - SPOT)), thin.expect(lambda s: abs(s - SPOT)))

    def test_the_distribution_is_moved_onto_the_market_forward(self):
        # A model median 1% above the close is a drift the model has no skill on; it is removed.
        levels, returns = fo.return_quantiles(_forecast()["indices"][0]["horizons"], SPOT, 5.0, LEVELS)
        dist = fo.Distribution(levels, [r + 0.01 for r in returns], SPOT, 24_990.0)
        self.assertAlmostEqual(sum(dist.levels) / len(dist.levels), 24_990.0, delta=0.5)


class SessionTests(unittest.TestCase):
    HOLIDAYS = [date(2026, 11, 10)]

    def test_sessions_left_count_whole_sessions_and_what_remains_of_today(self):
        self.assertEqual(fo.sessions_left(NOW, EXPIRY, self.HOLIDAYS), 5.0)  # 11, 12, 13, 16, 17 Nov
        midday = datetime(2026, 11, 11, 8, 0, tzinfo=timezone.utc)  # 13:30 IST, two hours of 6.25 left
        self.assertAlmostEqual(fo.sessions_left(midday, EXPIRY, self.HOLIDAYS), 4.0 + 2 / 6.25)
        after = datetime(2026, 11, 11, 10, 30, tzinfo=timezone.utc)  # 16:00 IST
        self.assertEqual(fo.sessions_left(after, EXPIRY, self.HOLIDAYS), 4.0)


class StructureTests(unittest.TestCase):
    CONDOR = fo.Structure("condor", "RANGE", [
        fo.Leg("p1", "PE", 24_300, -1), fo.Leg("p0", "PE", 24_000, 1),
        fo.Leg("c1", "CE", 25_700, -1), fo.Leg("c0", "CE", 26_000, 1)], const=300.0)

    def test_a_condor_is_worth_between_nothing_and_its_width(self):
        self.assertEqual(self.CONDOR.payoff(25_000), 300.0)
        self.assertEqual(self.CONDOR.payoff(23_000), 0.0)
        self.assertEqual(self.CONDOR.payoff(27_000), 0.0)
        self.assertEqual(self.CONDOR.payoff(24_150), 150.0)

    def test_legs_are_bought_at_the_ask_and_sold_at_the_bid(self):
        quotes = {"p1": {"bid": 60, "ask": 61}, "p0": {"bid": 30, "ask": 31},
                  "c1": {"bid": 50, "ask": 51}, "c0": {"bid": 20, "ask": 21}}
        q = self.CONDOR.quote(quotes)
        self.assertEqual(q["ask"], 300 - 60 + 31 - 50 + 21)  # cost = width less the credit
        self.assertEqual(q["bid"], 300 - 61 + 30 - 51 + 20)
        self.assertIsNone(self.CONDOR.quote({"p1": quotes["p1"]}))


class SignalTests(unittest.TestCase):
    def _run(self, market_sigma, forecast=None, held=()):
        return fo.option_signals(forecast or _forecast(), _rows(), NOW, quote_fn=_market(market_sigma),
                                 price_source="live order book (Upstox)", held=dict(held))

    def test_options_priced_narrower_than_the_model_buy_a_straddle(self):
        out = self._run(0.008)
        [s] = out["signals"]
        self.assertEqual((s.side, s.instrument["structure"], s.analyst), ("MOVE", "straddle", "options_model"))
        self.assertGreaterEqual(s.edge, orch.MIN_EDGE)
        self.assertEqual(s.instrument["label"], "Nifty 50 straddle at 25,000, 17 Nov 2026")
        self.assertIn("17 Nov 2026, 5.0 sessions left", {f["value"] for f in s.facts})
        self.assertIn(s.instrument["id"], out["quotes"])

    def test_options_priced_wider_than_the_model_sell_the_range_with_wings(self):
        [s] = self._run(0.0125)["signals"]
        self.assertIn(s.side, ("RANGE", "BULL", "BEAR"))
        self.assertTrue(any(leg["sign"] > 0 for leg in s.instrument["legs"]))  # never uncovered
        self.assertLessEqual(s.price, s.instrument["const"])  # costs at most its width

    def test_a_fair_market_is_not_traded(self):
        self.assertEqual(self._run(0.01)["signals"], [])

    def test_a_market_far_from_the_model_is_a_likely_error(self):
        out = self._run(0.004)
        self.assertEqual(out["signals"], [])
        self.assertIn("likely model or data error", out["skipped"][0]["reason"])

    def test_a_stale_forecast_is_not_traded(self):
        out = self._run(0.008, forecast=_forecast(as_of="2026-11-02"))
        self.assertEqual(out["signals"], [])
        self.assertIn("2026-11-02 close", out["skipped"][0]["reason"])

    def test_held_structures_are_valued_for_exits(self):
        [s] = self._run(0.008)["signals"]
        out = self._run(0.008, held={s.instrument["id"]: s.instrument})
        self.assertGreater(out["values"][s.instrument["id"]], s.price)


class QuoteTests(unittest.TestCase):
    def test_last_trades_are_widened_and_need_volume_to_count(self):
        def get(url, timeout, headers):
            body = {"data": {"candles": []}} if "intraday" in url else \
                {"data": {"candles": [["2026-11-09T00:00:00+05:30", 280, 297, 153, 230.5, 7_296_770, 794_950]]}}
            return SimpleNamespace(json=lambda: body, raise_for_status=lambda: None)

        q = fo.candle_quotes(["NSE_FO|1"], now=NOW, http_get=get)["NSE_FO|1"]
        self.assertAlmostEqual(q["ask"], 230.5 * (1 + fo.CLOSE_HALF_SPREAD))
        self.assertAlmostEqual(q["bid"], 230.5 * (1 - fo.CLOSE_HALF_SPREAD))
        self.assertEqual(q["at"], "2026-11-09T15:30:00+05:30")
        self.assertTrue(fo._tradeable(q, 65))
        self.assertFalse(fo._tradeable({**q, "volume": 99 * 65}, 65))

    def test_the_book_gives_best_bid_and_ask_and_needs_the_token(self):
        seen = {}

        def get(url, timeout, headers):
            seen.update(headers)
            body = {"data": {"NSE_FO:NIFTY25050CE": {
                "instrument_token": "NSE_FO|1", "last_price": 230.5, "volume": 1000,
                "depth": {"buy": [{"price": 230.0, "quantity": 65}], "sell": [{"price": 231.0, "quantity": 130}]}}}}
            return SimpleNamespace(json=lambda: body, raise_for_status=lambda: None)

        q = fo.book_quotes(["NSE_FO|1"], "tok", now=NOW, http_get=get)["NSE_FO|1"]
        self.assertEqual((q["bid"], q["ask"], q["source"]), (230.0, 231.0, "book"))
        self.assertEqual(seen["Authorization"], "Bearer tok")

    def test_without_a_token_prices_are_last_trades(self):
        _fn, label = fo.quote_source(None, now=NOW)
        self.assertIn("last trades", label)
        self.assertEqual(fo.quote_source("tok", now=NOW)[1], "live order book (Upstox)")


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        ledger.ensure_schema(self.conn)
        ledger.open_fund(self.conn, "f", 10_000.0)
        self.spec = {**fo.instrument("NIFTY", StructureTests.CONDOR, EXPIRY)}

    def test_a_structure_costs_its_price_and_is_worth_its_bid_in_dollars(self):
        ledger.buy(self.conn, "f", self.spec, "RANGE", 100, 160.0, fx=88.0)
        snap = ledger.nav(self.conn, "f", {self.spec["id"]: {"bid": 190.0, "ask": 195.0}}, {"INR": 88.0})
        [pos] = snap["positions"]
        self.assertEqual(pos["kind"], "option")
        self.assertAlmostEqual(pos["collateral_usd"], 16_000 / 88)
        self.assertAlmostEqual(pos["pnl_usd"], 3_000 / 88, places=3)
        self.assertEqual(snap["net_exposure_by_underlying"]["india_equity"], 0.0)  # a condor leans neither way
        self.assertEqual(ledger.instrument_spec(self.conn, "f", self.spec["id"])["legs"], self.spec["legs"])

    def test_a_bearish_structure_counts_as_short_exposure(self):
        ledger.buy(self.conn, "f", {**self.spec, "id": "nse_fo:NIFTY:call_spread:x"}, "BEAR", 100, 88.0, fx=88.0)
        self.assertAlmostEqual(ledger.nav(self.conn, "f", {}, {"INR": 88.0})["net_exposure_by_underlying"]["india_equity"],
                               -100.0)

    def test_a_structure_without_legs_is_refused(self):
        with self.assertRaises(ledger.LedgerError):
            ledger.buy(self.conn, "f", {**self.spec, "legs": []}, "RANGE", 1, 100.0, fx=88.0)

    def test_an_expired_structure_settles_at_its_value(self):
        ledger.buy(self.conn, "f", self.spec, "RANGE", 100, 160.0, fx=88.0)
        ledger.sell(self.conn, "f", self.spec["id"], "RANGE", 100, 0.0, fx=88.0, action="expire")
        account = self.conn.execute("SELECT cash, realized_pnl FROM fund_account").fetchone()
        self.assertAlmostEqual(account[1], -16_000 / 88)
        self.assertAlmostEqual(account[0], 10_000 - 16_000 / 88)


class PlanTests(unittest.TestCase):
    def _signal(self, price=160.0):
        inst = fo.instrument("NIFTY", StructureTests.CONDOR, EXPIRY)
        return orch.Signal("options_model", inst, "RANGE", price, 0.08, orch.DEFAULT_TRUST)

    def _snapshot(self, positions=()):
        return {"nav_usd": 10_000.0, "cash_usd": 6_000.0, "positions": list(positions), "by_asset_class": {},
                "net_exposure_by_underlying": {}}

    def test_a_structure_risks_two_percent_sized_in_rupees(self):
        [order] = orch.plan(self._snapshot(), [self._signal()], {}, {}, rates={"INR": 88.0})["orders"]
        self.assertAlmostEqual(order.notional / 88.0, orch.OPTION_RISK * 10_000)
        self.assertEqual(order.attribution, {"options_model": 1.0})

    def test_one_structure_per_index_and_expiry(self):
        s = self._signal()
        held = [{"instrument_id": s.instrument["id"].replace("condor", "straddle"), "side": "MOVE", "kind": "option",
                 "quantity": 1, "mark": 300.0, "attribution": {}}]
        out = orch.plan(self._snapshot(held), [s], {}, {}, rates={"INR": 88.0})
        self.assertEqual(out["orders"], [])
        self.assertIn("already holds a structure", out["passed"][0]["reason"])

    def test_a_structure_the_model_now_values_well_below_its_bid_is_sold(self):
        held = [{"instrument_id": "nse_fo:NIFTY:condor:2026-11-17:x", "label": "condor", "side": "RANGE",
                 "kind": "option", "quantity": 10, "mark": 200.0, "attribution": {"options_model": 1.0}}]
        out = orch.plan(self._snapshot(held), [], {}, {}, option_values={"nse_fo:NIFTY:condor:2026-11-17:x": 170.0})
        self.assertEqual([(o.action, o.purpose) for o in out["orders"]], [("sell", "exit")])
        keep = orch.plan(self._snapshot(held), [], {}, {}, option_values={"nse_fo:NIFTY:condor:2026-11-17:x": 192.0})
        self.assertEqual(keep["orders"], [])

    def test_rupee_futures_are_sized_in_rupees(self):
        # Before 2026-11-10 a rupee leg was sized as if its price were in dollars: 88 times too small.
        copper = {"id": "mcx:COPPER", "label": "Copper", "venue": "mcx", "asset_class": "commodity",
                  "underlying": "copper", "currency": "INR", "multiplier": 2500.0}
        s = orch.Signal("pairs_model", copper, "LONG", 0.0, 0.003, orch.DEFAULT_TRUST, pair_key="copper-aluminium")
        [order] = orch.plan(self._snapshot(), [s], {"mcx:COPPER": {"bid": 1000.0, "ask": 1001.0}}, {},
                            rates={"INR": 88.0})["orders"]
        self.assertAlmostEqual(order.notional / 88.0, orch.PAIR_RISK * 10_000)


class PutSleeveTests(unittest.TestCase):
    """The put premium sleeve: Nifty's expiry nearest 21 sessions, the put spread every time."""

    def _rows(self):
        far = int(datetime(2026, 12, 8, 15, 30, tzinfo=fo.IST).timestamp() * 1000)  # 20 sessions out
        return _rows() + [{**r, "expiry": far, "instrument_key": r["instrument_key"] + "D"} for r in _rows()]

    def _market(self, sigma, sessions=20.0):
        base = _market(sigma, sessions)

        def quote_fn(keys):
            plain = base([k[:-1] if k.endswith("D") else k for k in keys])
            return {k: plain[k[:-1] if k.endswith("D") else k] for k in keys
                    if (k[:-1] if k.endswith("D") else k) in plain}
        return quote_fn

    def test_the_sleeve_sells_the_put_spread_even_when_the_market_is_fair(self):
        out = fo.put_spread_signal(_forecast(), self._rows(), NOW, quote_fn=self._market(0.01),
                                   price_source="live order book (Upstox)")
        s = out["signal"]
        self.assertEqual((s.analyst, s.side, s.instrument["structure"]), ("put_premium", "BULL", "put_spread"))
        self.assertEqual(s.instrument["expiry"], "2026-12-08")  # not the 17 Nov weekly, 5 sessions out
        self.assertEqual(s.reason, fo.SLEEVE_RULE)
        self.assertLess(s.price, s.instrument["const"])  # it collects a credit
        self.assertTrue(all(leg["type"] == "PE" for leg in s.instrument["legs"]))

    def test_no_expiry_in_the_band_or_a_thin_leg_means_no_trade_and_says_why(self):
        none = fo.put_spread_signal(_forecast(), _rows(), NOW, quote_fn=_market(0.01), price_source="x")
        self.assertIsNone(none["signal"])
        self.assertIn("15 to 30 sessions", none["skipped"]["reason"])
        base = self._market(0.01)

        def thin(keys):
            return {k: {**q, "bid": 0.0} if "|PE" in k else q for k, q in base(keys).items()}
        out = fo.put_spread_signal(_forecast(), self._rows(), NOW, quote_fn=thin, price_source="x")
        self.assertIsNone(out["signal"])
        self.assertIn("too thin", out["skipped"]["reason"])

    def test_an_order_from_a_rule_carries_the_rule_as_its_reason(self):
        out = fo.put_spread_signal(_forecast(), self._rows(), NOW, quote_fn=self._market(0.01), price_source="x")
        [order] = orch.plan({"nav_usd": 10_000.0, "cash_usd": 6_000.0, "positions": [], "by_asset_class": {},
                             "net_exposure_by_underlying": {}}, [out["signal"]], {}, {}, rates={"INR": 88.0})["orders"]
        self.assertEqual(order.reason, fo.SLEEVE_RULE)
        self.assertEqual(order.attribution, {"put_premium": 1.0})
        self.assertAlmostEqual(order.notional / 88.0, orch.OPTION_RISK * 10_000)


if __name__ == "__main__":
    unittest.main()
