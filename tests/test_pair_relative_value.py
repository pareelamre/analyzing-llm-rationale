"""Relative value on the commodity pairs: the World Bank reader, the rule, its ledger, and how the fund
enters and exits on it (commodity_pairs.RV_*, pair_history, fund_orchestrator). Made-up prices."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from analyzing_llm_rationale import commodity_pairs as cp  # noqa: E402
from analyzing_llm_rationale import fund_orchestrator as orch  # noqa: E402
from analyzing_llm_rationale import pair_history as ph  # noqa: E402

_spec = importlib.util.spec_from_file_location("fund_cycle_rv", _ROOT / "scripts" / "fund_cycle.py")
cycle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cycle)

PAIR = {p["key"]: p for p in cp.PAIRS}
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


def _xlsx(rows: list[list]) -> bytes:
    """A minimal .xlsx with one sheet, "Monthly Prices", of inline strings and numbers."""
    def cell(ref, v):
        if isinstance(v, str):
            return f'<c r="{ref}" t="inlineStr"><is><t>{v}</t></is></c>'
        return f'<c r="{ref}"><v>{v}</v></c>'
    body = "".join(f'<row r="{i + 1}">' + "".join(cell(f"{chr(65 + j)}{i + 1}", v) for j, v in enumerate(r)
                                                 if v is not None) + "</row>" for i, r in enumerate(rows))
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rel = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/workbook.xml", f'<workbook {ns} {rel}><sheets><sheet name="Monthly Prices" sheetId="1" '
                                      'r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr("xl/worksheets/sheet1.xml", f"<worksheet {ns}><sheetData>{body}</sheetData></worksheet>")
    return buf.getvalue()


def _pink_rows(n=130):
    rows = [["World Bank Commodity Price Data"], [None, "Copper", "Gold", "Silver", "Platinum"],
            [None, "($/mt)", "($/troy oz)", "($/troy oz)", "($/troy oz)"]]
    y, m = 2015, 1
    for i in range(n):
        rows.append([f"{y}M{m:02d}", 6000.0 + i, 1200.0 + i, "…" if i == 0 else 16.0, 900.0])
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return rows


class Resp:
    def __init__(self, status=200, text="", content=b""):
        self.status_code, self.text, self.content = status, text, content


class PinkSheetTests(unittest.TestCase):
    def test_reads_months_and_skips_missing_values(self):
        out = ph.parse_pink_sheet(_xlsx(_pink_rows()), ("gold", "silver"))
        self.assertEqual(out["gold"][0], ("2015-01", 1200.0))
        self.assertEqual(out["silver"][0], ("2015-02", 16.0))  # "…" in January is left out
        self.assertEqual(len(out["gold"]), 130)

    def test_a_missing_column_or_short_history_is_a_feed_error(self):
        with self.assertRaises(cp.FeedError):
            ph.parse_pink_sheet(_xlsx(_pink_rows()), ("nickel",))
        with self.assertRaises(cp.FeedError):
            ph.parse_pink_sheet(_xlsx(_pink_rows(50)), ("gold",))
        with self.assertRaises(cp.FeedError):
            ph.parse_pink_sheet(b"not a spreadsheet", ("gold",))

    def test_the_link_comes_from_the_page_and_falls_back_when_it_cannot(self):
        link = "https://thedocs.worldbank.org/en/doc/abc-0050012027/related/CMO-Historical-Data-Monthly.xlsx"
        page = f'<a href="{link}">Monthly prices</a>'
        self.assertEqual(ph.pink_sheet_url(http_get=lambda *a, **k: Resp(text=page)), link)
        self.assertEqual(ph.pink_sheet_url(http_get=lambda *a, **k: Resp(status=503)), ph.PINK_SHEET_FALLBACK)
        self.assertEqual(ph.pink_sheet_url(http_get=lambda *a, **k: Resp(text="<html></html>")), ph.PINK_SHEET_FALLBACK)

    def test_fetch_monthly_adds_gold_silver_platinum_and_survives_an_outage(self):
        def http_get(url, params=None, **_):
            if url == ph.PINK_SHEET_PAGE:
                return Resp(text="")
            if url == ph.PINK_SHEET_FALLBACK:
                return Resp(content=_xlsx(_pink_rows()))
            return Resp(text="observation_date,v\n2026-09-01,10\n2026-10-01,11\n")

        monthly = ph.fetch_monthly(date(2026, 11, 17), http_get=http_get)
        self.assertEqual(monthly["gold"][0], ("2015-01", 1200.0))
        self.assertIn("platinum", monthly)

        def down(url, params=None, **_):
            if "worldbank" in url:
                raise ConnectionError("no route")
            return Resp(text="observation_date,v\n2026-09-01,10\n")

        monthly = ph.fetch_monthly(date(2026, 11, 17), http_get=down)
        self.assertNotIn("gold", monthly)
        self.assertIn("copper", monthly)

    def test_the_source_label_names_each_feed(self):
        self.assertEqual(ph.source_label(PAIR["copper-gold"]), "IMF monthly prices (FRED) and World Bank Pink Sheet monthly prices")
        self.assertEqual(ph.source_label(PAIR["brent-wti"]), "FRED daily spot prices, monthly averages")


def _ctx(z, story=None):
    return {"z": z, "percentile": 0.98 if z > 0 else 0.02, "ratio": 2.07, "usual": 1.36, "window_months": 120,
            "story": story}


class RuleTests(unittest.TestCase):
    def test_a_dear_first_leg_sells_the_ratio_and_a_cheap_one_buys_it(self):
        rv = cp.relative_value(PAIR["zinc-lead"], _ctx(2.95))
        self.assertEqual((rv["rich"], rv["cheap"], rv["side"], rv["signal"]), ("Zinc", "Lead", -1, True))
        rv = cp.relative_value(PAIR["nickel-copper"], _ctx(-1.6))
        self.assertEqual((rv["rich"], rv["cheap"], rv["side"]), ("Copper", "Nickel", 1))

    def test_inside_the_entry_line_it_reads_but_does_not_trade(self):
        rv = cp.relative_value(PAIR["copper-aluminium"], _ctx(1.19))
        self.assertEqual((rv["rich"], rv["side"], rv["signal"]), ("Copper", None, False))
        rv = cp.relative_value(PAIR["copper-zinc"], _ctx(0.3))
        self.assertEqual((rv["rich"], rv["cheap"]), (None, None))
        self.assertIsNone(cp.relative_value(PAIR["copper-zinc"], None))

    def test_wti_gas_trades_from_its_own_nearer_line(self):
        # Owner, 2026-11-25: start trading WTI / gas; its unseen years hold up from 0.85.
        rv = cp.relative_value(PAIR["wti-natgas"], _ctx(0.87))
        self.assertEqual((rv["rich"], rv["cheap"], rv["side"], rv["signal"], rv["entry_z"]),
                         ("WTI", "Natural gas", -1, True, 0.85))
        self.assertFalse(cp.relative_value(PAIR["wti-natgas"], _ctx(0.8))["signal"])
        other = cp.relative_value(PAIR["zinc-lead"], _ctx(0.87))
        self.assertEqual((other["signal"], other["entry_z"]), (False, 1.5))

    def test_a_lasting_cause_holds_the_trade_back(self):
        story = {"driver": "lasting", "cause": "Indonesia's smelters kept nickel in surplus. More text."}
        rv = cp.relative_value(PAIR["nickel-copper"], _ctx(-2.1, story))
        self.assertEqual((rv["side"], rv["signal"]), (1, False))
        self.assertIn("lasting (Indonesia's smelters kept nickel in surplus)", rv["held_back"])
        temporary = cp.relative_value(PAIR["zinc-lead"], _ctx(2.9, {"driver": "temporary", "cause": "Ore shortage."}))
        self.assertTrue(temporary["signal"])

    def test_expected_return_blends_the_unseen_average_with_the_pairs_own_record(self):
        prior = cp.RV_RECORD["unseen"]["avg_return"]
        own = cp.RV_BY_PAIR["copper-aluminium"]
        exp = cp.rv_expected("copper-aluminium", {"closed": 2, "avg_return": -0.1})
        want = (cp.RV_PRIOR_TRADES * prior + own["trades"] * own["avg_return"] + 2 * -0.1) / (cp.RV_PRIOR_TRADES + own["trades"] + 2)
        self.assertAlmostEqual(exp["per_trade"], round(want, 5))
        self.assertEqual(cp.rv_expected("no-such-pair")["per_trade"], round(prior, 5))


def _row(key, ratios, start=date(2026, 9, 1)):
    days = [start + timedelta(days=i) for i in range(len(ratios))]
    lb = np.zeros(len(ratios))
    return {"key": key, "_days": days, "_la": np.log(ratios), "_lb": lb}


class LedgerTests(unittest.TestCase):
    def test_a_signal_opens_once_and_closes_back_near_normal_with_its_return(self):
        ledger = []
        view = cp.relative_value(PAIR["zinc-lead"], _ctx(2.5))
        row = _row("zinc-lead", [2.0, 2.0])
        self.assertEqual(cp.update_rv_ledger(ledger, [row], {"zinc-lead": view}, NOW), 1)
        self.assertEqual(cp.update_rv_ledger(ledger, [row], {"zinc-lead": view}, NOW), 0)
        self.assertEqual((ledger[0]["side"], ledger[0]["rule"]), (-1, "relative_value"))
        row = _row("zinc-lead", [2.0, 2.0, 1.9, 1.8])
        cp.update_rv_ledger(ledger, [row], {"zinc-lead": cp.relative_value(PAIR["zinc-lead"], _ctx(0.4))}, NOW)
        e = ledger[0]
        self.assertEqual((e["status"], e["sessions"]), ("back_to_normal", 2))  # opened on the second day
        # Sold the ratio from 2.0 to 1.8: half the log move per dollar on both legs, less costs.
        self.assertAlmostEqual(e["return"], round(-1 * np.log(1.8 / 2.0) / 2 - 2 * cp.COST_PER_SIDE, 6))

    def test_crossing_the_normal_or_the_time_limit_also_closes(self):
        ledger = [{"pair": "zinc-lead", "entry_date": "2026-09-01", "side": -1, "status": "open", "entry_z": 2.5}]
        cp.update_rv_ledger(ledger, [_row("zinc-lead", [2.0, 1.5])],
                            {"zinc-lead": cp.relative_value(PAIR["zinc-lead"], _ctx(-0.9))}, NOW)
        self.assertEqual(ledger[0]["status"], "back_to_normal")
        n = cp.RV_MAX_MONTHS * cp.RV_SESSIONS_PER_MONTH + 1
        ledger = [{"pair": "zinc-lead", "entry_date": "2026-09-01", "side": -1, "status": "open", "entry_z": 2.5}]
        cp.update_rv_ledger(ledger, [_row("zinc-lead", [2.0] * n)],
                            {"zinc-lead": {**cp.relative_value(PAIR["zinc-lead"], _ctx(2.4)), "signal": False}}, NOW)
        self.assertEqual(ledger[0]["status"], "timed_out")

    def test_no_view_today_leaves_an_open_trade_alone(self):
        ledger = [{"pair": "zinc-lead", "entry_date": "2026-09-01", "side": -1, "status": "open", "entry_z": 2.5}]
        cp.update_rv_ledger(ledger, [_row("zinc-lead", [2.0, 1.0])], {}, NOW)
        self.assertEqual(ledger[0]["status"], "open")


def _rv_row(key, z, side, signal=True, held_back=None, date_="2026-11-20"):
    return {"key": key, "date": date_, "z": -3.7, "relative_value": {
        "z": z, "percentile": 0.98, "ratio": 2.07, "usual": 1.36, "window_months": 120,
        "rich": "Zinc" if z > 0 else "Lead", "cheap": "Lead" if z > 0 else "Zinc", "side": side, "signal": signal,
        "held_back": held_back, "expected": {"per_trade": 0.0626, "prior_trades": 20, "backtest_trades": 14, "live_trades": 0},
        "backtest": {"trades": 14, "won": 0.714, "avg_return": 0.0687}}}


ZINC = {"id": "mcx:ZINCMINI", "label": "Zinc mini (MCX)", "venue": "mcx", "asset_class": "commodity"}
LEAD = {"id": "mcx:LEADMINI", "label": "Lead mini (MCX)", "venue": "mcx", "asset_class": "commodity"}
CRUDE = {"id": "mcx:CRUDEOILM", "label": "Crude mini (MCX)", "venue": "mcx", "asset_class": "commodity"}
GAS = {"id": "mcx:NATGASMINI", "label": "Gas mini (MCX)", "venue": "mcx", "asset_class": "commodity"}
UNIVERSE = [{"key": "zinc-lead", "label": "Zinc / Lead (MCX)", "a": ZINC, "b": LEAD},
            {"key": "crude-natgas", "label": "Crude / Natural gas (MCX)", "a": CRUDE, "b": GAS}]
TRACKER = {"relative_value": {"rule": {"entry_z": 1.5, "exit_z": 0.5, "max_months": 24, "max_sessions": 504},
                              "backtest": cp.RV_RECORD}}


class FundTests(unittest.TestCase):
    def test_a_relative_value_signal_sells_the_dear_leg_and_buys_the_cheap_one(self):
        tracker = {**TRACKER, "pairs": [_rv_row("zinc-lead", 2.95, -1)]}
        signals, skipped = orch.pair_signals(tracker, UNIVERSE)
        self.assertEqual({(s.instrument["id"], s.side) for s in signals}, {("mcx:ZINCMINI", "SHORT"), ("mcx:LEADMINI", "LONG")})
        self.assertEqual(skipped, [])
        facts = {f["label"]: f["value"] for f in signals[0].facts}
        self.assertIn("Zinc dear against Lead: price ratio 2.07 against a 10-year normal of 1.36 (z +2.95, 98th percentile)",
                      facts["relative value"])
        self.assertIn("|z| >= 1.5 against the 10-year normal", facts["entry rule"])
        self.assertIn("on 2005-2026, unseen when the settings were picked: 72 trades, won 64%", facts["relative-value backtest"])
        self.assertEqual(facts["60-session stretch"], "z -3.70 (shown, not traded)")
        self.assertAlmostEqual(signals[0].edge, 0.0626)
        self.assertIn("3.0 standard deviations above its 10-year normal", signals[0].reason)

    def test_the_mcx_crude_gas_pair_reads_the_wti_gas_tracker_row(self):
        tracker = {**TRACKER, "pairs": [_rv_row("wti-natgas", -1.8, 1)]}
        signals, _ = orch.pair_signals(tracker, UNIVERSE)
        self.assertEqual({(s.pair_key, s.instrument["id"], s.side) for s in signals},
                         {("crude-natgas", "mcx:CRUDEOILM", "LONG"), ("crude-natgas", "mcx:NATGASMINI", "SHORT")})

    def test_held_back_or_inside_the_line_trades_nothing(self):
        tracker = {**TRACKER, "pairs": [_rv_row("zinc-lead", 2.95, -1, signal=False, held_back="the cause looks lasting")]}
        signals, skipped = orch.pair_signals(tracker, UNIVERSE)
        self.assertEqual(signals, [])
        self.assertIn("held back: the cause looks lasting", skipped[0]["reason"])
        tracker = {**TRACKER, "pairs": [_rv_row("zinc-lead", 1.2, None, signal=False)]}
        self.assertEqual(orch.pair_signals(tracker, UNIVERSE), ([], []))

    def test_exits_back_near_normal_against_the_view_or_after_the_limit(self):
        rule = TRACKER["relative_value"]["rule"]
        row = _rv_row("copper-aluminium", 1.19, None, signal=False)
        # The November 2026 copper / aluminium trade: long copper while copper is the dear leg.
        why = orch.pair_exit_reason(row, {"sessions": 4, "side": 1}, rv_rule=rule)
        self.assertIn("further from its 10-year normal (z +1.19)", why)
        self.assertIsNone(orch.pair_exit_reason(row, {"sessions": 4, "side": -1}, rv_rule=rule))
        near = _rv_row("zinc-lead", 0.3, None, signal=False)
        self.assertIn("back near its 10-year normal", orch.pair_exit_reason(near, {"sessions": 4, "side": -1}, rv_rule=rule))
        self.assertIn("504-session limit", orch.pair_exit_reason(row, {"sessions": 504, "side": -1}, rv_rule=rule))
        # No long view today: keep the trade.
        self.assertIsNone(orch.pair_exit_reason({"z": 0.1}, {"sessions": 4, "side": 1}, rv_rule=rule))
        # A tracker from before relative value keeps the 60-session band.
        self.assertIn("spread back inside", orch.pair_exit_reason({"z": 0.1}, {"sessions": 4}, rv_rule=None))

    def test_plan_closes_both_legs_of_a_trade_against_the_view(self):
        held = [{"instrument_id": "mcx:COPPER", "side": "LONG", "kind": "linear", "quantity": 1, "mark": 900.0,
                 "pair_key": "copper-aluminium", "attribution": {"pairs_model": 1.0}},
                {"instrument_id": "mcx:ALUMINI", "side": "SHORT", "kind": "linear", "quantity": 1, "mark": 250.0,
                 "pair_key": "copper-aluminium", "attribution": {"pairs_model": 1.0}}]
        snapshot = {"nav_usd": 10_000.0, "cash_usd": 6_000.0, "positions": held, "by_asset_class": {},
                    "net_exposure_by_underlying": {}}
        out = orch.plan(snapshot, [], {}, {}, held_pairs={"copper-aluminium": {"sessions": 4, "side": 1}},
                        pair_rows={"copper-aluminium": _rv_row("copper-aluminium", 1.19, None, signal=False)},
                        rv_rule=TRACKER["relative_value"]["rule"])
        self.assertEqual(sorted(o.instrument["id"] for o in out["orders"]), ["mcx:ALUMINI", "mcx:COPPER"])
        self.assertTrue(all(o.purpose == "exit" for o in out["orders"]))

    def test_an_owner_hold_keeps_the_pair_open(self):
        held = [{"instrument_id": "mcx:COPPER", "side": "LONG", "kind": "linear", "quantity": 1, "mark": 900.0,
                 "pair_key": "copper-aluminium", "attribution": {"pairs_model": 1.0}},
                {"instrument_id": "mcx:ALUMINI", "side": "SHORT", "kind": "linear", "quantity": 1, "mark": 250.0,
                 "pair_key": "copper-aluminium", "attribution": {"pairs_model": 1.0}}]
        snapshot = {"nav_usd": 10_000.0, "cash_usd": 6_000.0, "positions": held, "by_asset_class": {},
                    "net_exposure_by_underlying": {}}
        out = orch.plan(snapshot, [], {}, {}, held_pairs={"copper-aluminium": {"sessions": 4, "side": 1}},
                        pair_rows={"copper-aluminium": _rv_row("copper-aluminium", 1.19, None, signal=False)},
                        rv_rule=TRACKER["relative_value"]["rule"],
                        owner_holds=[{"id": "hold-1", "pair_key": "copper-aluminium"}], entries=False)
        self.assertEqual(out["orders"], [])
        [note] = out["passed"]
        self.assertIn("owner hold hold-1: kept open although this trade needs", note["reason"])

    def test_the_owner_file_holds_copper_aluminium(self):
        actions = json.loads((Path(__file__).resolve().parents[1] / "configs" / "fund_owner_actions.json").read_text())
        self.assertEqual([h["pair_key"] for h in actions["holds"]], ["copper-aluminium"])

    def test_the_cycle_reads_the_held_side_on_the_ratio(self):
        universe = [{"key": "copper-aluminium", "a": {"id": "mcx:COPPER"}, "b": {"id": "mcx:ALUMINI"}}]
        positions = [{"pair_key": "copper-aluminium", "instrument_id": "mcx:ALUMINI", "side": "SHORT",
                      "opened_at": "2026-11-16T08:01:17+00:00"},
                     {"pair_key": "copper-aluminium", "instrument_id": "mcx:COPPER", "side": "LONG",
                      "opened_at": "2026-11-16T08:01:17+00:00"},
                     {"pair_key": "gone-pair", "instrument_id": "mcx:X", "side": "LONG", "opened_at": "2026-11-16T08:01:17+00:00"}]
        held = cycle.held_pair_entries(positions, universe, date(2026, 11, 23))
        self.assertEqual(held["copper-aluminium"]["side"], 1)
        self.assertNotIn("side", held["gone-pair"])


class PayloadTests(unittest.TestCase):
    def test_the_payload_carries_each_pairs_view_and_the_rule_record(self):
        n = 40
        start = date(2026, 9, 1)
        days = [start + timedelta(days=i) for i in range(n)]
        series = {("zinc", "india"): [(d, 300.0 + (i % 3)) for i, d in enumerate(days)],
                  ("lead", "india"): [(d, 150.0) for d in days]}
        long_run = {"pairs": {"zinc-lead": {**_ctx(2.9), "tide": None}}, "driver_record": {}}
        rv_ledger: list = []
        payload, _ = cp.run(series, [], NOW, long_run=long_run, rv_ledger=rv_ledger)
        row = next(p for p in payload["pairs"] if p["key"] == "zinc-lead")
        self.assertEqual((row["relative_value"]["side"], row["relative_value"]["signal"]), (-1, True))
        self.assertEqual(row["relative_value"]["open_trade"]["side"], -1)
        self.assertEqual(payload["relative_value"]["rule"]["entry_z"], cp.RV_ENTRY_Z)
        self.assertEqual(payload["relative_value"]["live"]["open"], 1)
        self.assertEqual(len(rv_ledger), 1)
        payload, _ = cp.run(series, [], NOW)
        self.assertIsNone(payload["relative_value"])


if __name__ == "__main__":
    unittest.main()
