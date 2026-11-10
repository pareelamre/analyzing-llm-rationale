from __future__ import annotations

import json
import sqlite3
import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import fund_ledger as ledger  # noqa: E402
from analyzing_llm_rationale import fund_pm_live as live  # noqa: E402
from analyzing_llm_rationale import fund_weather as fw  # noqa: E402

NOW = datetime(2026, 11, 10, 8, 0, tzinfo=timezone.utc)  # 16:00 in Manila


def _at(h, m=0):
    return datetime(2026, 11, 10, h, m, tzinfo=timezone.utc)


def _event(event_id="9", ranges=("28°C or below", "29°C", "30°C", "31°C or higher")):
    return {"id": event_id, "title": "Highest temperature in Manila on November 10?", "endDate": "2026-11-10T12:00:00Z",
            "markets": [{"groupItemTitle": t, "clobTokenIds": json.dumps([f"y{i}", f"n{i}"]),
                         "description": "... Ninoy Aquino International Airport (RPLL) ... hourly temperature remark ..."}
                        for i, t in enumerate(ranges)]}


class ReadingTests(unittest.TestCase):
    def test_the_t_group_governs_and_the_body_counts_without_it(self):
        self.assertEqual(live.metar_celsius("METAR KLGA 100151Z 31011G19KT 10SM 12/03 A2998 RMK AO2 T01220028 $"), 12.2)
        self.assertEqual(live.metar_celsius("METAR KORD 100151Z 31011KT 10SM M03/M08 A2998 RMK AO2 T10281078"), -2.8)
        self.assertEqual(live.metar_celsius("METAR RPLL 100200Z 07012KT 9999 FEW025 29/22 Q1015 NOSIG"), 29.0)
        self.assertEqual(live.metar_celsius("METAR UUWW 100200Z 00000MPS CAVOK M11/M17 Q1030"), -11.0)
        self.assertIsNone(live.metar_celsius("METAR XXXX 100200Z NIL"))

    def test_special_reports_do_not_count(self):
        rows = [{"icaoId": "RPLL", "obsTime": int(_at(5).timestamp()), "metarType": "METAR",
                 "rawOb": "METAR RPLL 100500Z 07012KT 9999 31/22 Q1015"},
                {"icaoId": "RPLL", "obsTime": int(_at(5, 20).timestamp()), "metarType": "SPECI",
                 "rawOb": "SPECI RPLL 100520Z 07012KT 9999 33/22 Q1015"}]

        def get(url, params, timeout, headers):
            return SimpleNamespace(json=lambda: rows, raise_for_status=lambda: None, status_code=200)
        self.assertEqual(live.readings(["RPLL"], http_get=get), {"RPLL": [(_at(5), 31.0)]})


class BookTests(unittest.TestCase):
    def test_the_walk_takes_cheap_levels_up_to_the_cap_and_stops_at_99_cents(self):
        fill = live.walk([(0.80, 10), (0.95, 1000), (0.995, 1000)], cap_usd=50)
        self.assertAlmostEqual(fill["cost"] + fill["fee"], 50.0)
        self.assertEqual(fill["best_ask"], 0.80)
        self.assertLess(fill["avg_price"], 0.95)
        self.assertIsNone(live.walk([(0.995, 1000)]))
        self.assertIsNone(live.walk([]))

    def test_the_fee_is_polymarkets_weather_rate(self):
        fill = live.walk([(0.50, 1)], cap_usd=100)
        self.assertAlmostEqual(fill["fee"], 0.05 * 0.5 * 0.5)


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        ledger.ensure_schema(self.conn)
        self.ev = live.parse_open_event(_event())
        self.obs = {"RPLL": [(_at(0), 29.0), (_at(5), 30.4)]}  # 30.4 rounds to 30: "28 or below" and "29" are out

    def test_an_open_event_parses_with_its_station_day_and_tokens(self):
        self.assertEqual((self.ev["station"], self.ev["unit"], self.ev["day"]), ("RPLL", "C", date(2026, 11, 10)))
        self.assertEqual(self.ev["ranges"][1]["no_token"], "n1")

    def test_decided_ranges_are_noted_once_and_filled_when_something_is_for_sale(self):
        books = {"n0": [], "n1": [(0.97, 500)]}
        out = live.record(self.conn, "f", NOW, [self.ev], self.obs, book=lambda t: books.get(t, []))
        self.assertEqual(out, {"decided_new": 2, "filled_now": 1})
        books["n0"] = [(0.98, 500)]
        again = live.record(self.conn, "f", NOW, [self.ev], self.obs, book=lambda t: books.get(t, []))
        self.assertEqual(again, {"decided_new": 0, "filled_now": 1})  # the No on "28 or below" filled later
        third = live.record(self.conn, "f", NOW, [self.ev], self.obs, book=lambda t: books.get(t, []))
        self.assertEqual(third["filled_now"], 0)  # never filled twice
        s = live.summary(self.conn, "f")
        self.assertEqual((s["decided_seen"], s["fills"], s["settled"], s["verdict"]), (2, 2, 0, "pending"))

    def test_fills_are_graded_at_settlement(self):
        live.record(self.conn, "f", NOW, [self.ev], self.obs, book=lambda t: [(0.97, 500)] if t == "n1" else [])
        resolved = _event()
        for i, m in enumerate(resolved["markets"]):
            m["closed"] = True
            m["outcomePrices"] = json.dumps(["1", "0"] if i == 2 else ["0", "1"])  # 30°C won

        def get(url, params, timeout, headers):
            return SimpleNamespace(json=lambda: resolved, raise_for_status=lambda: None, status_code=200)
        self.assertEqual(live.grade(self.conn, "f", NOW, http_get=get), 1)
        s = live.summary(self.conn, "f")
        self.assertEqual((s["settled"], s["losses"]), (1, 0))
        spent = 100.0
        shares = spent / (0.97 + fw.polymarket_fee(0.97))
        self.assertAlmostEqual(s["mean_return"], round((shares - spent) / spent, 5))

    def test_an_unknown_station_is_skipped_rather_than_guessed(self):
        ev = {**self.ev, "station": "ZZZZ"}
        self.assertEqual(live.record(self.conn, "f", NOW, [ev], {"ZZZZ": self.obs["RPLL"]}, book=lambda t: [])["decided_new"], 0)


if __name__ == "__main__":
    unittest.main()
