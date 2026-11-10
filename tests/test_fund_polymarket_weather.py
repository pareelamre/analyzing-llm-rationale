from __future__ import annotations

import importlib.util
import json
import math
import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("fund_polymarket_weather_eval",
                                               _ROOT / "scripts" / "fund_polymarket_weather_eval.py")
pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pm)


class RangeTests(unittest.TestCase):
    def test_every_range_title_reads_as_half_degree_bounds(self):
        self.assertEqual(pm.bounds_of("29°C"), ("C", (28.5, 29.5)))
        self.assertEqual(pm.bounds_of("28°C or below"), ("C", (-math.inf, 28.5)))
        self.assertEqual(pm.bounds_of("38°C or higher"), ("C", (37.5, math.inf)))
        self.assertEqual(pm.bounds_of("-3°C"), ("C", (-3.5, -2.5)))
        self.assertEqual(pm.bounds_of("60-61°F"), ("F", (59.5, 61.5)))
        self.assertEqual(pm.bounds_of("60–61 °F"), ("F", (59.5, 61.5)))
        self.assertEqual(pm.bounds_of("78°F or higher"), ("F", (77.5, math.inf)))
        self.assertIsNone(pm.bounds_of("Other"))

    def test_rounding_follows_the_markets_own_examples(self):
        for c, want in ((13.9, 14), (17.4, 17), (16.5, 17), (-1.7, -2), (-8.5, -8)):
            self.assertEqual(pm.settle_value(c, "C"), want, c)
        for c, want in ((18.3, 65), (17.8, 64), (19.0, 66), (-22.5, -8)):
            self.assertEqual(pm.settle_value(c, "F"), want, c)

    def test_a_range_is_decided_one_whole_degree_past_its_edge(self):
        self.assertIsNone(pm.decided((28.5, 29.5), 29))
        self.assertEqual(pm.decided((28.5, 29.5), 30), "NO")
        self.assertEqual(pm.decided((-math.inf, 28.5), 29), "NO")
        self.assertIsNone(pm.decided((37.5, math.inf), 38))
        self.assertEqual(pm.decided((37.5, math.inf), 39), "YES")


class EventTests(unittest.TestCase):
    def _event(self, rule="hourly temperature remark", winner="29°C"):
        titles = ["28°C or below", "29°C", "30°C", "31°C or higher"]
        return {"id": 1, "title": "Highest temperature in Manila on November 9?", "endDate": "2026-11-09T12:00:00Z",
                "startDate": "2026-11-07T08:00:00Z",
                "markets": [{"groupItemTitle": t, "outcomePrices": json.dumps(["1" if t == winner else "0",
                                                                               "0" if t == winner else "1"]),
                             "clobTokenIds": json.dumps([f"tok{i}", f"no{i}"]),
                             "description": f"... Ninoy Aquino International Airport (RPLL) ... {rule} ..."}
                            for i, t in enumerate(titles)]}

    def test_an_event_on_the_metar_rule_parses_with_its_station_day_and_winner(self):
        ev = pm.parse_event(self._event())
        self.assertEqual((ev["station"], ev["unit"], ev["day"]), ("RPLL", "C", date(2026, 11, 9)))
        self.assertEqual([r["yes"] for r in ev["ranges"]], [False, True, False, False])
        self.assertEqual(ev["ranges"][0]["token"], "tok0")

    def test_events_on_other_rules_or_without_one_winner_are_left_out(self):
        self.assertIsNone(pm.parse_event(self._event(rule="Wunderground daily high")))
        self.assertIsNone(pm.parse_event(self._event(winner="nothing")))

    def test_forecasts_are_always_six_hours_older_than_the_decision_for_the_whole_local_day(self):
        ev = pm.parse_event(self._event())
        manila = "Asia/Manila"  # the 9th ends at 16:00 UTC
        # At 08:00 UTC on the 9th a day-old run is 14 hours older than the day's last hour needs.
        self.assertEqual(pm.lead_for(ev, manila, datetime(2026, 11, 9, 8, 0, tzinfo=timezone.utc)), 1)
        # At 20:00 UTC on the 8th it would be only 4 hours older: two days back.
        self.assertEqual(pm.lead_for(ev, manila, datetime(2026, 11, 8, 20, 0, tzinfo=timezone.utc)), 2)
        self.assertEqual(pm.lead_for(ev, "America/New_York", datetime(2026, 11, 8, 20, 0, tzinfo=timezone.utc)), 2)

    def test_a_price_is_the_last_point_by_the_decision_and_not_too_old(self):
        at = datetime(2026, 11, 8, 20, 0, tzinfo=timezone.utc)
        ts = int(at.timestamp())
        history = [{"t": ts - 7200, "p": 0.3}, {"t": ts - 60, "p": 0.35}, {"t": ts + 60, "p": 0.9}]
        self.assertEqual(pm.price_at(history, at), 0.35)
        self.assertIsNone(pm.price_at(history[:1], datetime(2026, 11, 8, 23, 0, tzinfo=timezone.utc)))

    def test_the_deciding_reading_is_the_first_that_puts_the_range_out_of_reach(self):
        at = lambda h: datetime(2026, 11, 9, h, 0, tzinfo=timezone.utc)  # noqa: E731
        readings = [(at(1), 28.2), (at(3), 29.6), (at(5), 30.4), (at(7), 29.0)]
        # "29°C" is lost once the rounded high reaches 30 (29.6 rounds to 30).
        self.assertEqual(pm.decided_at(readings, (28.5, 29.5), "C", "NO"), at(3))
        self.assertIsNone(pm.decided_at(readings[:1], (28.5, 29.5), "C", "NO"))


if __name__ == "__main__":
    unittest.main()
