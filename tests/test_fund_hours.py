from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import fund_hours as fh  # noqa: E402
from analyzing_llm_rationale import fund_options as fo  # noqa: E402
from analyzing_llm_rationale.fund_orchestrator import Signal  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
#: Upstox's row for 2026-11-10 (Diwali Balipratipada): NSE and BSE shut, MCX's evening session only.
BALIPRATIPADA = [{"date": "2026-11-10", "holiday_type": "TRADING_HOLIDAY",
                  "closed_exchanges": ["NSE", "NFO", "CDS", "BSE", "BFO", "BCD"],
                  "open_exchanges": [{"exchange": "MCX", "start_time": 1794310200000, "end_time": 1794335100000}]}]


def at(day: int, hour: int, minute: int = 0, month: int = 11) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=IST)


def signal(iid: str, pair_key=None) -> Signal:
    return Signal(analyst="x", instrument={"id": iid, "label": iid}, side="LONG", price=1.0, edge=0.1,
                  trust=1.0, pair_key=pair_key)


class SessionTests(unittest.TestCase):
    def test_nse_options_trade_only_in_the_weekday_session(self):
        self.assertTrue(fh.is_open("nse_fo", at(16, 13, 30), [])[0])  # Monday
        closed, why = fh.is_open("nse_fo", at(16, 1, 30), [])
        self.assertFalse(closed)
        self.assertEqual(why, "NSE F&O is closed now (it trades 09:15 to 15:30 IST on weekdays)")
        self.assertFalse(fh.is_open("nse_fo", at(14, 13, 30), [])[0])  # Saturday
        self.assertFalse(fh.is_open("nse_fo", at(16, 15, 30), [])[0])  # the close itself

    def test_a_holiday_closes_nse_and_leaves_mcx_its_evening_session(self):
        self.assertFalse(fh.is_open("nse_fo", at(10, 13, 30), BALIPRATIPADA)[0])
        self.assertFalse(fh.is_open("mcx", at(10, 13, 30), BALIPRATIPADA)[0])
        self.assertTrue(fh.is_open("mcx", at(10, 17, 30), BALIPRATIPADA)[0])
        self.assertEqual(fh.session("mcx", at(10, 0).date(), BALIPRATIPADA), (at(10, 17), at(10, 23, 55)))

    def test_mcx_closes_later_while_the_us_is_on_standard_time(self):
        # US daylight time ended on 2026-11-01.
        self.assertFalse(fh.is_open("mcx", at(15, 23, 40, month=10), [])[0])
        self.assertTrue(fh.is_open("mcx", at(16, 23, 40), [])[0])
        self.assertFalse(fh.is_open("mcx", at(16, 23, 56), [])[0])
        self.assertFalse(fh.is_open("mcx", at(16, 8, 59), [])[0])

    def test_without_the_holiday_list_weekdays_count_as_sessions(self):
        self.assertTrue(fh.is_open("nse_fo", at(10, 13, 30), None)[0])

    def test_event_markets_never_close_and_perps_follow_kalshi(self):
        for venue in ("kalshi", "polymarket"):
            self.assertTrue(fh.is_open(venue, at(14, 3), [])[0])
        self.assertTrue(fh.is_open("kalshi_perp", at(14, 3), [], quote={"open": True})[0])
        self.assertTrue(fh.is_open("kalshi_perp", at(14, 3), [])[0])
        self.assertEqual(fh.is_open("kalshi_perp", at(14, 3), [], quote={"open": False}),
                         (False, "Kalshi's perp market is closed now"))

    def test_the_holiday_route_is_read_and_a_failure_is_none(self):
        class Response:
            def __init__(self, body):
                self.body = body

            def raise_for_status(self):
                pass

            def json(self):
                return self.body

        rows = fh.fetch_calendar(at(10, 0).date(), http_get=lambda url, **_: Response(
            {"status": "success", "data": BALIPRATIPADA}))
        self.assertEqual(rows, BALIPRATIPADA)

        def down(url, **_):
            raise fh.requests.ConnectionError("down")
        self.assertIsNone(fh.fetch_calendar(at(10, 0).date(), http_get=down))


class LivePriceTests(unittest.TestCase):
    def test_a_price_counts_only_from_the_session_under_way(self):
        now = at(16, 13, 31)
        self.assertTrue(fh.live({"at": at(16, 13, 30).isoformat()}, "mcx", now, []))
        # Friday's close, which is what bought copper at 1,308.4 on 2026-11-16.
        self.assertFalse(fh.live({"at": at(13, 15, 30).isoformat()}, "mcx", now, []))
        self.assertTrue(fh.live({"source": "book"}, "nse_fo", now, []))
        self.assertFalse(fh.live({"bid": 1.0, "ask": 1.0}, "mcx", now, []))
        self.assertFalse(fh.live(None, "mcx", now, []))
        self.assertTrue(fh.live({"bid": 0.4, "ask": 0.5}, "kalshi", now, []))

    def test_a_spread_is_as_old_as_its_stalest_leg(self):
        legs = [fo.Leg("short", "PE", 100.0, -1), fo.Leg("long", "PE", 90.0, 1)]
        s = fo.Structure("put_spread", "BULL", legs, 10.0)
        q = s.quote({"short": {"bid": 5.0, "ask": 5.2, "at": at(16, 13, 30).isoformat()},
                     "long": {"bid": 2.0, "ask": 2.1, "at": at(16, 10, 0).isoformat()}})
        self.assertEqual(q["at"], at(16, 10, 0).isoformat())
        book = s.quote({"short": {"bid": 5.0, "ask": 5.2, "source": "book"},
                        "long": {"bid": 2.0, "ask": 2.1, "source": "book"}})
        self.assertEqual(book.get("source"), "book")


class GateTests(unittest.TestCase):
    def test_closed_or_stale_venues_wait_and_pairs_move_together(self):
        now = at(16, 13, 31)
        fresh, stale = {"at": at(16, 13, 30).isoformat()}, {"at": at(13, 15, 30).isoformat()}
        quotes = {"mcx:COPPER": fresh, "mcx:ALUMINI": fresh, "mcx:GOLDPETAL": stale, "mcx:SILVERMIC": fresh,
                  "mcx:CRUDEOILM": fresh, "kalshi_perp:KXWTIFUTURE": {"open": False},
                  "polymarket:x": {"bid": 0.4, "ask": 0.5}}
        items = [signal("mcx:COPPER", "cu-al"), signal("mcx:ALUMINI", "cu-al"),
                 signal("mcx:GOLDPETAL", "au-ag"), signal("mcx:SILVERMIC", "au-ag"),
                 signal("mcx:CRUDEOILM", "crude"), signal("kalshi_perp:KXWTIFUTURE", "crude"),
                 signal("polymarket:x")]
        kept, closed = fh.gate(items, quotes, now, [])
        self.assertEqual([s.instrument["id"] for s in kept], ["mcx:COPPER", "mcx:ALUMINI", "polymarket:x"])
        self.assertEqual({c["instrument"]: c["reason"] for c in closed}, {
            "mcx:GOLDPETAL": "no MCX price from this session yet",
            "mcx:SILVERMIC": "no MCX price from this session yet",
            "mcx:CRUDEOILM": "Kalshi's perp market is closed now",
            "kalshi_perp:KXWTIFUTURE": "Kalshi's perp market is closed now",
        })

    def test_at_night_only_the_markets_that_never_close_trade(self):
        now = at(17, 1, 31)
        spread = signal("nse_fo:NIFTY:put_spread:2026-12-08:24350/23600")
        quotes = {spread.instrument["id"]: {"bid": 90.0, "ask": 94.0, "at": at(16, 15, 0).isoformat()},
                  "mcx:COPPER": {"at": at(16, 23, 30).isoformat()}, "kalshi:K": {"bid": 0.2, "ask": 0.3}}
        kept, closed = fh.gate([spread, signal("mcx:COPPER"), signal("kalshi:K")], quotes, now, [])
        self.assertEqual([s.instrument["id"] for s in kept], ["kalshi:K"])
        self.assertEqual(len(closed), 2)


if __name__ == "__main__":
    unittest.main()
