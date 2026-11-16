"""When each venue the fund trades is open (docs/fund.md, "Trading hours").

The fund trades a venue only while it is open, and only at a price from that session:

- Kalshi and Polymarket event markets: always open.
- Kalshi perps and futures: their own schedule (Kalshi's margin-markets route says is_open).
- NSE and BSE derivatives: 09:15-15:30 IST on weekdays.
- MCX: 09:00-23:30 IST on weekdays, to 23:55 while New York is on standard time.

Holidays and special sessions come from Upstox's open holiday route (no account), which names
the exchanges a day closes and the hours of any that stay open, such as MCX's evening session
on an NSE holiday. Without that list, weekdays count as regular sessions, and the rule that a
price must come from today's session still keeps the fund off a closed market.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import requests

IST = timezone(timedelta(hours=5, minutes=30))
NEW_YORK = ZoneInfo("America/New_York")
HOLIDAYS_URL = "https://api.upstox.com/v2/market/holidays/{day}"
#: Venues that never close.
ALWAYS = frozenset({"kalshi", "polymarket"})
#: The fund's venue: the exchange code in Upstox's holiday list, and its regular session.
SESSIONS = {"nse_fo": ("NFO", time(9, 15), time(15, 30)), "bse_fo": ("BFO", time(9, 15), time(15, 30)),
            "mcx": ("MCX", time(9, 0), time(23, 30))}
#: MCX trades later while US markets are on standard time.
MCX_WINTER_CLOSE = time(23, 55)
NAMES = {"nse_fo": "NSE F&O", "bse_fo": "BSE F&O", "mcx": "MCX", "kalshi_perp": "Kalshi's perp market"}
HOURS = {"nse_fo": "09:15 to 15:30 IST on weekdays", "bse_fo": "09:15 to 15:30 IST on weekdays",
         "mcx": "09:00 to 23:30 IST on weekdays, 23:55 in the US winter"}

Calendar = Optional[Sequence[Mapping[str, Any]]]


def fetch_calendar(day: date, *, http_get: Callable[..., Any] = requests.get) -> Calendar:
    """Upstox's holiday rows for ``day`` (an empty list on an ordinary day), or None if unreachable."""
    try:
        r = http_get(HOLIDAYS_URL.format(day=day.isoformat()), timeout=20, headers={"Accept": "application/json"})
        r.raise_for_status()
        body = r.json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(body, dict) or body.get("status") != "success":
        return None
    return [row for row in body.get("data") or [] if isinstance(row, dict)]


def session(venue: str, day: date, calendar: Calendar) -> Optional[tuple[datetime, datetime]]:
    """The venue's session on ``day`` (IST), or None if it does not trade that day."""
    exchange, start, end = SESSIONS[venue]
    for row in calendar or []:
        if row.get("date") != day.isoformat():
            continue
        for special in row.get("open_exchanges") or []:
            if special.get("exchange") == exchange and special.get("start_time") and special.get("end_time"):
                return (datetime.fromtimestamp(special["start_time"] / 1000, IST),
                        datetime.fromtimestamp(special["end_time"] / 1000, IST))
        if exchange in (row.get("closed_exchanges") or []):
            return None
    if day.weekday() >= 5:
        return None
    if venue == "mcx" and not datetime.combine(day, time(12), NEW_YORK).dst():
        end = MCX_WINTER_CLOSE
    return datetime.combine(day, start, IST), datetime.combine(day, end, IST)


def is_open(venue: str, now: datetime, calendar: Calendar, *,
            quote: Optional[Mapping[str, Any]] = None) -> tuple[bool, str]:
    """Whether ``venue`` trades at ``now``, and why not. A Kalshi perp's quote carries its own
    open flag; a venue the fund does not know is left to the orchestrator."""
    if venue in ALWAYS:
        return True, ""
    if venue == "kalshi_perp":
        if (quote or {}).get("open") is False:
            return False, f"{NAMES[venue]} is closed now"
        return True, ""
    if venue not in SESSIONS:
        return True, ""
    hours = session(venue, now.astimezone(IST).date(), calendar)
    if hours is None or not hours[0] <= now < hours[1]:
        return False, f"{NAMES[venue]} is closed now (it trades {HOURS[venue]})"
    return True, ""


def live(quote: Optional[Mapping[str, Any]], venue: str, now: datetime, calendar: Calendar) -> bool:
    """Whether a quote's price is from the venue's session now under way. A live order book
    counts; a last trade counts if it is from this session; a quote with no time is trusted
    only on a venue with no session (event markets and Kalshi's perps)."""
    if venue not in SESSIONS:
        return True
    if not quote:
        return False
    if quote.get("source") == "book":
        return True
    try:
        at = datetime.fromisoformat(str(quote.get("at")))
    except ValueError:
        return False
    hours = session(venue, now.astimezone(IST).date(), calendar)
    return hours is not None and hours[0] <= at.astimezone(IST) <= now


def _venue(instrument_id: str) -> str:
    return instrument_id.split(":", 1)[0]


def gate(items: Iterable[Any], quotes: Mapping[str, Mapping[str, Any]], now: datetime,
         calendar: Calendar) -> tuple[list, list[dict]]:
    """Split signals or orders into those whose venue is open with a price from this session,
    and the rest, with why. Both legs of a pair go or stay together."""
    items = list(items)
    why: dict[int, str] = {}
    for n, item in enumerate(items):
        iid = item.instrument["id"]
        venue = _venue(iid)
        ok, reason = is_open(venue, now, calendar, quote=quotes.get(iid))
        if ok and not live(quotes.get(iid), venue, now, calendar):
            ok, reason = False, f"no {NAMES.get(venue, venue)} price from this session yet"
        if not ok:
            why[n] = reason
    blocked_pairs = {items[n].pair_key: reason for n, reason in why.items() if items[n].pair_key}
    kept, closed = [], []
    for n, item in enumerate(items):
        reason = why.get(n) or (blocked_pairs.get(item.pair_key) if item.pair_key else None)
        if reason:
            closed.append({"instrument": item.instrument["id"],
                           "label": item.instrument.get("label") or item.instrument["id"], "reason": reason})
        else:
            kept.append(item)
    return kept, closed
