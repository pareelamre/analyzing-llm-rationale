"""When a piece of evidence was published, and how much that should count (owner, 2026-11-11).

The last 30 days carry the most signal for a forecast. Evidence in that window keeps its full
ranking weight; older evidence halves every HALF_LIFE_DAYS past it, down to FLOOR, so background
can still be picked when nothing newer covers the question. Undated evidence sits in between.
The forecast prompt shows the evidence as a timeline in the same bands.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Mapping, Optional, Tuple

RECENT_WINDOW_DAYS = 30
LATEST_DAYS = 7
HALF_LIFE_DAYS = 30
FLOOR = 0.25
UNDATED_WEIGHT = 0.6

BAND_LATEST = f"Last {LATEST_DAYS} days"
BAND_RECENT = f"{LATEST_DAYS + 1} to {RECENT_WINDOW_DAYS} days ago"
BAND_OLDER = f"Older than {RECENT_WINDOW_DAYS} days (background; may be superseded)"
BAND_UNDATED = "Undated"


def parse_publish_date(value: object) -> Optional[datetime]:
    """A UTC datetime from the forms sources use: ISO 8601 (APIs), RFC 822 (RSS pubDate, as in
    "Tue, 10 Nov 2026 14:00:00 GMT") and GDELT's compact "20261110T140000Z"; else None."""
    if isinstance(value, datetime):
        parsed: Optional[datetime] = value
    elif not isinstance(value, str) or not value.strip():
        return None
    else:
        text = value.strip()
        parsed = None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            pass
        if parsed is None:
            compact = re.fullmatch(r"(\d{4})(\d{2})(\d{2})T?(\d{2})(\d{2})(\d{2})Z?", text)
            try:
                parsed = datetime(*map(int, compact.groups()), tzinfo=timezone.utc) if compact else None
            except ValueError:
                parsed = None
        if parsed is None:
            try:
                parsed = parsedate_to_datetime(text)
            except (TypeError, ValueError, IndexError):
                parsed = None
        if parsed is None:
            day = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
            try:
                parsed = datetime(int(day[1]), int(day[2]), int(day[3])) if day else None
            except ValueError:
                parsed = None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december")
_MONTH = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_LINK_DATES = (
    re.compile(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?=[/.?#]|$)"),  # /2026/06/18/
    re.compile(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)"),     # 2026-06-18
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"),       # 20261004
)
_WORD_DATES = (
    (re.compile(rf"\b{_MONTH}\.? (\d{{1,2}}),? (20\d{{2}})\b", re.I), (3, 1, 2)),  # November 9, 2026
    (re.compile(rf"\b(\d{{1,2}}) {_MONTH}\.?,? (20\d{{2}})\b", re.I), (3, 2, 1)),  # 9 November 2026
)


def _day(year: object, month: object, day: object) -> Optional[datetime]:
    if isinstance(month, str) and not month.isdigit():
        month = next(i for i, name in enumerate(_MONTHS, 1) if name.startswith(month.lower()[:3]))
    try:
        return datetime(int(year), int(month), int(day), tzinfo=timezone.utc)
    except ValueError:
        return None


def inferred_date(article: Mapping[str, object], now: Optional[datetime] = None) -> Tuple[Optional[datetime], str]:
    """For evidence with no publish date (web results often have none): the date in its link
    (/2026/06/18/, 2026-06-18, -20261004-) or in its title or link words ("Iran Update:
    November 9, 2026"), and which one it came from. A date in the future does not count."""
    now = now or datetime.now(timezone.utc)
    latest = now + timedelta(days=2)
    url = str(article.get("url") or "")
    path = "/" + url.split("://", 1)[-1].partition("/")[2]  # the link without its host
    title = str(article.get("title") or "")
    for pattern in _LINK_DATES:
        for match in pattern.finditer(path):
            found = _day(*match.groups())
            if found is not None and found <= latest:
                return found, "link"
    for text, where in ((title, "title"), (re.sub(r"[-_/+]", " ", path), "link")):
        for pattern, (y, m, d) in _WORD_DATES:
            for match in pattern.finditer(text):
                found = _day(match.group(y), match.group(m), match.group(d))
                if found is not None and found <= latest:
                    return found, where
    return None, ""


def article_date(article: Mapping[str, object], now: Optional[datetime] = None) -> Tuple[Optional[datetime], str]:
    """When the evidence is from: its publish date, else a date in its link or title."""
    published = parse_publish_date(article.get("publish_date"))
    if published is not None:
        return published, "published"
    return inferred_date(article, now)


def age_days(published: Optional[datetime], now: datetime) -> Optional[float]:
    return None if published is None else (now - published).total_seconds() / 86400.0


def recency_weight(published: Optional[datetime], now: datetime) -> float:
    """1 inside the last RECENT_WINDOW_DAYS, then halving every HALF_LIFE_DAYS down to FLOOR."""
    age = age_days(published, now)
    if age is None:
        return UNDATED_WEIGHT
    if age <= RECENT_WINDOW_DAYS:
        return 1.0
    return max(FLOOR, 0.5 ** ((age - RECENT_WINDOW_DAYS) / HALF_LIFE_DAYS))


def timeline_band(published: Optional[datetime], now: datetime) -> str:
    age = age_days(published, now)
    if age is None:
        return BAND_UNDATED
    if age <= LATEST_DAYS:
        return BAND_LATEST
    if age <= RECENT_WINDOW_DAYS:
        return BAND_RECENT
    return BAND_OLDER
