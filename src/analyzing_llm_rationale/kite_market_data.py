"""Private Kite Connect quote snapshots and historical candles in exchange units."""
from __future__ import annotations

import math
import re
from datetime import datetime, time, timedelta, timezone
from typing import Any

from .kite_account import KiteAccountError, request_data

_SYMBOL = re.compile(r"(?:NSE|BSE|NFO|BFO|CDS|BCD|MCX):[A-Z0-9][A-Z0-9 &_.-]{0,99}\Z")
_IST = timezone(timedelta(hours=5, minutes=30))
# Conservative hourly bound; the broker may impose tighter per-request limits.
_MAX_DAYS = {"minute": 60, "3minute": 100, "5minute": 100, "10minute": 100, "15minute": 200, "30minute": 200, "60minute": 365, "day": 2000}


def _symbol(value: str) -> str:
    if not isinstance(value, str) or not _SYMBOL.fullmatch(value.strip(" ").upper()):
        raise KiteAccountError("Use an exchange:symbol identifier, for example NSE:INFY.")
    return value.strip(" ").upper()


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _timestamp(value: str, *, end: bool = False) -> datetime:
    try:
        if not isinstance(value, str) or len(value) > 40:
            raise ValueError
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            stamp = datetime.combine(datetime.fromisoformat(value).date(), time(23, 59, 59) if end else time())
        else:
            stamp = datetime.fromisoformat(re.sub(r"([+-]\d\d)(\d\d)$", r"\1:\2", value.replace("Z", "+00:00")))
        return stamp.replace(tzinfo=_IST) if stamp.tzinfo is None else stamp.astimezone(_IST)
    except (ValueError, TypeError, OverflowError):
        raise KiteAccountError("Use valid ISO dates or datetimes; dates and naive times use IST.") from None


def quotes(credentials: dict[str, str], instruments: list[str]) -> dict[str, Any]:
    if not isinstance(instruments, list) or not 1 <= len(instruments) <= 500:
        raise KiteAccountError("Request between 1 and 500 Kite instruments.")
    instruments = list(dict.fromkeys(_symbol(value) for value in instruments))
    data = request_data(credentials, "/quote", "quotes", params=[("i", symbol) for symbol in instruments])
    result = {}
    for symbol in instruments:
        if symbol not in data:
            continue
        raw = data[symbol]
        if (
            not isinstance(raw, dict) or not _number(raw.get("last_price"))
            or not isinstance(raw.get("instrument_token"), int) or isinstance(raw["instrument_token"], bool)
            or not 0 < raw["instrument_token"] <= 2**32 - 1
        ):
            raise KiteAccountError("Kite returned an invalid quote.", 502)
        ohlc = raw.get("ohlc", {})
        if not isinstance(ohlc, dict) or any(not _number(ohlc[key]) for key in ("open", "high", "low", "close") if key in ohlc):
            raise KiteAccountError("Kite returned invalid quote OHLC values.", 502)
        item = {"instrument_token": raw["instrument_token"], "last_price": raw["last_price"], "ohlc": {key: ohlc[key] for key in ("open", "high", "low", "close") if key in ohlc}}
        for key in ("volume", "oi", "average_price"):
            if key in raw:
                if not _number(raw[key]):
                    raise KiteAccountError("Kite returned invalid quote values.", 502)
                item[key] = raw[key]
        for key in ("timestamp", "last_trade_time"):
            if raw.get(key) is not None:
                try:
                    item[key] = _timestamp(raw[key]).isoformat()
                except KiteAccountError:
                    raise KiteAccountError("Kite returned an invalid quote timestamp.", 502) from None
        result[symbol] = item
    return {"quotes": result, "missing_instruments": [symbol for symbol in instruments if symbol not in result]}


def history(credentials: dict[str, str], instrument: str, interval: str, start: str, end: str) -> dict[str, Any]:
    instrument = _symbol(instrument)
    if interval not in _MAX_DAYS:
        raise KiteAccountError("Unsupported Kite candle interval.")
    first, last = _timestamp(start), _timestamp(end, end=True)
    if last < first or last - first >= timedelta(days=_MAX_DAYS[interval]):
        raise KiteAccountError(f"Choose an ordered date range shorter than {_MAX_DAYS[interval]} days for this interval.")
    snapshot = quotes(credentials, [instrument])["quotes"].get(instrument)
    if snapshot is None:
        raise KiteAccountError("Kite has no current instrument matching that symbol.", 404)
    token = snapshot["instrument_token"]
    data = request_data(credentials, f"/instruments/historical/{token}/{interval}", "history", params={
        "from": first.strftime("%Y-%m-%d %H:%M:%S"), "to": last.strftime("%Y-%m-%d %H:%M:%S"), "oi": 1,
    })
    candles = data.get("candles")
    if not isinstance(candles, list) or len(candles) > 100_000:
        raise KiteAccountError("Kite returned invalid historical candles.", 502)
    previous = None
    normalized = []
    for candle in candles:
        if not isinstance(candle, list) or len(candle) not in (6, 7) or any(not _number(value) for value in candle[1:]):
            raise KiteAccountError("Kite returned invalid OHLCV candles.", 502)
        try:
            stamp = _timestamp(candle[0])
        except KiteAccountError:
            raise KiteAccountError("Kite returned an invalid candle timestamp.", 502) from None
        if stamp < first or stamp > last or (previous is not None and stamp <= previous):
            raise KiteAccountError("Kite returned candles outside the requested range or order.", 502)
        if candle[2] < max(candle[1], candle[3], candle[4]) or candle[3] > min(candle[1], candle[2], candle[4]):
            raise KiteAccountError("Kite returned inconsistent candle prices.", 502)
        previous = stamp
        normalized.append([stamp.isoformat(), *candle[1:]])
    return {"instrument": instrument, "instrument_token": token, "interval": interval, "timezone": "Asia/Kolkata", "from": first.isoformat(), "to": last.isoformat(), "candles": normalized}
