"""Weather market-making test: does quoting inside Kalshi's daily-high books earn the spread?

    python scripts/fund_weather_mm_eval.py [--cache DIR] [--out reports/fund_weather_mm_eval.json]

Pre-registered in docs/fund.md ("Weather market-making test", 2026-11-19, before any price was
read). Every minute whose book has a bid and an ask at least 3 cents apart, a paper bid one cent
above the best bid and a paper ask one cent below the best ask rest for the next minute, 10
contracts each, within a net position of 50 contracts per market. A trade in that minute fills
the paper bid when its taker sold Yes at or below it, and the paper ask when its taker bought Yes
at or above it, at the paper price. Fills are held to settlement. Makers pay no fee on these
series (fee type "quadratic"); a 0.0175 x p x (1 - p) maker fee is reported for description.

Data: Kalshi's public API (settled events with their markets, one-minute event candlesticks,
the trade feed). Nothing here trades or needs a key.
"""
from __future__ import annotations

import argparse
import bisect
import json
import random
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from zoneinfo import ZoneInfo

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import fund_weather as fw  # noqa: E402

FIRST, LAST = date(2026, 9, 24), date(2026, 11, 18)
QUOTE_SIZE, MAX_POSITION = 10, 50
MIN_SPREAD, TICK = 0.03, 0.01
MAKER_FEE_RATE = 0.0175  # description only: these series charge makers nothing
MIN_FILLS = 1000
BOOTSTRAP, SEED = 2000, 20261119
CHUNK_S = 12 * 3600
REQUEST_GAP_S = 0.06
EPS = 1e-9


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------

def _cached(path: Path, fetch: Callable[[], Any]) -> Any:
    if path.is_file():
        return json.loads(path.read_text())
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def _get(session: requests.Session, url: str, params: dict) -> Any:
    for attempt in range(6):
        resp = session.get(url, params=params, timeout=60, headers={"User-Agent": fw.USER_AGENT})
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        resp.raise_for_status()
        time.sleep(REQUEST_GAP_S)
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def _ts(text: str) -> int:
    return int(datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp())


def trades_for(session: requests.Session, ticker: str) -> list[dict]:
    out, cursor = [], None
    while True:
        params = {"ticker": ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        body = _get(session, f"{fw.KALSHI_API}/markets/trades", params)
        out.extend(body.get("trades") or [])
        cursor = body.get("cursor")
        if not cursor or not body.get("trades"):
            return out


def collect(city: fw.City, cache: Path, session: requests.Session) -> list[dict]:
    """Each settled market in the window: result, one-minute book and trades."""
    events = _cached(cache / city.series / "events.json", lambda: fw.events(city.series, "settled", http_get=session.get))
    out = []
    for ev in events:
        ticker = ev.get("event_ticker", "")
        day = fw.event_day(ticker)
        if not day or not FIRST <= day <= LAST:
            continue
        markets = [m for m in ev.get("markets") or [] if m.get("result") in ("yes", "no")]
        if not markets:
            continue
        lo = min(_ts(m["open_time"]) for m in markets)
        hi = max(_ts(m["close_time"]) for m in markets)
        books: dict[str, list] = defaultdict(list)
        for start in range(lo, hi, CHUNK_S):
            end = min(hi, start + CHUNK_S)
            body = _cached(cache / city.series / "candles" / f"{ticker}_{start}.json",
                           lambda start=start, end=end, ticker=ticker: _get(
                               session, f"{fw.KALSHI_API}/series/{city.series}/events/{ticker}/candlesticks",
                               {"start_ts": start, "end_ts": end, "period_interval": 1}))
            for mt, rows in zip(body.get("market_tickers") or [], body.get("market_candlesticks") or []):
                books[mt].extend(rows or [])
        for m in markets:
            trades = _cached(cache / city.series / "trades" / f"{m['ticker']}.json",
                             lambda m=m: trades_for(session, m["ticker"]))
            out.append({"city": city.name, "tz": city.tz, "day": day.isoformat(), "ticker": m["ticker"],
                        "result": 1.0 if m["result"] == "yes" else 0.0,
                        "book": book_from_candles(books.get(m["ticker"]) or []), "trades": trades})
    return out


def book_from_candles(rows: Iterable[dict]) -> dict[int, tuple[Optional[float], Optional[float]]]:
    """Minute end (unix seconds) -> (best yes bid, best yes ask) at that minute's close."""
    book = {}
    for r in rows:
        bid = fw._price((r.get("yes_bid") or {}).get("close_dollars"))
        ask = fw._price((r.get("yes_ask") or {}).get("close_dollars"))
        book[int(r["end_period_ts"])] = (bid, ask)
    return book


# --------------------------------------------------------------------------------------------
# Paper market maker
# --------------------------------------------------------------------------------------------

def _mid_at(book: dict[int, tuple], minutes: list[int], at: int) -> Optional[float]:
    """The mid of the last two-sided book at or before ``at``."""
    i = bisect.bisect_right(minutes, at) - 1
    while i >= 0:
        bid, ask = book[minutes[i]]
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        i -= 1
    return None


def simulate_market(book: dict[int, tuple], trades: Iterable[dict], result: float) -> list[dict]:
    """Every paper fill on one market under the pre-registered rule."""
    flow = sorted(
        ((_ts(t["created_time"]), t) for t in trades if not t.get("is_block_trade")),
        key=lambda x: x[0])
    times = [ts for ts, _ in flow]
    minutes = sorted(book)
    fills, position = [], 0
    for end in minutes:
        bid, ask = book[end]
        if bid is None or ask is None or ask - bid < MIN_SPREAD - EPS:
            continue
        q_bid, q_ask = round(bid + TICK, 2), round(ask - TICK, 2)
        room_buy = min(QUOTE_SIZE, MAX_POSITION - position)
        room_sell = min(QUOTE_SIZE, MAX_POSITION + position)
        i = bisect.bisect_right(times, end)
        while i < len(flow) and flow[i][0] <= end + 60:
            ts, t = flow[i]
            i += 1
            price = fw._price(t.get("yes_price_dollars"))
            size = int(float(t.get("count_fp") or t.get("count") or 0))
            if price is None or size <= 0:
                continue
            side = t.get("taker_side")
            if side == "no" and price <= q_bid + EPS and room_buy > 0:
                q = min(size, room_buy)
                room_buy -= q
                position += q
                fills.append(_fill(+1, q_bid, q, ts, end, book, minutes, result))
            elif side == "yes" and price >= q_ask - EPS and room_sell > 0:
                q = min(size, room_sell)
                room_sell -= q
                position -= q
                fills.append(_fill(-1, q_ask, q, ts, end, book, minutes, result))
    return fills


def _fill(side: int, price: float, qty: int, ts: int, quoted: int, book: dict, minutes: list[int], result: float) -> dict:
    mid = _mid_at(book, minutes, quoted)
    later = {k: _mid_at(book, minutes, ts + 60 * k) for k in (5, 60)}
    return {
        "side": side, "price": price, "qty": qty, "ts": ts,
        "profit": side * (result - price),  # per contract, before fees
        "maker_fee": MAKER_FEE_RATE * price * (1 - price),
        "half_spread": side * (mid - price) if mid is not None else None,
        "markout_5m": side * (later[5] - price) if later[5] is not None else None,
        "markout_60m": side * (later[60] - price) if later[60] is not None else None,
    }


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------

def _bucket(ts: int, tz: str, day: date) -> str:
    local = datetime.fromtimestamp(ts, tz=ZoneInfo(tz))
    if local.date() < day:
        return "before the day"
    if local.hour < 12:
        return "0-12"
    return "12-18" if local.hour < 18 else "18-24"


def per_contract(rows: list[dict], *, fee: bool = False) -> Optional[float]:
    n = sum(r["qty"] for r in rows)
    if not n:
        return None
    return sum(r["qty"] * (r["profit"] - (r["maker_fee"] if fee else 0.0)) for r in rows) / n


def _weighted(rows: list[dict], key: str) -> Optional[float]:
    use = [r for r in rows if r.get(key) is not None]
    n = sum(r["qty"] for r in use)
    return sum(r["qty"] * r[key] for r in use) / n if n else None


def bootstrap(by_day: dict[str, list[dict]], *, fee: bool = False) -> Optional[list[float]]:
    days = sorted(by_day)
    if len(days) < 5:
        return None
    rng = random.Random(SEED)
    stats = []
    for _ in range(BOOTSTRAP):
        rows = [r for d in (rng.choice(days) for _ in days) for r in by_day[d]]
        v = per_contract(rows, fee=fee)
        if v is not None:
            stats.append(v)
    stats.sort()
    return [round(stats[int(0.025 * len(stats))], 5), round(stats[int(0.975 * len(stats)) - 1], 5)]


def summarize(rows: list[dict]) -> dict:
    return {
        "contracts": sum(r["qty"] for r in rows), "fills": len(rows),
        "profit_per_contract": _r(per_contract(rows)), "total_profit": round(sum(r["qty"] * r["profit"] for r in rows), 2),
        "half_spread": _r(_weighted(rows, "half_spread")), "markout_5m": _r(_weighted(rows, "markout_5m")),
        "markout_60m": _r(_weighted(rows, "markout_60m")),
    }


def _r(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(x, 5)


def report(markets: list[dict]) -> dict:
    fills = []
    quoted_markets = 0
    for m in markets:
        f = simulate_market(m["book"], m["trades"], m["result"])
        day = date.fromisoformat(m["day"])
        for row in f:
            row.update({"city": m["city"], "day": m["day"], "bucket": _bucket(row["ts"], m["tz"], day)})
        fills.extend(f)
        quoted_markets += any(b is not None and a is not None and a - b >= MIN_SPREAD - EPS
                              for b, a in m["book"].values())
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in fills:
        by_day[r["day"]].append(r)
    pooled = summarize(fills)
    ci = bootstrap(by_day)
    go = pooled["contracts"] >= MIN_FILLS and ci is not None and ci[0] > 0
    return {
        "test": "Weather market-making test (docs/fund.md, pre-registered 2026-11-19)",
        "window": [FIRST.isoformat(), LAST.isoformat()],
        "markets": len(markets), "markets_quoted": quoted_markets, "days": len({m["day"] for m in markets}),
        "rule": {"quote_size": QUOTE_SIZE, "max_position": MAX_POSITION, "min_spread": MIN_SPREAD, "improve_by": TICK,
                 "maker_fee": 0.0, "maker_fee_description_rate": MAKER_FEE_RATE},
        "pooled": {**pooled, "profit_per_contract_95ci": ci,
                   "with_maker_fee": {"profit_per_contract": _r(per_contract(fills, fee=True)),
                                      "profit_per_contract_95ci": bootstrap(by_day, fee=True)}},
        "verdict": "GO" if go else "NO-GO",
        "by_city": {c: summarize([r for r in fills if r["city"] == c]) for c in sorted({r["city"] for r in fills})},
        "by_local_time": {b: summarize([r for r in fills if r["bucket"] == b])
                          for b in ("before the day", "0-12", "12-18", "18-24")},
        "by_side": {"bought Yes": summarize([r for r in fills if r["side"] > 0]),
                    "sold Yes": summarize([r for r in fills if r["side"] < 0])},
        "sources": "Kalshi public API: settled events with nested markets, one-minute event candlesticks, trade feed",
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=Path(".cache/weather_mm"))
    parser.add_argument("--out", type=Path, default=Path("reports/fund_weather_mm_eval.json"))
    parser.add_argument("--cities", help="comma-separated series, for a dry run")
    args = parser.parse_args(argv)
    session = requests.Session()
    cities = [c for c in fw.CITIES if not args.cities or c.series in args.cities.split(",")]
    markets = []
    for city in cities:
        got = collect(city, args.cache, session)
        print(f"{city.series}: {len(got)} markets", file=sys.stderr, flush=True)
        markets.extend(got)
    out = report(markets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("markets", "markets_quoted", "days", "pooled", "verdict")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
