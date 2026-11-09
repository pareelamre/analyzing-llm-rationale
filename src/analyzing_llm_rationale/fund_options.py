"""Index options for the Foresea Fund (docs/fund.md, "Options").

The options analyst prices NSE index options with the index model's own forecast of where each
index will close (index_forecast) and buys a structure when the model values it above what it
costs by the fund's usual margin:

- a straddle (side MOVE) when the model expects a wider range than the options price in;
- an iron condor (RANGE) when it expects a narrower one;
- a put credit spread (BULL) or a call credit spread (BEAR) when only one side of the range is
  priced too wide.

Every structure has defined risk and is held as one position: it costs its premium (a credit
structure costs its width less the credit) and is worth between nothing and its width at expiry,
so the most the fund can lose is what it paid. It never writes an uncovered option.

Valuation: the model's quantiles for 1, 5 and 21 sessions (with its 1% and 99% ones where it
publishes them), interpolated to the time left to expiry by the square root of time and
extended past the outermost along the last segment, so the tails are never fatter than the
model's own. Thin tails can only undervalue a straddle, never overvalue it, and a credit
structure's wings sit inside the 10%-90% range, so what lies beyond barely moves its value.
The distribution is then moved so its mean is the market's own forward (put-call parity at the
strike nearest the money). The index model has no skill on direction, so the analyst trades
only the width and shape of the range, never a drift.

Prices: with an Upstox Analytics Token (UPSTOX_ANALYTICS_TOKEN: read-only, valid a year) the live
order book; without one, each option's last trade from Upstox's open candle routes, widened by
CLOSE_HALF_SPREAD on each side, and only for options that traded at least MIN_CLOSE_VOLUME_LOTS
lots in that session, so a stale print is never taken for a price.

Index options settle in cash at the index's close on expiry day; the fund books the structure's
value at that close.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib.parse import quote

import requests

from . import fund_orchestrator as orch

IST = timezone(timedelta(hours=5, minutes=30))
UPSTOX_BASE_URL = "https://api.upstox.com"
USER_AGENT = "Foresea/fund (+https://foresea.ink)"
SESSION_OPEN, SESSION_CLOSE = time(9, 15), time(15, 30)
#: The index's close is final a little after the session ends.
SETTLE_AFTER = time(15, 45)

#: NSE index options the index model forecasts: option symbol -> (model's index name, spot key).
INDEX_OPTIONS: dict[str, tuple[str, str]] = {
    "NIFTY": ("Nifty 50", "NSE_INDEX|Nifty 50"),
    "BANKNIFTY": ("Nifty Bank", "NSE_INDEX|Nifty Bank"),
    "FINNIFTY": ("Nifty Financial Services", "NSE_INDEX|Nifty Fin Service"),
    "MIDCPNIFTY": ("Nifty Midcap Select", "NSE_INDEX|NIFTY MID SELECT"),
    "NIFTYNXT50": ("Nifty Next 50", "NSE_INDEX|Nifty Next 50"),
}
#: Sessions to expiry the analyst trades: less leaves no time to be right, more is beyond the
#: model's longest horizon (21 sessions) by more than it can stretch.
MIN_SESSIONS_LEFT, MAX_SESSIONS_LEFT = 2.0, 25.0
#: The model's forecast must come from a recent close.
FORECAST_MAX_AGE_DAYS = 4
#: Last-trade prices stand in for the book only this far apart on each side.
CLOSE_HALF_SPREAD = 0.02
MIN_CLOSE_VOLUME_LOTS = 100
#: A live leg whose book is wider than this share of its mid is too thin to trade.
MAX_LIVE_SPREAD = 0.20
#: When the model's straddle and the market's differ by more than this, the model or the data
#: is more likely wrong than the market (as MAX_DISAGREEMENT is for event contracts).
MAX_VOL_GAP = 0.50
#: Short strikes near the model's quartiles, protective wings near its 10% and 90% levels.
SHORT_QUANTILES, WING_QUANTILES = (0.25, 0.75), (0.10, 0.90)
GRID = 2000
QUOTE_BATCH = 400

HttpGet = Callable[..., Any]
QuoteFn = Callable[[Sequence[str]], dict]


@dataclass
class Leg:
    key: str
    type: str  # CE or PE
    strike: float
    sign: int  # +1 bought, -1 sold
    symbol: str = ""
    lot_size: float = 1.0

    def intrinsic(self, level: float) -> float:
        return max(level - self.strike, 0.0) if self.type == "CE" else max(self.strike - level, 0.0)


@dataclass
class Structure:
    kind: str  # straddle, condor, put_spread, call_spread
    side: str
    legs: list[Leg]
    const: float = 0.0  # a credit structure's width, so its value never goes below zero
    meta: dict = field(default_factory=dict)

    def payoff(self, level: float) -> float:
        return self.const + sum(leg.sign * leg.intrinsic(level) for leg in self.legs)

    def quote(self, quotes: Mapping[str, Mapping[str, Any]]) -> Optional[dict]:
        """What the structure costs (ask) and sells for (bid): each leg bought at its ask and
        sold at its bid."""
        bid = ask = self.const
        for leg in self.legs:
            q = quotes.get(leg.key) or {}
            if q.get("bid") is None or q.get("ask") is None:
                return None
            bid += leg.sign * (q["bid"] if leg.sign > 0 else q["ask"])
            ask += leg.sign * (q["ask"] if leg.sign > 0 else q["bid"])
        return {"bid": max(bid, 0.0), "ask": ask}


SIDES = {"straddle": "MOVE", "condor": "RANGE", "put_spread": "BULL", "call_spread": "BEAR"}


def _num(x: float) -> str:
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"


def _rupees(x: float) -> str:
    return f"₹{x:,.2f}"


def _expiry_date(ms: Any) -> Optional[date]:
    try:
        return datetime.fromtimestamp(float(ms) / 1000, IST).date()
    except (TypeError, ValueError, OverflowError):
        return None


def option_contracts(rows: Iterable[Mapping[str, Any]], symbol: str, today: date) -> dict[date, list[dict]]:
    """An Upstox instrument list's live options on ``symbol``, by expiry."""
    out: dict[date, list[dict]] = {}
    for r in rows:
        if r.get("segment") != "NSE_FO" or r.get("underlying_symbol") != symbol or r.get("instrument_type") not in ("CE", "PE"):
            continue
        expiry = _expiry_date(r.get("expiry"))
        if expiry is None or expiry < today or r.get("strike_price") is None:
            continue
        out.setdefault(expiry, []).append(dict(r))
    return out


def sessions_left(at: datetime, expiry: date, holidays: Iterable[date]) -> float:
    """Sessions from ``at`` to the close on ``expiry``, counting what is left of today's."""
    off = set(holidays)
    local = at.astimezone(IST)
    day, now = local.date(), local.time()

    def session(d: date) -> bool:
        return d.weekday() < 5 and d not in off

    total = 0.0
    if day <= expiry and session(day):
        if now < SESSION_OPEN:
            total += 1.0
        elif now < SESSION_CLOSE:
            whole = datetime.combine(day, SESSION_CLOSE) - datetime.combine(day, SESSION_OPEN)
            total += (datetime.combine(day, SESSION_CLOSE) - datetime.combine(day, now)) / whole
    d = day + timedelta(days=1)
    while d <= expiry:
        total += session(d)
        d += timedelta(days=1)
    return total


def _by_root_time(table: Sequence[tuple[float, Sequence[float]]], tau: float) -> list[float]:
    """Values given at a few horizons, at ``tau`` sessions: interpolated (and beyond the ends,
    scaled) by the square root of time."""
    table = sorted(table)
    first, last = table[0], table[-1]
    if tau <= first[0]:
        return [r * math.sqrt(tau / first[0]) for r in first[1]]
    if tau >= last[0]:
        return [r * math.sqrt(tau / last[0]) for r in last[1]]
    for (h1, r1), (h2, r2) in zip(table, table[1:]):
        if h1 <= tau <= h2:
            w = (math.sqrt(tau) - math.sqrt(h1)) / (math.sqrt(h2) - math.sqrt(h1))
            return [a + w * (b - a) for a, b in zip(r1, r2)]
    raise ValueError("horizons out of order")


def return_quantiles(horizons: Sequence[Mapping[str, Any]], close: float, tau: float,
                     levels_p: Sequence[float]) -> tuple[list[float], list[float]]:
    """Probability levels and log-return quantiles for ``tau`` sessions from the model's 1-, 5-
    and 21-session ones, with its 1% and 99% ones when every horizon has them."""
    tails = [h.get("tail_quantiles") or {} for h in horizons]
    extra = sorted({float(p) for t in tails for p in t}) if all(tails) else []
    levels = sorted([*levels_p, *extra])

    def row(h: Mapping[str, Any]) -> list[float]:
        by_p = {**dict(zip(map(float, levels_p), h["quantiles"])),
                **{float(p): q for p, q in (h.get("tail_quantiles") or {}).items()}}
        return [math.log(float(by_p[p]) / close) for p in levels]

    return levels, _by_root_time([(float(h["h"]), row(h)) for h in horizons], tau)


_Z_GRID: list[float] = []


def _z_grid() -> list[float]:
    if not _Z_GRID:
        unit = NormalDist()
        _Z_GRID.extend(unit.inv_cdf((k + 0.5) / GRID) for k in range(GRID))
    return _Z_GRID


def _interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Piecewise-linear through (xs, ys), extended past the ends along the end segments."""
    if x <= xs[0]:
        i = 0
    elif x >= xs[-1]:
        i = len(xs) - 2
    else:
        i = next(j for j in range(len(xs) - 1) if xs[j] <= x <= xs[j + 1])
    return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])


class Distribution:
    """The model's closing levels at expiry on an equal-probability grid, with mean ``forward``."""

    def __init__(self, levels_p: Sequence[float], returns: Sequence[float], spot: float, forward: float):
        knots = [NormalDist().inv_cdf(p) for p in levels_p]
        grid = [_interp(z, knots, returns) for z in _z_grid()]
        mean = sum(math.exp(r) for r in grid) / len(grid)
        shift = math.log(forward / spot) - math.log(mean)
        self.levels = sorted(spot * math.exp(r + shift) for r in grid)

    def expect(self, payoff: Callable[[float], float]) -> float:
        return sum(payoff(s) for s in self.levels) / len(self.levels)

    def quantile(self, p: float) -> float:
        return self.levels[min(len(self.levels) - 1, max(0, int(p * len(self.levels))))]


# Quotes ---------------------------------------------------------------------------------------

def _get(url: str, *, http_get: HttpGet, token: Optional[str] = None) -> Any:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = http_get(url, timeout=20, headers=headers)
    resp.raise_for_status()
    return resp.json()


def book_quotes(keys: Sequence[str], token: str, *, now: datetime, http_get: HttpGet = requests.get) -> dict[str, dict]:
    """Best bid and ask for each key from Upstox's full market quote (needs the token)."""
    out: dict[str, dict] = {}
    keys = list(dict.fromkeys(keys))
    for i in range(0, len(keys), QUOTE_BATCH):
        chunk = ",".join(quote(k, safe="") for k in keys[i:i + QUOTE_BATCH])
        body = _get(f"{UPSTOX_BASE_URL}/v2/market-quote/quotes?instrument_key={chunk}", http_get=http_get, token=token)
        for item in (body.get("data") or {}).values():
            depth = item.get("depth") or {}
            buy, sell = (depth.get("buy") or [{}])[0], (depth.get("sell") or [{}])[0]
            bid, ask = float(buy.get("price") or 0.0), float(sell.get("price") or 0.0)
            out[item.get("instrument_token")] = {
                "bid": bid if bid > 0 else None, "ask": ask if ask > 0 else None,
                "last": float(item.get("last_price") or 0.0) or None, "volume": float(item.get("volume") or 0.0),
                "at": now.isoformat(), "source": "book"}
    return out


def _latest_candle(key: str, now: datetime, *, http_get: HttpGet) -> Optional[tuple[datetime, float, float]]:
    """The newest candle Upstox's open routes have for ``key``: this session's 30-minute bars
    while it trades, else the last daily bar. Returns (time it closed, close, session volume)."""
    k = quote(key, safe="")
    try:
        bars = _get(f"{UPSTOX_BASE_URL}/v3/historical-candle/intraday/{k}/minutes/30", http_get=http_get)
        bars = (bars.get("data") or {}).get("candles") or []
    except (requests.RequestException, ValueError):
        bars = []
    if bars:
        newest = max(bars, key=lambda c: c[0])
        at = datetime.fromisoformat(newest[0]) + timedelta(minutes=30)
        return min(at, now), float(newest[4]), sum(float(c[5] or 0.0) for c in bars)
    today = now.astimezone(IST).date()
    path = f"/v3/historical-candle/{k}/days/1/{today.isoformat()}/{(today - timedelta(days=7)).isoformat()}"
    days = (_get(f"{UPSTOX_BASE_URL}{path}", http_get=http_get).get("data") or {}).get("candles") or []
    if not days:
        return None
    newest = max(days, key=lambda c: c[0])
    closed = datetime.combine(datetime.fromisoformat(newest[0]).date(), SESSION_CLOSE, IST)
    return closed, float(newest[4]), float(newest[5] or 0.0)


def candle_quotes(keys: Sequence[str], *, now: datetime, http_get: HttpGet = requests.get) -> dict[str, dict]:
    """Last-trade quotes with no account: each price widened by CLOSE_HALF_SPREAD each side."""
    out: dict[str, dict] = {}
    for key in dict.fromkeys(keys):
        try:
            candle = _latest_candle(key, now, http_get=http_get)
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError):
            continue
        if candle is None or candle[1] <= 0:
            continue
        at, last, volume = candle
        out[key] = {"bid": last * (1 - CLOSE_HALF_SPREAD), "ask": last * (1 + CLOSE_HALF_SPREAD), "last": last,
                    "volume": volume, "at": at.isoformat(), "source": "last trade"}
    return out


def quote_source(token: Optional[str], *, now: datetime, http_get: HttpGet = requests.get) -> tuple[QuoteFn, str]:
    """The quote function the fund uses, and how its prices should be described. A rejected
    token falls back to last trades rather than leaving positions unpriced."""
    def last_trades(keys: Sequence[str]) -> dict:
        return candle_quotes(keys, now=now, http_get=http_get)

    if not token:
        return last_trades, f"last trades (Upstox open data), {CLOSE_HALF_SPREAD:.0%} taken off each side"

    def book(keys: Sequence[str]) -> dict:
        try:
            return book_quotes(keys, token, now=now, http_get=http_get)
        except (requests.RequestException, ValueError):
            return last_trades(keys)
    return book, "live order book (Upstox)"


def _tradeable(q: Optional[Mapping[str, Any]], lot_size: float) -> bool:
    if not q or q.get("bid") is None or q.get("ask") is None or q["ask"] <= 0:
        return False
    if q.get("source") == "book":
        mid = (q["bid"] + q["ask"]) / 2
        return q["bid"] > 0 and (q["ask"] - q["bid"]) <= MAX_LIVE_SPREAD * mid
    return float(q.get("volume") or 0.0) >= MIN_CLOSE_VOLUME_LOTS * lot_size


# Structures and the analyst ------------------------------------------------------------------

def _leg(contract: Mapping[str, Any], sign: int) -> Leg:
    return Leg(key=contract["instrument_key"], type=contract["instrument_type"], strike=float(contract["strike_price"]),
               sign=sign, symbol=str(contract.get("trading_symbol") or ""), lot_size=float(contract.get("lot_size") or 1))


def _nearest(contracts: Mapping[float, Mapping[str, Any]], level: float, *, below: Optional[float] = None,
             above: Optional[float] = None) -> Optional[Mapping[str, Any]]:
    strikes = [k for k in contracts if (below is None or k < below) and (above is None or k > above)]
    return contracts[min(strikes, key=lambda k: abs(k - level))] if strikes else None


def candidate_structures(contracts: Sequence[Mapping[str, Any]], forward: float, dist: Distribution) -> list[Structure]:
    """The straddle at the money, and credit spreads and a condor with short strikes near the
    model's quartiles and wings near its 10% and 90% levels."""
    calls = {float(c["strike_price"]): c for c in contracts if c["instrument_type"] == "CE"}
    puts = {float(c["strike_price"]): c for c in contracts if c["instrument_type"] == "PE"}
    both = sorted(set(calls) & set(puts), key=lambda k: abs(k - forward))
    if not both:
        return []
    atm = both[0]
    out = [Structure("straddle", SIDES["straddle"], [_leg(calls[atm], 1), _leg(puts[atm], 1)], meta={"atm": atm})]
    put_short = _nearest(puts, dist.quantile(SHORT_QUANTILES[0]), below=atm + 1e-9)
    put_wing = put_short and _nearest(puts, dist.quantile(WING_QUANTILES[0]), below=float(put_short["strike_price"]))
    call_short = _nearest(calls, dist.quantile(SHORT_QUANTILES[1]), above=atm - 1e-9)
    call_wing = call_short and _nearest(calls, dist.quantile(WING_QUANTILES[1]), above=float(call_short["strike_price"]))
    if put_short and put_wing:
        width = float(put_short["strike_price"]) - float(put_wing["strike_price"])
        out.append(Structure("put_spread", SIDES["put_spread"], [_leg(put_short, -1), _leg(put_wing, 1)], const=width))
    if call_short and call_wing:
        width = float(call_wing["strike_price"]) - float(call_short["strike_price"])
        out.append(Structure("call_spread", SIDES["call_spread"], [_leg(call_short, -1), _leg(call_wing, 1)], const=width))
    if put_short and put_wing and call_short and call_wing:
        width = max(out[-2].const, out[-1].const)
        out.append(Structure("condor", SIDES["condor"], out[-2].legs + out[-1].legs, const=width))
    return out


def _label(name: str, s: Structure, expiry: date) -> str:
    puts = sorted(leg.strike for leg in s.legs if leg.type == "PE")
    calls = sorted(leg.strike for leg in s.legs if leg.type == "CE")
    when = f"{expiry.day} {expiry:%b %Y}"
    if s.kind == "straddle":
        return f"{name} straddle at {_num(calls[0])}, {when}"
    if s.kind == "put_spread":
        return f"{name} put credit spread {'/'.join(map(_num, puts))}, {when}"
    if s.kind == "call_spread":
        return f"{name} call credit spread {'/'.join(map(_num, calls))}, {when}"
    return f"{name} iron condor {'/'.join(map(_num, puts))} puts and {'/'.join(map(_num, calls))} calls, {when}"


def _legs_text(s: Structure) -> str:
    kind = {"CE": "call", "PE": "put"}
    return "; ".join(f"{'buy' if leg.sign > 0 else 'sell'} {_num(leg.strike)} {kind[leg.type]}" for leg in s.legs)


def instrument(symbol: str, s: Structure, expiry: date) -> dict:
    """The fund's instrument for a structure; its spec is kept with the position (fund_ledger)."""
    name, spot_key = INDEX_OPTIONS[symbol]
    strikes = "/".join(_num(leg.strike).replace(",", "") for leg in s.legs)
    return {"id": f"nse_fo:{symbol}:{s.kind}:{expiry.isoformat()}:{strikes}", "label": _label(name, s, expiry),
            "venue": "nse_fo", "asset_class": "option", "underlying": "india_equity", "currency": "INR",
            "multiplier": 1.0, "kind": "option", "structure": s.kind, "symbol": symbol, "index": name,
            "spot_key": spot_key, "expiry": expiry.isoformat(), "const": s.const,
            "legs": [vars(leg) for leg in s.legs]}


def structure_of(spec: Mapping[str, Any]) -> Structure:
    return Structure(spec["structure"], SIDES[spec["structure"]], [Leg(**leg) for leg in spec["legs"]],
                     const=float(spec.get("const") or 0.0))


def structure_quotes(specs: Mapping[str, Mapping[str, Any]], quote_fn: QuoteFn) -> dict[str, dict]:
    """Bid and ask for each held structure from its legs' quotes."""
    legs = [leg["key"] for spec in specs.values() for leg in spec["legs"]]
    raw = quote_fn(legs) if legs else {}
    out = {}
    for iid, spec in specs.items():
        q = structure_of(spec).quote(raw)
        if q is not None:
            out[iid] = q
    return out


def _forward(contracts: Sequence[Mapping[str, Any]], quotes: Mapping[str, Mapping[str, Any]], spot: float) -> Optional[tuple[float, float]]:
    """The market's forward from put-call parity at the quoted strike nearest the money."""
    calls = {float(c["strike_price"]): c["instrument_key"] for c in contracts if c["instrument_type"] == "CE"}
    puts = {float(c["strike_price"]): c["instrument_key"] for c in contracts if c["instrument_type"] == "PE"}
    for k in sorted(set(calls) & set(puts), key=lambda k: abs(k - spot)):
        c, p = quotes.get(calls[k]), quotes.get(puts[k])
        if c and p and c.get("bid") is not None and p.get("bid") is not None:
            mid_c, mid_p = (c["bid"] + c["ask"]) / 2, (p["bid"] + p["ask"]) / 2
            return k + mid_c - mid_p, k
    return None


def _wanted(contracts: Sequence[Mapping[str, Any]], spot: float, rough: Distribution) -> list[Mapping[str, Any]]:
    """The few options a cycle needs quoted: calls and puts at the two strikes nearest the
    money (for the forward and the straddle), puts near the model's lower short and wing
    levels and calls near its upper ones."""
    calls = {float(c["strike_price"]): c for c in contracts if c["instrument_type"] == "CE"}
    puts = {float(c["strike_price"]): c for c in contracts if c["instrument_type"] == "PE"}
    picked: dict[str, Mapping[str, Any]] = {}
    for k in sorted(set(calls) & set(puts), key=lambda k: abs(k - spot))[:2]:
        picked[calls[k]["instrument_key"]], picked[puts[k]["instrument_key"]] = calls[k], puts[k]
    for p, side in ((WING_QUANTILES[0], puts), (SHORT_QUANTILES[0], puts), (SHORT_QUANTILES[1], calls),
                    (WING_QUANTILES[1], calls)):
        c = _nearest(side, rough.quantile(p))
        if c is not None:
            picked[c["instrument_key"]] = c
    return list(picked.values())


def option_signals(forecast: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], now: datetime, *, quote_fn: QuoteFn,
                   price_source: str, held: Mapping[str, Mapping[str, Any]] = (), trust: float = orch.DEFAULT_TRUST,
                   analyst: str = "options_model") -> dict[str, Any]:
    """The options analyst's cycle: a signal per index whose best structure beats its cost by
    MIN_EDGE after costs, the model's value of each held structure (for exits), quotes for
    both, and the indices skipped, with why."""
    by_name = {f["name"]: f for f in forecast.get("indices") or []}
    holidays = [date.fromisoformat(d) for d in forecast.get("holidays") or []]
    levels_p = [float(p) for p in (forecast.get("model") or {}).get("quantiles") or (0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95)]
    held = dict(held)
    signals: list[orch.Signal] = []
    skipped: list[dict] = []
    values: dict[str, float] = {}
    quotes: dict[str, dict] = {}
    today = now.astimezone(IST).date()
    for symbol, (name, spot_key) in INDEX_OPTIONS.items():
        fc = by_name.get(name)
        if not fc or not fc.get("horizons"):
            continue
        as_of = date.fromisoformat(fc["as_of"])
        if (today - as_of).days > FORECAST_MAX_AGE_DAYS:
            skipped.append({"instrument": f"nse_fo:{symbol}", "label": f"{name} options",
                            "reason": f"the index model's latest forecast is from the {as_of} close"})
            continue
        chains = option_contracts(rows, symbol, today)
        expiries = sorted(chains)
        held_expiries = {date.fromisoformat(s["expiry"]) for s in held.values() if s.get("symbol") == symbol}
        spot_q = quote_fn([spot_key]).get(spot_key) or {}
        spot = spot_q.get("last")
        if not spot:
            continue
        priced_at = datetime.fromisoformat(spot_q["at"]) if spot_q.get("at") else now
        trade_expiry = next((e for e in expiries
                             if MIN_SESSIONS_LEFT <= sessions_left(priced_at, e, holidays) <= MAX_SESSIONS_LEFT), None)
        for expiry in sorted({e for e in (trade_expiry, *held_expiries) if e is not None and e in chains}):
            tau = sessions_left(priced_at, expiry, holidays)
            if tau <= 0:
                continue
            levels, returns = return_quantiles(fc["horizons"], float(fc["close"]), tau, levels_p)
            rough = Distribution(levels, returns, spot, spot)
            wanted = _wanted(chains[expiry], spot, rough)
            raw = quote_fn([c["instrument_key"] for c in wanted])
            fwd = _forward(wanted, raw, spot)
            if fwd is None:
                skipped.append({"instrument": f"nse_fo:{symbol}", "label": f"{name} options",
                                "reason": f"no quote for the {expiry} options near the money"})
                continue
            forward, parity_strike = fwd
            dist = Distribution(levels, returns, spot, forward)
            for iid, spec in held.items():
                if spec.get("symbol") == symbol and spec.get("expiry") == expiry.isoformat():
                    s = structure_of(spec)
                    values[iid] = dist.expect(s.payoff)
                    q = s.quote(raw)
                    if q is not None:
                        quotes[iid] = q
            if expiry != trade_expiry:
                continue
            candidates = candidate_structures(wanted, forward, dist)
            straddle = candidates[0] if candidates else None
            market = straddle.quote(raw) if straddle else None
            if not market:
                continue
            model_straddle = dist.expect(straddle.payoff)
            mid = (market["bid"] + market["ask"]) / 2
            if mid <= 0 or abs(model_straddle / mid - 1) > MAX_VOL_GAP:
                skipped.append({"instrument": f"nse_fo:{symbol}", "label": f"{name} options",
                                "reason": f"the model's straddle ({_rupees(model_straddle)}) is too far from the "
                                          f"market's ({_rupees(mid)}); treated as a likely model or data error"})
                continue
            best = None
            for s in candidates:
                if not all(_tradeable(raw.get(leg.key), leg.lot_size) for leg in s.legs):
                    continue
                q = s.quote(raw)
                if q is None or q["ask"] <= 0:
                    continue
                worth = dist.expect(s.payoff)
                edge = (worth - q["ask"]) / q["ask"] - orch.OPTION_COST
                if best is None or edge > best[0]:
                    best = (edge, s, q, worth)
            if best is None or best[0] < orch.MIN_EDGE:
                continue
            edge, s, q, worth = best
            inst = instrument(symbol, s, expiry)
            quotes[inst["id"]] = q
            source = f"index model, {as_of} close"
            signals.append(orch.Signal(
                analyst=analyst, instrument=inst, side=s.side, price=q["ask"], edge=edge, trust=trust,
                facts=[orch._fact("index", f"{name} at {_num(spot)}", f"Upstox, {spot_q.get('source', 'quote')}"),
                       orch._fact("expiry", f"{expiry.day} {expiry:%b %Y}, {tau:.1f} sessions left", "NSE"),
                       orch._fact("structure", _legs_text(s), "options analyst"),
                       orch._fact("cost, the most it can lose", f"{_rupees(q['ask'])} per unit", price_source),
                       orch._fact("model value at expiry", f"{_rupees(worth)} per unit", source),
                       orch._fact("model range at expiry, 10% to 90%",
                                  f"{_num(round(dist.quantile(0.10)))} to {_num(round(dist.quantile(0.90)))}", source),
                       orch._fact("market forward", f"{_num(round(forward, 1))}",
                                  f"put-call parity at {_num(parity_strike)}"),
                       orch._fact("expected profit per rupee at risk after costs", orch._pct(edge), "orchestrator")]))
    return {"signals": signals, "skipped": skipped, "values": values, "quotes": quotes}


def settlement_level(spec: Mapping[str, Any], *, http_get: HttpGet = requests.get) -> Optional[float]:
    """The index's close on the structure's expiry day, once Upstox has it."""
    expiry = spec["expiry"]
    path = f"/v3/historical-candle/{quote(spec['spot_key'], safe='')}/days/1/{expiry}/{expiry}"
    try:
        candles = (_get(f"{UPSTOX_BASE_URL}{path}", http_get=http_get).get("data") or {}).get("candles") or []
    except (requests.RequestException, ValueError):
        return None
    for c in candles:
        if str(c[0])[:10] == expiry:
            return float(c[4])
    return None


def expired(spec: Mapping[str, Any], now: datetime) -> bool:
    local = now.astimezone(IST)
    expiry = date.fromisoformat(spec["expiry"])
    return expiry < local.date() or (expiry == local.date() and local.time() >= SETTLE_AFTER)
