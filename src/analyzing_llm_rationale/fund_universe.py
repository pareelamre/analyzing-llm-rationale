"""The Foresea Fund's tradable universe: the indices and commodity pairs with the most volume.

docs/fund.md asks for the 10 indices and 10 commodity pairs that trade the most, across
venues. Volume is each instrument's average daily notional in USD over the last two weeks:

- Indian index futures (NSE, BSE) and MCX commodity futures: Upstox's open daily candles,
  volume (lots) x close x the contract's quantity multiplier, summed over the two nearest
  expiries, converted at the day's USD/INR.
- Kalshi perpetuals and futures: the public margin market list's 24-hour notional.

Sector indices without futures (Nifty IT, Nifty Auto, ...) are not tradable and so not in the
fund's universe, though the Indices page still forecasts them. A pair ranks by its thinner leg.
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional, Sequence
from urllib.parse import quote

import requests

HttpGet = Callable[..., Any]

IST = timezone(timedelta(hours=5, minutes=30))
UPSTOX_BASE_URL = "https://api.upstox.com"
UPSTOX_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/{exchange}.json.gz"
KALSHI_MARGIN_MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/margin/markets"
USER_AGENT = "Foresea/fund-universe"
#: Kalshi reports a side with no resting order as this many dollars.
KALSHI_NO_ORDER = 1e12
#: Upstox daily candles averaged over this many calendar days (about ten sessions).
VOLUME_WINDOW_DAYS = 16
#: Expiries per underlying counted toward its volume; later ones barely trade.
VOLUME_EXPIRIES = 2
UNIVERSE_SIZE = 10

#: The economic exposure an instrument carries, which the orchestrator nets across venues.
MCX_UNDERLYING = {
    "GOLD": "gold", "GOLDM": "gold", "GOLDTEN": "gold", "GOLDPETAL": "gold", "GOLDGUINEA": "gold",
    "SILVER": "silver", "SILVERM": "silver", "SILVERMIC": "silver", "SILVER100": "silver",
    "CRUDEOIL": "crude", "CRUDEOILM": "crude", "NATURALGAS": "natgas", "NATGASMINI": "natgas",
    "COPPER": "copper", "ZINC": "zinc", "ZINCMINI": "zinc", "ALUMINIUM": "aluminium", "ALUMINI": "aluminium",
    "LEAD": "lead", "LEADMINI": "lead", "NICKEL": "nickel",
}
#: The MCX contract the fund trades for each underlying: the smallest liquid lot, so a $10,000
#: account can hold one at the 10% position limit without leaning on fractional contracts.
MCX_TRADED = {
    "gold": "GOLDPETAL", "silver": "SILVERMIC", "crude": "CRUDEOILM", "natgas": "NATGASMINI",
    "copper": "COPPER", "zinc": "ZINCMINI", "aluminium": "ALUMINI", "lead": "LEADMINI", "nickel": "NICKEL",
}
KALSHI_UNDERLYING = {
    "KXGOLDPERP": "gold", "KXSILVERPERP": "silver", "KXCOPPERPERP": "copper", "KXALUMINUMPERP": "aluminium",
    "KXPLATINUMPERP": "platinum", "KXPALLADIUMPERP": "palladium", "KXWTIFUTURE": "crude",
    "KXBRENTFUTURE": "brent", "KXHHGASFUTURE": "natgas",
    "KXUS500PERP": "us_equity", "KXUSTECH100PERP": "us_equity", "KXUS2000PERP": "us_equity",
}
INDEX_LABELS = {
    "NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank", "FINNIFTY": "Nifty Financial Services",
    "MIDCPNIFTY": "Nifty Midcap Select", "NIFTYNXT50": "Nifty Next 50", "NIFTYFPI": "Nifty FPI 150",
    "SENSEX": "Sensex", "BANKEX": "Bankex", "SENSEX50": "Sensex 50", "FOCIT": "BSE Focused IT",
    "KXUS500PERP": "US 500 (Kalshi perp)", "KXUSTECH100PERP": "US Tech 100 (Kalshi perp)",
    "KXUS2000PERP": "US 2000 (Kalshi perp)",
}

#: Candidate pairs. cross_venue: the same underlying on MCX and Kalshi, where the gap is
#: rupee, duty and timing; spread: two underlyings whose ratio is watched. Legs name an
#: underlying and a venue.
PAIR_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"key": "gold-mcx-kalshi", "label": "Gold: MCX / Kalshi perp", "kind": "cross_venue",
     "a": ("gold", "mcx"), "b": ("gold", "kalshi_perp")},
    {"key": "silver-mcx-kalshi", "label": "Silver: MCX / Kalshi perp", "kind": "cross_venue",
     "a": ("silver", "mcx"), "b": ("silver", "kalshi_perp")},
    {"key": "crude-mcx-kalshi", "label": "Crude: MCX / Kalshi WTI future", "kind": "cross_venue",
     "a": ("crude", "mcx"), "b": ("crude", "kalshi_perp")},
    {"key": "copper-mcx-kalshi", "label": "Copper: MCX / Kalshi perp", "kind": "cross_venue",
     "a": ("copper", "mcx"), "b": ("copper", "kalshi_perp")},
    {"key": "aluminium-mcx-kalshi", "label": "Aluminium: MCX / Kalshi perp", "kind": "cross_venue",
     "a": ("aluminium", "mcx"), "b": ("aluminium", "kalshi_perp")},
    {"key": "gold-silver", "label": "Gold / Silver (MCX)", "kind": "spread",
     "a": ("gold", "mcx"), "b": ("silver", "mcx")},
    {"key": "crude-natgas", "label": "Crude / Natural gas (MCX)", "kind": "spread",
     "a": ("crude", "mcx"), "b": ("natgas", "mcx")},
    {"key": "copper-gold", "label": "Copper / Gold (MCX)", "kind": "spread",
     "a": ("copper", "mcx"), "b": ("gold", "mcx")},
    {"key": "copper-zinc", "label": "Copper / Zinc (MCX)", "kind": "spread",
     "a": ("copper", "mcx"), "b": ("zinc", "mcx")},
    {"key": "copper-aluminium", "label": "Copper / Aluminium (MCX)", "kind": "spread",
     "a": ("copper", "mcx"), "b": ("aluminium", "mcx")},
    {"key": "zinc-lead", "label": "Zinc / Lead (MCX)", "kind": "spread",
     "a": ("zinc", "mcx"), "b": ("lead", "mcx")},
    {"key": "gold-silver-kalshi", "label": "Gold / Silver (Kalshi perps)", "kind": "spread",
     "a": ("gold", "kalshi_perp"), "b": ("silver", "kalshi_perp")},
)


class FeedError(RuntimeError):
    pass


@dataclass(frozen=True)
class Instrument:
    """One tradable contract. ``multiplier`` turns a quoted price into one contract's value."""

    id: str
    label: str
    venue: str  # nse_fo, bse_fo, mcx, kalshi_perp
    asset_class: str  # index, commodity
    underlying: str
    currency: str  # INR or USD
    multiplier: float
    key: str  # Upstox instrument key or Kalshi ticker
    expiry: Optional[date] = None
    volume_usd: float = 0.0
    meta: dict = field(default_factory=dict, compare=False, hash=False)


def _get_json(url: str, *, http_get: HttpGet, accept: str = "application/json") -> Any:
    resp = http_get(url, timeout=30, headers={"Accept": accept, "User-Agent": USER_AGENT})
    if getattr(resp, "status_code", None) != 200:
        raise FeedError(f"{url} returned HTTP {getattr(resp, 'status_code', None)}")
    return resp


def upstox_futures(exchange: str, *, http_get: HttpGet = requests.get) -> list[dict]:
    """Every listed future on an Upstox exchange file (NSE, BSE or MCX), nearest expiry first."""
    raw = _get_json(UPSTOX_INSTRUMENTS_URL.format(exchange=exchange), http_get=http_get, accept="application/gzip").content
    rows = [r for r in json.loads(gzip.decompress(raw)) if r.get("instrument_type") == "FUT" and r.get("instrument_key")]
    return sorted(rows, key=lambda r: r.get("expiry") or 0)


def kalshi_margin_markets(*, http_get: HttpGet = requests.get) -> list[dict]:
    """Kalshi's perpetuals and futures, with their book and 24-hour notional (public route)."""
    body = _get_json(KALSHI_MARGIN_MARKETS_URL, http_get=http_get).json()
    return list((body or {}).get("markets") or [])


def daily_candles(key: str, start: date, through: date, *, http_get: HttpGet = requests.get) -> list[list]:
    """Upstox daily candles [ts, open, high, low, close, volume, oi] for one instrument."""
    path = f"/v3/historical-candle/{quote(key, safe='')}/days/1/{through.isoformat()}/{start.isoformat()}"
    body = _get_json(f"{UPSTOX_BASE_URL}{path}", http_get=http_get).json()
    if not isinstance(body, dict) or body.get("status") != "success":
        raise FeedError(f"Upstox {path} returned no candles")
    return list((body.get("data") or {}).get("candles") or [])


def _upstox_volume_usd(contracts: Sequence[dict], today: date, usdinr: float, *, http_get: HttpGet) -> float:
    """Average daily notional of an underlying's nearest contracts, in USD."""
    total, sessions = 0.0, set()
    for row in contracts[:VOLUME_EXPIRIES]:
        try:
            candles = daily_candles(row["instrument_key"], today - timedelta(days=VOLUME_WINDOW_DAYS), today, http_get=http_get)
        except (FeedError, requests.RequestException, ValueError):
            continue
        multiplier = float(row.get("qty_multiplier") or 1.0)
        for candle in candles:
            try:
                close, volume = float(candle[4]), float(candle[5] or 0)
            except (IndexError, TypeError, ValueError):
                continue
            if math.isfinite(close) and close > 0:
                total += volume * close * multiplier
                sessions.add(str(candle[0])[:10])
    return total / max(1, len(sessions)) / usdinr


def _upstox_instrument(row: dict, *, venue: str, asset_class: str, underlying: str, label: str, volume: float) -> Instrument:
    return Instrument(
        id=f"{venue}:{row.get('underlying_symbol') or row.get('name')}",
        label=label, venue=venue, asset_class=asset_class, underlying=underlying, currency="INR",
        multiplier=float(row.get("qty_multiplier") or 1.0), key=row["instrument_key"],
        expiry=datetime.fromtimestamp(row["expiry"] / 1000, IST).date() if row.get("expiry") else None,
        volume_usd=volume, meta={"contract": row.get("trading_symbol"), "lot_size": row.get("lot_size"),
                                 "tick_size": row.get("tick_size")},
    )


def _kalshi_instrument(market: dict, *, asset_class: str) -> Instrument:
    ticker = market["ticker"]
    return Instrument(
        id=f"kalshi_perp:{ticker}", label=INDEX_LABELS.get(ticker) or f"{market.get('title') or ticker} (Kalshi)",
        venue="kalshi_perp", asset_class=asset_class, underlying=KALSHI_UNDERLYING.get(ticker, ticker.lower()),
        currency="USD", multiplier=1.0, key=ticker,
        volume_usd=float(market.get("volume_24h_notional_value_dollars") or 0.0),
        meta={"contract_size": market.get("contract_size"), "tick_size": market.get("tick_size")},
    )


def _active(market: dict) -> bool:
    return market.get("status") == "active"


def rank_indices(nse: Sequence[dict], bse: Sequence[dict], kalshi: Sequence[dict], today: date, usdinr: float, *,
                 http_get: HttpGet = requests.get, size: int = UNIVERSE_SIZE) -> list[Instrument]:
    """The ``size`` index futures and perps with the most daily notional."""
    out: list[Instrument] = []
    for venue, rows in (("nse_fo", nse), ("bse_fo", bse)):
        by_symbol: dict[str, list[dict]] = {}
        for row in rows:
            if row.get("underlying_type") == "INDEX":
                by_symbol.setdefault(row.get("underlying_symbol") or row.get("name"), []).append(row)
        for symbol, contracts in by_symbol.items():
            volume = _upstox_volume_usd(contracts, today, usdinr, http_get=http_get)
            out.append(_upstox_instrument(contracts[0], venue=venue, asset_class="index",
                                          underlying="india_equity", label=INDEX_LABELS.get(symbol, symbol), volume=volume))
    out += [_kalshi_instrument(m, asset_class="index") for m in kalshi if m.get("asset_class") == "Indices" and _active(m)]
    return sorted((i for i in out if i.volume_usd > 0), key=lambda i: -i.volume_usd)[:size]


def commodity_legs(mcx: Sequence[dict], kalshi: Sequence[dict], today: date, usdinr: float, *,
                   http_get: HttpGet = requests.get) -> dict[tuple[str, str], Instrument]:
    """The traded instrument for each (underlying, venue), its volume the whole underlying's."""
    by_symbol: dict[str, list[dict]] = {}
    for row in mcx:
        symbol = row.get("underlying_symbol") or row.get("name")
        if symbol in MCX_UNDERLYING:
            by_symbol.setdefault(symbol, []).append(row)
    volume_by_underlying: dict[str, float] = {}
    for symbol, contracts in by_symbol.items():
        underlying = MCX_UNDERLYING[symbol]
        volume_by_underlying[underlying] = volume_by_underlying.get(underlying, 0.0) + _upstox_volume_usd(
            contracts, today, usdinr, http_get=http_get)
    legs: dict[tuple[str, str], Instrument] = {}
    for underlying, symbol in MCX_TRADED.items():
        contracts = by_symbol.get(symbol)
        if contracts:
            legs[(underlying, "mcx")] = _upstox_instrument(
                contracts[0], venue="mcx", asset_class="commodity", underlying=underlying,
                label=f"{underlying.title()} (MCX {symbol})", volume=volume_by_underlying.get(underlying, 0.0))
    for market in kalshi:
        underlying = KALSHI_UNDERLYING.get(market.get("ticker", ""))
        if underlying and market.get("asset_class") in ("Metals", "Energy") and _active(market):
            legs[(underlying, "kalshi_perp")] = _kalshi_instrument(market, asset_class="commodity")
    return legs


def rank_pairs(legs: dict[tuple[str, str], Instrument], *, size: int = UNIVERSE_SIZE) -> list[dict[str, Any]]:
    """The ``size`` candidate pairs whose thinner leg trades the most."""
    ranked = []
    for pair in PAIR_CANDIDATES:
        a, b = legs.get(pair["a"]), legs.get(pair["b"])
        if a is None or b is None:
            continue
        ranked.append({**pair, "a": a, "b": b, "volume_usd": min(a.volume_usd, b.volume_usd)})
    return sorted((p for p in ranked if p["volume_usd"] > 0), key=lambda p: -p["volume_usd"])[:size]


def build_universe(today: date, usdinr: float, *, http_get: HttpGet = requests.get) -> dict[str, Any]:
    """Fetch every list and rank both sleeves. Raises FeedError when a list is unavailable."""
    kalshi = kalshi_margin_markets(http_get=http_get)
    indices = rank_indices(upstox_futures("NSE", http_get=http_get), upstox_futures("BSE", http_get=http_get),
                           kalshi, today, usdinr, http_get=http_get)
    legs = commodity_legs(upstox_futures("MCX", http_get=http_get), kalshi, today, usdinr, http_get=http_get)
    return {"as_of": today.isoformat(), "usdinr": usdinr, "indices": indices, "pairs": rank_pairs(legs)}


def kalshi_quote(market: dict) -> dict[str, Optional[float]]:
    """Bid, ask and mid of a Kalshi margin market; a side with no order is None."""
    def side(value: Any) -> Optional[float]:
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        return price if 0 < price < KALSHI_NO_ORDER else None

    bid, ask = side(market.get("bid")), side(market.get("ask"))
    mid = (bid + ask) / 2 if bid is not None and ask is not None else None
    return {"bid": bid, "ask": ask, "mid": mid, "reference": side(market.get("reference_price"))}


def describe(instrument: Instrument) -> dict[str, Any]:
    """A JSON-ready view of an instrument for payloads and the ledger."""
    return {
        "id": instrument.id, "label": instrument.label, "venue": instrument.venue,
        "asset_class": instrument.asset_class, "underlying": instrument.underlying,
        "currency": instrument.currency, "multiplier": instrument.multiplier, "key": instrument.key,
        "expiry": instrument.expiry.isoformat() if instrument.expiry else None,
        "volume_usd": round(instrument.volume_usd, 2), **{k: v for k, v in instrument.meta.items() if v is not None},
    }


def universe_payload(universe: dict[str, Any]) -> dict[str, Any]:
    return {
        "as_of": universe["as_of"], "usdinr": universe["usdinr"],
        "indices": [describe(i) for i in universe["indices"]],
        "pairs": [{**{k: v for k, v in p.items() if k not in ("a", "b")}, "a": describe(p["a"]), "b": describe(p["b"])}
                  for p in universe["pairs"]],
    }
