"""The Foresea Fund's orchestrator: turns analysts' signals into paper trades (docs/fund.md).

Models advise; this code decides. Each cycle it:

1. collects signals in one shape from every analyst it has (a Signal: instrument, side, the
   edge the analyst expects per dollar after costs, how far that analyst is trusted, and the
   facts behind it);
2. exits positions whose reason is gone (a binary side the model no longer favours, a pair
   back inside its exit band or past its holding limit);
3. sizes new positions small: a fraction of Kelly scaled by trust for event contracts, a fixed
   risk budget scaled by trust for pairs, under hard limits (10% of the fund per position, 40%
   per asset class, 15% net per underlying, 20% kept in cash);
4. offsets any underlying beyond its limit with the cheapest liquid leg on another venue (a
   Kalshi perp for MCX metals and crude, say) rather than letting concentration build;
5. writes every order's attribution and its fact sheet, so the trade note can only say what
   the numbers say (fund_notes).

Analysts in this first version (and why some are absent):
- event_model: the on-device model's nightly forecasts on the Edge board.
- pairs_model: the commodity-pairs tracker, only for pairs whose 16-year backtest made money
  after costs. A stretched pair with a losing backtest is skipped with that reason recorded.
- The index model forecasts ranges, not direction (its median sits within 0.1% of the close),
  so it sizes risk but opens no index trades until a directional analyst joins (step 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

POSITION_CAP = 0.10
CLASS_CAP = 0.40
UNDERLYING_CAP = 0.15
CASH_FLOOR = 0.20
KELLY_FRACTION = 0.25
PAIR_RISK = 0.05
#: Smallest edge per dollar worth paying the spread for.
MIN_EDGE = 0.05
#: Trust in an analyst with no graded record yet; rises with its record (step 5).
DEFAULT_TRUST = 0.25
#: Binary prices outside this band leave too little to win or too much to lose on a model error.
PRICE_BAND = (0.03, 0.97)
#: A forecast older than this no longer backs a trade.
MAX_FORECAST_AGE_H = 48.0
#: Kalshi charges a fee on event contracts; this conservative per-dollar cost is taken off the edge.
EVENT_COST = 0.02
#: Orders smaller than this are not worth the fees and noise.
MIN_ORDER_USD = 5.0


@dataclass
class Signal:
    analyst: str
    instrument: dict
    side: str
    price: float  # the entry the edge assumes (ask for a buy)
    edge: float  # expected profit per dollar committed, after costs
    trust: float
    facts: list[dict] = field(default_factory=list)
    pair_key: Optional[str] = None
    kelly: Optional[float] = None  # binary only: Kelly fraction of the fund


@dataclass
class Order:
    action: str  # buy or sell
    instrument: dict
    side: str
    quantity: float
    price: float
    purpose: str  # view, exit, hedge
    attribution: dict
    facts: list[dict]
    pair_key: Optional[str] = None
    reason: str = ""

    @property
    def notional(self) -> float:
        return self.quantity * self.price * float(self.instrument.get("multiplier") or 1.0)


def _fact(label: str, value: str, source: str) -> dict:
    return {"label": label, "value": value, "source": source}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _cents(x: float) -> str:
    return f"{x * 100:.0f}¢"


def _age_hours(stamp: Any, now: datetime) -> Optional[float]:
    try:
        then = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (now - then).total_seconds() / 3600.0


def event_instrument(row: Mapping[str, Any]) -> dict:
    platform = str(row.get("platform") or "kalshi").lower()
    ident = str(row.get("ident") or "")
    return {"id": f"{platform}:{ident}", "label": row.get("question") or ident, "venue": platform,
            "asset_class": "event", "underlying": f"{platform}:{ident}", "currency": "USD", "multiplier": 1.0,
            "key": ident}


def event_signals(board: Iterable[Mapping[str, Any]], now: datetime, *, analyst: str = "event_model",
                  model_label: str = "the on-device model", trust: float = DEFAULT_TRUST) -> list[Signal]:
    """A signal per Edge board row where the forecast beats the price after costs."""
    out = []
    for row in board:
        p, bid, ask = row.get("model_probability"), row.get("market_bid"), row.get("market_ask")
        age = _age_hours(row.get("forecasted_at"), now)
        if p is None or bid is None or ask is None or age is None or age > MAX_FORECAST_AGE_H:
            continue
        p, bid, ask = float(p), float(bid), float(ask)
        for side, prob, price in (("YES", p, ask), ("NO", 1.0 - p, 1.0 - bid)):
            if not PRICE_BAND[0] <= price <= PRICE_BAND[1]:
                continue
            edge = (prob - price) / price - EVENT_COST
            if edge < MIN_EDGE:
                continue
            source = f"{model_label} forecast {str(row.get('forecasted_at'))[:16].replace('T', ' ')} UTC"
            out.append(Signal(
                analyst=analyst, instrument=event_instrument(row), side=side, price=price, edge=edge, trust=trust,
                kelly=(prob - price) / (1.0 - price),
                facts=[_fact("question", str(row.get("question") or row.get("ident")), "venue"),
                       _fact(f"model probability of {side}", _pct(prob), source),
                       _fact(f"price to buy {side}", _cents(price), f"{row.get('platform')} order book"),
                       _fact("expected profit per dollar after costs", _pct(edge), "orchestrator")],
            ))
    return out


def pair_signals(tracker: Mapping[str, Any], universe_pairs: Sequence[Mapping[str, Any]], *,
                 analyst: str = "pairs_model", trust: float = DEFAULT_TRUST) -> tuple[list[Signal], list[dict]]:
    """Signals for stretched fund pairs whose backtest made money; and the skipped ones, with why."""
    rows = {p.get("key"): p for p in tracker.get("pairs") or []}
    backtest = (tracker.get("backtest") or {}).get("pairs") or {}
    rule = tracker.get("rule") or {}
    signals, skipped = [], []
    for pair in universe_pairs:
        row, record = rows.get(pair["key"]), backtest.get(pair["key"])
        if row is None or row.get("z") is None:
            continue
        if not row.get("stretched"):
            continue
        if not record or float(record.get("avg_return") or 0.0) <= 0.0:
            skipped.append({"pair": pair["key"], "z": row["z"],
                            "reason": f"stretched at z={row['z']:.2f} but its backtest averaged "
                                      f"{_pct(float((record or {}).get('avg_return') or 0.0))} per signal"})
            continue
        z, beta = float(row["z"]), float(row.get("beta") or 1.0)
        # z < 0: leg a is cheap against b, so buy a and sell b (and the reverse).
        a_side, b_side = ("LONG", "SHORT") if z < 0 else ("SHORT", "LONG")
        facts = [_fact("pair", pair.get("label") or pair["key"], "fund universe"),
                 _fact("spread z-score", f"{z:.2f}", f"pairs tracker, data through {row.get('date')}"),
                 _fact("entry rule", f"|z| >= {rule.get('entry_z', 2.0)}", "pairs tracker"),
                 _fact("backtest average per signal after costs", _pct(float(record['avg_return'])),
                       f"pairs backtest {tracker.get('backtest', {}).get('window', '')}".strip()),
                 _fact("backtest win rate", _pct(float(record.get("win_rate") or 0.0)), "pairs backtest"),
                 _fact("hedge ratio", f"{beta:.2f}", "pairs tracker")]
        # Equal-dollar legs; the regression hedge ratio is reported, not yet applied.
        for leg, side in (("a", a_side), ("b", b_side)):
            signals.append(Signal(analyst=analyst, instrument=dict(pair[leg]), side=side, price=0.0,
                                  edge=float(record["avg_return"]), trust=trust, facts=facts, pair_key=pair["key"]))
    return signals, skipped


def _attribution(signals: Sequence[Signal]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for s in signals:
        weights[s.analyst] = weights.get(s.analyst, 0.0) + max(s.trust * s.edge, 1e-9)
    total = sum(weights.values())
    return {k: round(v / total, 4) for k, v in weights.items()}


def plan(snapshot: Mapping[str, Any], signals: Sequence[Signal], quotes: Mapping[str, Mapping[str, Any]],
         hedges: Mapping[str, dict], *, forecasts: Mapping[str, Mapping[str, Any]] = (),
         held_pairs: Mapping[str, Mapping[str, Any]] = (), pair_rows: Mapping[str, Mapping[str, Any]] = (),
         exit_z: float = 0.5, max_hold: int = 20) -> dict[str, Any]:
    """Orders for one cycle from the fund's nav() ``snapshot``, ``signals`` and ``quotes``.

    ``hedges`` maps an underlying to the instrument that offsets it. ``forecasts`` maps an event
    instrument id to its latest fresh forecast row (model_probability, forecasted_at): a held
    side is sold only when such a forecast values it below its bid, so positions the model has
    not looked at again are kept. ``held_pairs`` maps a held pair to the sessions it has been
    held, and ``pair_rows`` gives each pair's latest tracker row.
    """
    nav = float(snapshot["nav_usd"])
    cash = float(snapshot["cash_usd"])
    orders: list[Order] = []
    passed: list[dict] = []
    held = {(p["instrument_id"], p["side"]): p for p in snapshot.get("positions") or []}
    by_class = dict(snapshot.get("by_asset_class") or {})
    exposure = dict(snapshot.get("net_exposure_by_underlying") or {})
    forecasts = dict(forecasts)

    # 1. Exits: a held event side a fresh forecast now values below its bid; a pair back in its
    #    band or held too long.
    for (iid, side), pos in held.items():
        row = forecasts.get(iid)
        if pos["kind"] != "binary" or row is None or pos.get("mark") is None or row.get("model_probability") is None:
            continue
        p_yes = float(row["model_probability"])
        worth = p_yes if side == "YES" else 1.0 - p_yes
        if worth < float(pos["mark"]):
            source = f"forecast {str(row.get('forecasted_at'))[:16].replace('T', ' ')} UTC"
            orders.append(Order("sell", {"id": iid, "label": pos.get("label")}, side, pos["quantity"], pos["mark"],
                                "exit", pos.get("attribution") or {},
                                [_fact("position", f"{side} {pos.get('label') or iid}", "fund ledger"),
                                 _fact(f"model probability of {side}", _pct(worth), source),
                                 _fact("bid now", _cents(float(pos["mark"])), "order book")],
                                reason="the latest forecast values this side below what it sells for"))
    for key, entry in dict(held_pairs).items():
        row = dict(pair_rows).get(key) or {}
        z, sessions = row.get("z"), int(entry.get("sessions") or 0)
        if (z is not None and abs(float(z)) < exit_z) or sessions >= max_hold:
            why = f"spread back inside |z| < {exit_z}" if z is not None and abs(float(z)) < exit_z else \
                  f"held {sessions} sessions, the {max_hold}-session limit"
            for (iid, side), pos in held.items():
                if pos.get("pair_key") == key and pos.get("mark") is not None:
                    orders.append(Order("sell", {"id": iid, "label": pos.get("label")}, side, pos["quantity"],
                                        pos["mark"], "exit", pos.get("attribution") or {},
                                        [_fact("pair", key, "fund ledger"), _fact("reason", why, "pairs tracker")],
                                        pair_key=key, reason=why))

    # 2. Entries, best edge x trust first, inside every limit.
    budget = max(0.0, cash - CASH_FLOOR * nav)
    for s in sorted(signals, key=lambda s: -(s.edge * s.trust)):
        iid = s.instrument["id"]
        if (iid, s.side) in held:
            continue
        quote = quotes.get(iid) or {}
        if s.instrument.get("asset_class") == "event":
            price = s.price
            target = min(POSITION_CAP, KELLY_FRACTION * s.trust * float(s.kelly or 0.0)) * nav
        else:
            price = quote.get("ask") if s.side == "LONG" else quote.get("bid")
            if price is None:
                passed.append({"instrument": iid, "reason": "no price to trade at"})
                continue
            target = min(POSITION_CAP, PAIR_RISK * s.trust * 4) * nav  # trust 0.25 -> 5% per leg
        cls = s.instrument.get("asset_class", "")
        target = min(target, CLASS_CAP * nav - by_class.get(cls, 0.0), budget)
        if target < MIN_ORDER_USD:
            passed.append({"instrument": iid, "reason": "no room under the fund's limits"})
            continue
        qty = target / (price * float(s.instrument.get("multiplier") or 1.0))
        orders.append(Order("buy", s.instrument, s.side, qty, price, "view", _attribution([s]),
                            s.facts + [_fact("size", f"{target / nav * 100:.1f}% of the fund", "orchestrator")],
                            pair_key=s.pair_key, reason=f"edge {_pct(s.edge)} per dollar after costs"))
        budget -= target
        by_class[cls] = by_class.get(cls, 0.0) + target
        sign = -1.0 if s.side == "SHORT" else 1.0
        exposure[s.instrument.get("underlying", iid)] = exposure.get(s.instrument.get("underlying", iid), 0.0) + sign * target

    # 3. Hedges: offset any non-event underlying beyond its limit on another venue.
    for underlying, net in exposure.items():
        excess = abs(net) - UNDERLYING_CAP * nav
        hedge = hedges.get(underlying)
        if excess < MIN_ORDER_USD or hedge is None or ":" in underlying:
            continue
        side = "SHORT" if net > 0 else "LONG"
        quote = quotes.get(hedge["id"]) or {}
        price = quote.get("bid") if side == "SHORT" else quote.get("ask")
        if price is None or excess > budget:
            passed.append({"instrument": hedge["id"], "reason": f"could not hedge {underlying}"})
            continue
        orders.append(Order("buy", hedge, side, excess / (price * float(hedge.get("multiplier") or 1.0)), price,
                            "hedge", {"risk_overlay": 1.0},
                            [_fact("underlying", underlying, "fund ledger"),
                             _fact("net exposure", f"${abs(net):,.0f}", "fund ledger"),
                             _fact("limit", f"${UNDERLYING_CAP * nav:,.0f}", "orchestrator"),
                             _fact("hedge leg", hedge.get("label") or hedge["id"], "fund universe")],
                            reason=f"{underlying} exposure above {UNDERLYING_CAP:.0%} of the fund"))
        budget -= excess
    return {"orders": orders, "passed": passed}
