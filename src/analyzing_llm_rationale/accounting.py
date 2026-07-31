"""Prediction-market account accounting primitives.

The benchmark account is intentionally conservative:

- open positions are marked to liquidation value using bid prices;
- there is no abstract sell operation for Kalshi-style binary contracts;
- buying the opposite side nets reciprocal YES/NO pairs at $1.00 per pair.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

try:  # Keep the core package importable when the optional serve extra is absent.
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
except Exception:  # pragma: no cover - exercised only in minimal installs.
    metrics = trace = None  # type: ignore[assignment]
    Status = StatusCode = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

if trace is not None:
    _tracer = trace.get_tracer("foresea.accounting")
else:
    _tracer = None

if metrics is not None:
    _meter = metrics.get_meter("foresea.accounting")
    _account_runs = _meter.create_counter("accounting.mark_to_market_runs", unit="1")
    _account_duration = _meter.create_histogram("accounting.mark_to_market_duration", unit="s")
    _realized_pairs_counter = _meter.create_counter("accounting.realized_pairs", unit="contracts")
    _shadow_runs = _meter.create_counter("accounting.shadow_mark_to_market_runs", unit="1")
    _shadow_duration = _meter.create_histogram("accounting.shadow_mark_to_market_duration", unit="s")
    _shadow_trades = _meter.create_counter("accounting.shadow_mark_to_market_trades", unit="1")
else:
    _meter = _account_runs = _account_duration = _realized_pairs_counter = None
    _shadow_runs = _shadow_duration = _shadow_trades = None


YES = "YES"
NO = "NO"
_SIDES = {YES, NO}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("__dt__")
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _price(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out > 1.0 and out <= 100.0:
        out /= 100.0
    if out < 0.0 or out > 1.0:
        return None
    return out


def _side(value: Any) -> str:
    side = str(value or "").strip().upper()
    if side not in _SIDES:
        raise ValueError("side must be YES or NO")
    return side


def _opposite(side: str) -> str:
    side = _side(side)
    return NO if side == YES else YES


@dataclass(frozen=True)
class MarketQuote:
    """Bid/ask quote for a binary market's YES contract.

    NO prices are inferred from the reciprocal binary contract when explicit NO
    quotes are not supplied: NO bid = 1 - YES ask, NO ask = 1 - YES bid.
    """

    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    yes_probability: Optional[float] = None
    ts: Optional[Any] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MarketQuote":
        yes_bid = _price(data.get("yes_bid", data.get("market_bid")))
        yes_ask = _price(data.get("yes_ask", data.get("market_ask")))
        no_bid = _price(data.get("no_bid"))
        no_ask = _price(data.get("no_ask"))
        yes_probability = _price(data.get("yes_probability", data.get("market_probability")))
        return cls(
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            yes_probability=yes_probability,
            ts=data.get("ts") or data.get("snapshot_ts"),
        )

    def ask(self, side: str) -> Optional[float]:
        side = _side(side)
        if side == YES:
            return self.yes_ask if self.yes_ask is not None else self.yes_probability
        if self.no_ask is not None:
            return self.no_ask
        if self.yes_bid is not None:
            return 1.0 - self.yes_bid
        if self.yes_probability is not None:
            return 1.0 - self.yes_probability
        return None

    def bid(self, side: str) -> Optional[float]:
        side = _side(side)
        if side == YES:
            return self.yes_bid
        if self.no_bid is not None:
            return self.no_bid
        if self.yes_ask is not None:
            return 1.0 - self.yes_ask
        return None


@dataclass
class Position:
    platform: str
    ident: str
    side: str
    quantity: float = 0.0
    cost_basis: float = 0.0

    @property
    def avg_entry_price(self) -> float:
        return self.cost_basis / self.quantity if self.quantity > 0 else 0.0


@dataclass
class TradeFill:
    platform: str
    ident: str
    side: str
    quantity: float
    price: float
    fee: float
    ts: Any
    settlement_status: str
    realized_pairs: float = 0.0
    realized_pnl: float = 0.0
    cash_delta: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "ident": self.ident,
            "side": self.side,
            "quantity": round(self.quantity, 6),
            "price": round(self.price, 6),
            "fee": round(self.fee, 6),
            "ts": _iso(self.ts),
            "settlement_status": self.settlement_status,
            "realized_pairs": round(self.realized_pairs, 6),
            "realized_pnl": round(self.realized_pnl, 6),
            "cash_delta": round(self.cash_delta, 6),
        }


class PredictionMarketAccount:
    """Cash account with reciprocal binary-contract netting."""

    def __init__(self, starting_cash: float = 10_000.0):
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.trades: List[TradeFill] = []
        self.positions: Dict[Tuple[str, str, str], Position] = {}

    def _position(self, platform: str, ident: str, side: str) -> Position:
        key = (platform, ident, _side(side))
        pos = self.positions.get(key)
        if pos is None:
            pos = Position(platform=platform, ident=ident, side=key[2])
            self.positions[key] = pos
        return pos

    def _active_position(self, platform: str, ident: str, side: str) -> Optional[Position]:
        pos = self.positions.get((platform, ident, _side(side)))
        if pos is None or pos.quantity <= 1e-12:
            return None
        return pos

    def buy(
        self,
        *,
        platform: str,
        ident: str,
        side: str,
        quantity: float,
        price: float,
        fee: float = 0.0,
        ts: Any = None,
    ) -> TradeFill:
        """Buy YES or NO contracts, netting any opposite-side inventory first.

        Realized PnL for netted pairs is $1.00 per pair minus the entry-price
        basis of the old side and the price plus fee for the new side. Prior
        fees were already paid from cash when those old contracts were opened.
        """
        side = _side(side)
        quantity = float(quantity)
        price = float(price)
        fee = float(fee or 0.0)
        if quantity <= 0:
            raise ValueError("quantity must be greater than 0")
        if not (0.0 <= price <= 1.0):
            raise ValueError("price must be between 0 and 1")
        if fee < 0:
            raise ValueError("fee must be non-negative")

        ts = ts or _now()
        total_cost = quantity * price + fee
        cash_before = self.cash
        self.cash -= total_cost
        self.fees_paid += fee

        realized_pairs = 0.0
        realized_pnl = 0.0
        remaining = quantity
        opposite_pos = self._active_position(platform, ident, _opposite(side))
        if opposite_pos is not None:
            realized_pairs = min(remaining, opposite_pos.quantity)
            if realized_pairs > 0:
                fee_alloc = fee * (realized_pairs / quantity)
                old_basis = opposite_pos.avg_entry_price * realized_pairs
                new_basis = price * realized_pairs
                payout = realized_pairs
                realized_pnl = payout - old_basis - new_basis - fee_alloc
                self.cash += payout
                self.realized_pnl += realized_pnl
                opposite_pos.quantity -= realized_pairs
                opposite_pos.cost_basis -= old_basis
                if opposite_pos.quantity <= 1e-12:
                    opposite_pos.quantity = 0.0
                    opposite_pos.cost_basis = 0.0
                remaining -= realized_pairs

        if remaining > 1e-12:
            pos = self._position(platform, ident, side)
            pos.quantity += remaining
            pos.cost_basis += remaining * price

        fill = TradeFill(
            platform=platform,
            ident=ident,
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            ts=ts,
            settlement_status="realized" if realized_pairs > 0 else "open",
            realized_pairs=realized_pairs,
            realized_pnl=realized_pnl,
            cash_delta=self.cash - cash_before,
        )
        self.trades.append(fill)
        if realized_pairs > 0 and _realized_pairs_counter is not None:
            _realized_pairs_counter.add(realized_pairs, {"platform": platform.lower() or "unknown"})
        return fill

    def settle_market(self, *, platform: str, ident: str, outcome: int, ts: Any = None) -> Dict[str, Any]:
        """Settle open inventory against a final binary outcome."""
        ts = ts or _now()
        payout = 0.0
        settled_basis = 0.0
        settled_contracts = 0.0
        for side in (YES, NO):
            pos = self._active_position(platform, ident, side)
            if pos is None:
                continue
            wins = (side == YES and int(outcome) == 1) or (side == NO and int(outcome) == 0)
            if wins:
                payout += pos.quantity
            settled_contracts += pos.quantity
            settled_basis += pos.cost_basis
            pos.quantity = 0.0
            pos.cost_basis = 0.0
        self.cash += payout
        realized_pnl = payout - settled_basis
        self.realized_pnl += realized_pnl
        return {
            "platform": platform,
            "ident": ident,
            "outcome": int(outcome),
            "ts": _iso(ts),
            "settlement_status": "settled",
            "settled_contracts": round(settled_contracts, 6),
            "payout": round(payout, 6),
            "realized_pnl": round(realized_pnl, 6),
        }

    def open_positions(self) -> List[Position]:
        return [p for p in self.positions.values() if p.quantity > 1e-12]

    def liquidation_value(
        self,
        quotes: Mapping[Tuple[str, str], MarketQuote | Mapping[str, Any]],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        total = 0.0
        illiquid: List[Dict[str, Any]] = []
        for pos in self.open_positions():
            raw_quote = quotes.get((pos.platform, pos.ident)) or quotes.get(("", pos.ident))
            quote = raw_quote if isinstance(raw_quote, MarketQuote) else MarketQuote.from_mapping(raw_quote or {})
            bid = quote.bid(pos.side)
            if bid is None or bid <= 0.0:
                illiquid.append({
                    "platform": pos.platform,
                    "ident": pos.ident,
                    "side": pos.side,
                    "quantity": round(pos.quantity, 6),
                    "reason": "zero_or_missing_bid",
                })
                bid = 0.0
            total += pos.quantity * bid
        return total, illiquid

    def snapshot(
        self,
        quotes: Optional[Mapping[Tuple[str, str], MarketQuote | Mapping[str, Any]]] = None,
        *,
        ts: Any = None,
    ) -> Dict[str, Any]:
        quotes = quotes or {}
        liquidation_value, illiquid = self.liquidation_value(quotes)
        account_value = self.cash + liquidation_value
        return {
            "ts": _iso(ts or _now()),
            "value_method": "mark_to_market_bid_liquidation",
            "cash": round(self.cash, 6),
            "liquidation_value": round(liquidation_value, 6),
            "account_value": round(account_value, 6),
            "unrealized_pnl": round(account_value - self.cash - sum(p.cost_basis for p in self.open_positions()), 6),
            "realized_pnl": round(self.realized_pnl, 6),
            "fees_paid": round(self.fees_paid, 6),
            "open_positions": [
                {
                    "platform": p.platform,
                    "ident": p.ident,
                    "side": p.side,
                    "quantity": round(p.quantity, 6),
                    "cost_basis": round(p.cost_basis, 6),
                    "avg_entry_price": round(p.avg_entry_price, 6),
                }
                for p in self.open_positions()
            ],
            "illiquid_positions": illiquid,
        }


def _row_sort_ts(row: Mapping[str, Any]) -> str:
    return str(_iso(row.get("snapshot_ts") or row.get("ts") or "") or "")


def _row_side(row: Mapping[str, Any]) -> Optional[str]:
    model_p = _price(row.get("model_probability"))
    if model_p is None or model_p == 0.5:
        return None
    return YES if model_p > 0.5 else NO


def _row_edge_side(row: Mapping[str, Any], min_edge: float) -> Optional[str]:
    model_p = _price(row.get("model_probability"))
    market_p = _price(row.get("market_probability"))
    if model_p is None or market_p is None:
        return None
    edge = model_p - market_p
    if abs(edge) < min_edge or abs(edge) <= 1e-12:
        return None
    return YES if edge > 0 else NO


def _row_quote(row: Mapping[str, Any]) -> MarketQuote:
    return MarketQuote.from_mapping(row)


def _quote_map(
    quotes: Optional[Mapping[Any, MarketQuote | Mapping[str, Any]]],
) -> Dict[Tuple[str, str], MarketQuote | Mapping[str, Any]]:
    out: Dict[Tuple[str, str], MarketQuote | Mapping[str, Any]] = {}
    for key, quote in (quotes or {}).items():
        if isinstance(key, tuple) and len(key) >= 2:
            out[(str(key[0] or ""), str(key[1] or ""))] = quote
        else:
            out[("", str(key or ""))] = quote
    return out


def _event_sort_key(value: Any) -> str:
    iso = _iso(value)
    if not iso:
        return ""
    return iso.replace("Z", "+00:00")


def simulate_shadow_mark_to_market_account(
    rows: Iterable[Mapping[str, Any]],
    *,
    latest_quotes: Optional[Mapping[Any, MarketQuote | Mapping[str, Any]]] = None,
    starting_cash: float = 10_000.0,
    target_contracts: float = 1.0,
    min_edge: float = 0.0,
    fee_fn: Optional[Callable[[Any, float, float], float]] = None,
    max_trades: Optional[int] = None,
) -> Dict[str, Any]:
    """Simulate a persistent shadow strategy ledger from forecast snapshots.

    Trade events happen at ``snapshot_ts`` and settlement events happen later at
    ``resolved_ts``/``close_time``. A market is settled at most once, and no
    further trades are accepted after settlement. Unlike the legacy benchmark
    helper, the strategy trades only when the model disagrees with the market:
    model > market buys YES, model < market buys NO.
    """
    row_list = sorted([dict(r) for r in rows], key=_row_sort_ts)
    start = time.perf_counter()
    span_cm = _tracer.start_as_current_span("account.shadow_mark_to_market") if _tracer else None
    span = None
    if span_cm is not None:
        span = span_cm.__enter__()
        span.set_attribute("items.count", len(row_list))
        span.set_attribute("account.value_method", "bid_liquidation")
        span.set_attribute("strategy.min_edge", float(min_edge))
    try:
        account = PredictionMarketAccount(starting_cash=starting_cash)
        current_quotes = _quote_map(latest_quotes)
        settlements_by_market: Dict[Tuple[str, str], Dict[str, Any]] = {}
        events: List[Tuple[str, int, int, Dict[str, Any]]] = []
        seq = 0
        for row in row_list:
            platform = str(row.get("platform") or "")
            ident = str(row.get("ident") or "")
            if not platform or not ident:
                continue
            events.append((_event_sort_key(row.get("snapshot_ts") or row.get("ts")), 1, seq, row))
            seq += 1
            if row.get("resolved") and row.get("outcome") is not None:
                key = (platform, ident)
                settlements_by_market.setdefault(key, {
                    "platform": platform,
                    "ident": ident,
                    "outcome": int(row["outcome"]),
                    "ts": row.get("resolved_ts") or row.get("close_time") or row.get("snapshot_ts"),
                })
        for item in settlements_by_market.values():
            events.append((_event_sort_key(item.get("ts")), 0, seq, item))
            seq += 1
        events.sort(key=lambda e: (e[0], e[1], e[2]))

        trade_count = 0
        skipped_no_edge = 0
        skipped_unexecutable = 0
        settled_markets: set[Tuple[str, str]] = set()
        settlements: List[Dict[str, Any]] = []
        value_curve: List[Dict[str, Any]] = []

        for _ts_key, priority, _seq, payload in events:
            platform = str(payload.get("platform") or "")
            ident = str(payload.get("ident") or "")
            market_key = (platform, ident)
            if priority == 0:
                if market_key in settled_markets:
                    continue
                settlement = account.settle_market(
                    platform=platform,
                    ident=ident,
                    outcome=int(payload["outcome"]),
                    ts=payload.get("ts"),
                )
                settled_markets.add(market_key)
                settlements.append(settlement)
                snap = account.snapshot(current_quotes, ts=payload.get("ts"))
                snap["event_type"] = "settlement"
                snap["event_market"] = {"platform": platform, "ident": ident}
                value_curve.append(snap)
                continue

            row = payload
            if market_key in settled_markets:
                continue
            side = _row_edge_side(row, min_edge)
            if not side:
                skipped_no_edge += 1
                continue
            quote = _row_quote(row)
            current_quotes.setdefault((platform, ident), quote)
            ask = quote.ask(side)
            if ask is None or ask <= 0.0 or ask >= 1.0:
                skipped_unexecutable += 1
                logger.warning("skipping shadow trade without executable ask", extra={
                    "platform": platform,
                    "side": side,
                })
                continue
            opposite = account._active_position(platform, ident, _opposite(side))
            same = account._active_position(platform, ident, side)
            buy_qty = 0.0
            if opposite is not None:
                buy_qty += opposite.quantity
            if same is None:
                buy_qty += target_contracts
            elif same.quantity < target_contracts:
                buy_qty += target_contracts - same.quantity
            if buy_qty <= 1e-12:
                continue
            if max_trades is not None and trade_count >= max_trades:
                break
            notional = buy_qty * ask
            explicit_fee = row.get("fee")
            fee = float(explicit_fee) if explicit_fee not in (None, "") else (
                float(fee_fn(platform, notional, ask)) if fee_fn else 0.0
            )
            fill = account.buy(
                platform=platform,
                ident=ident,
                side=side,
                quantity=buy_qty,
                price=ask,
                fee=fee,
                ts=row.get("snapshot_ts") or row.get("ts"),
            )
            trade_count += 1
            if _shadow_trades is not None:
                _shadow_trades.add(1, {"platform": platform.lower() or "unknown"})
            snap = account.snapshot(current_quotes, ts=row.get("snapshot_ts") or row.get("ts"))
            snap["event_type"] = "trade"
            snap["trade_status"] = fill.settlement_status
            snap["event_market"] = {"platform": platform, "ident": ident, "side": side}
            value_curve.append(snap)

        final = account.snapshot(current_quotes)
        final.update({
            "strategy": "edge_shadow_ledger",
            "starting_cash": round(starting_cash, 6),
            "return": round((final["account_value"] / starting_cash) - 1.0, 6) if starting_cash else None,
            "target_contracts": round(float(target_contracts), 6),
            "min_edge": round(float(min_edge), 6),
            "n_trades": trade_count,
            "n_settlements": len(settlements),
            "n_open_positions": len(final["open_positions"]),
            "n_illiquid_positions": len(final["illiquid_positions"]),
            "n_skipped_no_edge": skipped_no_edge,
            "n_skipped_unexecutable": skipped_unexecutable,
            "trades": [t.as_dict() for t in account.trades],
            "settlements": settlements,
            "value_curve": value_curve,
            "notes": [
                "Shadow ledger: trades are inferred from model-vs-market edge at forecast time, not sent to an exchange.",
                "Open positions are valued at bid-side liquidation prices.",
                "Resolved markets settle into cash once per market before later trades are considered.",
                "Kalshi-style exits are represented as buying the opposite contract and netting YES/NO pairs at $1.00.",
            ],
        })
        if _shadow_runs is not None:
            _shadow_runs.add(1, {"outcome": "success"})
        if span is not None:
            span.set_attribute("outcome", "success")
            span.set_attribute("trades.count", trade_count)
            span.set_attribute("settlements.count", len(settlements))
            span.set_attribute("positions.open_count", final["n_open_positions"])
        return final
    except Exception as exc:
        if _shadow_runs is not None:
            _shadow_runs.add(1, {"outcome": "failure"})
        if span is not None and Status is not None and StatusCode is not None:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
        raise
    finally:
        elapsed = time.perf_counter() - start
        if _shadow_duration is not None:
            _shadow_duration.record(elapsed, {"operation": "shadow_mark_to_market"})
        if span_cm is not None:
            span_cm.__exit__(None, None, None)


def simulate_mark_to_market_account(
    rows: Iterable[Mapping[str, Any]],
    *,
    latest_quotes: Optional[Mapping[Any, MarketQuote | Mapping[str, Any]]] = None,
    starting_cash: float = 10_000.0,
    target_contracts: float = 1.0,
    max_trades: Optional[int] = None,
) -> Dict[str, Any]:
    """Simulate one benchmark account and return its bid-liquidation value.

    Each row is treated as a model decision for one market. The account targets
    ``target_contracts`` on the model's current side. If the side changes, the
    account buys the opposite contract to close through reciprocal netting, then
    opens the new target side. Open inventory is valued at bid liquidation prices
    from ``latest_quotes`` (or the row quote when no latest quote is supplied).
    """
    row_list = sorted([dict(r) for r in rows], key=_row_sort_ts)
    start = time.perf_counter()
    span_cm = _tracer.start_as_current_span("account.mark_to_market") if _tracer else None
    span = None
    if span_cm is not None:
        span = span_cm.__enter__()
        span.set_attribute("items.count", len(row_list))
        span.set_attribute("account.value_method", "bid_liquidation")
    try:
        account = PredictionMarketAccount(starting_cash=starting_cash)
        current_quotes = _quote_map(latest_quotes)
        trade_count = 0
        settlements: List[Dict[str, Any]] = []
        value_curve: List[Dict[str, Any]] = []

        for row in row_list:
            side = _row_side(row)
            platform = str(row.get("platform") or "")
            ident = str(row.get("ident") or "")
            if not side or not platform or not ident:
                continue
            quote = _row_quote(row)
            current_quotes.setdefault((platform, ident), quote)
            ask = quote.ask(side)
            if ask is None or ask <= 0.0 or ask >= 1.0:
                logger.warning("skipping account trade without executable ask", extra={
                    "platform": platform,
                    "side": side,
                })
                continue
            opposite = account._active_position(platform, ident, _opposite(side))
            same = account._active_position(platform, ident, side)
            buy_qty = 0.0
            if opposite is not None:
                buy_qty += opposite.quantity
            if same is None:
                buy_qty += target_contracts
            elif same.quantity < target_contracts:
                buy_qty += target_contracts - same.quantity
            if buy_qty <= 1e-12:
                continue
            if max_trades is not None and trade_count >= max_trades:
                break
            fee = float(row.get("fee") or 0.0)
            fill = account.buy(
                platform=platform,
                ident=ident,
                side=side,
                quantity=buy_qty,
                price=ask,
                fee=fee,
                ts=row.get("snapshot_ts") or row.get("ts"),
            )
            trade_count += 1
            if row.get("outcome") is not None and row.get("resolved"):
                settlements.append(account.settle_market(
                    platform=platform,
                    ident=ident,
                    outcome=int(row["outcome"]),
                    ts=row.get("resolved_ts"),
                ))
            snap = account.snapshot(current_quotes, ts=row.get("snapshot_ts") or row.get("ts"))
            snap["trade_status"] = fill.settlement_status
            value_curve.append(snap)

        final = account.snapshot(current_quotes)
        final.update({
            "starting_cash": round(starting_cash, 6),
            "return": round((final["account_value"] / starting_cash) - 1.0, 6) if starting_cash else None,
            "n_trades": trade_count,
            "n_open_positions": len(final["open_positions"]),
            "n_illiquid_positions": len(final["illiquid_positions"]),
            "trades": [t.as_dict() for t in account.trades],
            "settlements": settlements,
            "value_curve": value_curve,
            "notes": [
                "Open positions are valued at bid-side liquidation prices.",
                "Kalshi-style exits are represented as buying the opposite contract and netting YES/NO pairs at $1.00.",
            ],
        })
        if _account_runs is not None:
            _account_runs.add(1, {"outcome": "success"})
        if span is not None:
            span.set_attribute("outcome", "success")
            span.set_attribute("trades.count", trade_count)
            span.set_attribute("positions.open_count", final["n_open_positions"])
        return final
    except Exception as exc:
        if _account_runs is not None:
            _account_runs.add(1, {"outcome": "failure"})
        if span is not None and Status is not None and StatusCode is not None:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
        raise
    finally:
        elapsed = time.perf_counter() - start
        if _account_duration is not None:
            _account_duration.record(elapsed, {"operation": "mark_to_market"})
        if span_cm is not None:
            span_cm.__exit__(None, None, None)
