"""Prediction-market mark-to-market accounting.

Open positions are valued at bid-side liquidation prices. Kalshi-style exits
are represented by buying the opposite side and netting YES/NO pairs at $1.00.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("foresea.accounting")
meter = metrics.get_meter("foresea.accounting")
mark_to_market_runs = meter.create_counter("accounting.mark_to_market_runs", unit="1")
mark_to_market_duration = meter.create_histogram("accounting.mark_to_market_duration", unit="s")
realized_pairs_counter = meter.create_counter("accounting.realized_pairs", unit="contracts")

YES = "YES"
NO = "NO"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _price(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if 1.0 < out <= 100.0:
        out /= 100.0
    if out < 0.0 or out > 1.0:
        return None
    return out


def _side(value: Any) -> str:
    side = str(value or "").strip().upper()
    if side not in {YES, NO}:
        raise ValueError("side must be YES or NO")
    return side


def _opposite(side: str) -> str:
    return NO if _side(side) == YES else YES


@dataclass(frozen=True)
class MarketQuote:
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    yes_probability: Optional[float] = None
    ts: Optional[Any] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MarketQuote":
        return cls(
            yes_bid=_price(data.get("yes_bid", data.get("market_bid"))),
            yes_ask=_price(data.get("yes_ask", data.get("market_ask"))),
            no_bid=_price(data.get("no_bid")),
            no_ask=_price(data.get("no_ask")),
            yes_probability=_price(data.get("yes_probability", data.get("market_probability"))),
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
        if key not in self.positions:
            self.positions[key] = Position(platform=platform, ident=ident, side=key[2])
        return self.positions[key]

    def _active_position(self, platform: str, ident: str, side: str) -> Optional[Position]:
        pos = self.positions.get((platform, ident, _side(side)))
        return pos if pos is not None and pos.quantity > 1e-12 else None

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
        side = _side(side)
        quantity = float(quantity)
        price = float(price)
        fee = float(fee or 0.0)
        if quantity <= 0:
            raise ValueError("quantity must be greater than 0")
        if not 0.0 <= price <= 1.0:
            raise ValueError("price must be between 0 and 1")
        if fee < 0:
            raise ValueError("fee must be non-negative")

        ts = ts or _now()
        cash_before = self.cash
        self.cash -= quantity * price + fee
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
                realized_pnl = realized_pairs - old_basis - new_basis - fee_alloc
                self.cash += realized_pairs
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
        if realized_pairs > 0:
            realized_pairs_counter.add(realized_pairs, {"platform": platform.lower() or "unknown"})
        return fill

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
        liquidation_value, illiquid = self.liquidation_value(quotes or {})
        account_value = self.cash + liquidation_value
        open_basis = sum(p.cost_basis for p in self.open_positions())
        return {
            "ts": _iso(ts or _now()),
            "value_method": "mark_to_market_bid_liquidation",
            "cash": round(self.cash, 6),
            "liquidation_value": round(liquidation_value, 6),
            "account_value": round(account_value, 6),
            "unrealized_pnl": round(liquidation_value - open_basis, 6),
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


def simulate_mark_to_market_account(
    rows: Iterable[Mapping[str, Any]],
    *,
    latest_quotes: Optional[Mapping[Any, MarketQuote | Mapping[str, Any]]] = None,
    starting_cash: float = 10_000.0,
    target_contracts: float = 1.0,
    max_trades: Optional[int] = None,
) -> Dict[str, Any]:
    """Simulate one benchmark account and return bid-liquidation value."""
    row_list = sorted([dict(r) for r in rows], key=_row_sort_ts)
    start = time.perf_counter()
    with tracer.start_as_current_span("accounting.mark_to_market") as span:
        span.set_attributes({
            "items.count": len(row_list),
            "account.value_method": "bid_liquidation",
        })
        try:
            account = PredictionMarketAccount(starting_cash=starting_cash)
            current_quotes: Dict[Tuple[str, str], MarketQuote | Mapping[str, Any]] = {}
            final_quotes = _quote_map(latest_quotes)
            trade_count = 0
            value_curve: List[Dict[str, Any]] = []

            for row in row_list:
                side = _row_side(row)
                platform = str(row.get("platform") or "")
                ident = str(row.get("ident") or "")
                if not side or not platform or not ident:
                    continue
                quote = MarketQuote.from_mapping(row)
                current_quotes[(platform, ident)] = quote
                ask = quote.ask(side)
                if ask is None or ask <= 0.0 or ask >= 1.0:
                    logger.warning("skipping account trade without executable ask")
                    continue

                buy_qty = 0.0
                opposite = account._active_position(platform, ident, _opposite(side))
                same = account._active_position(platform, ident, side)
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

                fill = account.buy(
                    platform=platform,
                    ident=ident,
                    side=side,
                    quantity=buy_qty,
                    price=ask,
                    fee=float(row.get("fee") or 0.0),
                    ts=row.get("snapshot_ts") or row.get("ts"),
                )
                trade_count += 1
                snap = account.snapshot(current_quotes, ts=row.get("snapshot_ts") or row.get("ts"))
                snap["trade_status"] = fill.settlement_status
                value_curve.append(snap)

            final = account.snapshot({**current_quotes, **final_quotes})
            final.update({
                "starting_cash": round(starting_cash, 6),
                "return": round((final["account_value"] / starting_cash) - 1.0, 6)
                if starting_cash else None,
                "n_trades": trade_count,
                "n_open_positions": len(final["open_positions"]),
                "n_illiquid_positions": len(final["illiquid_positions"]),
                "trades": [t.as_dict() for t in account.trades],
                "value_curve": value_curve,
                "notes": [
                    "Open positions are valued at bid-side liquidation prices.",
                    "Exits are modeled as buying the opposite contract and netting YES/NO pairs at $1.00.",
                ],
            })
            mark_to_market_runs.add(1, {"outcome": "success"})
            span.set_attributes({
                "outcome": "success",
                "trades.count": trade_count,
                "positions.open_count": final["n_open_positions"],
            })
            return final
        except Exception as exc:
            mark_to_market_runs.add(1, {"outcome": "failure"})
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            raise
        finally:
            mark_to_market_duration.record(
                time.perf_counter() - start,
                {"operation": "mark_to_market"},
            )
