"""Prediction-market mark-to-market accounting.

Open positions are valued at bid-side liquidation prices. Kalshi-style exits
are represented by buying the opposite side and netting YES/NO pairs at $1.00.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("foresea.accounting")
meter = metrics.get_meter("foresea.accounting")
mark_to_market_runs = meter.create_counter("accounting.mark_to_market_runs", unit="1")
mark_to_market_duration = meter.create_histogram("accounting.mark_to_market_duration", unit="s")
realized_pairs_counter = meter.create_counter("accounting.realized_pairs", unit="contracts")
shadow_mark_to_market_runs = meter.create_counter("accounting.shadow_mark_to_market_runs", unit="1")
shadow_mark_to_market_duration = meter.create_histogram(
    "accounting.shadow_mark_to_market_duration", unit="s"
)
shadow_mark_to_market_trades = meter.create_counter(
    "accounting.shadow_mark_to_market_trades", unit="1"
)
market_follow_mark_to_market_runs = meter.create_counter(
    "accounting.market_follow_mark_to_market_runs", unit="1"
)
validated_kelly_runs = meter.create_counter("accounting.validated_kelly_runs", unit="1")
validated_kelly_duration = meter.create_histogram(
    "accounting.validated_kelly_duration", unit="s"
)
validated_kelly_trades = meter.create_counter(
    "accounting.validated_kelly_trades", unit="1"
)

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


def _row_edge_side(row: Mapping[str, Any], min_edge: float) -> Optional[str]:
    model_p = _price(row.get("model_probability"))
    market_p = _price(row.get("market_probability"))
    if model_p is None or market_p is None:
        return None
    edge = model_p - market_p
    if abs(edge) < min_edge or abs(edge) <= 1e-12:
        return None
    return YES if edge > 0 else NO


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


def _kelly_fraction(p_win: float, p_side: float) -> float:
    """Full-Kelly fraction of bankroll for a binary bet: f* = (p*b - q) / b,
    b = decimal payout odds on the side actually held. Uncapped and
    un-fractioned -- callers apply their own kelly_fraction multiplier and a
    max-concentration cap; this just returns the raw formula's output
    (can be negative, meaning no edge on this side)."""
    if p_side <= 0.0 or p_side >= 1.0:
        return 0.0
    odds = (1.0 - p_side) / p_side
    if odds <= 0:
        return 0.0
    q_win = 1.0 - p_win
    return (p_win * odds - q_win) / odds


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
    """Simulate a persistent shadow ledger from forecast snapshots.

    Trade events happen at snapshot time; settlement events happen later at
    resolved_ts/close_time. The strategy trades only model-vs-market edge:
    model > market buys YES, model < market buys NO.
    """
    row_list = sorted([dict(r) for r in rows], key=_row_sort_ts)
    start = time.perf_counter()
    with tracer.start_as_current_span("accounting.shadow_mark_to_market") as span:
        span.set_attributes({
            "items.count": len(row_list),
            "account.value_method": "bid_liquidation",
            "strategy.min_edge": float(min_edge),
        })
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
                    settlements_by_market.setdefault((platform, ident), {
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
                quote = MarketQuote.from_mapping(row)
                current_quotes.setdefault((platform, ident), quote)
                ask = quote.ask(side)
                if ask is None or ask <= 0.0 or ask >= 1.0:
                    skipped_unexecutable += 1
                    logger.warning("skipping shadow trade without executable ask")
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
                shadow_mark_to_market_trades.add(1, {"platform": platform.lower() or "unknown"})
                snap = account.snapshot(current_quotes, ts=row.get("snapshot_ts") or row.get("ts"))
                snap["event_type"] = "trade"
                snap["trade_status"] = fill.settlement_status
                snap["event_market"] = {"platform": platform, "ident": ident, "side": side}
                value_curve.append(snap)

            final = account.snapshot(current_quotes)
            if trade_count > 0:
                # Extend the curve to right now even when every position has
                # already settled -- otherwise an account that's fully in
                # cash freezes its last plotted point at its last trade/
                # settlement time, which can be days old, and looks
                # (wrongly) stale/behind next to accounts still holding open
                # positions (which always get marked to the current moment).
                mtm_snap = dict(final)
                mtm_snap["event_type"] = "mark_to_market"
                value_curve.append(mtm_snap)
            final.update({
                "strategy": "edge_shadow_ledger",
                "starting_cash": round(starting_cash, 6),
                "return": round((final["account_value"] / starting_cash) - 1.0, 6)
                if starting_cash else None,
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
            shadow_mark_to_market_runs.add(1, {"outcome": "success"})
            span.set_attributes({
                "outcome": "success",
                "trades.count": trade_count,
                "settlements.count": len(settlements),
                "positions.open_count": final["n_open_positions"],
            })
            return final
        except Exception as exc:
            shadow_mark_to_market_runs.add(1, {"outcome": "failure"})
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            raise
        finally:
            shadow_mark_to_market_duration.record(
                time.perf_counter() - start,
                {"operation": "shadow_mark_to_market"},
            )


def simulate_validated_kelly_account(
    rows: Iterable[Mapping[str, Any]],
    *,
    latest_quotes: Optional[Mapping[Any, MarketQuote | Mapping[str, Any]]] = None,
    starting_cash: float = 10_000.0,
    kelly_fraction: float = 0.5,
    fixed_fraction: Optional[float] = None,
    strategy_name: Optional[str] = None,
    max_concentration: float = 0.15,
    market_shrinkage: float = 0.0,
    max_drawdown: Optional[float] = None,
    require_validated: bool = True,
    follow_model_call: bool = False,
    fade: bool = False,
    min_edge: float = 0.0,
    min_price: float = 0.20,
    max_price: float = 0.80,
    fee_fn: Optional[Callable[[Any, float, float], float]] = None,
    max_trades: Optional[int] = None,
) -> Dict[str, Any]:
    """Real position ledger (one settle per market) sized by fractional Kelly
    on the *current* account value and capped per-market to bound tail risk
    from any single miscalibrated call. ``market_shrinkage`` tempers the
    calibrated probability toward the executable market price before sizing.
    When ``max_drawdown`` is set, new purchases halt once the account's
    current total value (cash plus open positions marked to market) has
    drawn down past that fraction of its running high-watermark peak,
    resuming automatically once value recovers above the floor.

    This is the strategy actually meant to be traded, unlike
    simulate_shadow_mark_to_market_account's fixed $1 target_contracts (a
    diagnostic default) or paper_pnl's per-snapshot backtest scoring (a
    research tool, not an executable position). Two things distinguish it:

    1. Position size is a *fraction of the live account value*, recomputed at
       every trade -- gains from earlier markets are reinvested (they raise
       the stake on the next one), and losses genuinely shrink it. Capped at
       max_concentration so no single market can consume more than that
       fraction of the book, regardless of how confident the call is -- this
       directly targets the failure mode found in production: a model with a
       78% resolved win rate was still net-negative because a handful of
       oversized, wrong, high-conviction bets outweighed many small correct
       ones.
    2. Each row is expected to carry (attached by the caller from historical
       resolved data): 'calibrated_model_probability' -- P(YES) to use for
       Kelly sizing of whichever side ends up traded -- and '_edge_validated'
       (True only if this row is approved to trade under the current
       strategy). With require_validated=True (the point of this strategy),
       rows without that field or set to False are skipped entirely: no
       proven edge, no trade, regardless of how large the raw disagreement
       looks. What populates these two fields depends on the strategy the
       caller is building (see track_record_live.py):
         - follow (fade=False): calibrated_model_probability is the model's
           own walk-forward isotonic fit; _edge_validated means this
           disagreement bucket has *proven positive* skill
           (edge_calibration's skill_significant).
         - fade (fade=True): the traded side is the *opposite* of the
           model's implied side -- betting with the market specifically in
           the regime proven, across every tracked model, to be where the
           model is worst (the 20pp+ bucket has statistically significant
           *negative* skill everywhere, not just "no proven edge yet").
           calibrated_model_probability there is the walk-forward empirical
           win rate of that fade bet within its own resolved history, and
           _edge_validated means enough of that history exists to trust the
           estimate.
    """
    row_list = sorted([dict(r) for r in rows], key=_row_sort_ts)
    start = time.perf_counter()
    with tracer.start_as_current_span("accounting.validated_kelly") as span:
        span.set_attributes({
            "items.count": len(row_list),
            "account.value_method": "bid_liquidation",
            "strategy.kelly_fraction": float(kelly_fraction),
            "strategy.max_concentration": float(max_concentration),
            "strategy.market_shrinkage": float(market_shrinkage),
            "strategy.max_drawdown": float(max_drawdown) if max_drawdown is not None else -1.0,
            "strategy.require_validated": bool(require_validated),
            "strategy.follow_model_call": bool(follow_model_call),
            "strategy.fade": bool(fade),
        })
        try:
            account = PredictionMarketAccount(starting_cash=starting_cash)
            drawdown_limit = (
                max(0.0, min(float(max_drawdown), 0.99))
                if max_drawdown is not None else None
            )
            peak_account_value = float(starting_cash)
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
                    settlements_by_market.setdefault((platform, ident), {
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
            skipped_not_validated = 0
            skipped_unexecutable = 0
            skipped_drawdown_limit = 0
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
                    peak_account_value = max(peak_account_value, float(snap["account_value"]))
                    continue

                row = payload
                if market_key in settled_markets:
                    continue
                model_side = _row_side(row) if follow_model_call else _row_edge_side(row, min_edge)
                if not model_side:
                    skipped_no_edge += 1
                    continue
                side = _opposite(model_side) if fade else model_side
                if require_validated and not row.get("_edge_validated"):
                    skipped_not_validated += 1
                    continue
                quote = MarketQuote.from_mapping(row)
                current_quotes.setdefault((platform, ident), quote)
                ask = quote.ask(side)
                if ask is None or ask <= 0.0 or ask >= 1.0:
                    skipped_unexecutable += 1
                    logger.warning("skipping validated-kelly trade without executable ask")
                    continue
                if ask < min_price or ask > max_price:
                    # A blended/aggregate calibrated probability is only as
                    # good as how well it represents the *specific* case it's
                    # applied to. Near a price extreme (e.g. a $0.001 "will
                    # this happen" longshot), the market's own price is
                    # already a far stronger, more specific signal than the
                    # bucket-wide average -- applying that average there
                    # produced multi-hundred-thousand-contract "lottery
                    # ticket" positions in testing (capped in dollars, but a
                    # near-certain full loss of that capped stake, repeatedly).
                    skipped_unexecutable += 1
                    continue

                # Size against the *live* account value (cash + open positions
                # marked to market) so earlier gains/losses genuinely compound
                # into this stake, then cap at max_concentration.
                liq_value, _illiquid = account.liquidation_value(current_quotes)
                account_value = account.cash + liq_value
                peak_account_value = max(peak_account_value, account_value)
                model_p = float(
                    row.get("calibrated_model_probability")
                    if row.get("calibrated_model_probability") is not None
                    else (row.get("model_probability") or 0.5)
                )
                market_p = float(row.get("market_probability") or 0.5)
                p_side_mkt = market_p if side == YES else (1.0 - market_p)
                p_side_model = model_p if side == YES else (1.0 - model_p)
                shrinkage = max(0.0, min(float(market_shrinkage), 1.0))
                p_win = p_side_model + shrinkage * (p_side_mkt - p_side_model)
                if fixed_fraction is None:
                    raw_kelly = _kelly_fraction(p_win, p_side_mkt)
                    sized_fraction = max(0.0, min(kelly_fraction * raw_kelly, max_concentration))
                else:
                    sized_fraction = max(0.0, min(float(fixed_fraction), max_concentration))
                if sized_fraction <= 1e-9 or account_value <= 0:
                    skipped_no_edge += 1
                    continue
                target_contracts = (sized_fraction * account_value) / ask

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

                notional = buy_qty * ask
                explicit_fee = row.get("fee")
                def _fee_for(
                    notional_value: float,
                    explicit_fee: Any = explicit_fee,
                    platform: str = platform,
                    ask: float = ask,
                    fee_fn: Optional[Callable[[Any, float, float], float]] = fee_fn,
                ) -> float:
                    return float(explicit_fee) if explicit_fee not in (None, "") else (
                        float(fee_fn(platform, notional_value, ask)) if fee_fn else 0.0
                    )

                if drawdown_limit is not None:
                    # Gate on *total* account value (cash + open positions
                    # marked to market), not cash alone -- cash alone falls
                    # every time a position opens even though that value
                    # never left the account, which falsely and permanently
                    # locked out accounts simply for holding several
                    # concurrent positions. Halt new positions only once
                    # realized/marked value has actually drawn down past the
                    # floor; trading resumes automatically once it recovers.
                    drawdown_floor = peak_account_value * (1.0 - drawdown_limit)
                    if account_value < drawdown_floor:
                        skipped_drawdown_limit += 1
                        continue
                fee = _fee_for(notional)
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
                validated_kelly_trades.add(1, {"platform": platform.lower() or "unknown"})
                snap = account.snapshot(current_quotes, ts=row.get("snapshot_ts") or row.get("ts"))
                snap["event_type"] = "trade"
                snap["trade_status"] = fill.settlement_status
                snap["event_market"] = {"platform": platform, "ident": ident, "side": side}
                snap["sized_fraction"] = round(sized_fraction, 6)
                value_curve.append(snap)
                peak_account_value = max(peak_account_value, float(snap["account_value"]))

            final = account.snapshot(current_quotes)
            if trade_count > 0:
                # Extend the curve to right now even when every position has
                # already settled -- otherwise an account that's fully in
                # cash freezes its last plotted point at its last trade/
                # settlement time, which can be days old, and looks
                # (wrongly) stale/behind next to accounts still holding open
                # positions (which always get marked to the current moment).
                mtm_snap = dict(final)
                mtm_snap["event_type"] = "mark_to_market"
                value_curve.append(mtm_snap)
            final.update({
                "strategy": strategy_name or (
                    "fade_kelly_ledger" if fade else
                    ("validated_kelly_ledger" if require_validated else "capped_half_kelly_ledger")
                ),
                "starting_cash": round(starting_cash, 6),
                "return": round((final["account_value"] / starting_cash) - 1.0, 6)
                if starting_cash else None,
                "kelly_fraction": round(float(kelly_fraction), 6),
                "fixed_fraction": (
                    round(max(0.0, float(fixed_fraction)), 6)
                    if fixed_fraction is not None else None
                ),
                "max_concentration": round(float(max_concentration), 6),
                "market_shrinkage": round(max(0.0, min(float(market_shrinkage), 1.0)), 6),
                "max_drawdown": drawdown_limit,
                "min_edge": round(float(min_edge), 6),
                "min_price": round(float(min_price), 6),
                "max_price": round(float(max_price), 6),
                "fade": bool(fade),
                "n_trades": trade_count,
                "n_settlements": len(settlements),
                "n_open_positions": len(final["open_positions"]),
                "n_illiquid_positions": len(final["illiquid_positions"]),
                "n_skipped_no_edge": skipped_no_edge,
                "n_skipped_not_validated": skipped_not_validated,
                "n_skipped_unexecutable": skipped_unexecutable,
                "n_skipped_drawdown_limit": skipped_drawdown_limit,
                "trades": [t.as_dict() for t in account.trades],
                "settlements": settlements,
                "value_curve": value_curve,
                "notes": [
                    "Sized by fractional Kelly against the live account value -- gains "
                    "and losses compound into the next stake, capped at max_concentration.",
                    (
                        "Fades the model: trades the *opposite* of its implied side, "
                        "in the regime proven to have significantly negative model "
                        "skill -- no other bets are taken."
                        if fade else
                        (
                            "Only trades disagreement buckets with proven, statistically "
                            "significant historical edge (edge_calibration skill_significant); "
                            "no other bets are taken regardless of raw disagreement size."
                            if require_validated else
                            "Follows every executable model call using walk-forward calibrated "
                            "probabilities; this is a historical sizing comparison, not a "
                            "validated live-trading recommendation."
                        )
                    ),
                    f"Skips trades priced outside [{min_price}, {max_price}]: a blended "
                    "calibrated probability isn't specific enough to size a bet near a "
                    "price extreme, where the market's own price is already a far "
                    "stronger signal than the bucket-wide average.",
                    "Open positions are valued at bid-side liquidation prices.",
                    "Resolved markets settle into cash once per market before later trades are considered.",
                    (
                        f"New positions halt once total account value draws down "
                        f"{drawdown_limit:.0%} from its high-watermark peak, resuming "
                        "automatically once value recovers above that floor."
                        if drawdown_limit is not None else
                        "No portfolio-level drawdown floor is configured."
                    ),
                    (
                        f"Sizing shrinks calibrated probabilities {max(0.0, min(float(market_shrinkage), 1.0)):.0%} "
                        "toward the executable market price."
                    ),
                    "Kalshi-style exits are represented as buying the opposite contract and netting YES/NO pairs at $1.00.",
                ],
            })
            validated_kelly_runs.add(1, {"outcome": "success"})
            span.set_attributes({
                "outcome": "success",
                "trades.count": trade_count,
                "settlements.count": len(settlements),
                "positions.open_count": final["n_open_positions"],
            })
            return final
        except Exception as exc:
            validated_kelly_runs.add(1, {"outcome": "failure"})
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            raise
        finally:
            validated_kelly_duration.record(
                time.perf_counter() - start,
                {"operation": "validated_kelly"},
            )


def simulate_market_follow_mark_to_market_account(
    rows: Iterable[Mapping[str, Any]],
    *,
    latest_quotes: Optional[Mapping[Any, MarketQuote | Mapping[str, Any]]] = None,
    starting_cash: float = 10_000.0,
    target_contracts: float = 1.0,
    fee_fn: Optional[Callable[[Any, float, float], float]] = None,
    max_trades: Optional[int] = None,
) -> Dict[str, Any]:
    """Simulate an active crowd baseline that buys the market-favored side."""
    baseline_rows: List[Dict[str, Any]] = []
    skipped_no_direction = 0
    for row in rows:
        item = dict(row)
        market_probability = _price(item.get("market_probability"))
        if market_probability is None or market_probability == 0.5:
            skipped_no_direction += 1
            continue
        item["model_probability"] = 1.0 if market_probability > 0.5 else 0.0
        baseline_rows.append(item)

    with tracer.start_as_current_span("accounting.market_follow_mark_to_market") as span:
        span.set_attributes({
            "items.count": len(baseline_rows),
            "items.skipped_no_direction": skipped_no_direction,
            "account.value_method": "bid_liquidation",
        })
        try:
            account = simulate_shadow_mark_to_market_account(
                baseline_rows,
                latest_quotes=latest_quotes,
                starting_cash=starting_cash,
                target_contracts=target_contracts,
                min_edge=0.0,
                fee_fn=fee_fn,
                max_trades=max_trades,
            )
            account["strategy"] = "market_follow_baseline"
            account["n_skipped_no_direction"] = skipped_no_direction
            account["notes"] = [
                "Active MTM baseline: buys the market-favored side at snapshot time.",
                *account.get("notes", []),
            ]
            market_follow_mark_to_market_runs.add(1, {"outcome": "success"})
            span.set_attributes({
                "outcome": "success",
                "trades.count": int(account.get("n_trades") or 0),
                "positions.open_count": int(account.get("n_open_positions") or 0),
            })
            return account
        except Exception as exc:
            market_follow_mark_to_market_runs.add(1, {"outcome": "failure"})
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            raise


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
