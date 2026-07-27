"""Deterministic evaluation and portfolio accounting for resolved forecasts."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence


def _as_utc(value: datetime | str, *, field: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _probability(value: Any, *, field: str) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return probability


@dataclass(frozen=True)
class ResolvedForecast:
    forecast_id: str
    platform: str
    market_id: str
    model: str
    forecasted_at: datetime
    resolved_at: datetime
    model_probability: float
    market_probability: float
    outcome: int
    domain: str = "other"
    horizon: str = "unknown"
    market_bid: float | None = None
    market_ask: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "forecasted_at", _as_utc(self.forecasted_at, field="forecasted_at")
        )
        object.__setattr__(self, "resolved_at", _as_utc(self.resolved_at, field="resolved_at"))
        object.__setattr__(
            self,
            "model_probability",
            _probability(self.model_probability, field="model_probability"),
        )
        object.__setattr__(
            self,
            "market_probability",
            _probability(self.market_probability, field="market_probability"),
        )
        if self.outcome not in {0, 1}:
            raise ValueError("outcome must be binary")
        if self.resolved_at < self.forecasted_at:
            raise ValueError("resolved_at cannot precede forecasted_at")
        for name in ("market_bid", "market_ask"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _probability(value, field=name))

    @classmethod
    def from_ledger(cls, row: Mapping[str, Any]) -> "ResolvedForecast":
        return cls(
            forecast_id=str(row["forecast_id"]),
            platform=str(row["platform"]),
            market_id=str(row["ident"]),
            model=str(row["model"]),
            forecasted_at=row["forecasted_at"],
            resolved_at=row["resolved_at"],
            model_probability=row["model_probability"],
            market_probability=row["market_probability"],
            outcome=int(row["outcome"]),
            domain=str(row.get("domain") or "other"),
            horizon=str(row.get("horizon") or "unknown"),
            market_bid=row.get("market_bid"),
            market_ask=row.get("market_ask"),
        )


@dataclass(frozen=True)
class Trade:
    trade_id: str
    opened_at: datetime
    settled_at: datetime
    side: str
    entry_price: float
    outcome: int
    requested_fraction: float
    fee_fraction: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "opened_at", _as_utc(self.opened_at, field="opened_at"))
        object.__setattr__(self, "settled_at", _as_utc(self.settled_at, field="settled_at"))
        if self.side not in {"YES", "NO"}:
            raise ValueError("side must be YES or NO")
        if not 0.0 < self.entry_price < 1.0:
            raise ValueError("entry_price must be strictly between 0 and 1")
        if self.outcome not in {0, 1}:
            raise ValueError("outcome must be binary")
        if self.settled_at <= self.opened_at:
            raise ValueError("settled_at must be after opened_at")
        if not 0.0 < self.requested_fraction <= 1.0:
            raise ValueError("requested_fraction must be in (0, 1]")
        if self.fee_fraction < 0.0:
            raise ValueError("fee_fraction cannot be negative")

    @property
    def won(self) -> bool:
        return (self.outcome == 1) == (self.side == "YES")


def brier_score(forecasts: Sequence[ResolvedForecast], *, market: bool = False) -> float | None:
    if not forecasts:
        return None
    field = "market_probability" if market else "model_probability"
    return sum((getattr(row, field) - row.outcome) ** 2 for row in forecasts) / len(forecasts)


def log_loss(forecasts: Sequence[ResolvedForecast], *, epsilon: float = 1e-12) -> float | None:
    if not forecasts:
        return None
    total = 0.0
    for row in forecasts:
        probability = min(1.0 - epsilon, max(epsilon, row.model_probability))
        total -= row.outcome * math.log(probability)
        total -= (1 - row.outcome) * math.log(1.0 - probability)
    return total / len(forecasts)


def reliability_buckets(
    forecasts: Sequence[ResolvedForecast], *, bins: int = 10
) -> list[dict[str, Any]]:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    grouped: list[list[ResolvedForecast]] = [[] for _ in range(bins)]
    for row in forecasts:
        index = min(int(row.model_probability * bins), bins - 1)
        grouped[index].append(row)

    result: list[dict[str, Any]] = []
    for index, rows in enumerate(grouped):
        if not rows:
            continue
        mean_probability = sum(row.model_probability for row in rows) / len(rows)
        outcome_rate = sum(row.outcome for row in rows) / len(rows)
        result.append(
            {
                "bucket": f"{index / bins:.1f}-{(index + 1) / bins:.1f}",
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "n": len(rows),
                "mean_probability": mean_probability,
                "outcome_rate": outcome_rate,
                "calibration_gap": outcome_rate - mean_probability,
            }
        )
    return result


def domain_probability_buckets(
    forecasts: Sequence[ResolvedForecast],
    *,
    bins: int = 10,
    prior_strength: float = 20.0,
    min_domain_n: int = 30,
) -> list[dict[str, Any]]:
    """Estimate domain buckets with shrinkage toward each global bucket."""
    if prior_strength < 0.0:
        raise ValueError("prior_strength cannot be negative")
    global_buckets = {
        int(min(row["lower"] * bins, bins - 1)): row
        for row in reliability_buckets(forecasts, bins=bins)
    }
    grouped: dict[tuple[str, int], list[ResolvedForecast]] = {}
    domain_counts: dict[str, int] = {}
    for row in forecasts:
        index = min(int(row.model_probability * bins), bins - 1)
        grouped.setdefault((row.domain, index), []).append(row)
        domain_counts[row.domain] = domain_counts.get(row.domain, 0) + 1

    result: list[dict[str, Any]] = []
    for (domain, index), rows in sorted(grouped.items()):
        global_bucket = global_buckets[index]
        successes = sum(row.outcome for row in rows)
        global_rate = float(global_bucket["outcome_rate"])
        shrunk_rate = (
            successes + prior_strength * global_rate
        ) / (len(rows) + prior_strength)
        result.append(
            {
                "domain": domain,
                "bucket": global_bucket["bucket"],
                "n": len(rows),
                "domain_n": domain_counts[domain],
                "eligible_for_domain_model": domain_counts[domain] >= min_domain_n,
                "mean_probability": sum(row.model_probability for row in rows) / len(rows),
                "raw_outcome_rate": successes / len(rows),
                "shrunk_outcome_rate": shrunk_rate,
                "global_outcome_rate": global_rate,
                "prior_strength": prior_strength,
            }
        )
    return result


def evaluation_report(
    forecasts: Sequence[ResolvedForecast],
    *,
    bins: int = 10,
    prior_strength: float = 20.0,
    min_domain_n: int = 30,
) -> dict[str, Any]:
    model_brier = brier_score(forecasts)
    market_brier = brier_score(forecasts, market=True)
    return {
        "n": len(forecasts),
        "model_brier": model_brier,
        "market_brier": market_brier,
        "skill_vs_market": (
            market_brier - model_brier
            if market_brier is not None and model_brier is not None
            else None
        ),
        "log_loss": log_loss(forecasts),
        "calibration": reliability_buckets(forecasts, bins=bins),
        "domain_probability_buckets": domain_probability_buckets(
            forecasts,
            bins=bins,
            prior_strength=prior_strength,
            min_domain_n=min_domain_n,
        ),
    }


def _execution(forecast: ResolvedForecast) -> tuple[str, float]:
    if forecast.model_probability >= forecast.market_probability:
        price = (
            forecast.market_ask
            if forecast.market_ask is not None
            else forecast.market_probability
        )
        return "YES", price
    yes_bid = (
        forecast.market_bid
        if forecast.market_bid is not None
        else forecast.market_probability
    )
    return "NO", 1.0 - yes_bid


def build_trades(
    forecasts: Iterable[ResolvedForecast],
    *,
    min_edge: float = 0.05,
    requested_fraction: float = 0.02,
    fee_fraction: float = 0.0,
    one_per_market_model: bool = True,
) -> list[Trade]:
    if min_edge < 0.0:
        raise ValueError("min_edge cannot be negative")
    ordered = sorted(
        forecasts,
        key=lambda row: (row.forecasted_at, row.forecast_id),
    )
    seen: set[tuple[str, str, str]] = set()
    trades: list[Trade] = []
    for forecast in ordered:
        cohort = (forecast.platform, forecast.market_id, forecast.model)
        if one_per_market_model and cohort in seen:
            continue
        side, entry_price = _execution(forecast)
        model_side_probability = (
            forecast.model_probability
            if side == "YES"
            else 1.0 - forecast.model_probability
        )
        edge = model_side_probability - entry_price
        if edge < min_edge or not 0.0 < entry_price < 1.0:
            continue
        seen.add(cohort)
        trades.append(
            Trade(
                trade_id=forecast.forecast_id,
                opened_at=forecast.forecasted_at,
                settled_at=forecast.resolved_at,
                side=side,
                entry_price=entry_price,
                outcome=forecast.outcome,
                requested_fraction=requested_fraction,
                fee_fraction=fee_fraction,
            )
        )
    return trades


def simulate_compounded_portfolio(
    trades: Sequence[Trade],
    *,
    initial_bankroll: float = 100.0,
    max_total_exposure: float = 0.25,
) -> dict[str, Any]:
    """Run cash accounting with settlements and entries ordered in event time."""
    if initial_bankroll <= 0.0:
        raise ValueError("initial_bankroll must be positive")
    if not 0.0 < max_total_exposure <= 1.0:
        raise ValueError("max_total_exposure must be in (0, 1]")
    trade_ids = [trade.trade_id for trade in trades]
    if len(trade_ids) != len(set(trade_ids)):
        raise ValueError("trade_id values must be unique")

    events: list[tuple[datetime, int, str, Trade]] = []
    for trade in trades:
        events.append((trade.opened_at, 1, trade.trade_id, trade))
        events.append((trade.settled_at, 0, trade.trade_id, trade))
    events.sort(key=lambda event: (event[0], event[1], event[2]))

    cash = float(initial_bankroll)
    open_cost = 0.0
    positions: dict[str, tuple[Trade, float, float]] = {}
    total_fees = 0.0
    skipped = 0
    peak_equity = float(initial_bankroll)
    max_drawdown = 0.0
    curve: list[dict[str, Any]] = []

    for event_at, event_priority, trade_id, trade in events:
        if event_priority == 0:
            position = positions.pop(trade_id, None)
            if position is None:
                continue
            _, stake, _fee = position
            open_cost -= stake
            if trade.won:
                cash += stake / trade.entry_price
            event_type = "settlement"
        else:
            equity = cash + open_cost
            exposure_limit = max_total_exposure * equity
            exposure_available = max(0.0, exposure_limit - open_cost)
            desired_stake = trade.requested_fraction * equity
            affordable_stake = cash / (1.0 + trade.fee_fraction)
            stake = min(desired_stake, exposure_available, affordable_stake)
            if stake <= 1e-12:
                skipped += 1
                continue
            fee = stake * trade.fee_fraction
            cash -= stake + fee
            open_cost += stake
            total_fees += fee
            positions[trade_id] = (trade, stake, fee)
            event_type = "entry"

        equity = cash + open_cost
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0.0
        max_drawdown = max(max_drawdown, drawdown)
        curve.append(
            {
                "ts": event_at.isoformat(),
                "event": event_type,
                "trade_id": trade_id,
                "cash": round(cash, 8),
                "open_exposure": round(open_cost, 8),
                "equity": round(equity, 8),
            }
        )

    final_equity = cash + open_cost
    return {
        "initial_bankroll": initial_bankroll,
        "final_bankroll": final_equity,
        "compound_return": final_equity / initial_bankroll - 1.0,
        "max_drawdown": max_drawdown,
        "total_fees": total_fees,
        "n_trades": len(trades),
        "n_opened": len(trades) - skipped,
        "n_skipped_for_exposure": skipped,
        "open_positions": len(positions),
        "equity_curve": curve,
    }
