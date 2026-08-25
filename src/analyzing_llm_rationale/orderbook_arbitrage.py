"""Depth-aware, read-only complement-arbitrage analysis for binary markets.

The scanner deliberately does not submit orders.  It evaluates whether buying
both complementary contracts can be filled below a $1 settlement payout after
an explicit per-leg fee assumption.  Callers must still verify that the two
contracts share identical resolution criteria before treating a result as an
opportunity.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

Level = Tuple[float, float]


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _levels(raw: Any) -> List[Level]:
    """Normalise common CLOB/Kalshi depth formats to ascending price levels."""
    values = raw if isinstance(raw, list) else []
    levels: List[Level] = []
    for value in values:
        if isinstance(value, dict):
            price = _number(value.get("price"))
            size = _number(value.get("size", value.get("quantity")))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            price, size = _number(value[0]), _number(value[1])
        else:
            continue
        if price is not None and size is not None and 0 < price < 1 and size > 0:
            levels.append((price, size))
    return sorted(levels)


def polymarket_ask_levels(orderbook: Dict[str, Any]) -> List[Level]:
    """Return executable ask depth from a Polymarket CLOB orderbook."""
    return _levels(orderbook.get("asks"))


def kalshi_complement_ask_levels(orderbook: Dict[str, Any]) -> Tuple[List[Level], List[Level]]:
    """Convert Kalshi YES/NO bid books into executable YES/NO ask depth.

    A YES ask is the complement of a NO bid, and vice versa.  This makes the
    two-leg calculation use tradable prices rather than midpoint or last price.
    """
    yes_bids = _levels(orderbook.get("yes"))
    no_bids = _levels(orderbook.get("no"))
    yes_asks = sorted((1.0 - price, size) for price, size in no_bids)
    no_asks = sorted((1.0 - price, size) for price, size in yes_bids)
    return yes_asks, no_asks


def scan_complement_arbitrage(
    yes_asks: Iterable[Level],
    no_asks: Iterable[Level],
    *,
    fee_bps_per_leg: float = 0.0,
    min_net_edge: float = 0.0,
    latency_bps_per_leg: float = 0.0,
    requested_quantity: float | None = None,
) -> Dict[str, Any]:
    """Match two outcome books level-by-level using a live-market FOK simulation.

    ``fee_bps_per_leg`` is intentionally caller-supplied: venue fees can vary
    by account and contract.  The returned signal is therefore a candidate, not
    an execution instruction or a claim of guaranteed profit. ``latency_bps_per_leg``
    reserves an adverse-price allowance on both legs. If ``requested_quantity``
    is present, the simulated pair is fill-or-kill: a partial apparent edge is
    not reported as an executable fill.
    """
    fee_bps = max(0.0, float(fee_bps_per_leg))
    min_edge = max(0.0, float(min_net_edge))
    latency_bps = max(0.0, float(latency_bps_per_leg))
    requested = None if requested_quantity is None else float(requested_quantity)
    if requested is not None and requested <= 0:
        raise ValueError("requested_quantity must be greater than 0 when supplied")
    yes = sorted((float(price), float(size)) for price, size in yes_asks)
    no = sorted((float(price), float(size)) for price, size in no_asks)
    yes = [(price, size) for price, size in yes if 0 < price < 1 and size > 0]
    no = [(price, size) for price, size in no if 0 < price < 1 and size > 0]

    matches: List[Dict[str, float]] = []
    yes_index = no_index = 0
    yes_remaining = no_remaining = 0.0
    requested_remaining = requested
    while yes_index < len(yes) and no_index < len(no):
        yes_price, yes_size = yes[yes_index]
        no_price, no_size = no[no_index]
        if yes_remaining <= 0:
            yes_remaining = yes_size
        if no_remaining <= 0:
            no_remaining = no_size
        quantity = min(yes_remaining, no_remaining)
        if requested_remaining is not None:
            quantity = min(quantity, requested_remaining)
        entry_cost = yes_price + no_price
        fees = entry_cost * (fee_bps / 10_000.0)
        latency_cost = entry_cost * ((2.0 * latency_bps) / 10_000.0)
        net_edge = 1.0 - entry_cost - fees - latency_cost
        if net_edge >= min_edge:
            matches.append({
                "quantity": round(quantity, 6),
                "yes_ask": round(yes_price, 6),
                "no_ask": round(no_price, 6),
                "entry_cost": round(entry_cost, 6),
                "fees_per_pair": round(fees, 6),
                "latency_cost_per_pair": round(latency_cost, 6),
                "net_edge_per_pair": round(net_edge, 6),
                "net_profit": round(quantity * net_edge, 6),
            })
            if requested_remaining is not None:
                requested_remaining -= quantity
                if requested_remaining <= 1e-12:
                    requested_remaining = 0.0
                    break
        else:
            # Both books are consumed in ascending-ask order; once a pair no
            # longer clears the net threshold, later paired levels cannot
            # improve it. Do not represent a thin partial sweep as executable.
            break
        yes_remaining -= quantity
        no_remaining -= quantity
        if yes_remaining <= 1e-12:
            yes_index += 1
            yes_remaining = 0.0
        if no_remaining <= 1e-12:
            no_index += 1
            no_remaining = 0.0

    total_quantity = sum(match["quantity"] for match in matches)
    total_profit = sum(match["net_profit"] for match in matches)
    best = matches[0] if matches else None
    fully_fillable = requested is None or total_quantity >= requested - 1e-9
    candidate = bool(matches) and fully_fillable
    return {
        "candidate": candidate,
        "fee_bps_per_leg": fee_bps,
        "min_net_edge": min_edge,
        "latency_bps_per_leg": latency_bps,
        "requested_quantity": requested,
        "fill_policy": "fill_or_kill_pair" if requested is not None else "sweep_available_depth",
        "fully_fillable": fully_fillable,
        "unfilled_quantity": round(max(0.0, (requested or 0.0) - total_quantity), 6),
        "executable_pairs": len(matches),
        "executable_quantity": round(total_quantity, 6),
        "estimated_net_profit": round(total_profit, 6),
        "best_net_edge_per_pair": best["net_edge_per_pair"] if best else None,
        "levels": matches,
        "warning": (
            "Live-market simulation only: based on one orderbook snapshot with a "
            "fill-or-kill paired assumption. Verify identical resolution rules, venue "
            "fees, available balance, and real-world non-atomic leg risk before any order decision."
        ),
    }
