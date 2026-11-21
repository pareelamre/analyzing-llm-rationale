"""Risk limits for the Foresea Fund: the worst a position can lose, and alerts when one is losing.

The owner asked (2026-11-21) that no position blindside the fund. A Kalshi No on "any AI lab
solves a Millennium Prize Problem" was bought at 94 cents for 6% of the fund and lost most of
its value in an afternoon when a proof was claimed. Nothing in the fund said how much any one
bet could lose, or warned when one was losing.

Every position and every new order now has a worst-case loss:

- a prediction-market contract: its whole stake (it pays 0 if it loses);
- an option structure: its stake, which the fund sizes as the most the structure can lose;
- a futures leg: its notional times a severe 20-session move for its asset class
  (STRESS_MOVE; MCX copper fell 12% in the five sessions to 2026-11-20).

New orders are sized so one position's worst case is at most MAX_LOSS_PER_POSITION of the
fund, and so the worst cases of everything held stay under MAX_LOSS_TOTAL. Held positions are
never sold by these rules: selling a prediction market at its bid on a scare locks in the loss
(the Millennium market can still settle No). Instead `review` flags them on the board.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

#: The most one new position may lose, as a share of the fund.
MAX_LOSS_PER_POSITION = 0.03
#: The most everything held may lose together; above it no new risk is added.
MAX_LOSS_TOTAL = 0.40
#: A severe 20-session move for a futures leg, by asset class.
STRESS_MOVE: dict[str, float] = {"commodity": 0.15, "index": 0.12}
DEFAULT_STRESS = 0.15
#: Alerts on held positions.
ALERT_LOSS = 0.02          # losing this share of the fund
ALERT_STAKE_LOST = 0.5     # or this share of its own stake
ALERT_DROP = 0.01          # or fell this share of the fund since the last board


def kind_of(asset_class: str, side: str, option_sides: Iterable[str]) -> str:
    if asset_class == "event":
        return "binary"
    if side in set(option_sides):
        return "option"
    return "linear"


def loss_per_dollar(kind: str, asset_class: str = "") -> float:
    """The worst-case loss for each dollar of stake."""
    if kind in ("binary", "option"):
        return 1.0
    return STRESS_MOVE.get(asset_class, DEFAULT_STRESS)


def position_worst_case(pos: Mapping[str, Any]) -> float:
    """What a held position can still lose from its current value."""
    value = max(0.0, float(pos.get("value_usd") or 0.0))
    return value * loss_per_dollar(str(pos.get("kind") or ""), str(pos.get("asset_class") or ""))


def entry_worst_case(pos: Mapping[str, Any]) -> float:
    stake = max(0.0, float(pos.get("collateral_usd") or 0.0))
    return stake * loss_per_dollar(str(pos.get("kind") or ""), str(pos.get("asset_class") or ""))


def portfolio_worst_case(positions: Iterable[Mapping[str, Any]]) -> float:
    return sum(position_worst_case(p) for p in positions)


def entry_room(nav: float, worst_total: float, per_dollar: float, *, legs: int = 1) -> float:
    """The largest stake a new position may take, per leg, under both limits."""
    if nav <= 0 or per_dollar <= 0:
        return 0.0
    single = MAX_LOSS_PER_POSITION * nav / per_dollar
    left = max(0.0, MAX_LOSS_TOTAL * nav - worst_total) / (per_dollar * legs)
    return min(single, left)


def review(snapshot: Mapping[str, Any], previous: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Each held position's worst case and alerts, for the board. ``previous`` is the last
    published board, for the drop since then."""
    nav = float(snapshot.get("nav_usd") or 0.0)
    before = {(p.get("instrument_id"), p.get("side")): float(p.get("value_usd") or 0.0)
              for p in (previous or {}).get("positions") or []}
    rows, alerts, over_limit = [], [], []
    for pos in snapshot.get("positions") or []:
        worst = position_worst_case(pos)
        at_entry = entry_worst_case(pos)
        pnl = float(pos.get("pnl_usd") or 0.0)
        stake = float(pos.get("collateral_usd") or 0.0)
        flags = []
        if nav > 0 and -pnl >= ALERT_LOSS * nav:
            flags.append(f"losing {-pnl / nav:.1%} of the fund")
        if stake > 0 and -pnl >= ALERT_STAKE_LOST * stake:
            flags.append(f"lost {-pnl / stake:.0%} of its stake")
        key = (pos.get("instrument_id"), pos.get("side"))
        if nav > 0 and key in before:
            drop = before[key] - float(pos.get("value_usd") or 0.0)
            if drop >= ALERT_DROP * nav:
                flags.append(f"fell ${drop:,.0f} since the last update")
        row = {"label": pos.get("label"), "side": pos.get("side"), "kind": pos.get("kind"),
               "worst_case_usd": round(worst, 2), "worst_case_share": round(worst / nav, 4) if nav else None,
               "entry_worst_case_usd": round(at_entry, 2), "pnl_usd": round(pnl, 2), "alerts": flags}
        if nav > 0 and worst > MAX_LOSS_PER_POSITION * nav * 1.0001:
            # Opened before the limit (or grown since): it is shown, not sold.
            over_limit.append({"label": pos.get("label"), "side": pos.get("side"),
                               "worst_case_share": row["worst_case_share"]})
        rows.append(row)
        if flags:
            alerts.append({"label": pos.get("label"), "side": pos.get("side"), "alerts": flags})
    total = sum(r["worst_case_usd"] for r in rows)
    rows.sort(key=lambda r: -r["worst_case_usd"])
    return {
        "limits": {"max_loss_per_position": MAX_LOSS_PER_POSITION, "max_loss_total": MAX_LOSS_TOTAL,
                   "stress_move": STRESS_MOVE, "alert_loss": ALERT_LOSS, "alert_stake_lost": ALERT_STAKE_LOST,
                   "alert_drop": ALERT_DROP},
        "worst_case_total_usd": round(total, 2),
        "worst_case_total_share": round(total / nav, 4) if nav else None,
        "new_risk_paused": bool(nav) and total >= MAX_LOSS_TOTAL * nav,
        "positions": rows, "alerts": alerts,
        "over_limit": sorted(over_limit, key=lambda r: -(r["worst_case_share"] or 0)),
    }
