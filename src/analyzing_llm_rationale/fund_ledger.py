"""The Foresea Fund's paper ledger: one USD account holding every kind of position.

Two kinds of position, valued differently (docs/fund.md):

- binary: a Yes/No event contract (Kalshi, Polymarket). It costs price x quantity and is worth
  the side's bid x quantity, as the agent accounts have always been marked.
- linear: a future or perpetual (NSE/BSE, MCX, Kalshi perps), long or short. Opening one sets
  aside its full notional in USD as collateral, so the fund never borrows; it is worth that
  collateral plus its profit, sign x (mark - entry) x quantity x multiplier, in USD at the
  current USD/INR for rupee contracts.

Every fill records why it happened: the analysts behind it and their weights (attribution),
its purpose (a view or a hedge), the trade note and whether that note is the model's own words
or the plain fact sheet, and the evidence the note may cite. Positions carry a quantity-weighted
attribution so profit and loss can be split back to the analysts that caused it.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

#: A position at or below this size is closed (float dust from partial fills).
MIN_QUANTITY = 1e-9
LINEAR_SIDES = ("LONG", "SHORT")
BINARY_SIDES = ("YES", "NO")


class LedgerError(ValueError):
    pass


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fund_account (
            fund_id TEXT PRIMARY KEY,
            starting_value REAL NOT NULL,
            cash REAL NOT NULL,
            realized_pnl REAL NOT NULL DEFAULT 0,
            fees_paid REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fund_positions (
            fund_id TEXT NOT NULL,
            instrument_id TEXT NOT NULL,
            side TEXT NOT NULL,
            kind TEXT NOT NULL,
            venue TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            underlying TEXT NOT NULL,
            key TEXT NOT NULL,
            label TEXT,
            currency TEXT NOT NULL,
            multiplier REAL NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            collateral_usd REAL NOT NULL,
            attribution_json TEXT NOT NULL DEFAULT '{}',
            pair_key TEXT,
            opened_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (fund_id, instrument_id, side)
        );
        CREATE TABLE IF NOT EXISTS fund_trades (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            fund_id TEXT NOT NULL,
            cycle_id TEXT,
            instrument_id TEXT NOT NULL,
            side TEXT NOT NULL,
            action TEXT NOT NULL,
            purpose TEXT NOT NULL DEFAULT 'view',
            pair_key TEXT,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            fx REAL NOT NULL DEFAULT 1,
            notional_usd REAL NOT NULL,
            fee_usd REAL NOT NULL DEFAULT 0,
            cash_delta_usd REAL NOT NULL,
            realized_pnl_usd REAL NOT NULL DEFAULT 0,
            attribution_json TEXT NOT NULL DEFAULT '{}',
            note TEXT,
            note_source TEXT,
            evidence_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS fund_nav (
            ts TEXT NOT NULL,
            fund_id TEXT NOT NULL,
            nav_usd REAL NOT NULL,
            cash_usd REAL NOT NULL,
            detail_json TEXT NOT NULL,
            PRIMARY KEY (fund_id, ts)
        );
    """)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _kind(side: str) -> str:
    side = side.upper()
    if side in LINEAR_SIDES:
        return "linear"
    if side in BINARY_SIDES:
        return "binary"
    raise LedgerError(f"unknown side {side!r}")


def _usd(amount: float, currency: str, fx: float) -> float:
    """Convert ``amount`` in ``currency`` to USD; ``fx`` is units of that currency per USD."""
    if currency == "USD":
        return amount
    if not fx or fx <= 0:
        raise LedgerError(f"no USD rate for {currency}")
    return amount / fx


def open_fund(conn: sqlite3.Connection, fund_id: str, starting_value: float, *, cash: Optional[float] = None,
              now: Optional[str] = None) -> None:
    ensure_schema(conn)
    if conn.execute("SELECT 1 FROM fund_account WHERE fund_id = ?", (fund_id,)).fetchone():
        raise LedgerError(f"fund {fund_id!r} already exists")
    ts = now or _now()
    conn.execute("INSERT INTO fund_account (fund_id, starting_value, cash, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                 (fund_id, starting_value, starting_value if cash is None else cash, ts, ts))


def _merge_attribution(old: Mapping[str, float], old_qty: float, new: Mapping[str, float], new_qty: float) -> dict[str, float]:
    total = old_qty + new_qty
    names = set(old) | set(new)
    merged = {n: (old.get(n, 0.0) * old_qty + new.get(n, 0.0) * new_qty) / total for n in names} if total > 0 else dict(new)
    norm = sum(merged.values())
    return {n: w / norm for n, w in merged.items() if w > 0} if norm > 0 else {}


def _record(conn: sqlite3.Connection, fund_id: str, instrument: Mapping[str, Any], side: str, action: str, *,
            quantity: float, price: float, fx: float, notional_usd: float, fee_usd: float, cash_delta: float,
            realized: float, attribution: Mapping[str, float], purpose: str, pair_key: Optional[str],
            cycle_id: Optional[str], note: Optional[str], note_source: Optional[str],
            evidence: Sequence[Mapping[str, Any]], now: str) -> str:
    trade_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO fund_trades (id, ts, fund_id, cycle_id, instrument_id, side, action, purpose, pair_key, quantity,"
        " price, fx, notional_usd, fee_usd, cash_delta_usd, realized_pnl_usd, attribution_json, note, note_source,"
        " evidence_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (trade_id, now, fund_id, cycle_id, instrument["id"], side, action, purpose, pair_key, quantity, price, fx,
         notional_usd, fee_usd, cash_delta, realized, json.dumps(dict(attribution), sort_keys=True), note,
         note_source, json.dumps(list(evidence), sort_keys=True)))
    conn.execute("UPDATE fund_account SET cash = cash + ?, realized_pnl = realized_pnl + ?, fees_paid = fees_paid + ?,"
                 " updated_at = ? WHERE fund_id = ?", (cash_delta, realized, fee_usd, now, fund_id))
    return trade_id


def buy(conn: sqlite3.Connection, fund_id: str, instrument: Mapping[str, Any], side: str, quantity: float,
        price: float, *, fx: float = 1.0, fee_usd: float = 0.0, attribution: Mapping[str, float] = (),
        purpose: str = "view", pair_key: Optional[str] = None, cycle_id: Optional[str] = None,
        note: Optional[str] = None, note_source: Optional[str] = None,
        evidence: Sequence[Mapping[str, Any]] = (), action: str = "open", now: Optional[str] = None) -> str:
    """Open or add to ``side`` of ``instrument`` (a fund_universe.describe() dict or alike).

    Binary sides cost price x quantity; linear sides set aside their notional as collateral.
    Fails, changing nothing, if cash would go negative.
    """
    side = side.upper()
    kind = _kind(side)
    if quantity <= MIN_QUANTITY or price <= 0:
        raise LedgerError("a fill needs a positive quantity and price")
    currency = instrument.get("currency", "USD")
    multiplier = float(instrument.get("multiplier") or 1.0)
    notional = _usd(price * quantity * multiplier, currency, fx)
    account = conn.execute("SELECT cash FROM fund_account WHERE fund_id = ?", (fund_id,)).fetchone()
    if account is None:
        raise LedgerError(f"no fund {fund_id!r}")
    if account["cash"] < notional + fee_usd - 1e-9:
        raise LedgerError(f"{instrument['id']}: needs ${notional + fee_usd:,.2f}, cash is ${account['cash']:,.2f}")
    ts = now or _now()
    attribution = dict(attribution or {})
    held = conn.execute("SELECT * FROM fund_positions WHERE fund_id = ? AND instrument_id = ? AND side = ?",
                        (fund_id, instrument["id"], side)).fetchone()
    if held is None:
        conn.execute(
            "INSERT INTO fund_positions (fund_id, instrument_id, side, kind, venue, asset_class, underlying, key, label,"
            " currency, multiplier, quantity, entry_price, collateral_usd, attribution_json, pair_key, opened_at,"
            " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fund_id, instrument["id"], side, kind, instrument.get("venue", ""), instrument.get("asset_class", ""),
             instrument.get("underlying", ""), instrument.get("key", instrument["id"]), instrument.get("label"),
             currency, multiplier, quantity, price, notional, json.dumps(attribution, sort_keys=True), pair_key,
             ts, ts))
    else:
        total = held["quantity"] + quantity
        merged = _merge_attribution(json.loads(held["attribution_json"]), held["quantity"], attribution, quantity)
        conn.execute(
            "UPDATE fund_positions SET quantity = ?, entry_price = ?, collateral_usd = collateral_usd + ?,"
            " attribution_json = ?, updated_at = ? WHERE fund_id = ? AND instrument_id = ? AND side = ?",
            (total, (held["entry_price"] * held["quantity"] + price * quantity) / total, notional,
             json.dumps(merged, sort_keys=True), ts, fund_id, instrument["id"], side))
        action = "add" if action == "open" else action
    return _record(conn, fund_id, instrument, side, action, quantity=quantity, price=price, fx=fx,
                   notional_usd=notional, fee_usd=fee_usd, cash_delta=-(notional + fee_usd), realized=-fee_usd,
                   attribution=attribution, purpose=purpose, pair_key=pair_key, cycle_id=cycle_id, note=note,
                   note_source=note_source, evidence=evidence, now=ts)


def position_pnl_usd(position: Mapping[str, Any], mark: float, fx: float, quantity: Optional[float] = None) -> float:
    """Profit of ``quantity`` (default all) of a position at ``mark``, in USD."""
    qty = position["quantity"] if quantity is None else quantity
    if position["kind"] == "binary":
        return qty * mark - position["collateral_usd"] * qty / position["quantity"]
    sign = 1.0 if position["side"] == "LONG" else -1.0
    gain = sign * (mark - position["entry_price"]) * qty * position["multiplier"]
    return _usd(gain, position["currency"], fx)


def sell(conn: sqlite3.Connection, fund_id: str, instrument_id: str, side: str, quantity: float, price: float, *,
         fx: float = 1.0, fee_usd: float = 0.0, purpose: str = "view", pair_key: Optional[str] = None,
         cycle_id: Optional[str] = None, note: Optional[str] = None, note_source: Optional[str] = None,
         evidence: Sequence[Mapping[str, Any]] = (), action: Optional[str] = None,
         now: Optional[str] = None) -> str:
    """Reduce or close ``side`` of a held instrument at ``price``; realizes its share of profit."""
    side = side.upper()
    held = conn.execute("SELECT * FROM fund_positions WHERE fund_id = ? AND instrument_id = ? AND side = ?",
                        (fund_id, instrument_id, side)).fetchone()
    if held is None or held["quantity"] <= MIN_QUANTITY:
        raise LedgerError(f"no {side} position in {instrument_id}")
    qty = min(quantity, held["quantity"])
    share = qty / held["quantity"]
    collateral = held["collateral_usd"] * share
    pnl = position_pnl_usd(held, price, fx, qty)
    ts = now or _now()
    remaining = held["quantity"] - qty
    if remaining <= MIN_QUANTITY:
        conn.execute("DELETE FROM fund_positions WHERE fund_id = ? AND instrument_id = ? AND side = ?",
                     (fund_id, instrument_id, side))
    else:
        conn.execute("UPDATE fund_positions SET quantity = ?, collateral_usd = collateral_usd - ?, updated_at = ?"
                     " WHERE fund_id = ? AND instrument_id = ? AND side = ?",
                     (remaining, collateral, ts, fund_id, instrument_id, side))
    instrument = {"id": instrument_id}
    return _record(conn, fund_id, instrument, side, action or ("close" if remaining <= MIN_QUANTITY else "reduce"),
                   quantity=qty, price=price, fx=fx, notional_usd=collateral + pnl, fee_usd=fee_usd,
                   cash_delta=collateral + pnl - fee_usd, realized=pnl - fee_usd,
                   attribution=json.loads(held["attribution_json"]), purpose=purpose, pair_key=pair_key,
                   cycle_id=cycle_id, note=note, note_source=note_source, evidence=evidence, now=ts)


def positions(conn: sqlite3.Connection, fund_id: str) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM fund_positions WHERE fund_id = ? AND quantity > ? ORDER BY asset_class, instrument_id",
        (fund_id, MIN_QUANTITY))]


def mark_for(position: Mapping[str, Any], quote: Optional[Mapping[str, Any]]) -> Optional[float]:
    """The price a position would close at: the bid for anything held long (a binary side
    included), the ask for a short; the last or mid price when the book is one-sided."""
    if not quote:
        return None
    if position["kind"] == "binary":
        return quote.get("no_bid" if position["side"] == "NO" else "yes_bid", quote.get("bid"))
    near = quote.get("bid") if position["side"] == "LONG" else quote.get("ask")
    return near if near is not None else (quote.get("last") if quote.get("last") is not None else quote.get("mid"))


def nav(conn: sqlite3.Connection, fund_id: str, quotes: Mapping[str, Mapping[str, Any]],
        fx: Mapping[str, float]) -> dict[str, Any]:
    """Account value at ``quotes`` (by instrument id), with exposure and P&L split by asset
    class, underlying and analyst. An unquoted position is held at its collateral."""
    account = conn.execute("SELECT * FROM fund_account WHERE fund_id = ?", (fund_id,)).fetchone()
    if account is None:
        raise LedgerError(f"no fund {fund_id!r}")
    rows, value, by_class, exposure, by_analyst = [], account["cash"], {}, {}, {}
    for pos in positions(conn, fund_id):
        rate = fx.get(pos["currency"], 1.0)
        mark = mark_for(pos, quotes.get(pos["instrument_id"]))
        pnl = position_pnl_usd(pos, mark, rate) if mark is not None else 0.0
        worth = pos["collateral_usd"] + pnl
        value += worth
        by_class[pos["asset_class"]] = by_class.get(pos["asset_class"], 0.0) + worth
        sign = -1.0 if pos["side"] == "SHORT" else 1.0
        exposure[pos["underlying"]] = exposure.get(pos["underlying"], 0.0) + sign * pos["collateral_usd"]
        for analyst, weight in json.loads(pos["attribution_json"]).items():
            by_analyst[analyst] = by_analyst.get(analyst, 0.0) + weight * pnl
        rows.append({**{k: pos[k] for k in ("instrument_id", "label", "side", "kind", "venue", "asset_class",
                                            "underlying", "quantity", "entry_price", "collateral_usd", "pair_key",
                                            "opened_at")},
                     "mark": mark, "value_usd": round(worth, 4), "pnl_usd": round(pnl, 4), "marked": mark is not None,
                     "attribution": json.loads(pos["attribution_json"])})
    realized_by_analyst = {}
    for trade in conn.execute("SELECT attribution_json, realized_pnl_usd FROM fund_trades WHERE fund_id = ?"
                              " AND realized_pnl_usd != 0", (fund_id,)):
        for analyst, weight in json.loads(trade["attribution_json"]).items():
            realized_by_analyst[analyst] = realized_by_analyst.get(analyst, 0.0) + weight * trade["realized_pnl_usd"]
    return {
        "fund_id": fund_id, "nav_usd": round(value, 4), "cash_usd": round(account["cash"], 4),
        "starting_value": account["starting_value"], "realized_pnl_usd": round(account["realized_pnl"], 4),
        "positions": rows, "by_asset_class": {k: round(v, 4) for k, v in by_class.items()},
        "net_exposure_by_underlying": {k: round(v, 4) for k, v in exposure.items()},
        "unrealized_pnl_by_analyst": {k: round(v, 4) for k, v in by_analyst.items()},
        "realized_pnl_by_analyst": {k: round(v, 4) for k, v in realized_by_analyst.items()},
    }


def record_nav(conn: sqlite3.Connection, snapshot: Mapping[str, Any], *, now: Optional[str] = None) -> None:
    conn.execute("INSERT OR REPLACE INTO fund_nav (ts, fund_id, nav_usd, cash_usd, detail_json) VALUES (?, ?, ?, ?, ?)",
                 (now or _now(), snapshot["fund_id"], snapshot["nav_usd"], snapshot["cash_usd"],
                  json.dumps(dict(snapshot), sort_keys=True, default=str)))
