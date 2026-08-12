"""Model-facing benchmark tools for prediction-market trading agents.

The tool names intentionally match the benchmark spec:

- place_trade: Kalshi YES/NO buy tool. Default mode is shadow, not live funds.
- web_search: multi-source news search (news_pipeline.NewsPipeline) with a
  small blacklist. Uses SCADS_AI_API_KEY, already required elsewhere in this
  codebase -- no separate key is needed.
- manage_notes: bounded persistent notes per agent.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlparse

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("foresea.benchmark_tools")
meter = metrics.get_meter("foresea.benchmark_tools")

tool_calls = meter.create_counter("benchmark_tools.calls", unit="1")
tool_duration = meter.create_histogram("benchmark_tools.duration", unit="s")
trade_actions = meter.create_counter("benchmark_tools.place_trade.actions", unit="1")
note_actions = meter.create_counter("benchmark_tools.manage_notes.actions", unit="1")
risk_guard_checks = meter.create_counter("benchmark_tools.risk_guard.checks", unit="1")
risk_guard_rejections = meter.create_counter("benchmark_tools.risk_guard.rejections", unit="1")
settlement_actions = meter.create_counter("benchmark_tools.settlements", unit="1")
fill_actions = meter.create_counter("benchmark_tools.place_trade.fills", unit="1")

BLACKLISTED_WEB_DOMAINS = ("coinmarketcap.com",)
MAX_NOTES_PER_AGENT = 50
MAX_NOTE_CHARS = 1200
WEB_SEARCH_SOURCES = ("web", "gdelt", "google-news", "rss", "newsapi", "open-meteo")
WEB_SEARCH_TOP_K = 5
DEFAULT_AGENT_ACCOUNT_VALUE = 10_000.0
DEFAULT_CONCENTRATION_LIMIT = 0.15
DEFAULT_PER_CYCLE_SPEND_LIMIT = 500.0
DEFAULT_CYCLE_MINUTES = 15
KALSHI_FEE_COEFFICIENT = 0.07
DEFAULT_SETTLEMENT_FEE_RATE = 0.014
IMMEDIATE_TIME_IN_FORCE = "immediate_or_cancel"


@dataclass(frozen=True)
class ToolContext:
    agent_id: str
    user_id: Optional[str] = None
    model: Optional[str] = None


@dataclass(frozen=True)
class RiskGuardPolicy:
    account_value: float
    concentration_limit: float
    per_cycle_spend_limit: float
    cycle_id: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded_agent_id(agent_id: str) -> str:
    cleaned = "".join(ch for ch in str(agent_id or "agent") if ch.isalnum() or ch in "-_.:")[:120]
    return cleaned or "agent"


def _notes_path() -> Path:
    return Path(
        os.environ.get("FORESEA_AGENT_NOTES_PATH")
        or Path(os.environ.get("TMPDIR", "/tmp")) / "foresea_agent_notes.json"
    )


def _ledger_path() -> Optional[Path]:
    raw = os.environ.get("FORESEA_AGENT_TOOL_LEDGER_PATH")
    if raw:
        return Path(raw)
    return Path(os.environ.get("TMPDIR", "/tmp")) / "foresea_agent_tool_ledger.jsonl"


def _account_db_path() -> Path:
    raw = os.environ.get("FORESEA_AGENT_ACCOUNT_DB_PATH")
    if raw:
        return Path(raw)
    return Path(os.environ.get("TMPDIR", "/tmp")) / "foresea_agent_accounts.sqlite"


def _load_notes(path: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    target = path or _notes_path()
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("agent notes file could not be read; starting empty", exc_info=True)
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for agent, notes in data.items():
        if isinstance(notes, list):
            out[str(agent)] = [dict(n) for n in notes if isinstance(n, dict)]
    return out


def _save_notes(data: Mapping[str, List[Dict[str, Any]]], path: Optional[Path] = None) -> None:
    target = path or _notes_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


def _record_ledger(event: Mapping[str, Any]) -> None:
    path = _ledger_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True) + "\n")


def _metric_attrs(tool: str, outcome: str) -> Dict[str, str]:
    return {"tool": tool, "outcome": outcome}


def _finish_tool(tool: str, start: float, outcome: str) -> None:
    attrs = _metric_attrs(tool, outcome)
    tool_calls.add(1, attrs)
    tool_duration.record(time.perf_counter() - start, attrs)


def _clean_side(value: Any) -> str:
    side = str(value or "").strip().lower()
    if side in {"yes", "y"}:
        return "yes"
    if side in {"no", "n"}:
        return "no"
    raise ValueError("side must be 'yes' or 'no'")


def _clean_ticker(value: Any) -> str:
    ticker = str(value or "").strip().upper()
    if not ticker:
        raise ValueError("ticker is required")
    if len(ticker) > 120:
        raise ValueError("ticker must be at most 120 characters")
    return ticker


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _current_cycle_id() -> str:
    explicit = str(os.environ.get("FORESEA_AGENT_CYCLE_ID") or "").strip()
    if explicit:
        return explicit[:120]
    minutes = int(_env_float("FORESEA_AGENT_CYCLE_MINUTES", DEFAULT_CYCLE_MINUTES) or DEFAULT_CYCLE_MINUTES)
    if minutes <= 0:
        raise ValueError("FORESEA_AGENT_CYCLE_MINUTES must be greater than 0")
    now = datetime.now(timezone.utc)
    slot = int(now.timestamp()) // (minutes * 60)
    return f"{minutes}m:{slot}"


def _risk_guard_policy() -> RiskGuardPolicy:
    account_value = _env_float("FORESEA_AGENT_ACCOUNT_VALUE", DEFAULT_AGENT_ACCOUNT_VALUE)
    if account_value <= 0:
        raise ValueError("FORESEA_AGENT_ACCOUNT_VALUE must be greater than 0")
    concentration_limit = _env_float("FORESEA_AGENT_CONCENTRATION_LIMIT", DEFAULT_CONCENTRATION_LIMIT)
    if not 0 < concentration_limit <= 1:
        raise ValueError("FORESEA_AGENT_CONCENTRATION_LIMIT must be between 0 and 1")
    per_cycle_spend_limit = _env_float(
        "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT",
        DEFAULT_PER_CYCLE_SPEND_LIMIT,
    )
    return RiskGuardPolicy(
        account_value=account_value,
        concentration_limit=concentration_limit,
        per_cycle_spend_limit=per_cycle_spend_limit,
        cycle_id=_current_cycle_id(),
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _kalshi_fee(price: float, quantity: float) -> float:
    return max(0.0, KALSHI_FEE_COEFFICIENT * quantity * price * (1.0 - price))


def _settlement_fee_rate() -> float:
    return _env_float("FORESEA_AGENT_SETTLEMENT_FEE_RATE", DEFAULT_SETTLEMENT_FEE_RATE)


def _order_fee(args: Mapping[str, Any], normalized: Mapping[str, Any]) -> float:
    for source in (args, normalized):
        for key in ("fee", "estimated_fee", "kalshi_fee"):
            if source.get(key) not in (None, ""):
                fee = float(source[key])
                if fee < 0:
                    raise ValueError("fee must be non-negative")
                return fee
    return _kalshi_fee(
        _as_float(normalized.get("price")),
        _as_float(normalized.get("quantity")),
    )


def _immediate_order_adjustments(args: Mapping[str, Any]) -> List[str]:
    warnings: List[str] = []
    requested_tif = str(args.get("time_in_force") or "").strip().lower()
    if requested_tif and requested_tif != IMMEDIATE_TIME_IN_FORCE:
        warnings.append(
            f"time_in_force={requested_tif!r} ignored; benchmark place_trade uses immediate_or_cancel only."
        )
    if bool(args.get("post_only", False)):
        warnings.append("post_only=true ignored; benchmark place_trade never posts resting orders.")
    requested_order_type = str(args.get("order_type") or "").strip().lower()
    if requested_order_type and requested_order_type != "limit":
        warnings.append("order_type ignored; benchmark place_trade uses immediate limit orders.")
    return warnings


def _nested_values(payload: Any) -> Iterable[Any]:
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            yield from _nested_values(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _nested_values(value)


def _first_float(mapping: Mapping[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            try:
                return float(mapping[key])
            except (TypeError, ValueError):
                continue
    return None


def _extract_filled_quantity(result: Mapping[str, Any], normalized: Mapping[str, Any], *, live: bool) -> tuple[float, str]:
    requested = _as_float(normalized.get("quantity"))
    if not live:
        return requested, "shadow_assumed_full"
    body = ((result.get("venue_response") or {}).get("body") or {})
    filled_keys = (
        "filled_quantity",
        "filled_size",
        "fill_quantity",
        "fill_size",
        "filled_count",
        "fill_count",
        "executed_quantity",
        "executed_size",
        "executed_count",
        "count_filled",
    )
    remaining_keys = (
        "remaining_quantity",
        "remaining_size",
        "remaining_count",
        "unfilled_quantity",
        "unfilled_size",
        "unfilled_count",
    )
    for item in _nested_values(body):
        if not isinstance(item, dict):
            continue
        filled = _first_float(item, filled_keys)
        if filled is not None:
            return max(0.0, min(requested, filled)), "venue_reported"
        remaining = _first_float(item, remaining_keys)
        if remaining is not None:
            return max(0.0, min(requested, requested - remaining)), "venue_reported_remaining"
    status_text = json.dumps(body, sort_keys=True).lower()[:3000]
    if any(token in status_text for token in ("filled", "executed")):
        return requested, "venue_status_assumed_full"
    if any(token in status_text for token in ("canceled", "cancelled", "expired", "rejected")):
        return 0.0, "venue_status_assumed_zero"
    return requested, "venue_unknown_assumed_full"


def _extract_fee_from_result(result: Mapping[str, Any]) -> Optional[float]:
    body = ((result.get("venue_response") or {}).get("body") or {})
    fee_keys = (
        "fee",
        "fees",
        "fee_amount",
        "total_fee",
        "total_fees",
        "taker_fee",
        "exchange_fee",
    )
    for item in _nested_values(body):
        if isinstance(item, dict):
            fee = _first_float(item, fee_keys)
            if fee is not None and fee >= 0:
                return fee
    return None


def _normalize_fill_for_accounting(
    *,
    args: Mapping[str, Any],
    result: Mapping[str, Any],
    normalized: Mapping[str, Any],
    guard: Mapping[str, Any],
    live: bool,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    filled_quantity, fill_status = _extract_filled_quantity(result, normalized, live=live)
    accounting_order = dict(normalized)
    accounting_order["quantity"] = filled_quantity
    accounting_order["estimated_notional"] = round(_as_float(normalized.get("price")) * filled_quantity, 6)
    accounting_guard = dict(guard)
    price = _as_float(accounting_order.get("price"))
    venue_fee = _extract_fee_from_result(result) if live else None
    fee = venue_fee if venue_fee is not None else (
        _order_fee(args, accounting_order) if filled_quantity > 0 else 0.0
    )
    accounting_guard.update({
        "requested_quantity": round(_as_float(normalized.get("quantity")), 6),
        "filled_quantity": round(filled_quantity, 6),
        "fill_status": fill_status,
        "filled_notional": round(price * filled_quantity, 6),
        "filled_fee": round(fee, 6),
        "fee_source": "venue" if venue_fee is not None else "estimated",
    })
    return accounting_order, accounting_guard


def _account_conn() -> sqlite3.Connection:
    path = _account_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _ensure_account_schema(conn)
    return conn


@contextmanager
def _account_transaction() -> Iterable[sqlite3.Connection]:
    conn = _account_conn()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _ensure_account_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS agent_accounts (
            agent_id TEXT PRIMARY KEY,
            starting_cash REAL NOT NULL,
            cash REAL NOT NULL,
            realized_pnl REAL NOT NULL DEFAULT 0,
            fees_paid REAL NOT NULL DEFAULT 0,
            settlement_fees_paid REAL NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS agent_positions (
            agent_id TEXT NOT NULL,
            platform TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            cost_basis REAL NOT NULL,
            avg_entry_price REAL NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (agent_id, platform, ticker, side)
        );
        CREATE TABLE IF NOT EXISTS agent_actions (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            mode TEXT,
            submitted INTEGER NOT NULL DEFAULT 0,
            platform TEXT,
            ticker TEXT,
            side TEXT,
            price REAL,
            quantity REAL,
            notional REAL,
            fee REAL NOT NULL DEFAULT 0,
            settlement_fee REAL NOT NULL DEFAULT 0,
            payout REAL NOT NULL DEFAULT 0,
            netting_payout REAL NOT NULL DEFAULT 0,
            cash_required REAL NOT NULL DEFAULT 0,
            cash_delta REAL NOT NULL DEFAULT 0,
            realized_pnl REAL NOT NULL DEFAULT 0,
            realized_pairs REAL NOT NULL DEFAULT 0,
            cycle_id TEXT,
            client_order_id TEXT,
            outcome TEXT,
            metadata_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_agent_actions_agent_cycle
            ON agent_actions(agent_id, cycle_id, action_type);
        CREATE TABLE IF NOT EXISTS agent_cycle_settlements (
            agent_id TEXT NOT NULL,
            cycle_id TEXT NOT NULL,
            checked_at TEXT NOT NULL,
            PRIMARY KEY (agent_id, cycle_id)
        );
        CREATE TABLE IF NOT EXISTS agent_cycles (
            agent_id TEXT NOT NULL,
            cycle_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            thesis TEXT,
            transcript_json TEXT,
            steps INTEGER,
            truncated INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (agent_id, cycle_id)
        );
        """
    )


def _ensure_agent_account(conn: sqlite3.Connection, agent_id: str, starting_cash: float) -> None:
    now = _now()
    conn.execute(
        """
        INSERT OR IGNORE INTO agent_accounts
            (agent_id, starting_cash, cash, realized_pnl, fees_paid, settlement_fees_paid, updated_at)
        VALUES (?, ?, ?, 0, 0, 0, ?)
        """,
        (agent_id, starting_cash, starting_cash, now),
    )


def _account_row(conn: sqlite3.Connection, agent_id: str, starting_cash: float) -> sqlite3.Row:
    _ensure_agent_account(conn, agent_id, starting_cash)
    row = conn.execute(
        "SELECT * FROM agent_accounts WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("agent account could not be initialized")
    return row


def _account_summary(conn: sqlite3.Connection, agent_id: str, starting_cash: float) -> Dict[str, Any]:
    row = _account_row(conn, agent_id, starting_cash)
    positions = [
        dict(p)
        for p in conn.execute(
            """
            SELECT platform, ticker, side, quantity, cost_basis, avg_entry_price
            FROM agent_positions
            WHERE agent_id = ? AND quantity > 0
            ORDER BY platform, ticker, side
            """,
            (agent_id,),
        )
    ]
    open_cost_basis = sum(float(p["cost_basis"]) for p in positions)
    return {
        "cash": round(float(row["cash"]), 6),
        "starting_cash": round(float(row["starting_cash"]), 6),
        "realized_pnl": round(float(row["realized_pnl"]), 6),
        "fees_paid": round(float(row["fees_paid"]), 6),
        "settlement_fees_paid": round(float(row["settlement_fees_paid"]), 6),
        "open_cost_basis": round(open_cost_basis, 6),
        "n_open_positions": len(positions),
        "open_positions": [
            {
                **p,
                "quantity": round(float(p["quantity"]), 6),
                "cost_basis": round(float(p["cost_basis"]), 6),
                "avg_entry_price": round(float(p["avg_entry_price"]), 6),
            }
            for p in positions
        ],
    }


def _upsert_position(
    conn: sqlite3.Connection,
    *,
    agent_id: str,
    platform: str,
    ticker: str,
    side: str,
    quantity: float,
    cost_basis: float,
) -> None:
    now = _now()
    if quantity <= 1e-12:
        conn.execute(
            "DELETE FROM agent_positions WHERE agent_id = ? AND platform = ? AND ticker = ? AND side = ?",
            (agent_id, platform, ticker, side),
        )
        return
    conn.execute(
        """
        INSERT INTO agent_positions
            (agent_id, platform, ticker, side, quantity, cost_basis, avg_entry_price, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(agent_id, platform, ticker, side) DO UPDATE SET
            quantity = excluded.quantity,
            cost_basis = excluded.cost_basis,
            avg_entry_price = excluded.avg_entry_price,
            updated_at = excluded.updated_at
        """,
        (agent_id, platform, ticker, side, quantity, cost_basis, cost_basis / quantity, now),
    )


def _opposite_side(side: str) -> str:
    return "no" if side == "yes" else "yes"


def _record_account_action(
    conn: sqlite3.Connection,
    *,
    agent_id: str,
    action_type: str,
    cycle_id: str,
    mode: Optional[str] = None,
    submitted: bool = False,
    platform: Optional[str] = None,
    ticker: Optional[str] = None,
    side: Optional[str] = None,
    price: Optional[float] = None,
    quantity: Optional[float] = None,
    notional: float = 0.0,
    fee: float = 0.0,
    settlement_fee: float = 0.0,
    payout: float = 0.0,
    netting_payout: float = 0.0,
    cash_required: float = 0.0,
    cash_delta: float = 0.0,
    realized_pnl: float = 0.0,
    realized_pairs: float = 0.0,
    client_order_id: Optional[str] = None,
    outcome: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    action_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO agent_actions (
            id, ts, agent_id, action_type, mode, submitted, platform, ticker, side,
            price, quantity, notional, fee, settlement_fee, payout, netting_payout,
            cash_required, cash_delta, realized_pnl, realized_pairs, cycle_id,
            client_order_id, outcome, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            action_id,
            _now(),
            agent_id,
            action_type,
            mode,
            int(bool(submitted)),
            platform,
            ticker,
            side,
            price,
            quantity,
            notional,
            fee,
            settlement_fee,
            payout,
            netting_payout,
            cash_required,
            cash_delta,
            realized_pnl,
            realized_pairs,
            cycle_id,
            client_order_id,
            outcome,
            json.dumps(dict(metadata or {}), sort_keys=True),
        ),
    )
    return action_id


def _record_rejected_account_action(
    *,
    agent_id: str,
    mode: str,
    ticker: str,
    side: str,
    normalized: Mapping[str, Any],
    guard: Mapping[str, Any],
) -> None:
    with _account_transaction() as conn:
        _ensure_agent_account(conn, agent_id, _as_float(guard.get("account_value"), DEFAULT_AGENT_ACCOUNT_VALUE))
        _record_account_action(
            conn,
            agent_id=agent_id,
            action_type="rejected_trade",
            mode=mode,
            submitted=False,
            platform="kalshi",
            ticker=ticker,
            side=side,
            price=_as_float(normalized.get("price")),
            quantity=_as_float(normalized.get("quantity")),
            notional=_as_float(guard.get("notional")),
            fee=_as_float(guard.get("fee")),
            netting_payout=_as_float(guard.get("netting_payout")),
            cash_required=_as_float(guard.get("cash_required")),
            cash_delta=_as_float(guard.get("cash_delta")),
            realized_pairs=_as_float(guard.get("netting_payout")),
            cycle_id=str(guard.get("cycle_id") or ""),
            client_order_id=normalized.get("exchange_order", {}).get("client_order_id"),
            outcome="rejected",
            metadata={"risk_guard": dict(guard)},
        )


def _apply_trade_to_account_tables(
    *,
    agent_id: str,
    policy: RiskGuardPolicy,
    mode: str,
    submitted: bool,
    ticker: str,
    side: str,
    normalized: Mapping[str, Any],
    guard: Mapping[str, Any],
) -> Dict[str, Any]:
    platform = "kalshi"
    opposite_side = _opposite_side(side)
    price = _as_float(normalized.get("price"))
    quantity = _as_float(normalized.get("quantity"))
    fee = _as_float(guard.get("filled_fee", guard.get("fee")))
    notional = price * quantity
    now = _now()
    with _account_transaction() as conn:
        row = _account_row(conn, agent_id, policy.account_value)
        cash_before = float(row["cash"])
        realized_pnl_before = float(row["realized_pnl"])
        fees_paid_before = float(row["fees_paid"])
        remaining = quantity
        realized_pairs = 0.0
        realized_pnl = 0.0
        opposite = conn.execute(
            """
            SELECT * FROM agent_positions
            WHERE agent_id = ? AND platform = ? AND ticker = ? AND side = ?
            """,
            (agent_id, platform, ticker, opposite_side),
        ).fetchone()
        if opposite is not None:
            opposite_qty = float(opposite["quantity"])
            if opposite_qty > 1e-12:
                realized_pairs = min(remaining, opposite_qty)
                old_basis = float(opposite["avg_entry_price"]) * realized_pairs
                new_basis = price * realized_pairs
                fee_alloc = fee * (realized_pairs / quantity) if quantity else 0.0
                realized_pnl = realized_pairs - old_basis - new_basis - fee_alloc
                new_opposite_qty = opposite_qty - realized_pairs
                new_opposite_basis = max(0.0, float(opposite["cost_basis"]) - old_basis)
                _upsert_position(
                    conn,
                    agent_id=agent_id,
                    platform=platform,
                    ticker=ticker,
                    side=opposite_side,
                    quantity=new_opposite_qty,
                    cost_basis=new_opposite_basis,
                )
                remaining -= realized_pairs

        if remaining > 1e-12:
            same = conn.execute(
                """
                SELECT * FROM agent_positions
                WHERE agent_id = ? AND platform = ? AND ticker = ? AND side = ?
                """,
                (agent_id, platform, ticker, side),
            ).fetchone()
            same_qty = float(same["quantity"]) if same is not None else 0.0
            same_basis = float(same["cost_basis"]) if same is not None else 0.0
            _upsert_position(
                conn,
                agent_id=agent_id,
                platform=platform,
                ticker=ticker,
                side=side,
                quantity=same_qty + remaining,
                cost_basis=same_basis + (remaining * price),
            )

        cash_delta = -notional - fee + realized_pairs
        cash_required = max(0.0, -cash_delta)
        cash_after = cash_before + cash_delta
        conn.execute(
            """
            UPDATE agent_accounts
            SET cash = ?, realized_pnl = ?, fees_paid = ?, updated_at = ?
            WHERE agent_id = ?
            """,
            (
                cash_after,
                realized_pnl_before + realized_pnl,
                fees_paid_before + fee,
                now,
                agent_id,
            ),
        )
        action_id = _record_account_action(
            conn,
            agent_id=agent_id,
            action_type="trade",
            mode=mode,
            submitted=submitted,
            platform=platform,
            ticker=ticker,
            side=side,
            price=price,
            quantity=quantity,
            notional=notional,
            fee=fee,
            netting_payout=realized_pairs,
            cash_required=cash_required,
            cash_delta=cash_delta,
            realized_pnl=realized_pnl,
            realized_pairs=realized_pairs,
            cycle_id=policy.cycle_id,
            client_order_id=normalized.get("exchange_order", {}).get("client_order_id"),
            outcome="realized" if realized_pairs > 0 else "open",
            metadata={"risk_guard": dict(guard)},
        )
        summary = _account_summary(conn, agent_id, policy.account_value)
    return {
        "action_id": action_id,
        "notional": round(notional, 6),
        "fee": round(fee, 6),
        "cash_required": round(cash_required, 6),
        "cash_delta": round(cash_delta, 6),
        "netting_payout": round(realized_pairs, 6),
        "realized_pairs": round(realized_pairs, 6),
        "realized_pnl": round(realized_pnl, 6),
        "account": summary,
    }


def _settle_agent_open_positions(agent_id: str, policy: RiskGuardPolicy) -> List[Dict[str, Any]]:
    """Settle resolved Kalshi markets once per agent/cycle before trading."""
    settled: List[Dict[str, Any]] = []
    with tracer.start_as_current_span("benchmark_tools.settle_agent_positions") as span:
        span.set_attributes({"agent.id": agent_id, "cycle.id": policy.cycle_id})
        try:
            from analyzing_llm_rationale import market_data

            with _account_transaction() as conn:
                _ensure_agent_account(conn, agent_id, policy.account_value)
                already = conn.execute(
                    """
                    SELECT 1 FROM agent_cycle_settlements
                    WHERE agent_id = ? AND cycle_id = ?
                    """,
                    (agent_id, policy.cycle_id),
                ).fetchone()
                if already is not None:
                    span.set_attribute("settlement.skipped", True)
                    return []
                markets = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT platform, ticker
                        FROM agent_positions
                        WHERE agent_id = ? AND quantity > 0
                        GROUP BY platform, ticker
                        """,
                        (agent_id,),
                    )
                ]
                for market in markets:
                    platform = str(market["platform"] or "").lower()
                    ticker = str(market["ticker"] or "").upper()
                    if platform != "kalshi" or not ticker:
                        continue
                    try:
                        outcome_value = market_data.resolve_kalshi(ticker)
                    except Exception:
                        logger.warning("agent settlement lookup failed ticker=%s", ticker, exc_info=True)
                        continue
                    if outcome_value is None:
                        continue
                    winning_side = "yes" if int(outcome_value) == 1 else "no"
                    positions = [
                        dict(row)
                        for row in conn.execute(
                            """
                            SELECT * FROM agent_positions
                            WHERE agent_id = ? AND platform = ? AND ticker = ?
                            """,
                            (agent_id, platform, ticker),
                        )
                    ]
                    if not positions:
                        continue
                    settled_contracts = sum(float(p["quantity"]) for p in positions)
                    settled_basis = sum(float(p["cost_basis"]) for p in positions)
                    payout = sum(
                        float(p["quantity"]) for p in positions if str(p["side"]) == winning_side
                    )
                    settlement_fee = payout * _settlement_fee_rate()
                    cash_delta = payout - settlement_fee
                    realized_pnl = cash_delta - settled_basis
                    row = _account_row(conn, agent_id, policy.account_value)
                    conn.execute(
                        """
                        UPDATE agent_accounts
                        SET cash = ?,
                            realized_pnl = ?,
                            settlement_fees_paid = ?,
                            updated_at = ?
                        WHERE agent_id = ?
                        """,
                        (
                            float(row["cash"]) + cash_delta,
                            float(row["realized_pnl"]) + realized_pnl,
                            float(row["settlement_fees_paid"]) + settlement_fee,
                            _now(),
                            agent_id,
                        ),
                    )
                    conn.execute(
                        """
                        DELETE FROM agent_positions
                        WHERE agent_id = ? AND platform = ? AND ticker = ?
                        """,
                        (agent_id, platform, ticker),
                    )
                    action_id = _record_account_action(
                        conn,
                        agent_id=agent_id,
                        action_type="settlement",
                        platform=platform,
                        ticker=ticker,
                        side=winning_side,
                        quantity=settled_contracts,
                        settlement_fee=settlement_fee,
                        payout=payout,
                        cash_delta=cash_delta,
                        realized_pnl=realized_pnl,
                        cycle_id=policy.cycle_id,
                        outcome=winning_side,
                        metadata={"settled_basis": round(settled_basis, 6)},
                    )
                    settled.append({
                        "action_id": action_id,
                        "ticker": ticker,
                        "outcome": winning_side,
                        "settled_contracts": round(settled_contracts, 6),
                        "payout": round(payout, 6),
                        "settlement_fee": round(settlement_fee, 6),
                        "realized_pnl": round(realized_pnl, 6),
                        "cash_delta": round(cash_delta, 6),
                    })
                conn.execute(
                    """
                    INSERT OR REPLACE INTO agent_cycle_settlements
                        (agent_id, cycle_id, checked_at)
                    VALUES (?, ?, ?)
                    """,
                    (agent_id, policy.cycle_id, _now()),
                )
            settlement_actions.add(len(settled), {"outcome": "success"})
            span.set_attributes({"settlement.count": len(settled), "outcome": "success"})
            return settled
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            settlement_actions.add(1, {"outcome": "failure"})
            logger.warning("agent settlement pass failed", exc_info=True)
            return []


def _iter_ledger_events() -> Iterable[Dict[str, Any]]:
    path = _ledger_path()
    if path is None or not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("skipping malformed agent trade ledger line")
                    continue
                if isinstance(event, dict):
                    events.append(event)
    except Exception:
        logger.warning("agent trade ledger could not be read; starting empty", exc_info=True)
    return events


def _market_cost_basis(account: Any, ticker: str) -> float:
    return sum(
        float(pos.cost_basis)
        for pos in account.open_positions()
        if str(pos.platform).lower() == "kalshi" and str(pos.ident).upper() == ticker
    )


def _load_guard_account(agent_id: str, policy: RiskGuardPolicy) -> tuple[Any, float]:
    from analyzing_llm_rationale.accounting import PredictionMarketAccount

    account = PredictionMarketAccount(starting_cash=policy.account_value)
    with _account_transaction() as conn:
        row = _account_row(conn, agent_id, policy.account_value)
        account.cash = float(row["cash"])
        account.realized_pnl = float(row["realized_pnl"])
        account.fees_paid = float(row["fees_paid"])
        for pos in conn.execute(
            """
            SELECT platform, ticker, side, quantity, cost_basis
            FROM agent_positions
            WHERE agent_id = ? AND quantity > 0
            """,
            (agent_id,),
        ):
            loaded = account._position(str(pos["platform"]), str(pos["ticker"]), str(pos["side"]))
            loaded.quantity = float(pos["quantity"])
            loaded.cost_basis = float(pos["cost_basis"])
        cycle_spend = float(
            conn.execute(
                """
                SELECT COALESCE(SUM(cash_required), 0)
                FROM agent_actions
                WHERE agent_id = ? AND cycle_id = ? AND action_type = 'trade'
                """,
                (agent_id, policy.cycle_id),
            ).fetchone()[0]
        )
    return account, cycle_spend


def _check_trade_guards(
    *,
    args: Mapping[str, Any],
    normalized: Mapping[str, Any],
    agent_id: str,
    ticker: str,
    side: str,
) -> tuple[bool, Dict[str, Any], RiskGuardPolicy]:
    policy = _risk_guard_policy()
    settlements = _settle_agent_open_positions(agent_id, policy)
    account, cycle_spend_before = _load_guard_account(agent_id, policy)
    price = _as_float(normalized.get("price"))
    quantity = _as_float(normalized.get("quantity"))
    fee = _order_fee(args, normalized)
    cash_before = float(account.cash)
    fill = account.buy(
        platform="kalshi",
        ident=ticker,
        side=side,
        quantity=quantity,
        price=price,
        fee=fee,
    )
    market_cost_basis_after = _market_cost_basis(account, ticker)
    concentration_cap = policy.account_value * policy.concentration_limit
    cash_required = max(0.0, -float(fill.cash_delta))
    cycle_spend_after = cycle_spend_before + cash_required
    reasons: List[str] = []
    if market_cost_basis_after > concentration_cap + 1e-9:
        reasons.append("concentration_limit")
    if cash_required > cash_before + 1e-9:
        reasons.append("solvency")
    if cycle_spend_after > policy.per_cycle_spend_limit + 1e-9:
        reasons.append("per_cycle_spend")

    detail = {
        "allowed": not reasons,
        "reasons": reasons,
        "account_value": round(policy.account_value, 6),
        "cash_before": round(cash_before, 6),
        "concentration_limit": policy.concentration_limit,
        "concentration_cap": round(concentration_cap, 6),
        "market_cost_basis_after": round(market_cost_basis_after, 6),
        "per_cycle_spend_limit": round(policy.per_cycle_spend_limit, 6),
        "cycle_id": policy.cycle_id,
        "cycle_spend_before": round(cycle_spend_before, 6),
        "cycle_spend_after": round(cycle_spend_after, 6),
        "notional": round(price * quantity, 6),
        "fee": round(fee, 6),
        "netting_payout": round(float(fill.realized_pairs), 6),
        "cash_required": round(cash_required, 6),
        "cash_delta": round(float(fill.cash_delta), 6),
        "settlements_before_trade": settlements,
    }
    outcome = "allowed" if detail["allowed"] else "rejected"
    risk_guard_checks.add(1, {"outcome": outcome})
    if reasons:
        risk_guard_rejections.add(1, {"reason": reasons[0]})
    return detail["allowed"], detail, policy


def place_trade(args: Mapping[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    """Place a Kalshi trade through the benchmark tool surface.

    By default this records a shadow action and returns the normalized Kalshi
    order. Set FORESEA_AGENT_PLACE_TRADE_MODE=live to call the real trading
    path, which still requires the existing trading env gates and credentials.
    """
    start = time.perf_counter()
    tool = "place_trade"
    agent_id = _bounded_agent_id(ctx.agent_id)
    with tracer.start_as_current_span("benchmark_tools.place_trade") as span:
        span.set_attributes({
            "agent.id": agent_id,
            "tool.name": tool,
            "market.venue": "kalshi",
        })
        try:
            from analyzing_llm_rationale import trading

            side = _clean_side(args.get("side") or args.get("outcome"))
            ticker = _clean_ticker(args.get("ticker") or args.get("ident"))
            order = {
                "platform": "kalshi",
                "action": "buy",
                "outcome": side,
                "order_type": "limit",
                "ticker": ticker,
                "price": args.get("price"),
                "quantity": args.get("quantity", 1),
                "time_in_force": IMMEDIATE_TIME_IN_FORCE,
                "post_only": False,
                "client_order_id": str(args.get("client_order_id") or f"foresea-agent-{uuid.uuid4()}"),
            }
            execution_warnings = _immediate_order_adjustments(args)
            for key in ("reduce_only", "cancel_order_on_pause", "subaccount"):
                if key in args:
                    order[key] = args[key]
            mode = str(os.environ.get("FORESEA_AGENT_PLACE_TRADE_MODE", "shadow")).strip().lower()
            if mode not in {"shadow", "live"}:
                raise ValueError("FORESEA_AGENT_PLACE_TRADE_MODE must be 'shadow' or 'live'")

            preview = trading.preview_order(order)
            normalized = preview.get("normalized_order") or {}
            allowed, guard, policy = _check_trade_guards(
                args=args,
                normalized=normalized,
                agent_id=agent_id,
                ticker=ticker,
                side=side,
            )
            if not allowed:
                event = {
                    "ts": _now(),
                    "agent_id": agent_id,
                    "tool": tool,
                    "ok": False,
                    "rejected": True,
                    "rejection_reasons": guard["reasons"],
                    "mode": mode,
                    "submitted": False,
                    "platform": "kalshi",
                    "ticker": ticker,
                    "side": side,
                    "price": normalized.get("price"),
                    "quantity": normalized.get("quantity"),
                    "client_order_id": normalized.get("exchange_order", {}).get("client_order_id"),
                    "cycle_id": guard["cycle_id"],
                    "notional": guard["notional"],
                    "fee": guard["fee"],
                    "netting_payout": guard["netting_payout"],
                    "cash_required": guard["cash_required"],
                    "cash_delta": guard["cash_delta"],
                    "risk_guard": guard,
                }
                _record_ledger(event)
                _record_rejected_account_action(
                    agent_id=agent_id,
                    mode=mode,
                    ticker=ticker,
                    side=side,
                    normalized=normalized,
                    guard=guard,
                )
                span.set_attributes({
                    "outcome": "rejected",
                    "risk_guard.allowed": False,
                    "risk_guard.reason": guard["reasons"][0] if guard["reasons"] else "unknown",
                    "trade.mode": mode,
                    "trade.submitted": False,
                })
                _finish_tool(tool, start, "rejected")
                return {
                    "ok": False,
                    "tool": tool,
                    "rejected": True,
                    "reason": guard["reasons"][0] if guard["reasons"] else "risk_guard",
                    "message": "Trade rejected by benchmark risk guards before execution.",
                    "mode": mode,
                    "submitted": False,
                    "normalized_order": normalized,
                    "risk_guard": guard,
                    "execution": {
                        "immediate_only": True,
                        "time_in_force": IMMEDIATE_TIME_IN_FORCE,
                        "requested_quantity": normalized.get("quantity"),
                        "filled_quantity": 0.0,
                        "fill_status": "rejected_before_execution",
                    },
                    "warnings": execution_warnings + preview.get("warnings", []),
                }

            if mode == "live":
                order.update({"execute": True, "confirmation": trading.CONFIRMATION_PHRASE})
                result = trading.place_order(order, user_id=f"agent:{agent_id}")
                submitted = True
                normalized = result.get("normalized_order") or {}
            else:
                result = preview
                submitted = False
            live = mode == "live"
            accounting_normalized, accounting_guard = _normalize_fill_for_accounting(
                args=args,
                result=result,
                normalized=normalized,
                guard=guard,
                live=live,
            )
            fill_status = str(accounting_guard.get("fill_status") or "unknown")
            filled_quantity = _as_float(accounting_guard.get("filled_quantity"))
            fill_outcome = (
                "none" if filled_quantity <= 1e-12
                else "full" if abs(filled_quantity - _as_float(normalized.get("quantity"))) <= 1e-12
                else "partial"
            )

            account_update = _apply_trade_to_account_tables(
                agent_id=agent_id,
                policy=policy,
                mode=mode,
                submitted=submitted,
                ticker=ticker,
                side=side,
                normalized=accounting_normalized,
                guard=accounting_guard,
            )
            event = {
                "ts": _now(),
                "agent_id": agent_id,
                "tool": tool,
                "ok": True,
                "rejected": False,
                "mode": mode,
                "submitted": submitted,
                "platform": "kalshi",
                "ticker": ticker,
                "side": side,
                "price": normalized.get("price"),
                "quantity": accounting_normalized.get("quantity"),
                "requested_quantity": normalized.get("quantity"),
                "filled_quantity": accounting_guard["filled_quantity"],
                "fill_status": fill_status,
                "client_order_id": normalized.get("exchange_order", {}).get("client_order_id"),
                "cycle_id": accounting_guard["cycle_id"],
                "notional": account_update["notional"],
                "fee": account_update["fee"],
                "netting_payout": account_update["netting_payout"],
                "cash_required": account_update["cash_required"],
                "cash_delta": account_update["cash_delta"],
                "risk_guard": accounting_guard,
            }
            _record_ledger(event)
            trade_actions.add(1, {"mode": mode, "submitted": str(submitted).lower()})
            fill_actions.add(1, {"mode": mode, "outcome": fill_outcome})
            span.set_attributes({
                "outcome": "success",
                "risk_guard.allowed": True,
                "trade.mode": mode,
                "trade.submitted": submitted,
                "trade.fill_outcome": fill_outcome,
                "trade.fill_status": fill_status,
            })
            _finish_tool(tool, start, "success")
            return {
                "ok": True,
                "tool": tool,
                "mode": mode,
                "submitted": submitted,
                "message": (
                    "Live Kalshi order submitted."
                    if submitted
                    else "Shadow Kalshi trade recorded; no exchange order was submitted."
                ),
                "normalized_order": normalized,
                "risk_guard": accounting_guard,
                "execution": {
                    "immediate_only": True,
                    "time_in_force": IMMEDIATE_TIME_IN_FORCE,
                    "requested_quantity": normalized.get("quantity"),
                    "filled_quantity": accounting_guard["filled_quantity"],
                    "fill_status": fill_status,
                    "fill_outcome": fill_outcome,
                    "unfilled_quantity_cancelled": round(
                        max(0.0, _as_float(normalized.get("quantity")) - filled_quantity),
                        6,
                    ),
                },
                "account": account_update["account"],
                "action_id": account_update["action_id"],
                "warnings": execution_warnings + result.get("warnings", []),
            }
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            _finish_tool(tool, start, "failure")
            logger.warning("benchmark place_trade failed", exc_info=True)
            return {"ok": False, "tool": tool, "error": str(exc)}


def _domain_blacklisted(url: str, blacklist: Iterable[str] = BLACKLISTED_WEB_DOMAINS) -> bool:
    host = (urlparse(url).netloc or "").lower()
    return any(host == domain or host.endswith("." + domain) for domain in blacklist)


def web_search(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Run a bounded multi-source news search (the open web, GDELT, Google
    News, RSS, and NewsAPI when NEWSAPI_KEY is set) with blacklist filtering.
    Query planning and per-article summarization use SCADS_AI_API_KEY,
    already required elsewhere in this codebase -- no separate (e.g. OpenAI)
    key is needed."""
    start = time.perf_counter()
    tool = "web_search"
    query = str(args.get("query") or "").strip()
    with tracer.start_as_current_span("benchmark_tools.web_search") as span:
        span.set_attributes({
            "tool.name": tool,
            "query.length": len(query),
            "web.blacklist.count": len(BLACKLISTED_WEB_DOMAINS),
        })
        try:
            if not query:
                raise ValueError("query is required")
            from analyzing_llm_rationale.news_pipeline import NewsPipeline

            pipeline = NewsPipeline(fetch_sources=WEB_SEARCH_SOURCES)
            articles = pipeline.fetch_summarize_rank(query, top_k=WEB_SEARCH_TOP_K)
            sources: List[Dict[str, str]] = []
            blocked: List[Dict[str, Any]] = []
            for article in articles:
                url = str(article.get("url") or "")
                if not url:
                    continue
                entry = {"title": str(article.get("title") or url), "url": url}
                (blocked if _domain_blacklisted(url) else sources).append(entry)
            summary = "\n\n".join(
                f"{article.get('title') or article.get('url')}: {article['summary']}"
                for article in articles
                if article.get("summary") and article.get("url")
                and not _domain_blacklisted(str(article["url"]))
            )
            span.set_attributes({
                "outcome": "success",
                "gen_ai.provider.name": "scads",
                "web.citations.count": len(sources),
                "web.blocked_results.count": len(blocked),
            })
            _finish_tool(tool, start, "success")
            return {
                "ok": True,
                "tool": tool,
                "query": query,
                "summary": summary,
                "sources": sources,
                "blacklisted_domains": list(BLACKLISTED_WEB_DOMAINS),
                "blocked_results": len(blocked),
            }
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            _finish_tool(tool, start, "failure")
            logger.warning("benchmark web_search failed", exc_info=True)
            return {"ok": False, "tool": tool, "error": str(exc)}


def manage_notes(args: Mapping[str, Any], ctx: ToolContext, *, path: Optional[Path] = None) -> Dict[str, Any]:
    """Store, search, edit, list, and delete bounded notes across cycles."""
    start = time.perf_counter()
    tool = "manage_notes"
    agent_id = _bounded_agent_id(ctx.agent_id)
    action = str(args.get("action") or "list").strip().lower()
    with tracer.start_as_current_span("benchmark_tools.manage_notes") as span:
        span.set_attributes({
            "agent.id": agent_id,
            "tool.name": tool,
            "notes.action": action,
        })
        try:
            data = _load_notes(path)
            notes = data.setdefault(agent_id, [])
            now = _now()
            result: Dict[str, Any]
            if action == "add":
                text = str(args.get("text") or "").strip()
                if not text:
                    raise ValueError("text is required for add")
                if len(text) > MAX_NOTE_CHARS:
                    raise ValueError(f"notes are limited to {MAX_NOTE_CHARS} characters")
                if len(notes) >= MAX_NOTES_PER_AGENT:
                    raise ValueError(f"agent already has {MAX_NOTES_PER_AGENT} notes")
                note = {
                    "id": str(args.get("id") or uuid.uuid4()),
                    "text": text,
                    "tags": [str(t)[:40] for t in (args.get("tags") or [])][:10],
                    "created_at": now,
                    "updated_at": now,
                }
                notes.append(note)
                result = {"note": note}
            elif action == "edit":
                note_id = str(args.get("id") or "")
                text = str(args.get("text") or "").strip()
                if not note_id or not text:
                    raise ValueError("id and text are required for edit")
                if len(text) > MAX_NOTE_CHARS:
                    raise ValueError(f"notes are limited to {MAX_NOTE_CHARS} characters")
                note = next((n for n in notes if str(n.get("id")) == note_id), None)
                if note is None:
                    raise ValueError("note not found")
                note["text"] = text
                note["updated_at"] = now
                if "tags" in args:
                    note["tags"] = [str(t)[:40] for t in (args.get("tags") or [])][:10]
                result = {"note": note}
            elif action == "delete":
                note_id = str(args.get("id") or "")
                before = len(notes)
                data[agent_id] = [n for n in notes if str(n.get("id")) != note_id]
                result = {"deleted": before - len(data[agent_id])}
            elif action == "search":
                query = str(args.get("query") or "").strip().lower()
                matches = [
                    n for n in notes
                    if not query
                    or query in str(n.get("text", "")).lower()
                    or any(query in str(t).lower() for t in n.get("tags", []))
                ]
                result = {"notes": matches[:MAX_NOTES_PER_AGENT]}
            elif action == "list":
                result = {"notes": notes[:MAX_NOTES_PER_AGENT]}
            else:
                raise ValueError("action must be one of add, edit, delete, search, list")
            _save_notes(data, path)
            note_actions.add(1, {"action": action, "outcome": "success"})
            span.set_attributes({
                "outcome": "success",
                "notes.count": len(data.get(agent_id, [])),
            })
            _finish_tool(tool, start, "success")
            return {"ok": True, "tool": tool, "action": action, **result}
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            note_actions.add(1, {"action": action or "unknown", "outcome": "failure"})
            _finish_tool(tool, start, "failure")
            if not isinstance(exc, ValueError):
                logger.warning("benchmark manage_notes failed", exc_info=True)
            return {"ok": False, "tool": tool, "action": action, "error": str(exc)}


def observation(payload: Mapping[str, Any]) -> str:
    """Compact JSON string for the ReAct observation channel."""
    return json.dumps(payload, sort_keys=True, default=str)[:4000]
