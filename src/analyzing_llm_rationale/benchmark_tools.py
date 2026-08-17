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
# Deliberately larger than agent_trading_tick.py's order-notional cap (8% of
# account value) so one cycle has room for more than a single max-sized
# order -- a per-cycle cap stricter than the single-order cap would make the
# order cap meaningless. Hit exactly this with the old flat $500/cycle limit
# once the order cap was raised past it.
DEFAULT_PER_CYCLE_SPEND_LIMIT_PCT = 0.20
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


def _clean_ticker(value: Any, *, platform: str = "kalshi") -> str:
    # Kalshi tickers are conventionally uppercase (e.g. "KXFED-25DEC-T");
    # Polymarket idents are lowercase-hyphenated slugs, and every downstream
    # lookup (fetch_polymarket, resolve_polymarket, position/quote keys) is
    # case-sensitive against that exact slug, so only Kalshi gets uppercased.
    ticker = str(value or "").strip()
    if platform == "kalshi":
        ticker = ticker.upper()
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
    per_cycle_spend_limit_pct = _env_float(
        "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT",
        DEFAULT_PER_CYCLE_SPEND_LIMIT_PCT,
    )
    if per_cycle_spend_limit_pct <= 0:
        raise ValueError("FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT must be greater than 0")
    return RiskGuardPolicy(
        account_value=account_value,
        concentration_limit=concentration_limit,
        per_cycle_spend_limit=account_value * per_cycle_spend_limit_pct,
        cycle_id=_current_cycle_id(),
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


# Memoized per process (not a "rate" -> value dict of size >1): a shadow-
# trading cycle is a fresh process each run (see agent_trading_tick.py), so
# this naturally refreshes every ~15 minutes without needing its own
# cache-invalidation logic. The "rate" key's presence (not its value)
# distinguishes "not yet looked up" from "looked up, no credentials/request
# failed" so a misconfigured or rate-limited endpoint isn't retried on every
# trade within one cycle.
_KALSHI_TAKER_FEE_RATE_CACHE: Dict[str, Optional[float]] = {}


def _first_valid_rate(raw: Any) -> Optional[float]:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if 0.0 <= raw < 1.0 else None
    if isinstance(raw, list):
        for entry in raw:
            rate = _first_valid_rate(entry)
            if rate is not None:
                return rate
        return None
    if isinstance(raw, dict):
        for key in ("rate", "fee_rate", "taker_fee_rate", "value"):
            if key in raw:
                rate = _first_valid_rate(raw[key])
                if rate is not None:
                    return rate
    return None


def _parse_taker_fee_rate(tiers: Mapping[str, Any]) -> Optional[float]:
    """Pull a usable taker rate out of /margin/fee_tiers's response.

    place_trade only ever takes liquidity (immediate-or-cancel, no resting
    orders), so taker is always the applicable rate -- maker never applies
    here. The exact response shape (a flat rate vs. a list of volume tiers)
    isn't verified against Kalshi's docs from this codebase; handle both
    defensively and return None rather than guess if nothing parses cleanly.
    """
    raw = tiers.get("taker_fee_rates", tiers.get("taker_fee_rate"))
    return _first_valid_rate(raw)


def _fetch_kalshi_taker_fee_rate() -> Optional[float]:
    try:
        from analyzing_llm_rationale import trading

        tiers = trading.get_kalshi_fee_tiers()
    except Exception:
        logger.warning(
            "Kalshi fee-tiers lookup failed; falling back to the estimated fee formula",
            exc_info=True,
        )
        return None
    rate = _parse_taker_fee_rate(tiers)
    if rate is None:
        logger.warning(
            "Kalshi fee-tiers response had no usable taker rate; falling back to the "
            "estimated fee formula: %r", tiers,
        )
    return rate


def _kalshi_taker_fee_rate() -> Optional[float]:
    if "rate" not in _KALSHI_TAKER_FEE_RATE_CACHE:
        _KALSHI_TAKER_FEE_RATE_CACHE["rate"] = _fetch_kalshi_taker_fee_rate()
    return _KALSHI_TAKER_FEE_RATE_CACHE["rate"]


def _kalshi_fee(price: float, quantity: float) -> float:
    rate = _kalshi_taker_fee_rate()
    if rate is not None:
        # Kalshi's fee-tiers endpoint documents this rate as applied
        # directly to notional, not plugged into the parabolic
        # price*(1-price) shape the flat KALSHI_FEE_COEFFICIENT estimate
        # below approximates.
        return max(0.0, rate * quantity * price)
    return max(0.0, KALSHI_FEE_COEFFICIENT * quantity * price * (1.0 - price))


def _settlement_fee_rate() -> float:
    return _env_float("FORESEA_AGENT_SETTLEMENT_FEE_RATE", DEFAULT_SETTLEMENT_FEE_RATE)


def _order_fee(args: Mapping[str, Any], normalized: Mapping[str, Any], *, platform: str = "kalshi") -> float:
    for source in (args, normalized):
        for key in ("fee", "estimated_fee", "kalshi_fee"):
            if source.get(key) not in (None, ""):
                fee = float(source[key])
                if fee < 0:
                    raise ValueError("fee must be non-negative")
                return fee
    if platform != "kalshi":
        # Polymarket's CLOB charges no per-trade maker/taker fee (unlike
        # Kalshi's tiered taker fee below) -- an explicit fee/estimated_fee
        # arg above still overrides this if a caller ever needs to model one.
        return 0.0
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


def _extract_filled_quantity(
    result: Mapping[str, Any],
    normalized: Mapping[str, Any],
    *,
    live: bool,
    shadow_marketable: bool = True,
    shadow_unfilled_status: str = "shadow_unfilled_below_market",
) -> tuple[float, str]:
    requested = _as_float(normalized.get("quantity"))
    if not live:
        if not shadow_marketable:
            # The requested price never crossed a live quote for this side
            # (or no live quote could be fetched at all), so a real IOC order
            # here would fill zero -- match that instead of assuming a fill
            # at a price no real book confirmed. shadow_unfilled_status
            # distinguishes the two reasons for the caller (see
            # _resolve_shadow_marketability).
            return 0.0, shadow_unfilled_status
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
    shadow_marketable: bool = True,
    shadow_unfilled_status: str = "shadow_unfilled_below_market",
    platform: str = "kalshi",
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    filled_quantity, fill_status = _extract_filled_quantity(
        result, normalized, live=live, shadow_marketable=shadow_marketable,
        shadow_unfilled_status=shadow_unfilled_status,
    )
    accounting_order = dict(normalized)
    accounting_order["quantity"] = filled_quantity
    accounting_order["estimated_notional"] = round(_as_float(normalized.get("price")) * filled_quantity, 6)
    accounting_guard = dict(guard)
    price = _as_float(accounting_order.get("price"))
    venue_fee = _extract_fee_from_result(result) if live else None
    fee = venue_fee if venue_fee is not None else (
        _order_fee(args, accounting_order, platform=platform) if filled_quantity > 0 else 0.0
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
    platform: str = "kalshi",
) -> None:
    if _use_datastore_account_store():
        _ds_record_rejected_account_action(
            agent_id=agent_id, mode=mode, ticker=ticker, side=side,
            normalized=normalized, guard=guard, platform=platform,
        )
        return
    with _account_transaction() as conn:
        _ensure_agent_account(conn, agent_id, _as_float(guard.get("account_value"), DEFAULT_AGENT_ACCOUNT_VALUE))
        _record_account_action(
            conn,
            agent_id=agent_id,
            action_type="rejected_trade",
            mode=mode,
            submitted=False,
            platform=platform,
            ticker=ticker,
            side=side,
            price=_as_float(normalized.get("price")),
            quantity=_as_float(normalized.get("quantity")),
            notional=_as_float(guard.get("notional")),
            fee=_as_float(guard.get("fee")),
            netting_payout=_as_float(guard.get("netting_payout")),
            cash_required=_as_float(guard.get("cash_required")),
            # A rejected order never touches the account -- cash_delta must
            # be 0, not the hypothetical delta the guard preview computed
            # before rejecting it. That hypothetical value is still visible
            # to callers under cash_required/metadata.risk_guard for context.
            cash_delta=0.0,
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
    platform: str = "kalshi",
) -> Dict[str, Any]:
    if _use_datastore_account_store():
        return _ds_apply_trade(
            agent_id=agent_id, policy=policy, mode=mode, submitted=submitted,
            ticker=ticker, side=side, normalized=normalized, guard=guard, platform=platform,
        )
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
    """Settle resolved Kalshi or Polymarket markets once per agent/cycle before trading."""
    if _use_datastore_account_store():
        return _ds_settle_agent_open_positions(agent_id, policy)
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
                    # Kalshi tickers are stored uppercase; Polymarket slugs are
                    # case-sensitive and must round-trip exactly as stored.
                    raw_ticker = str(market["ticker"] or "")
                    ticker = raw_ticker.upper() if platform == "kalshi" else raw_ticker
                    if not ticker or platform not in ("kalshi", "polymarket"):
                        continue
                    try:
                        outcome_value = (
                            market_data.resolve_kalshi(ticker)
                            if platform == "kalshi"
                            else market_data.resolve_polymarket(ticker)
                        )
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


# ---------------------------------------------------------------------------
# Datastore-backed account store
#
# _account_db_path() falls back to a file under /tmp when
# FORESEA_AGENT_ACCOUNT_DB_PATH is unset. That's fine for the scheduled
# GitHub Actions trading-tick workflow (which always sets it explicitly, to
# a file it downloads from and re-uploads to GCS after each run) but not for
# the live Cloud Run service, which has no persistent volume for it: that
# ledger silently resets on every deploy and diverges across concurrent
# instances. When the env var is unset -- the Cloud Run case -- the account
# store below is used instead, backed by Google Cloud Datastore: already
# this app's primary datastore (see server.py's _get_datastore()), already
# provisioned, no new GCP resources or IAM needed. The scheduled workflow's
# GCS-backed SQLite files are untouched by this.
#
# Each _ds_* write path is wrapped in a Datastore transaction (with retry on
# conflict via _ds_run_in_transaction), so concurrent trades against the
# same agent -- the expected case, since agent_id is the model name, not a
# per-session id -- can't corrupt the netting math the way two racing SQLite
# writers could. What this does NOT do is re-validate the risk-guard
# thresholds against post-conflict state inside that transaction; the guard
# decision is still made once, earlier, against a snapshot (same as the
# SQLite path today). So two concurrent trades that both look acceptable
# against a stale snapshot can still both apply -- correctly, with no data
# corruption, just possibly exceeding a limit in aggregate. Closing that
# fully would mean merging the guard check and the apply into one
# transaction, which is a larger control-flow change shared with the SQLite
# path; left as a follow-up rather than risking that rewrite here.
# ---------------------------------------------------------------------------

_DS_ACCOUNT_KIND = "AgentTradingAccount"
_DS_POSITION_KIND = "AgentTradingPosition"
_DS_ACTION_KIND = "AgentTradingAction"
_DS_CYCLE_SETTLEMENT_KIND = "AgentTradingCycleSettlement"
_DS_TRANSACTION_RETRIES = 3

_ds_account_client: Any = None


def _use_datastore_account_store() -> bool:
    return not os.environ.get("FORESEA_AGENT_ACCOUNT_DB_PATH")


def _get_account_datastore() -> Any:
    global _ds_account_client
    if _ds_account_client is None:
        try:
            from google.cloud import datastore as _ds

            _ds_account_client = _ds.Client()
        except Exception:
            logger.warning("agent trading Datastore client unavailable", exc_info=True)
    return _ds_account_client


def _ds_account_key(client: Any, agent_id: str) -> Any:
    return client.key(_DS_ACCOUNT_KIND, agent_id)


def _ds_position_key(client: Any, agent_id: str, platform: str, ticker: str, side: str) -> Any:
    return client.key(_DS_ACCOUNT_KIND, agent_id, _DS_POSITION_KIND, f"{platform}:{ticker}:{side}")


def _ds_action_key(client: Any, agent_id: str, action_id: str) -> Any:
    return client.key(_DS_ACCOUNT_KIND, agent_id, _DS_ACTION_KIND, action_id)


def _ds_cycle_settlement_key(client: Any, agent_id: str, cycle_id: str) -> Any:
    return client.key(_DS_ACCOUNT_KIND, agent_id, _DS_CYCLE_SETTLEMENT_KIND, cycle_id)


def _ds_new_account_entity(client: Any, agent_id: str, starting_cash: float) -> Any:
    from google.cloud import datastore as _ds

    entity = _ds.Entity(key=_ds_account_key(client, agent_id))
    entity.update({
        "starting_cash": starting_cash,
        "cash": starting_cash,
        "realized_pnl": 0.0,
        "fees_paid": 0.0,
        "settlement_fees_paid": 0.0,
        "updated_at": _now(),
    })
    return entity


def _ds_ensure_account(client: Any, agent_id: str, starting_cash: float) -> Any:
    entity = client.get(_ds_account_key(client, agent_id))
    if entity is None:
        entity = _ds_new_account_entity(client, agent_id, starting_cash)
        client.put(entity)
    return entity


def _ds_positions(client: Any, agent_id: str) -> List[Dict[str, Any]]:
    account_key = _ds_account_key(client, agent_id)
    return [
        dict(p)
        for p in client.query(kind=_DS_POSITION_KIND, ancestor=account_key).fetch()
        if float(p.get("quantity") or 0) > 1e-12
    ]


def _ds_account_summary(client: Any, agent_id: str, starting_cash: float) -> Dict[str, Any]:
    account = _ds_ensure_account(client, agent_id, starting_cash)
    positions = sorted(
        _ds_positions(client, agent_id),
        key=lambda p: (str(p.get("platform")), str(p.get("ticker")), str(p.get("side"))),
    )
    open_cost_basis = sum(float(p["cost_basis"]) for p in positions)
    return {
        "cash": round(float(account["cash"]), 6),
        "starting_cash": round(float(account["starting_cash"]), 6),
        "realized_pnl": round(float(account["realized_pnl"]), 6),
        "fees_paid": round(float(account["fees_paid"]), 6),
        "settlement_fees_paid": round(float(account["settlement_fees_paid"]), 6),
        "open_cost_basis": round(open_cost_basis, 6),
        "n_open_positions": len(positions),
        "open_positions": [
            {
                "platform": p["platform"],
                "ticker": p["ticker"],
                "side": p["side"],
                "quantity": round(float(p["quantity"]), 6),
                "cost_basis": round(float(p["cost_basis"]), 6),
                "avg_entry_price": round(float(p["avg_entry_price"]), 6),
            }
            for p in positions
        ],
    }


def _ds_upsert_position(
    client: Any,
    *,
    agent_id: str,
    platform: str,
    ticker: str,
    side: str,
    quantity: float,
    cost_basis: float,
) -> None:
    from google.cloud import datastore as _ds

    key = _ds_position_key(client, agent_id, platform, ticker, side)
    if quantity <= 1e-12:
        client.delete(key)
        return
    entity = _ds.Entity(key=key)
    entity.update({
        "platform": platform,
        "ticker": ticker,
        "side": side,
        "quantity": quantity,
        "cost_basis": cost_basis,
        "avg_entry_price": cost_basis / quantity,
        "updated_at": _now(),
    })
    client.put(entity)


def _ds_record_action(client: Any, *, agent_id: str, action_type: str, cycle_id: str, **fields: Any) -> str:
    from google.cloud import datastore as _ds

    action_id = str(uuid.uuid4())
    entity = _ds.Entity(
        key=_ds_action_key(client, agent_id, action_id),
        exclude_from_indexes=("metadata_json",),
    )
    metadata = fields.pop("metadata", None)
    entity.update({
        "ts": _now(),
        "agent_id": agent_id,
        "action_type": action_type,
        "cycle_id": cycle_id,
        "mode": fields.get("mode"),
        "submitted": bool(fields.get("submitted", False)),
        "platform": fields.get("platform"),
        "ticker": fields.get("ticker"),
        "side": fields.get("side"),
        "price": fields.get("price"),
        "quantity": fields.get("quantity"),
        "notional": fields.get("notional", 0.0),
        "fee": fields.get("fee", 0.0),
        "settlement_fee": fields.get("settlement_fee", 0.0),
        "payout": fields.get("payout", 0.0),
        "netting_payout": fields.get("netting_payout", 0.0),
        "cash_required": fields.get("cash_required", 0.0),
        "cash_delta": fields.get("cash_delta", 0.0),
        "realized_pnl": fields.get("realized_pnl", 0.0),
        "realized_pairs": fields.get("realized_pairs", 0.0),
        "client_order_id": fields.get("client_order_id"),
        "outcome": fields.get("outcome"),
        "metadata_json": json.dumps(dict(metadata or {}), sort_keys=True),
    })
    client.put(entity)
    return action_id


def _ds_run_in_transaction(client: Any, fn: Any) -> Any:
    """Run fn(txn) with a small retry loop for optimistic-concurrency
    conflicts. google.cloud.datastore transactions don't auto-retry, and a
    second writer to the same agent is the expected case here, not an edge
    case -- agent_id is the model name, not a per-session id."""
    from google.api_core import exceptions as _gax_exceptions

    last_exc: Optional[BaseException] = None
    for _ in range(_DS_TRANSACTION_RETRIES):
        try:
            with client.transaction() as txn:
                return fn(txn)
        except _gax_exceptions.Aborted as exc:
            last_exc = exc
            continue
    raise last_exc or RuntimeError("Datastore transaction retries exhausted")


def _ds_record_rejected_account_action(
    *,
    agent_id: str,
    mode: str,
    ticker: str,
    side: str,
    normalized: Mapping[str, Any],
    guard: Mapping[str, Any],
    platform: str = "kalshi",
) -> None:
    client = _get_account_datastore()
    if client is None:
        return

    def _run(_txn: Any) -> None:
        _ds_ensure_account(
            client, agent_id, _as_float(guard.get("account_value"), DEFAULT_AGENT_ACCOUNT_VALUE)
        )
        _ds_record_action(
            client,
            agent_id=agent_id,
            action_type="rejected_trade",
            cycle_id=str(guard.get("cycle_id") or ""),
            mode=mode,
            submitted=False,
            platform=platform,
            ticker=ticker,
            side=side,
            price=_as_float(normalized.get("price")),
            quantity=_as_float(normalized.get("quantity")),
            notional=_as_float(guard.get("notional")),
            fee=_as_float(guard.get("fee")),
            netting_payout=_as_float(guard.get("netting_payout")),
            cash_required=_as_float(guard.get("cash_required")),
            cash_delta=0.0,
            realized_pairs=_as_float(guard.get("netting_payout")),
            client_order_id=normalized.get("exchange_order", {}).get("client_order_id"),
            outcome="rejected",
            metadata={"risk_guard": dict(guard)},
        )

    _ds_run_in_transaction(client, _run)


def _ds_apply_trade(
    *,
    agent_id: str,
    policy: RiskGuardPolicy,
    mode: str,
    submitted: bool,
    ticker: str,
    side: str,
    normalized: Mapping[str, Any],
    guard: Mapping[str, Any],
    platform: str = "kalshi",
) -> Dict[str, Any]:
    client = _get_account_datastore()
    if client is None:
        # _load_guard_account degrades to an empty in-memory account when
        # Datastore is unreachable, so the guard check upstream may have
        # already said "allowed". Silently no-op-ing here would report a
        # trade as successful without ever persisting it -- fail loudly
        # instead, same as a real venue outage would.
        raise RuntimeError("Datastore is unavailable for the agent trading account store")
    opposite_side = _opposite_side(side)
    price = _as_float(normalized.get("price"))
    quantity = _as_float(normalized.get("quantity"))
    fee = _as_float(guard.get("filled_fee", guard.get("fee")))
    notional = price * quantity

    def _run(_txn: Any) -> Dict[str, Any]:
        account_entity = client.get(_ds_account_key(client, agent_id))
        if account_entity is None:
            account_entity = _ds_new_account_entity(client, agent_id, policy.account_value)
        cash_before = float(account_entity["cash"])
        realized_pnl_before = float(account_entity["realized_pnl"])
        fees_paid_before = float(account_entity["fees_paid"])

        remaining = quantity
        realized_pairs = 0.0
        realized_pnl = 0.0
        opposite = client.get(_ds_position_key(client, agent_id, platform, ticker, opposite_side))
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
                _ds_upsert_position(
                    client,
                    agent_id=agent_id,
                    platform=platform,
                    ticker=ticker,
                    side=opposite_side,
                    quantity=new_opposite_qty,
                    cost_basis=new_opposite_basis,
                )
                remaining -= realized_pairs

        if remaining > 1e-12:
            same = client.get(_ds_position_key(client, agent_id, platform, ticker, side))
            same_qty = float(same["quantity"]) if same is not None else 0.0
            same_basis = float(same["cost_basis"]) if same is not None else 0.0
            _ds_upsert_position(
                client,
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
        account_entity.update({
            "cash": cash_after,
            "realized_pnl": realized_pnl_before + realized_pnl,
            "fees_paid": fees_paid_before + fee,
            "updated_at": _now(),
        })
        client.put(account_entity)

        action_id = _ds_record_action(
            client,
            agent_id=agent_id,
            action_type="trade",
            cycle_id=policy.cycle_id,
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
            client_order_id=normalized.get("exchange_order", {}).get("client_order_id"),
            outcome="realized" if realized_pairs > 0 else "open",
            metadata={"risk_guard": dict(guard)},
        )
        summary = _ds_account_summary(client, agent_id, policy.account_value)
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

    return _ds_run_in_transaction(client, _run)


def _ds_settle_agent_open_positions(agent_id: str, policy: RiskGuardPolicy) -> List[Dict[str, Any]]:
    client = _get_account_datastore()
    if client is None:
        return []

    def _run(_txn: Any) -> List[Dict[str, Any]]:
        from analyzing_llm_rationale import market_data

        _ds_ensure_account(client, agent_id, policy.account_value)
        if client.get(_ds_cycle_settlement_key(client, agent_id, policy.cycle_id)) is not None:
            return []

        positions = _ds_positions(client, agent_id)
        # Kalshi tickers are stored uppercase; Polymarket slugs are
        # case-sensitive and must round-trip exactly as stored. Group by the
        # normalized (platform, ticker) key up front instead of re-deriving
        # it per comparison, so the grouping key and the filter can't drift.
        by_market: Dict[tuple[str, str], List[Any]] = {}
        for p in positions:
            plat = str(p["platform"]).lower()
            ticker = str(p["ticker"]).upper() if plat == "kalshi" else str(p["ticker"])
            by_market.setdefault((plat, ticker), []).append(p)

        settled: List[Dict[str, Any]] = []
        for (plat, ticker), market_positions in sorted(by_market.items()):
            if not ticker or plat not in ("kalshi", "polymarket"):
                continue
            try:
                outcome_value = (
                    market_data.resolve_kalshi(ticker)
                    if plat == "kalshi"
                    else market_data.resolve_polymarket(ticker)
                )
            except Exception:
                logger.warning("agent settlement lookup failed ticker=%s", ticker, exc_info=True)
                continue
            if outcome_value is None:
                continue
            winning_side = "yes" if int(outcome_value) == 1 else "no"
            settled_contracts = sum(float(p["quantity"]) for p in market_positions)
            settled_basis = sum(float(p["cost_basis"]) for p in market_positions)
            payout = sum(float(p["quantity"]) for p in market_positions if str(p["side"]) == winning_side)
            settlement_fee = payout * _settlement_fee_rate()
            cash_delta = payout - settlement_fee
            realized_pnl = cash_delta - settled_basis

            account_entity = client.get(_ds_account_key(client, agent_id))
            account_entity.update({
                "cash": float(account_entity["cash"]) + cash_delta,
                "realized_pnl": float(account_entity["realized_pnl"]) + realized_pnl,
                "settlement_fees_paid": float(account_entity["settlement_fees_paid"]) + settlement_fee,
                "updated_at": _now(),
            })
            client.put(account_entity)
            for p in market_positions:
                client.delete(_ds_position_key(client, agent_id, plat, ticker, str(p["side"])))
            action_id = _ds_record_action(
                client,
                agent_id=agent_id,
                action_type="settlement",
                cycle_id=policy.cycle_id,
                platform=plat,
                ticker=ticker,
                side=winning_side,
                quantity=settled_contracts,
                settlement_fee=settlement_fee,
                payout=payout,
                cash_delta=cash_delta,
                realized_pnl=realized_pnl,
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

        from google.cloud import datastore as _ds

        marker = _ds.Entity(key=_ds_cycle_settlement_key(client, agent_id, policy.cycle_id))
        marker.update({"checked_at": _now()})
        client.put(marker)
        return settled

    try:
        settled = _ds_run_in_transaction(client, _run)
        settlement_actions.add(len(settled), {"outcome": "success"})
        return settled
    except Exception:
        settlement_actions.add(1, {"outcome": "failure"})
        logger.warning("agent settlement pass failed (datastore)", exc_info=True)
        return []


def _ds_load_guard_account(agent_id: str, policy: RiskGuardPolicy) -> tuple[Any, float]:
    from analyzing_llm_rationale.accounting import PredictionMarketAccount

    client = _get_account_datastore()
    account = PredictionMarketAccount(starting_cash=policy.account_value)
    if client is None:
        return account, 0.0
    row = _ds_ensure_account(client, agent_id, policy.account_value)
    account.cash = float(row["cash"])
    account.realized_pnl = float(row["realized_pnl"])
    account.fees_paid = float(row["fees_paid"])
    for pos in _ds_positions(client, agent_id):
        loaded = account._position(str(pos["platform"]), str(pos["ticker"]), str(pos["side"]))
        loaded.quantity = float(pos["quantity"])
        loaded.cost_basis = float(pos["cost_basis"])
    account_key = _ds_account_key(client, agent_id)
    cycle_spend = sum(
        float(a.get("cash_required") or 0.0)
        for a in client.query(kind=_DS_ACTION_KIND, ancestor=account_key).fetch()
        if a.get("cycle_id") == policy.cycle_id and a.get("action_type") == "trade"
    )
    return account, cycle_spend


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


def _market_cost_basis(account: Any, ticker: str, *, platform: str = "kalshi") -> float:
    return sum(
        float(pos.cost_basis)
        for pos in account.open_positions()
        if str(pos.platform).lower() == platform and str(pos.ident) == ticker
    )


def _load_guard_account(agent_id: str, policy: RiskGuardPolicy) -> tuple[Any, float]:
    if _use_datastore_account_store():
        return _ds_load_guard_account(agent_id, policy)
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
    platform: str = "kalshi",
) -> tuple[bool, Dict[str, Any], RiskGuardPolicy]:
    policy = _risk_guard_policy()
    settlements = _settle_agent_open_positions(agent_id, policy)
    account, cycle_spend_before = _load_guard_account(agent_id, policy)
    price = _as_float(normalized.get("price"))
    quantity = _as_float(normalized.get("quantity"))
    fee = _order_fee(args, normalized, platform=platform)
    cash_before = float(account.cash)
    fill = account.buy(
        platform=platform,
        ident=ticker,
        side=side,
        quantity=quantity,
        price=price,
        fee=fee,
    )
    market_cost_basis_after = _market_cost_basis(account, ticker, platform=platform)
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


def _resolve_shadow_marketability(
    ticker: str, side: str, requested_price: float, *, platform: str = "kalshi"
) -> Dict[str, Any]:
    """Check a shadow-mode Kalshi or Polymarket order against a live quote before filling it.

    Shadow mode otherwise trusts whatever price the caller supplies, which
    lets a mispriced (or hallucinated) order "fill" at a price no real book
    would ever give -- and once that lands opposite an existing position,
    reciprocal netting turns it straight into manufactured profit. When a
    live quote is available we fill at the real ask (never worse for the
    caller than their own request) and only fill at all if the requested
    price actually crosses it, mirroring how the immediate-or-cancel orders
    this tool issues would behave against a real book.

    If the quote can't be fetched (unknown ticker, provider hiccup), the
    order doesn't fill -- a real IOC order has no book to route against
    either in that situation. This previously fell back to trusting the
    caller's price instead, reasoning that the netting-arb guard in
    `_check_trade_guards` would still catch the abuse case; it doesn't fully:
    that guard only fires on the *closing* leg of a netted pair, so a single
    directional entry booked at a fabricated price during a quote outage can
    sit open indefinitely, marked to market later against a real quote --
    silently inflating the shown unrealized P&L on exactly the leaderboard
    numbers a go/no-go call on live trading would be based on. A missed fill
    during a transient outage is a lost opportunity; a corrupted ledger entry
    is worse and harder to notice.
    """
    try:
        from analyzing_llm_rationale import market_data
        from analyzing_llm_rationale.accounting import MarketQuote

        raw_quote = (
            market_data.fetch_polymarket(slug=ticker)
            if platform != "kalshi"
            else market_data.fetch_kalshi(ticker)
        )
        quote = MarketQuote.from_mapping(raw_quote)
        real_ask = quote.ask(side)
    except Exception:
        real_ask = None

    if real_ask is None:
        return {
            "marketable": False,
            "price": requested_price,
            "real_ask": None,
            "status": "shadow_quote_unavailable",
        }

    marketable = requested_price + 1e-9 >= real_ask
    return {
        "marketable": marketable,
        "price": round(real_ask, 4) if marketable else requested_price,
        "real_ask": round(real_ask, 4),
        "status": "shadow_filled_at_market" if marketable else "shadow_unfilled_below_market",
    }


def place_trade(args: Mapping[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    """Place a Kalshi or Polymarket trade through the benchmark tool surface.

    This is shadow/paper trading only: it records a shadow action and returns
    the normalized order, and never calls trading.place_order. This tool is
    reachable from an autonomous LLM tool loop, which cannot supply real
    human confirmation and does not route through create_trading_run,
    execute_trading_run, the guardrail chain, or the kill switch -- real
    execution must go through POST /trading/preview then POST /trading/orders.
    """
    start = time.perf_counter()
    tool = "place_trade"
    agent_id = _bounded_agent_id(ctx.agent_id)
    with tracer.start_as_current_span("benchmark_tools.place_trade") as span:
        # Best-effort venue tag for tracing -- the validated value (which can
        # raise) is parsed just below, inside the try block that already
        # catches and reports every failure mode for this tool.
        span.set_attributes({
            "agent.id": agent_id,
            "tool.name": tool,
            "market.venue": str(args.get("platform") or "kalshi").strip().lower(),
        })
        try:
            from analyzing_llm_rationale import trading

            platform = trading._clean_platform(args.get("platform") or "kalshi")
            side = _clean_side(args.get("side") or args.get("outcome"))
            ticker = _clean_ticker(args.get("ticker") or args.get("ident"), platform=platform)
            order = {
                "platform": platform,
                "action": "buy",
                "outcome": side,
                "order_type": "limit",
                "ticker": ticker,
                "price": args.get("price"),
                "quantity": args.get("quantity", 1),
                # trading._preview_polymarket only accepts GTC/GTD for a
                # limit order (IOC-style limit orders aren't a Polymarket
                # CLOB concept) -- this is purely the informational
                # "exchange_order" shape trading.preview_order would submit;
                # the actual immediate-fill-or-nothing behavior below (via
                # _resolve_shadow_marketability / _extract_filled_quantity)
                # is identical for both venues and doesn't read this field.
                "time_in_force": IMMEDIATE_TIME_IN_FORCE if platform == "kalshi" else "GTC",
                "post_only": False,
                "client_order_id": str(args.get("client_order_id") or f"foresea-agent-{uuid.uuid4()}"),
            }
            if platform == "polymarket":
                # trading._preview_polymarket reads slug/market_id/token_id,
                # not ticker -- ticker doubles as the slug here so callers
                # keep using one ident field regardless of venue.
                order["slug"] = ticker
                if args.get("token_id"):
                    order["token_id"] = str(args["token_id"])
            execution_warnings = _immediate_order_adjustments(args)
            for key in ("reduce_only", "cancel_order_on_pause", "subaccount"):
                if key in args:
                    order[key] = args[key]
            mode = str(os.environ.get("FORESEA_AGENT_PLACE_TRADE_MODE", "shadow")).strip().lower()
            if mode != "shadow":
                # This tool is called from an autonomous LLM tool loop, which
                # cannot supply real human confirmation and does not route
                # through create_trading_run/execute_trading_run/the guardrail
                # chain/the kill switch. It must never place a real order --
                # real execution goes through /trading/preview -> /trading/orders,
                # which requires a signed-in human to type the confirmation
                # phrase and passes through _validate_live_trade_guardrails.
                raise ValueError(
                    "FORESEA_AGENT_PLACE_TRADE_MODE must be 'shadow'. The benchmark "
                    "tool surface is shadow/paper trading only and cannot place real "
                    "orders; use /trading/preview and /trading/orders for real execution."
                )

            shadow_marketable = True
            shadow_unfilled_status = "shadow_unfilled_below_market"
            try:
                requested_price = float(order.get("price"))
            except (TypeError, ValueError):
                requested_price = None
            if requested_price is not None and requested_price > 0:
                market_check = _resolve_shadow_marketability(ticker, side, requested_price, platform=platform)
                shadow_marketable = market_check["marketable"]
                if not shadow_marketable:
                    shadow_unfilled_status = market_check["status"]
                if market_check["real_ask"] is not None:
                    # Fill at the real ask (never worse than what was asked
                    # for) instead of whatever price the caller guessed.
                    order["price"] = market_check["price"]

            preview = trading.preview_order(order)
            normalized = preview.get("normalized_order") or {}
            allowed, guard, policy = _check_trade_guards(
                args=args,
                normalized=normalized,
                agent_id=agent_id,
                ticker=ticker,
                side=side,
                platform=platform,
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
                    "platform": platform,
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
                    platform=platform,
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

            result = preview
            submitted = False
            live = False
            accounting_normalized, accounting_guard = _normalize_fill_for_accounting(
                args=args,
                result=result,
                normalized=normalized,
                guard=guard,
                live=live,
                shadow_marketable=shadow_marketable,
                shadow_unfilled_status=shadow_unfilled_status,
                platform=platform,
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
                platform=platform,
            )
            event = {
                "ts": _now(),
                "agent_id": agent_id,
                "tool": tool,
                "ok": True,
                "rejected": False,
                "mode": mode,
                "submitted": submitted,
                "platform": platform,
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
                    f"Live {platform.capitalize()} order submitted."
                    if submitted
                    else f"Shadow {platform.capitalize()} trade recorded; no exchange order was submitted."
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
