"""Model-facing benchmark tools for prediction-market trading agents.

The tool names intentionally match the benchmark spec:

- place_trade: Kalshi YES/NO buy tool. Default mode is shadow, not live funds.
- web_search: OpenAI Responses API web_search wrapper with a small blacklist.
- manage_notes: bounded persistent notes per agent.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlparse

import requests
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

BLACKLISTED_WEB_DOMAINS = ("coinmarketcap.com",)
MAX_NOTES_PER_AGENT = 50
MAX_NOTE_CHARS = 1200
WEB_SEARCH_TIMEOUT_S = 120
DEFAULT_AGENT_ACCOUNT_VALUE = 10_000.0
DEFAULT_CONCENTRATION_LIMIT = 0.15
DEFAULT_PER_CYCLE_SPEND_LIMIT = 500.0
DEFAULT_CYCLE_MINUTES = 15
KALSHI_FEE_COEFFICIENT = 0.07


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
    cycle_spend = 0.0
    for event in _iter_ledger_events():
        if event.get("tool") != "place_trade":
            continue
        if _bounded_agent_id(str(event.get("agent_id") or "")) != agent_id:
            continue
        if event.get("rejected") or event.get("ok") is False:
            continue
        if str(event.get("platform") or "kalshi").lower() != "kalshi":
            continue
        ticker = str(event.get("ticker") or "").upper()
        side = str(event.get("side") or "").lower()
        if not ticker or side not in {"yes", "no"}:
            continue
        try:
            quantity = _as_float(event.get("quantity"))
            price = _as_float(event.get("price"))
            fee = _as_float(event.get("fee"), _kalshi_fee(price, quantity))
            fill = account.buy(
                platform="kalshi",
                ident=ticker,
                side=side,
                quantity=quantity,
                price=price,
                fee=fee,
                ts=event.get("ts"),
            )
        except Exception:
            logger.warning("skipping invalid agent trade ledger event", exc_info=True)
            continue
        if str(event.get("cycle_id") or "") == policy.cycle_id:
            cycle_spend += _as_float(event.get("cash_required"), max(0.0, -fill.cash_delta))
    return account, cycle_spend


def _check_trade_guards(
    *,
    args: Mapping[str, Any],
    normalized: Mapping[str, Any],
    agent_id: str,
    ticker: str,
    side: str,
) -> tuple[bool, Dict[str, Any]]:
    policy = _risk_guard_policy()
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
    }
    outcome = "allowed" if detail["allowed"] else "rejected"
    risk_guard_checks.add(1, {"outcome": outcome})
    if reasons:
        risk_guard_rejections.add(1, {"reason": reasons[0]})
    return detail["allowed"], detail


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
                "order_type": str(args.get("order_type") or "limit").lower(),
                "ticker": ticker,
                "price": args.get("price"),
                "quantity": args.get("quantity", 1),
                "client_order_id": str(args.get("client_order_id") or f"foresea-agent-{uuid.uuid4()}"),
            }
            for key in ("time_in_force", "post_only", "reduce_only", "cancel_order_on_pause", "subaccount"):
                if key in args:
                    order[key] = args[key]
            mode = str(os.environ.get("FORESEA_AGENT_PLACE_TRADE_MODE", "shadow")).strip().lower()
            if mode not in {"shadow", "live"}:
                raise ValueError("FORESEA_AGENT_PLACE_TRADE_MODE must be 'shadow' or 'live'")

            preview = trading.preview_order(order)
            normalized = preview.get("normalized_order") or {}
            allowed, guard = _check_trade_guards(
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
                    "warnings": preview.get("warnings", []),
                }

            if mode == "live":
                order.update({"execute": True, "confirmation": trading.CONFIRMATION_PHRASE})
                result = trading.place_order(order, user_id=f"agent:{agent_id}")
                submitted = True
                normalized = result.get("normalized_order") or {}
            else:
                result = preview
                submitted = False

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
            trade_actions.add(1, {"mode": mode, "submitted": str(submitted).lower()})
            span.set_attributes({
                "outcome": "success",
                "risk_guard.allowed": True,
                "trade.mode": mode,
                "trade.submitted": submitted,
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
                "risk_guard": guard,
                "warnings": result.get("warnings", []),
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


def _extract_text(payload: Mapping[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text
    chunks: List[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict):
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunks).strip()


def _extract_citations(payload: Mapping[str, Any]) -> List[Dict[str, str]]:
    citations: List[Dict[str, str]] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            for ann in content.get("annotations") or []:
                if not isinstance(ann, dict):
                    continue
                url = ann.get("url")
                if isinstance(url, str) and url:
                    citations.append({
                        "title": str(ann.get("title") or url),
                        "url": url,
                    })
    return citations


def web_search(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Run a bounded OpenAI web search with blacklist guidance and filtering."""
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
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for web_search")
            model = str(os.environ.get("FORESEA_WEB_SEARCH_MODEL") or "gpt-5-mini")
            payload = {
                "model": model,
                "tools": [{"type": "web_search", "search_context_size": "low"}],
                "input": query,
                "instructions": (
                    "Search the web for concise current evidence. Do not use results from "
                    + ", ".join(BLACKLISTED_WEB_DOMAINS)
                    + ". Return source-grounded facts and links."
                ),
            }
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=WEB_SEARCH_TIMEOUT_S,
            )
            response.raise_for_status()
            body = response.json()
            citations = [
                c for c in _extract_citations(body)
                if not _domain_blacklisted(c.get("url", ""))
            ]
            blocked = [
                c for c in _extract_citations(body)
                if _domain_blacklisted(c.get("url", ""))
            ]
            span.set_attributes({
                "outcome": "success",
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": model,
                "web.citations.count": len(citations),
                "web.blocked_results.count": len(blocked),
            })
            _finish_tool(tool, start, "success")
            return {
                "ok": True,
                "tool": tool,
                "query": query,
                "summary": _extract_text(body),
                "sources": citations,
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
