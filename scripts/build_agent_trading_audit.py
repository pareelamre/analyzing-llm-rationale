#!/usr/bin/env python3
"""Publish GitHub-backed audit artifacts for Agentic paper trades.

The live per-model SQLite ledger remains operational: it restores cash, open
positions, and execution state before a scheduled paper-trading cycle.  This
publisher copies immutable action records into
append-only monthly JSON archives committed to GitHub, as well as a small
recent index for fast reads.  Long-term audit retention therefore lives in the
repository, not in a new Cloud Run or database storage product.  The live
ledger is not compacted here because the paper PnL and equity curve are
recomputed from it; compaction must first preserve those aggregates.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import benchmark_tools  # noqa: E402
from analyzing_llm_rationale.config import load_model_configs  # noqa: E402

tracer = trace.get_tracer("foresea.agent_trading_audit")
meter = metrics.get_meter("foresea.agent_trading_audit")
audit_builds = meter.create_counter(
    "agent_trading.audit_artifact.builds",
    unit="1",
    description="Agent paper-trade audit artifact builds by outcome",
)
audit_records = meter.create_counter(
    "agent_trading.audit_artifact.records",
    unit="1",
    description="Bounded paper-trade audit records published",
)
audit_duration = meter.create_histogram(
    "agent_trading.audit_artifact.duration",
    unit="s",
    description="Agent paper-trade audit artifact build duration",
)
audit_archive_builds = meter.create_counter(
    "agent_trading.audit_archive.builds",
    unit="1",
    description="GitHub-backed agent paper-trade archive builds by outcome",
)
audit_archive_records = meter.create_counter(
    "agent_trading.audit_archive.records",
    unit="1",
    description="Paper-trade audit records written to the GitHub archive",
)
audit_archive_duration = meter.create_histogram(
    "agent_trading.audit_archive.duration",
    unit="s",
    description="GitHub-backed paper-trade audit archive build duration",
)

logger = logging.getLogger(__name__)

STORE_DIR = Path(os.environ.get("AGENT_TRADING_BOARD_STORE_DIR", "tmp/agent-trading-board"))
OUTPUT_PATH = Path(os.environ.get("AGENT_TRADING_AUDIT_OUTPUT", "static/agent_trading_audit_live.json"))
ARCHIVE_ROOT = Path(os.environ.get("AGENT_TRADING_AUDIT_ARCHIVE_ROOT", "static/agent_trading_audits"))
ARCHIVE_MANIFEST_PATH = Path(
    os.environ.get("AGENT_TRADING_AUDIT_ARCHIVE_MANIFEST", "static/agent_trading_audit_archive_manifest.json")
)
DEFAULT_PER_MODEL_LIMIT = 500
PER_MODEL_LIMIT = max(
    1,
    min(2_000, int(os.environ.get("AGENT_TRADING_AUDIT_PER_MODEL_LIMIT", str(DEFAULT_PER_MODEL_LIMIT)))),
)
_AUDITED_ACTIONS = ("trade", "rejected_trade", "settlement")


def _chat_capable_models() -> List[str]:
    models = load_model_configs(ROOT / "configs" / "models.yaml")
    return sorted(name for name, cfg in models.items() if cfg.chat_interface_enabled)


def _open_store(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path) if path.exists() else ":memory:")
    conn.row_factory = sqlite3.Row
    benchmark_tools._ensure_account_schema(conn)
    return conn


def _bounded_value(value: Any, *, max_chars: int = 320) -> Any:
    if isinstance(value, str):
        return value[:max_chars]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        return [_bounded_value(item, max_chars=max_chars) for item in value[:16]]
    if isinstance(value, dict):
        return {
            str(key)[:80]: _bounded_value(item, max_chars=max_chars)
            for key, item in list(value.items())[:32]
        }
    return str(value)[:max_chars]


def _audit_context(metadata_json: Any) -> Dict[str, Any]:
    """Return only the compact, intentionally-auditable metadata namespace."""
    try:
        metadata = json.loads(metadata_json) if metadata_json else {}
    except (TypeError, ValueError):
        return {"status": "metadata_unreadable"}
    if not isinstance(metadata, dict):
        return {"status": "metadata_unreadable"}
    audit = metadata.get("audit")
    if not isinstance(audit, dict) or int(audit.get("version") or 0) != 1:
        return {"status": "legacy_record"}
    return {"status": "recorded", **_bounded_value(audit)}


def _record_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    record = {
        "action_id": str(row["id"]),
        "recorded_at": str(row["ts"]),
        "agent_id": str(row["agent_id"]),
        "cycle_id": str(row["cycle_id"] or ""),
        "event_type": str(row["action_type"]),
        "mode": str(row["mode"] or "shadow"),
        "outcome": str(row["outcome"] or "recorded"),
        "market": {
            "platform": str(row["platform"] or ""),
            "ticker": str(row["ticker"] or ""),
            "side": str(row["side"] or ""),
        },
        "execution": {
            "submitted": bool(row["submitted"]),
            "price": row["price"],
            "quantity": row["quantity"],
            "notional": row["notional"],
            "fee": row["fee"],
            "settlement_fee": row["settlement_fee"],
            "cash_required": row["cash_required"],
            "cash_delta": row["cash_delta"],
            "netting_payout": row["netting_payout"],
            "realized_pnl": row["realized_pnl"],
        },
        "audit": _audit_context(row["metadata_json"]),
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    record["record_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return record


def _recent_audits(conn: sqlite3.Connection, model: str, limit: int) -> List[Dict[str, Any]]:
    placeholders = ",".join("?" for _ in _AUDITED_ACTIONS)
    rows = conn.execute(
        f"""
        SELECT id, ts, agent_id, action_type, mode, submitted, platform, ticker, side,
               price, quantity, notional, fee, settlement_fee, cash_required, cash_delta,
               netting_payout, realized_pnl, cycle_id, outcome, metadata_json
        FROM agent_actions
        WHERE agent_id = ? AND action_type IN ({placeholders})
        ORDER BY ts DESC, id DESC
        LIMIT ?
        """,
        (model, *_AUDITED_ACTIONS, limit),
    ).fetchall()
    return [_record_from_row(row) for row in rows]


def _all_audits(conn: sqlite3.Connection, model: str) -> List[Dict[str, Any]]:
    """Return immutable audit rows for archival, ordered deterministically."""
    placeholders = ",".join("?" for _ in _AUDITED_ACTIONS)
    rows = conn.execute(
        f"""
        SELECT id, ts, agent_id, action_type, mode, submitted, platform, ticker, side,
               price, quantity, notional, fee, settlement_fee, cash_required, cash_delta,
               netting_payout, realized_pnl, cycle_id, outcome, metadata_json
        FROM agent_actions
        WHERE agent_id = ? AND action_type IN ({placeholders})
        ORDER BY ts ASC, id ASC
        """,
        (model, *_AUDITED_ACTIONS),
    ).fetchall()
    return [_record_from_row(row) for row in rows]


def _month_key(recorded_at: Any) -> str:
    """Return a conservative calendar partition without trusting malformed timestamps."""
    value = str(recorded_at or "")
    return value[:7] if len(value) >= 7 and value[:4].isdigit() and value[4] == "-" else "unknown"


def _write_json_if_changed(path: Path, payload: Any) -> bool:
    """Atomically replace a GitHub-published JSON artifact only when it changes."""
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        if path.read_text(encoding="utf-8") == rendered:
            return False
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(rendered, encoding="utf-8")
    tmp.replace(path)
    return True


def _archive_manifest_path(path: Path) -> str:
    """Use repository-relative paths in production and portable paths in tests."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def build_audit_archive(
    *,
    store_dir: Path = STORE_DIR,
    archive_root: Path = ARCHIVE_ROOT,
    manifest_path: Path = ARCHIVE_MANIFEST_PATH,
    models: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Write per-model monthly audit archives and a small discoverability manifest.

    Existing period files are rebuilt from immutable action IDs rather than
    blindly appended, so retries are idempotent.  Once committed by the
    publisher workflow these files are the long-term audit copy in GitHub.
    """
    started = time.perf_counter()
    chosen_models = list(models) if models is not None else _chat_capable_models()
    with tracer.start_as_current_span("agent_trading.build_audit_archive") as span:
        span.set_attribute("agent.models", len(chosen_models))
        try:
            archives: Dict[str, List[Dict[str, Any]]] = {}
            record_count = 0
            for model in chosen_models:
                conn = _open_store(store_dir / model / "store.sqlite")
                try:
                    grouped: Dict[str, List[Dict[str, Any]]] = {}
                    for record in _all_audits(conn, str(model)):
                        grouped.setdefault(_month_key(record.get("recorded_at")), []).append(record)
                finally:
                    conn.close()

                periods: List[Dict[str, Any]] = []
                for month, records in sorted(grouped.items(), reverse=True):
                    records.sort(key=lambda item: (str(item.get("recorded_at") or ""), str(item.get("action_id") or "")))
                    path = archive_root / str(model) / f"{month}.json"
                    _write_json_if_changed(path, {"schema_version": 1, "agent_id": model, "month": month, "items": records})
                    periods.append({
                        "month": month,
                        "path": _archive_manifest_path(path),
                        "records": len(records),
                    })
                    record_count += len(records)
                archives[str(model)] = periods

            manifest = {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "mode": "shadow",
                "storage": "github_repository",
                "archives": archives,
            }
            _write_json_if_changed(manifest_path, manifest)
            audit_archive_builds.add(1, {"outcome": "success"})
            audit_archive_records.add(record_count, {"storage": "github_repository"})
            span.set_attributes({"outcome": "success", "audit.records": record_count})
            logger.info("agent paper-trade audit archive built models=%d records=%d", len(chosen_models), record_count)
            return manifest
        except Exception as exc:
            audit_archive_builds.add(1, {"outcome": "failure"})
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            logger.exception("agent paper-trade audit archive build failed")
            raise
        finally:
            audit_archive_duration.record(time.perf_counter() - started, {"operation": "build"})


def build_audit_artifact(
    *,
    store_dir: Path = STORE_DIR,
    models: Optional[Iterable[str]] = None,
    per_model_limit: int = PER_MODEL_LIMIT,
) -> Dict[str, Any]:
    """Build a bounded, independently fetchable audit index from all model ledgers."""
    started = time.perf_counter()
    chosen_models = list(models) if models is not None else _chat_capable_models()
    with tracer.start_as_current_span("agent_trading.build_audit_artifact") as span:
        span.set_attributes({"agent.models": len(chosen_models), "audit.per_model_limit": per_model_limit})
        try:
            audits: Dict[str, List[Dict[str, Any]]] = {}
            for model in chosen_models:
                conn = _open_store(store_dir / model / "store.sqlite")
                try:
                    audits[str(model)] = _recent_audits(conn, str(model), per_model_limit)
                finally:
                    conn.close()
            count = sum(len(records) for records in audits.values())
            artifact = {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "mode": "shadow",
                "retained_per_model": per_model_limit,
                "models": chosen_models,
                "audits": audits,
            }
            audit_builds.add(1, {"outcome": "success"})
            audit_records.add(count, {"source": "sqlite_ledger"})
            span.set_attributes({"outcome": "success", "audit.records": count})
            return artifact
        except Exception as exc:
            audit_builds.add(1, {"outcome": "failure"})
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            raise
        finally:
            audit_duration.record(time.perf_counter() - started, {"operation": "build"})


def main() -> int:
    artifact = build_audit_artifact()
    _write_json_if_changed(OUTPUT_PATH, artifact)
    archive = build_audit_archive()
    print(
        f"agent-trading-audit built models={len(artifact['models'])} "
        f"recent_records={sum(len(records) for records in artifact['audits'].values())} "
        f"archived_records={sum(period['records'] for periods in archive['archives'].values() for period in periods)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
