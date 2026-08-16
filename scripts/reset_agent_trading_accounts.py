#!/usr/bin/env python3
"""One-off admin tool: reset agentic-trading shadow accounts to a clean
starting state.

Used when a bug in place_trade/PredictionMarketAccount.buy() has already
been fixed (see PR #230, #233) but historical GCS-stored account data still
reflects trades made under the old, buggy code. Rather than trying to
surgically reconcile every possible historical anomaly, this resets the
named agents' cash/realized_pnl/fees back to starting_cash/0/0 and zeroes
out (never deletes) any open positions, so nothing entry-priced under the
old bug carries forward.

The full historical agent_actions/agent_positions row set is preserved for
audit -- only agent_accounts is directly reset, and position rows are
UPDATEd to zero quantity rather than removed. An admin_reset action row is
added recording the real cash_delta, so
agent_trading_stats.agent_equity_curve() shows the reset as a visible point
on the chart instead of silently diverging from the reset agent_accounts.cash.

Usage:
    python scripts/reset_agent_trading_accounts.py MODEL [MODEL ...]
    python scripts/reset_agent_trading_accounts.py --all

Downloads each model's store from GCS, applies the reset locally, uploads
the modified store back to the same path. Requires the same GCP credentials
already used by scripts/agent_trading_tick.py.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import benchmark_tools  # noqa: E402
from analyzing_llm_rationale.config import load_model_configs  # noqa: E402

STORE_BUCKET = "brave-drive-471109-d9-track-record-store"
DEFAULT_REASON = (
    "netting-arb pricing bug (fixed in PR #230/#233) let shadow trades "
    "manufacture unearned realized profit; resetting so no past bug carries forward"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _all_models() -> List[str]:
    models = load_model_configs(ROOT / "configs" / "models.yaml")
    return sorted(name for name, cfg in models.items() if cfg.chat_interface_enabled)


def reset_account(conn: sqlite3.Connection, agent_id: str, reason: str) -> Dict[str, Any]:
    """Reset one agent's account to starting_cash/0/0/0, zero out (not
    delete) its open positions, and record the adjustment as a visible
    admin_reset action. Returns the pre/post state for logging."""
    acct = conn.execute(
        "SELECT * FROM agent_accounts WHERE agent_id = ?", (agent_id,)
    ).fetchone()
    if acct is None:
        return {"agent_id": agent_id, "skipped": "no account row"}

    starting_cash = float(acct["starting_cash"])
    pre_cash = float(acct["cash"])
    pre_realized_pnl = float(acct["realized_pnl"])
    cash_delta = starting_cash - pre_cash

    with conn:
        conn.execute(
            "UPDATE agent_accounts SET cash = ?, realized_pnl = 0, fees_paid = 0, "
            "settlement_fees_paid = 0, updated_at = ? WHERE agent_id = ?",
            (starting_cash, _now(), agent_id),
        )
        # Zero out rather than delete -- an open position entry-priced under
        # the old, unvalidated shadow-fill code shouldn't carry into the
        # reset state, but the row itself stays for audit.
        n_positions = conn.execute(
            "UPDATE agent_positions SET quantity = 0, cost_basis = 0, avg_entry_price = 0 "
            "WHERE agent_id = ? AND quantity > 0",
            (agent_id,),
        ).rowcount
        conn.execute(
            """
            INSERT INTO agent_actions
                (id, ts, agent_id, action_type, cycle_id, outcome, cash_delta,
                 realized_pnl, metadata_json)
            VALUES (?, ?, ?, 'admin_reset', 'manual', 'reset', ?, ?, ?)
            """,
            (
                f"reset-{agent_id}-{uuid.uuid4()}",
                _now(),
                agent_id,
                cash_delta,
                -pre_realized_pnl,
                json.dumps({
                    "reason": reason,
                    "pre_reset_cash": pre_cash,
                    "pre_reset_realized_pnl": pre_realized_pnl,
                    "positions_zeroed": n_positions,
                }),
            ),
        )
    return {
        "agent_id": agent_id,
        "pre_cash": pre_cash,
        "post_cash": starting_cash,
        "pre_realized_pnl": pre_realized_pnl,
        "positions_zeroed": n_positions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="Model names to reset")
    parser.add_argument("--all", action="store_true", help="Reset every chat-capable model")
    parser.add_argument("--reason", default=DEFAULT_REASON)
    args = parser.parse_args()

    models = _all_models() if args.all else args.models
    if not models:
        parser.error("specify model names or --all")

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(STORE_BUCKET)

    with tempfile.TemporaryDirectory() as td:
        for model in models:
            blob_name = f"agent_trading_store__{model}.sqlite"
            blob = bucket.blob(blob_name)
            local_path = Path(td) / blob_name
            if not blob.exists():
                print(f"{model}: skipped (no store object in GCS)")
                continue
            blob.download_to_filename(str(local_path))

            conn = sqlite3.connect(str(local_path))
            conn.row_factory = sqlite3.Row
            benchmark_tools._ensure_account_schema(conn)
            result = reset_account(conn, model, args.reason)
            conn.close()

            if "skipped" in result:
                print(f"{model}: skipped ({result['skipped']})")
                continue

            blob.upload_from_filename(str(local_path))
            print(
                f"{model}: cash {result['pre_cash']:.2f} -> {result['post_cash']:.2f}, "
                f"realized_pnl {result['pre_realized_pnl']:.2f} -> 0.00, "
                f"{result['positions_zeroed']} position(s) zeroed"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
