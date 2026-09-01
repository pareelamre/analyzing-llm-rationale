import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_agent_trading_audit

from analyzing_llm_rationale import benchmark_tools


class AgentTradingAuditTests(unittest.TestCase):
    def test_trade_audit_context_is_persisted_with_the_action(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "store.sqlite"
            policy = benchmark_tools.RiskGuardPolicy(
                account_value=10_000.0,
                concentration_limit=0.15,
                per_cycle_spend_limit=2_000.0,
                cycle_id="audit-cycle",
                max_drawdown_limit=0.2,
                daily_risk_limit=3_000.0,
                max_open_markets=10,
                max_trades_per_cycle=6,
                duplicate_trade_cooldown_seconds=900,
            )
            audit = benchmark_tools._trade_audit_context(
                requested_price=0.42,
                requested_quantity=50,
                market_check={
                    "marketable": True,
                    "status": "shadow_filled_at_market",
                    "real_ask": 0.4,
                },
                sizing={"mode": "quarter_kelly", "applied": True, "target_quantity": 50},
                guard={"allowed": True, "cash_before": 10_000.0, "cycle_id": "audit-cycle"},
                fill_status="shadow_filled_at_market",
                filled_quantity=50,
            )
            with mock.patch.dict(os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(db_path)}, clear=False):
                update = benchmark_tools._apply_trade_to_account_tables(
                    agent_id="audit-model",
                    policy=policy,
                    mode="shadow",
                    submitted=False,
                    ticker="KXTEST-26",
                    side="yes",
                    normalized={"price": 0.4, "quantity": 50, "exchange_order": {}},
                    guard={"filled_fee": 0.1, "cycle_id": "audit-cycle"},
                    audit=audit,
                )
                with benchmark_tools._account_transaction() as conn:
                    row = conn.execute(
                        "SELECT metadata_json FROM agent_actions WHERE id = ?",
                        (update["action_id"],),
                    ).fetchone()

        metadata = json.loads(row["metadata_json"])
        self.assertEqual(metadata["audit"]["version"], 1)
        self.assertEqual(metadata["audit"]["quote"]["observed_ask"], 0.4)
        self.assertEqual(metadata["audit"]["execution"]["filled_quantity"], 50)

    def test_background_builder_keeps_a_compact_per_model_audit_index(self):
        with tempfile.TemporaryDirectory() as td:
            store_dir = Path(td) / "stores"
            model_dir = store_dir / "audit-model"
            model_dir.mkdir(parents=True)
            conn = sqlite3.connect(model_dir / "store.sqlite")
            conn.row_factory = sqlite3.Row
            benchmark_tools._ensure_account_schema(conn)
            action_id = benchmark_tools._record_account_action(
                conn,
                agent_id="audit-model",
                action_type="trade",
                cycle_id="audit-cycle",
                mode="shadow",
                platform="kalshi",
                ticker="KXTEST-26",
                side="yes",
                price=0.4,
                quantity=50,
                notional=20,
                fee=0.1,
                cash_required=20.1,
                cash_delta=-20.1,
                outcome="open",
                metadata={
                    "audit": {
                        "version": 1,
                        "quote": {"source": "live_venue_quote", "observed_ask": 0.4},
                        "risk": {"allowed": True},
                    },
                },
            )
            conn.commit()
            conn.close()

            artifact = build_agent_trading_audit.build_audit_artifact(
                store_dir=store_dir,
                models=["audit-model"],
                per_model_limit=10,
            )
            archive_root = Path(td) / "static" / "agent_trading_audits"
            manifest_path = Path(td) / "static" / "agent_trading_audit_archive_manifest.json"
            archive = build_agent_trading_audit.build_audit_archive(
                store_dir=store_dir,
                archive_root=archive_root,
                manifest_path=manifest_path,
                models=["audit-model"],
            )
            archived_path = Path(archive["archives"]["audit-model"][0]["path"])
            archived = json.loads(archived_path.read_text(encoding="utf-8"))

        records = artifact["audits"]["audit-model"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["action_id"], action_id)
        self.assertEqual(records[0]["audit"]["status"], "recorded")
        self.assertEqual(records[0]["audit"]["quote"]["observed_ask"], 0.4)
        self.assertEqual(len(records[0]["record_hash"]), 64)
        self.assertEqual(archive["storage"], "github_repository")
        self.assertEqual(archive["archives"]["audit-model"][0]["records"], 1)
        self.assertEqual(archived["items"][0]["action_id"], action_id)

    def test_retired_model_archives_remain_discoverable_without_a_live_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            archive_root = Path(td) / "static" / "agent_trading_audits"
            for model in ("kimi-k3", "scads-alias-code"):
                retired_path = archive_root / model / "2026-08.json"
                retired_path.parent.mkdir(parents=True)
                retired_path.write_text(json.dumps({
                    "schema_version": 1,
                    "agent_id": model,
                    "month": "2026-08",
                    "items": [{"action_id": f"{model}-retired-action"}],
                }), encoding="utf-8")
            manifest_path = Path(td) / "static" / "agent_trading_audit_archive_manifest.json"

            with mock.patch.object(build_agent_trading_audit, "ROOT", Path.cwd()):
                archive = build_agent_trading_audit.build_audit_archive(
                    store_dir=Path(td) / "stores",
                    archive_root=archive_root,
                    manifest_path=manifest_path,
                )

        self.assertIn("kimi-k3", archive["retired_models"])
        self.assertIn("scads-alias-code", archive["retired_models"])
        self.assertEqual(archive["archives"]["kimi-k3"][0]["records"], 1)
        self.assertEqual(archive["archives"]["scads-alias-code"][0]["records"], 1)


if __name__ == "__main__":
    unittest.main()
