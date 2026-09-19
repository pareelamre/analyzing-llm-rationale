"""The two agent account stores must book the same ledger.

benchmark_tools keeps the paper-trading account twice: SQLite for the
scheduled tick (FORESEA_AGENT_ACCOUNT_DB_PATH set) and Datastore for the
Cloud Run tool loop (unset). Every write path is a hand-kept copy, and the
copies have drifted three times, each found by accident:

  - _settlement_fee_rate was venue-blind on the Datastore side only;
  - _ds_apply_trade's cash_required floor could flip to min() and survive
    the whole suite, because nothing ran the _ds_* family;
  - the Datastore writers dropped the versioned ``audit`` block, so every
    Cloud Run trade published as "legacy_record".

This runs one fixed script through each backend's four dispatch points
(_apply_trade_to_account_tables, _record_rejected_account_action,
_settle_agent_open_positions, _load_guard_account) and compares everything
they book: every return value, every action row including its metadata,
the account, the positions, the settlement markers and what the risk guard
reads back. A future edit to one copy and not the other fails here.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzing_llm_rationale import benchmark_tools, market_data  # noqa: E402
from tests.test_benchmark_datastore_account import FakeDatastoreClient  # noqa: E402

AGENT = "parity-model"
CYCLE_1 = "parity-cycle-1"
CYCLE_2 = "parity-cycle-2"
SLUG = "will-it-rain-in-nyc"

# Kalshi KXNO resolves NO and KXYES resolves YES, so settlement pays out on
# one and not the other; the Polymarket slug resolves YES at no settlement
# fee. KXOPEN stays unresolved.
RESOLUTIONS = {"KXNO": 0, "KXYES": 1, SLUG: 1}

ACTION_FIELDS = (
    "action_type", "cycle_id", "mode", "submitted", "platform", "ticker", "side",
    "price", "quantity", "notional", "fee", "settlement_fee", "payout",
    "netting_payout", "cash_required", "cash_delta", "realized_pnl",
    "realized_pairs", "client_order_id", "outcome",
)
ACCOUNT_FIELDS = (
    "starting_cash", "cash", "realized_pnl", "fees_paid", "settlement_fees_paid",
    "high_watermark",
)
POSITION_FIELDS = ("platform", "ticker", "side", "quantity", "cost_basis", "avg_entry_price")


def _normalize(value):
    """Drop per-run identity (ids, clocks) and float noise; keep everything else."""
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items() if k not in ("action_id", "updated_at")}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def _canonical(rows):
    """Order-free comparison: each backend is free to iterate markets differently."""
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, default=str))


def _audit(status, **extra):
    return {"version": 1, "status": status, "execution": {"fill_status": status}, **extra}


class AccountBackendParityTests(unittest.TestCase):
    maxDiff = None

    def run_script(self, backend):
        """Run the script against one backend; return everything it booked."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = {"FORESEA_AGENT_CYCLE_ID": CYCLE_1}
        if backend == "sqlite":
            env["FORESEA_AGENT_ACCOUNT_DB_PATH"] = str(Path(tmp.name) / "accounts.sqlite")
        with mock.patch.dict(os.environ, env, clear=False):
            if backend == "datastore":
                os.environ.pop("FORESEA_AGENT_ACCOUNT_DB_PATH", None)
            self.assertEqual(benchmark_tools._use_datastore_account_store(), backend == "datastore")
            client = FakeDatastoreClient()
            with (
                mock.patch.object(benchmark_tools, "_ds_account_client", client),
                mock.patch.object(market_data, "resolve_kalshi", side_effect=RESOLUTIONS.get),
                mock.patch.object(market_data, "resolve_polymarket", side_effect=RESOLUTIONS.get),
            ):
                returns = self._script()
                booked = self._read_sqlite() if backend == "sqlite" else self._read_datastore(client)
        return {"returns": _normalize(returns), **booked}

    def _script(self):
        base = benchmark_tools._risk_guard_policy()
        cycle_1 = dataclasses.replace(base, account_value=1_000.0, cycle_id=CYCLE_1)
        cycle_2 = dataclasses.replace(cycle_1, cycle_id=CYCLE_2)
        returns = {}

        def trade(label, ticker, side, price, quantity, fee, *, platform="kalshi", audit=None):
            returns[label] = benchmark_tools._apply_trade_to_account_tables(
                agent_id=AGENT, policy=cycle_1, mode="shadow", submitted=False,
                ticker=ticker, side=side, platform=platform,
                normalized={"price": price, "quantity": quantity,
                            "exchange_order": {"client_order_id": f"coid-{label}"}},
                guard={"filled_fee": fee, "cycle_id": CYCLE_1, "allowed": True},
                audit=audit or _audit("shadow_filled_at_market", filled_quantity=quantity),
            )

        def guard(label, policy, ticker, side, platform="kalshi"):
            account, usage = benchmark_tools._load_guard_account(
                AGENT, policy, platform=platform, ticker=ticker, side=side,
            )
            returns[label] = {
                "usage": usage,
                "cash": account.cash,
                "realized_pnl": account.realized_pnl,
                "fees_paid": account.fees_paid,
                "positions": _canonical([
                    {"platform": p.platform, "ident": p.ident, "side": p.side,
                     "quantity": p.quantity, "cost_basis": p.cost_basis}
                    for p in account.open_positions()
                ]),
            }

        # Open, add to, partly net, then fully net one market.
        trade("open_yes", "KXNET", "yes", 0.40, 10.0, 0.35)
        trade("add_yes", "KXNET", "yes", 0.46, 5.0, 0.20)
        trade("partial_net", "KXNET", "no", 0.50, 6.0, 0.10)
        trade("full_net", "KXNET", "no", 0.55, 9.0, 0.10)
        # A remainder within MIN_POSITION_QUANTITY is closed as dust, not kept.
        trade("dust_open", "KXDUST", "yes", 0.30, 5.05, 0.05)
        trade("dust_close", "KXDUST", "no", 0.65, 5.0, 0.05)
        # Positions that settlement will see, on both venues.
        trade("settle_loser", "KXNO", "yes", 0.35, 8.0, 0.15)
        trade("settle_winner", "KXYES", "yes", 0.62, 4.0, 0.10)
        trade("polymarket", SLUG, "yes", 0.58, 10.0, 0.0, platform="polymarket")
        trade("left_open", "KXOPEN", "no", 0.44, 3.0, 0.05)
        # A system-initiated trade: booked the same, skipped by the guard replay.
        trade("rule_exit", "KXOPEN", "yes", 0.50, 1.0, 0.01,
              audit=_audit("shadow_filled_at_market", initiated_by="pre_expiry_exit_rule"))

        # Not added to returns: SQLite hands back the row id, Datastore None.
        benchmark_tools._record_rejected_account_action(
            agent_id=AGENT, mode="shadow", ticker="KXREJ", side="yes", platform="kalshi",
            normalized={"price": 0.41, "quantity": 30.0,
                        "exchange_order": {"client_order_id": "coid-rejected"}},
            guard={"allowed": False, "reasons": ["concentration_limit"], "cycle_id": CYCLE_1,
                   "notional": 12.3, "fee": 0.4, "netting_payout": 0.0, "cash_required": 12.7,
                   "account_value": 1_000.0},
            audit=_audit("rejected_before_execution", risk={"reasons": ["concentration_limit"]}),
        )
        benchmark_tools._record_pre_sizing_rejection(
            agent_id=AGENT, mode="shadow", ticker="KXPRE", side="no", platform="kalshi",
            price=None, quantity=5, reason="no_executable_price",
            initiated_by="pre_expiry_exit_rule",
        )

        guard("guard_cycle_1", cycle_1, "KXNET", "no")
        guard("guard_duplicate", cycle_1, "KXOPEN", "no")
        returns["settle_cycle_2"] = _canonical(
            _normalize(benchmark_tools._settle_agent_open_positions(AGENT, cycle_2))
        )
        returns["settle_cycle_2_again"] = benchmark_tools._settle_agent_open_positions(AGENT, cycle_2)
        guard("guard_cycle_2", cycle_2, SLUG, "no", platform="polymarket")
        return returns

    def _read_sqlite(self):
        with benchmark_tools._account_transaction() as conn:
            account = conn.execute(
                f"SELECT {', '.join(ACCOUNT_FIELDS)} FROM agent_accounts WHERE agent_id = ?", (AGENT,),
            ).fetchone()
            positions = conn.execute(
                f"SELECT {', '.join(POSITION_FIELDS)} FROM agent_positions WHERE agent_id = ?", (AGENT,),
            ).fetchall()
            actions = conn.execute(
                f"SELECT {', '.join(ACTION_FIELDS)}, metadata_json FROM agent_actions WHERE agent_id = ?",
                (AGENT,),
            ).fetchall()
            markers = conn.execute(
                "SELECT cycle_id FROM agent_cycle_settlements WHERE agent_id = ?", (AGENT,),
            ).fetchall()
        return self._booked(
            account=dict(account),
            positions=[dict(p) for p in positions],
            actions=[dict(a) for a in actions],
            markers=[m["cycle_id"] for m in markers],
        )

    def _read_datastore(self, client):
        accounts = client.entities_of_kind(benchmark_tools._DS_ACCOUNT_KIND)
        self.assertEqual(len(accounts), 1)
        return self._booked(
            account={k: accounts[0].get(k) for k in ACCOUNT_FIELDS},
            positions=[{k: p.get(k) for k in POSITION_FIELDS}
                       for p in client.entities_of_kind(benchmark_tools._DS_POSITION_KIND)],
            actions=[{**{k: a.get(k) for k in ACTION_FIELDS}, "metadata_json": a.get("metadata_json")}
                     for a in client.entities_of_kind(benchmark_tools._DS_ACTION_KIND)],
            markers=[m.key.name for m in client.entities_of_kind(benchmark_tools._DS_CYCLE_SETTLEMENT_KIND)],
        )

    @staticmethod
    def _booked(*, account, positions, actions, markers):
        normalized_actions = []
        for row in actions:
            metadata = json.loads(row.pop("metadata_json") or "{}")
            row["submitted"] = bool(row["submitted"])
            normalized_actions.append({**row, "metadata": metadata})
        return {
            "account": _normalize(account),
            "positions": _canonical(_normalize(positions)),
            "actions": _canonical(_normalize(normalized_actions)),
            "settlement_markers": sorted(markers),
        }

    def setUp(self):
        self.sqlite = self.run_script("sqlite")
        self.datastore = self.run_script("datastore")

    def test_the_script_exercises_every_write_path(self):
        """A parity check over an empty ledger would pass for free."""
        kinds = [a["action_type"] for a in self.sqlite["actions"]]
        self.assertEqual(kinds.count("trade"), 11)
        self.assertEqual(kinds.count("rejected_trade"), 2)
        self.assertEqual(kinds.count("settlement"), 3)
        outcomes = {a["outcome"] for a in self.sqlite["actions"]}
        self.assertTrue({"open", "realized", "rejected", "yes", "no"} <= outcomes)
        self.assertEqual(len(self.sqlite["returns"]["settle_cycle_2"]), 3)
        self.assertEqual(self.sqlite["returns"]["settle_cycle_2_again"], [])
        self.assertTrue(self.sqlite["returns"]["guard_duplicate"]["usage"]["duplicate_active"])
        self.assertEqual(
            {(p["ticker"], p["side"]) for p in self.sqlite["positions"]}, {("KXOPEN", "no")},
        )

    def test_both_backends_return_the_same_results(self):
        self.assertEqual(self.datastore["returns"], self.sqlite["returns"])

    def test_both_backends_book_the_same_action_rows(self):
        self.assertEqual(self.datastore["actions"], self.sqlite["actions"])

    def test_both_backends_end_with_the_same_account(self):
        self.assertEqual(self.datastore["account"], self.sqlite["account"])

    def test_both_backends_end_with_the_same_positions(self):
        self.assertEqual(self.datastore["positions"], self.sqlite["positions"])

    def test_both_backends_mark_the_same_cycles_settled(self):
        self.assertEqual(self.datastore["settlement_markers"], self.sqlite["settlement_markers"])


if __name__ == "__main__":
    unittest.main()
