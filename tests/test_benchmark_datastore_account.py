"""The Datastore branch of the agent account store, which nothing ran.

benchmark_tools keeps two implementations of the paper-trading account:
a local one and a Datastore one. The pair has form -- an earlier audit
found _settlement_fee_rate venue-blind on the Datastore side while the
local side was correct -- and the reason is structural. The unit suite
never constructs a Datastore client, and the emulator job in CI runs
only tests.test_twin_store_integration, so nothing in either job
executes benchmark_tools._ds_*.

What that hid this time:

    _ds_apply_trade:  cash_required = max(0.0, -cash_delta)
                                   -> min(0.0, -cash_delta)   SURVIVED

against the whole 2,034-test suite, while the same expression on the
local path (line 1255) and the two that read it back are all caught.

It is not cosmetic. _ds_apply_trade writes cash_required onto the action
entity, and _ds_load_guard_account replays those actions to rebuild
cycle_spend and daily_risk. Under min() every buy records zero cash
consumed, the rebuilt totals stay at zero, and the per-cycle spend cap
and daily risk cap never bind -- an agent could trade past both without
anything raising.

FakeDatastoreClient below is deliberately a little more than this one
test needs (query, delete, transaction) so the rest of the _ds_ family
can be covered without building it again.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.cloud import datastore  # noqa: E402
from scripts import build_agent_trading_audit  # noqa: E402

from analyzing_llm_rationale import benchmark_tools, market_data  # noqa: E402


class _FakeQuery:
    def __init__(self, store, kind, ancestor):
        self._store, self._kind, self._ancestor = store, kind, ancestor

    def fetch(self, limit=None):
        prefix = tuple(self._ancestor.flat_path) if self._ancestor is not None else ()
        found = [
            entity
            for path, entity in self._store.items()
            if path[: len(prefix)] == prefix
            and (self._kind is None or path[-2] == self._kind)
        ]
        return found[:limit] if limit else found


class FakeDatastoreClient:
    """Enough of google.cloud.datastore.Client for the account store.

    Keys are real datastore.Key objects, so the key helpers under test
    build them exactly as they do in production; only the storage and the
    transaction are stubbed. The transaction is a plain context manager
    because these tests are single-writer -- the retry loop around
    conflicts is a separate concern.
    """

    def __init__(self):
        self.store = {}

    def key(self, *path):
        return datastore.Key(*path, project="foresea-test")

    def get(self, key):
        return self.store.get(tuple(key.flat_path))

    def put(self, entity):
        self.store[tuple(entity.key.flat_path)] = entity

    def delete(self, key):
        self.store.pop(tuple(key.flat_path), None)

    def query(self, kind=None, ancestor=None, **_kwargs):
        return _FakeQuery(self.store, kind, ancestor)

    @contextlib.contextmanager
    def transaction(self):
        yield object()

    def entities_of_kind(self, kind):
        return [e for path, e in self.store.items() if path[-2] == kind]


class DatastoreCashRequiredTests(unittest.TestCase):
    AGENT = "agent-under-test"
    TICKER = "TEST-TICKER"

    def setUp(self):
        self.client = FakeDatastoreClient()
        previous = benchmark_tools._ds_account_client
        benchmark_tools._ds_account_client = self.client
        self.addCleanup(
            setattr, benchmark_tools, "_ds_account_client", previous
        )
        self.policy = benchmark_tools._risk_guard_policy()

    def _trade(self, *, side, price, quantity, fee, mode="open"):
        return benchmark_tools._ds_apply_trade(
            agent_id=self.AGENT,
            policy=self.policy,
            mode=mode,
            submitted=True,
            ticker=self.TICKER,
            side=side,
            normalized={"price": price, "quantity": quantity},
            guard={"fee": fee},
            platform="kalshi",
        )

    def test_a_buy_records_the_cash_it_actually_consumed(self):
        result = self._trade(side="yes", price=0.40, quantity=10.0, fee=0.35)
        self.assertAlmostEqual(result["cash_delta"], -4.35)
        self.assertAlmostEqual(result["cash_required"], 4.35)
        self.assertGreater(
            result["cash_required"], 0.0, "a buy that spent cash must record it"
        )

    def test_the_recorded_action_carries_the_figure_the_guards_replay(self):
        """_ds_load_guard_account rebuilds spend from these entities."""
        self._trade(side="yes", price=0.40, quantity=10.0, fee=0.35)
        actions = [
            entity
            for entity in self.client.store.values()
            if entity.get("action_type") == "trade"
        ]
        self.assertEqual(len(actions), 1)
        self.assertAlmostEqual(actions[0]["cash_required"], 4.35)

    def test_cash_required_tracks_the_size_of_the_trade(self):
        """Constant-zero and constant-anything both fail this."""
        small = self._trade(side="yes", price=0.10, quantity=1.0, fee=0.01)
        self.setUp()
        large = self._trade(side="yes", price=0.60, quantity=50.0, fee=1.20)
        self.assertAlmostEqual(small["cash_required"], 0.11)
        self.assertAlmostEqual(large["cash_required"], 31.20)
        self.assertGreater(large["cash_required"], small["cash_required"])

    def test_a_cash_returning_trade_records_zero_rather_than_a_negative(self):
        """What the floor is for: netting can hand cash back."""
        self._trade(side="yes", price=0.40, quantity=10.0, fee=0.35)
        netted = self._trade(
            side="no", price=0.10, quantity=10.0, fee=0.05, mode="close"
        )
        self.assertGreater(netted["cash_delta"], 0.0, "this trade should return cash")
        self.assertEqual(
            netted["cash_required"], 0.0, "a trade that returns cash consumes none"
        )


def _kalshi_quote(ticker, *, bid, ask):
    return {
        "platform": "Kalshi", "ident": ticker, "question": "Q?",
        "probability": (bid + ask) / 2, "yes_bid": bid, "yes_ask": ask,
        "close_time": "2026-09-01T00:00:00Z", "created_time": "2026-05-01T00:00:00Z",
    }


class DatastoreTradeAuditTests(unittest.TestCase):
    """place_trade on the Datastore store keeps the versioned audit block.

    FORESEA_AGENT_ACCOUNT_DB_PATH unset selects Datastore, which is the Cloud
    Run tool loop's path. _apply_trade_to_account_tables and
    _record_rejected_account_action took an ``audit`` argument and did not
    forward it, and the _ds_* writers stored only risk_guard. Every Datastore
    trade lost its requested order, quote, sizing, risk and fill status, and
    published as "legacy_record". The scheduled tick runs on SQLite, so the
    SQLite-only audit tests never saw it.
    """

    AGENT = "datastore-audit-model"
    RULE = "pre_expiry_exit_rule"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = mock.patch.dict(os.environ, {
            "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(tmp.name) / "ledger.jsonl"),
            "FORESEA_AGENT_NOTES_PATH": str(Path(tmp.name) / "notes.json"),
            "FORESEA_AGENT_PLACE_TRADE_MODE": "shadow",
            "FORESEA_AGENT_CYCLE_ID": "datastore-audit-cycle",
        }, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("FORESEA_AGENT_ACCOUNT_DB_PATH", None)
        self.assertTrue(benchmark_tools._use_datastore_account_store())

        self.client = FakeDatastoreClient()
        previous = benchmark_tools._ds_account_client
        benchmark_tools._ds_account_client = self.client
        self.addCleanup(setattr, benchmark_tools, "_ds_account_client", previous)
        resolve = mock.patch.object(market_data, "resolve_kalshi", return_value=None)
        resolve.start()
        self.addCleanup(resolve.stop)

    def place(self, ticker, *, side="yes", bid=0.40, ask=0.42, initiated_by=None):
        ctx = benchmark_tools.ToolContext(agent_id=self.AGENT, initiated_by=initiated_by)
        with mock.patch.object(market_data, "fetch_kalshi", return_value=_kalshi_quote(ticker, bid=bid, ask=ask)):
            return benchmark_tools.place_trade(
                {"ticker": ticker, "side": side, "price": 0.42, "quantity": 10}, ctx,
            )

    def recorded(self, ticker):
        rows = [
            e for e in self.client.entities_of_kind(benchmark_tools._DS_ACTION_KIND)
            if e.get("ticker") == ticker
        ]
        self.assertEqual(len(rows), 1, rows)
        return rows[0]

    def audit(self, ticker):
        return json.loads(self.recorded(ticker)["metadata_json"])["audit"]

    def assert_published_as_recorded(self, ticker):
        published = build_agent_trading_audit._audit_context(self.recorded(ticker)["metadata_json"])
        self.assertNotEqual(published["status"], "legacy_record")
        self.assertEqual(published["version"], 1)

    def test_a_fill_records_the_full_audit_block(self):
        self.assertTrue(self.place("KXFILL")["ok"])
        row = self.recorded("KXFILL")
        self.assertEqual(row["action_type"], "trade")
        audit = self.audit("KXFILL")
        self.assertEqual(audit["version"], 1)
        self.assertEqual(audit["requested_order"], {"price": 0.42, "quantity": 10})
        self.assertEqual(audit["quote"]["observed_ask"], 0.42)
        self.assertIn("risk", audit)
        self.assertEqual(audit["execution"]["filled_quantity"], 10.0)
        self.assertNotIn("initiated_by", audit, "an agent's own trade carries no tag")
        self.assertIn("risk_guard", json.loads(row["metadata_json"]))
        self.assert_published_as_recorded("KXFILL")

    def test_a_system_initiated_fill_records_who_placed_it(self):
        self.assertTrue(self.place("KXRULEFILL", initiated_by=self.RULE)["ok"])
        self.assertEqual(self.audit("KXRULEFILL")["initiated_by"], self.RULE)
        self.assert_published_as_recorded("KXRULEFILL")

    def test_a_guard_rejection_records_the_audit_block(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_CONCENTRATION_LIMIT": "0.0001"}):
            result = self.place("KXREJECT")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "concentration_limit")
        self.assertEqual(self.recorded("KXREJECT")["action_type"], "rejected_trade")
        audit = self.audit("KXREJECT")
        self.assertEqual(audit["version"], 1)
        self.assertEqual(audit["requested_order"], {"price": 0.42, "quantity": 10})
        self.assertIn("concentration_limit", audit["risk"]["reasons"])
        self.assertEqual(audit["execution"]["filled_quantity"], 0.0)
        self.assertNotIn("initiated_by", audit)
        self.assert_published_as_recorded("KXREJECT")

    def test_a_system_initiated_guard_rejection_records_who_placed_it(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_CONCENTRATION_LIMIT": "0.0001"}):
            self.assertFalse(self.place("KXRULEREJECT", initiated_by=self.RULE)["ok"])
        self.assertEqual(self.audit("KXRULEREJECT")["initiated_by"], self.RULE)

    def test_a_pre_sizing_rejection_records_the_audit_block(self):
        # yes_bid 0 leaves a NO order no executable price: refused before sizing.
        self.assertFalse(self.place("KXPRESIZE", side="no", bid=0.0, ask=0.02)["ok"])
        audit = self.audit("KXPRESIZE")
        self.assertEqual(audit["status"], "rejected_before_sizing")
        self.assertEqual(audit["risk"]["reasons"], ["no_executable_price"])
        self.assertNotIn("initiated_by", audit)
        self.assert_published_as_recorded("KXPRESIZE")

    def test_a_system_initiated_pre_sizing_rejection_records_who_placed_it(self):
        self.place("KXRULEPRESIZE", side="no", bid=0.0, ask=0.02, initiated_by=self.RULE)
        self.assertEqual(self.audit("KXRULEPRESIZE")["initiated_by"], self.RULE)

    def test_the_guard_replay_skips_a_system_fill_on_datastore_too(self):
        """_risk_usage reads initiated_by back out of the stored audit.

        With the block dropped, a system close on Datastore used up the agent's
        per-cycle trade count exactly like the agent's own order.
        """
        def cycle_trades():
            _account, usage = benchmark_tools._load_guard_account(
                self.AGENT, benchmark_tools._risk_guard_policy(),
                platform="kalshi", ticker="KXOTHER", side="yes",
            )
            return usage["cycle_trade_count"]

        self.assertTrue(self.place("KXSYSTEM", initiated_by=self.RULE)["ok"])
        self.assertEqual(cycle_trades(), 0)
        self.assertTrue(self.place("KXAGENT")["ok"])
        self.assertEqual(cycle_trades(), 1)


if __name__ == "__main__":
    unittest.main()
