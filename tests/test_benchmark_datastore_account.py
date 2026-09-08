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
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google.cloud import datastore  # noqa: E402

from analyzing_llm_rationale import benchmark_tools  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
