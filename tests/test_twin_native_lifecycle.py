"""Scoped native observations and restart recovery; no venue transport or model."""

import base64
import json
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from analyzing_llm_rationale.datastore_sql import Client
from analyzing_llm_rationale.twin.models import AccountScope, CommandState, canonical_instrument_id
from analyzing_llm_rationale.twin.native_lifecycle import (
    NativeLifecycleMaintenance,
    NativeOrderReader,
)
from analyzing_llm_rationale.twin.recovery import DatastoreLifecycleStore
from analyzing_llm_rationale.twin.runtime_app import _maintenance_operation
from analyzing_llm_rationale.twin.store import DatastoreTwinStore
from analyzing_llm_rationale.twin.worker import (
    DatastoreWorkerJobs,
    WorkerDegraded,
    WorkerJob,
    WorkerJobKind,
)
from tests.test_twin_recovery import NOW, prepared_unknown

CONDITION = "0x" + "c" * 64


class NativeOrderEndpointTests(unittest.TestCase):
    def test_live_native_account_binding_rejects_demo_credentials_before_reads(self):
        from unittest.mock import Mock

        from analyzing_llm_rationale.twin.reconcile import AccountReadError
        from analyzing_llm_rationale.twin.venue_account import complete_account_read_plan

        reader = Mock()
        with self.assertRaises(AccountReadError):
            complete_account_read_plan(
                "kalshi",
                reader=reader,
                connection_ref="live-connection",
                creds={
                    "kalshi_api_key_id": "fixture",
                    "kalshi_private_key": "fixture",
                    "kalshi_base_url": "https://demo-api.kalshi.co/trade-api/v2",
                },
            )
        reader.assert_not_called()

    def test_single_order_reads_use_the_existing_scoped_signed_transport(self):
        from analyzing_llm_rationale import venue_api

        with patch(
            "analyzing_llm_rationale.trading._kalshi_request",
            return_value={"order": {"order_id": "native-id"}},
        ) as send:
            result = venue_api.read(
                "kalshi",
                "order",
                {"order_id": "native-id"},
                access="account",
                creds={"fixture": "not-real"},
            )
        self.assertEqual(result["data"]["order"]["order_id"], "native-id")
        self.assertEqual(send.call_args.args, ("GET", "/portfolio/orders/native-id"))
        client = SimpleNamespace(host="https://clob.polymarket.com")
        from unittest.mock import Mock

        client._get = Mock(return_value={"id": "native-id"})
        client._l2_headers = Mock(return_value={"fixture": "not-real"})
        with patch("analyzing_llm_rationale.trading._polymarket_client", return_value=client):
            self.assertEqual(
                venue_api.read(
                    "polymarket",
                    "order",
                    {"order_id": "native-id"},
                    access="account",
                    creds={"fixture": "not-real"},
                )["data"]["id"],
                "native-id",
            )
        client._l2_headers.assert_called_once_with("GET", "/data/order/native-id")
        self.assertEqual(
            client._get.call_args.args, ("https://clob.polymarket.com/data/order/native-id",)
        )

    def test_order_reads_cannot_become_public_or_unsigned(self):
        from analyzing_llm_rationale import market_data, trading, venue_api

        for venue in ("kalshi", "polymarket"):
            with self.subTest(venue=venue):
                with self.assertRaises(market_data.MarketDataInputError):
                    venue_api.read(venue, "order", {"order_id": "native-id"})
                with self.assertRaises(trading.TradingNotConfiguredError):
                    venue_api.read(venue, "order", {"order_id": "native-id"}, access="account")


class NativeLifecycleTests(unittest.TestCase):
    venue = "kalshi"

    def make_client(self):
        return self.enterContext(closing(Client(path=str(Path(self.root) / "state.sqlite3"))))

    def setUp(self):
        self.root = self.enterContext(TemporaryDirectory())
        self.enterContext(
            patch.dict(
                "os.environ",
                {
                    "FORESEA_DATASTORE_BACKEND": "sql",
                    "FORESEA_TWIN_ORDER_ENCRYPTION_KEY": base64.urlsafe_b64encode(
                        b"0" * 32
                    ).decode(),
                },
            )
        )
        self.db = self.make_client()
        self.store, self.lifecycle, self.jobs = (
            DatastoreTwinStore(self.db),
            DatastoreLifecycleStore(self.db),
            DatastoreWorkerJobs(self.db),
        )
        self.creds = (
            {"kalshi_api_key_id": "fixture-key", "kalshi_private_key": "fixture-private-not-real"}
            if self.venue == "kalshi"
            else {
                "polymarket_api_key": "fixture-api-key",
                "polymarket_api_secret": "fixture-not-real",
                "polymarket_private_key": "fixture-not-real",
                "polymarket_api_passphrase": "fixture-not-real",
            }
        )
        account = (
            "kalshi-key-" + sha256(b"fixture-key").hexdigest()
            if self.venue == "kalshi"
            else "0x" + "a" * 40
        )
        self.scope = AccountScope(
            "native-lifecycle", "owner", self.venue, account, "live", "USD", "connection", 1, NOW
        )
        self.store.register_account(
            self.scope, venue_available_cash=Decimal("10"), loss_limit=Decimal("10")
        )
        _, base, _, _ = prepared_unknown(self.venue)
        self.intent = replace(
            base,
            account_scope_id=self.scope.id,
            instrument_id=canonical_instrument_id(
                venue=self.venue,
                environment="live",
                venue_instrument_id="KXTEST" if self.venue == "kalshi" else "market-1",
                condition_id=None if self.venue == "kalshi" else CONDITION,
            ),
        )
        self.order_id = "venue-order" if self.venue == "kalshi" else "0x" + "1" * 64
        packet = (
            None
            if self.venue == "kalshi"
            else {
                "protocol": "ctf_v2",
                "chain_id": 137,
                "intent_hash": self.intent.intent_hash,
                "venue_instrument_id": "market-1",
                "condition_id": CONDITION,
                "order_type": "FAK",
                "order": {
                    "tokenId": "token-1",
                    "side": "BUY",
                    "maker": account,
                    "makerAmount": "1000000",
                    "takerAmount": "2000000",
                },
            }
        )
        self.store.reserve_intent(
            self.intent,
            cash=Decimal("1"),
            max_loss=Decimal("1"),
            now=NOW,
            client_order_id=None if self.venue == "kalshi" else self.order_id,
            prepared_request=packet,
        )
        command = self.store.command_for_intent(self.intent)
        first = self.store.claim_command(
            command.id, worker_id="lost-sender", now=NOW, lease_seconds=1
        )
        self.store.transition_command(
            command.id,
            target=CommandState.SUBMISSION_UNKNOWN,
            fence=first.fence,
            worker_id=first.worker_id,
        )
        self.command = self.store.command_for_intent(self.intent)
        self.time, self.calls, self.order_reads = NOW + timedelta(seconds=3), [], 0
        self.hook = lambda: None
        self.order = (
            {
                "order_id": self.order_id,
                "client_order_id": self.command.client_order_id,
                "ticker": "KXTEST",
                "status": "executed",
                "action": "buy",
                "side": "yes",
                "initial_count_fp": "2",
                "fill_count_fp": "2",
                "remaining_count_fp": "0",
                "yes_price_dollars": ".5",
                "created_time": NOW.isoformat(),
                "subaccount_number": 0,
            }
            if self.venue == "kalshi"
            else {
                "id": self.order_id,
                "market": CONDITION,
                "asset_id": "token-1",
                "maker_address": account,
                "side": "BUY",
                "price": ".5",
                "original_size": "2",
                "size_matched": "2",
                "status": "MATCHED",
                "associate_trades": ["trade-1"],
                "created_at": int(NOW.timestamp()),
            }
        )
        self.fill = (
            {
                "fill_id": "fill-1",
                "order_id": self.order_id,
                "ticker": "KXTEST",
                "action": "buy",
                "side": "yes",
                "count_fp": "2",
                "yes_price_dollars": ".4",
                "fee_cost": ".01",
                "created_time": NOW.isoformat(),
                "subaccount_number": 0,
            }
            if self.venue == "kalshi"
            else {
                "id": "trade-1",
                "market": CONDITION,
                "asset_id": "token-1",
                "maker_address": account,
                "taker_order_id": self.order_id,
                "side": "BUY",
                "trader_side": "TAKER",
                "size": "2",
                "price": ".4",
                "fee_rate_bps": "0",
                "status": "CONFIRMED",
                "maker_orders": [],
                "match_time": str(int(NOW.timestamp())),
            }
        )
        self.native_client = SimpleNamespace(
            host="https://clob.polymarket.com",
            builder=SimpleNamespace(funder=account),
            signer=SimpleNamespace(address=lambda: account),
            _l2_headers=lambda *_: {"fixture": "not-real"},
            _get=self.trade_page,
        )
        self.enterContext(
            patch(
                "analyzing_llm_rationale.trading._polymarket_client",
                return_value=self.native_client,
            )
        )
        job = WorkerJob(
            "native-order-job",
            self.scope.id,
            WorkerJobKind.RECOVERY,
            {"command_id": self.command.id, "account_epoch_id": "1"},
            NOW + timedelta(seconds=180),
            created_at=NOW,
        )
        self.jobs.add(job)
        self.job = self.jobs.claim(job.id, worker_id="maintenance", now=NOW, lease_seconds=120)

    def reader(self, _venue, operation, parameters, **_kwargs):
        self.calls.append((operation, dict(parameters)))
        if self.venue == "polymarket" and operation == "fills":
            return {
                "data": self.trade_page("fixture", headers={}, params=parameters),
                "next_cursor": None,
            }
        if operation in {"order", "orders"}:
            self.order_reads += 1
            if self.order_reads == 2:
                self.hook()
            value = json.loads(json.dumps(self.order))
            data = (
                (value if self.venue == "polymarket" else {"order": value})
                if operation == "order"
                else {"orders": [value]}
            )
            return {"data": data, "next_cursor": None}
        if operation == "historical_orders":
            return {"data": {"orders": []}, "next_cursor": None}
        return {
            "data": {"fills": [json.loads(json.dumps(self.fill))] if operation == "fills" else []},
            "next_cursor": None,
        }

    def trade_page(self, _url, *, headers, params):
        self.calls.append(("native_trade", dict(params)))
        return {"data": [json.loads(json.dumps(self.fill))], "next_cursor": "LTE="}

    def operation(self):
        return NativeLifecycleMaintenance(
            self.store,
            self.lifecycle,
            self.jobs,
            resolve_reader=lambda scope: NativeOrderReader(
                scope, creds=self.creds, connection_ref=scope.connection_ref, reader=self.reader
            ),
            clock=lambda: self.time,
        )

    def test_private_recovery_applies_unique_native_fills_and_replay_once(self):
        result = _maintenance_operation(
            self.jobs, None, self.job, native_lifecycle=self.operation()
        )
        self.assertEqual(result["command_state"], "filled")
        self.assertEqual(result["filled_quantity"], "2")
        first = self.lifecycle.load(self.scope.id, self.command.id)
        self.assertEqual(first.fills[0].price, Decimal(".4"))
        self.assertEqual(self.operation()(self.job)["filled_quantity"], "2")
        self.assertEqual(self.lifecycle.load(self.scope.id, self.command.id), first)
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))

    def test_native_capture_passes_through_real_signed_read_contracts(self):
        from unittest.mock import Mock

        from analyzing_llm_rationale import venue_api

        fixture_reader = self.reader

        def kalshi_transport(_method, path, *, params, creds):
            operations = {
                "/portfolio/orders": "orders",
                "/historical/orders": "historical_orders",
                "/portfolio/fills": "fills",
                "/historical/fills": "historical_fills",
            }
            operation = operations.get(path, "order")
            return {**fixture_reader("kalshi", operation, params)["data"], "cursor": ""}

        def polymarket_transport(url, *, headers, params):
            if "/data/order/" in url:
                return json.loads(json.dumps(self.order))
            self.assertEqual(params.get("maker_address"), self.scope.venue_account_ref)
            return self.trade_page(url, headers=headers, params=params)

        self.reader = Mock(wraps=venue_api.read)
        with (
            patch("analyzing_llm_rationale.trading._kalshi_request", side_effect=kalshi_transport),
            patch.object(self.native_client, "_get", side_effect=polymarket_transport),
        ):
            self.assertEqual(self.operation()(self.job)["filled_quantity"], "2")
        reads = {call.args[1] for call in self.reader.call_args_list}
        if self.venue == "kalshi":
            self.assertTrue(
                {"order", "orders", "fills", "historical_orders", "historical_fills"} <= reads
            )
        else:
            self.assertIn("fills", reads)

    def test_missing_order_is_unknown_and_never_releases_a_hold(self):
        self.order = {}
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))
        self.assertEqual(
            self.store.command_for_intent(self.intent).state, CommandState.SUBMISSION_UNKNOWN
        )

    def test_order_without_positive_account_binding_never_reconciles(self):
        field = "subaccount_number" if self.venue == "kalshi" else "maker_address"
        for value in (None, False, "0"):
            with self.subTest(value=value):
                self.order[field] = value
                with self.assertRaises(WorkerDegraded):
                    self.operation()(self.job)
                self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))

    def test_fill_with_wrong_account_binding_never_reconciles(self):
        field = "subaccount_number" if self.venue == "kalshi" else "taker_order_id"
        self.fill[field] = False
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))

    def test_decimal_format_and_receive_time_changes_do_not_duplicate_fills(self):
        self.operation()(self.job)
        prior = self.lifecycle.load(self.scope.id, self.command.id)
        self.time += timedelta(seconds=1)
        self.fill["count_fp" if self.venue == "kalshi" else "size"] = "2.0000"
        self.fill["yes_price_dollars" if self.venue == "kalshi" else "price"] = ".4000"
        if self.venue == "kalshi":
            self.fill["fee_cost"] = ".01000"
        self.operation()(self.job)
        self.assertEqual(self.lifecycle.load(self.scope.id, self.command.id), prior)

    def test_zero_fill_cancellation_releases_once_and_late_fills_pause(self):
        self.order["status"] = "canceled" if self.venue == "kalshi" else "CANCELED"
        self.order["fill_count_fp" if self.venue == "kalshi" else "size_matched"] = "0"
        if self.venue == "polymarket":
            self.order["associate_trades"] = []
        original = self.reader

        def reader(venue, operation, parameters, **kwargs):
            value = original(venue, operation, parameters, **kwargs)
            if operation in {"fills", "historical_fills"}:
                value["data"]["fills" if venue == "kalshi" else "data"] = []
            return value

        self.reader = reader
        self.assertEqual(self.operation()(self.job)["command_state"], "cancelled")
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("0"))
        self.operation()(self.job)
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("0"))
        self.reader = original
        self.order["fill_count_fp" if self.venue == "kalshi" else "size_matched"] = "2"
        if self.venue == "polymarket":
            self.order["associate_trades"] = ["trade-1"]
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertEqual(len(self.store.recovery_attention(self.scope.id)), 1)
        self.assertEqual(
            self.lifecycle.load(self.scope.id, self.command.id).filled_quantity, Decimal("0")
        )

    def test_reported_quantity_without_fill_evidence_retains_cash(self):
        if self.venue == "kalshi":
            self.fill["count_fp"] = "1"
        else:
            self.fill["size"] = "1"
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))

    def test_same_native_fill_id_with_new_price_or_fee_requires_attention(self):
        self.operation()(self.job)
        self.fill["yes_price_dollars" if self.venue == "kalshi" else "price"] = ".3"
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertEqual(len(self.store.recovery_attention(self.scope.id)), 1)
        self.assertEqual(
            self.lifecycle.load(self.scope.id, self.command.id).fills[0].price, Decimal(".4")
        )

    def test_expiry_after_lifecycle_write_rolls_back_inventory_and_retains_hold(self):
        save = self.lifecycle.save

        def expire(*args, **kwargs):
            value = save(*args, **kwargs)
            self.time = NOW + timedelta(seconds=121)
            return value

        with (
            patch.object(self.lifecycle, "save", side_effect=expire),
            self.assertRaises(WorkerDegraded),
        ):
            self.operation()(self.job)
        self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))
        self.assertEqual(
            self.store.command_for_intent(self.intent).state, CommandState.ACKNOWLEDGED
        )

    def test_private_sdk_error_is_not_empty_inventory(self):
        def broken(*_args, **_kwargs):
            raise RuntimeError("fixture-private-sdk-text-not-for-telemetry")

        self.reader = broken
        with self.assertRaises(WorkerDegraded) as error:
            self.operation()(self.job)
        self.assertNotIn("private-sdk", str(error.exception))
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))

    def test_changed_order_or_refenced_job_cannot_apply_fills(self):
        self.hook = lambda: self.jobs.claim(
            self.job.id, worker_id="new-job", now=NOW + timedelta(seconds=121), lease_seconds=30
        )
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))

    def test_wrong_market_never_updates_inventory(self):
        self.order["ticker" if self.venue == "kalshi" else "asset_id"] = "wrong-market"
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))

    def test_wrong_action_never_updates_inventory(self):
        self.order["action" if self.venue == "kalshi" else "side"] = (
            "sell" if self.venue == "kalshi" else "SELL"
        )
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))

    def test_incomplete_cursor_pages_never_become_complete_inventory(self):
        fixture_reader = self.reader
        target = "historical_fills" if self.venue == "kalshi" else "fills"
        for mode in ("missing", "repeated", "truncated"):
            with self.subTest(mode=mode):

                def incomplete(venue, operation, parameters, mode=mode, **kwargs):
                    response = fixture_reader(venue, operation, parameters, **kwargs)
                    if operation == target:
                        if self.venue == "kalshi":
                            if mode == "missing":
                                response.pop("next_cursor")
                            elif mode == "truncated":
                                response["pagination_limit_reached"] = True
                            else:
                                response["next_cursor"] = "repeat"
                        else:
                            if mode == "missing":
                                response["data"].pop("next_cursor")
                            elif mode == "truncated":
                                response["data"]["data"] = None
                            else:
                                response["data"]["next_cursor"] = "repeat"
                    return response

                self.reader = incomplete
                with self.assertRaises(WorkerDegraded):
                    self.operation()(self.job)
                self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))
                self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))

    def test_missing_intent_never_reads_provider_or_changes_cash(self):
        self.db.delete(self.store._key(self.scope.id, "TwinIntent", self.intent.intent_hash))
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))

    def test_actual_fill_cost_above_reserved_cash_pauses_new_exposure(self):
        self.fill["fee_cost" if self.venue == "kalshi" else "fee"] = "2"
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))
        attention = self.store.recovery_attention(self.scope.id)
        self.assertEqual(attention[0]["reason"], "fill_cost_exceeds_reservation")
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))

    def test_reserved_command_is_not_promoted_into_a_submission(self):
        intent = replace(self.intent, id="never-dispatched", market_version="new-market-version")
        self.store.reserve_intent(intent, cash=Decimal("1"), max_loss=Decimal("1"), now=NOW)
        command = self.store.command_for_intent(intent)
        queued = WorkerJob(
            "never-dispatched-job",
            self.scope.id,
            WorkerJobKind.RECOVERY,
            {"command_id": command.id, "account_epoch_id": "1"},
            self.job.deadline,
            created_at=NOW,
        )
        self.jobs.add(queued)
        job = self.jobs.claim(queued.id, worker_id="maintenance", now=NOW, lease_seconds=120)
        with self.assertRaises(WorkerDegraded):
            self.operation()(job)
        self.assertEqual(self.store.command_for_intent(intent).state, CommandState.RESERVED)
        self.assertEqual(self.calls, [])

    def test_open_order_stays_pending_with_its_cash_hold(self):
        self.order["status"] = "resting" if self.venue == "kalshi" else "LIVE"
        self.order["fill_count_fp" if self.venue == "kalshi" else "size_matched"] = "1"
        self.order["remaining_count_fp"] = "1"
        self.fill["count_fp" if self.venue == "kalshi" else "size"] = "1"
        result = self.operation()(self.job)
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["command_state"], "partially_filled")
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))

    def test_pending_recovery_stages_one_delayed_restart_safe_followup(self):
        self.order["status"] = "resting" if self.venue == "kalshi" else "LIVE"
        self.order["fill_count_fp" if self.venue == "kalshi" else "size_matched"] = "1"
        self.order["remaining_count_fp"] = "1"
        self.fill["count_fp" if self.venue == "kalshi" else "size"] = "1"
        result = self.operation()(self.job)
        child_id = result["followup_job_id"]
        child = DatastoreWorkerJobs(self.db).get(child_id)
        self.assertEqual(child.account_scope_id, self.scope.id)
        self.assertEqual(child.payload["command_id"], self.command.id)
        self.assertEqual(child.deadline, self.job.deadline)
        self.assertGreater(child.not_before, self.time)
        self.assertIsNone(self.jobs.claim(child_id, worker_id="early", now=self.time))
        self.time += timedelta(seconds=1)
        self.assertEqual(self.operation()(self.job)["followup_job_id"], child_id)
        self.assertEqual(self.jobs.get(child_id), child)
        self.time = child.not_before
        self.assertIn(child_id, {item.id for item in self.jobs.due(now=self.time)})
        next_job = self.jobs.claim(child_id, worker_id="followup", now=self.time, lease_seconds=30)
        self.order["status"] = "canceled" if self.venue == "kalshi" else "CANCELED"
        self.order["remaining_count_fp"] = "0"
        final = self.operation()(next_job)
        self.assertEqual(final["status"], "complete")
        self.assertIsNone(final["followup_job_id"])

    def test_followup_persistence_failure_rolls_back_the_lifecycle_commit(self):
        self.order["status"] = "resting" if self.venue == "kalshi" else "LIVE"
        add = self.jobs._add_in_transaction

        def lose_child(child):
            add(child)
            raise RuntimeError("fixture-child-write-failed")

        with patch.object(self.jobs, "_add_in_transaction", side_effect=lose_child):
            with self.assertRaises(WorkerDegraded):
                self.operation()(self.job)
        self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))
        self.assertEqual(len(list(self.db.query(kind="TwinWorkerJob").fetch())), 1)

    def test_pending_recovery_has_finite_deadline_and_poll_budgets(self):
        self.order["status"] = "resting" if self.venue == "kalshi" else "LIVE"
        queued = WorkerJob(
            "last-recovery-round",
            self.scope.id,
            WorkerJobKind.RECOVERY,
            {
                "command_id": self.command.id,
                "account_epoch_id": "1",
                "recovery_root_id": self.job.id,
                "recovery_round_id": "16",
            },
            self.job.deadline,
            created_at=NOW,
        )
        self.jobs.add(queued)
        claimed = self.jobs.claim(queued.id, worker_id="last", now=NOW, lease_seconds=120)
        result = self.operation()(claimed)
        self.assertEqual(result["status"], "reconciliation_required")
        self.assertTrue(result["followup_budget_exhausted"])
        self.assertIsNone(result["followup_job_id"])
        self.assertEqual(len(list(self.db.query(kind="TwinWorkerJob").fetch())), 2)
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))

    def test_lost_followup_dispatch_keeps_a_durable_due_job(self):
        from unittest.mock import Mock

        self.order["status"] = "resting" if self.venue == "kalshi" else "LIVE"
        operation = self.operation()
        operation.dispatcher = Mock()
        operation.dispatcher.enqueue.side_effect = RuntimeError("fixture-lost-queue-ack")
        result = operation(self.job)
        child = self.jobs.get(result["followup_job_id"])
        self.assertEqual(result["status"], "pending")
        self.assertIn(child, DatastoreWorkerJobs(self.db).due(now=child.not_before))

    def test_pending_recovery_cannot_extend_its_original_deadline(self):
        self.order["status"] = "resting" if self.venue == "kalshi" else "LIVE"
        queued = WorkerJob(
            "short-recovery",
            self.scope.id,
            WorkerJobKind.RECOVERY,
            {"command_id": self.command.id, "account_epoch_id": "1"},
            NOW + timedelta(seconds=20),
            created_at=NOW,
        )
        self.jobs.add(queued)
        claimed = self.jobs.claim(queued.id, worker_id="short", now=NOW, lease_seconds=120)
        result = self.operation()(claimed)
        self.assertEqual(result["status"], "reconciliation_required")
        self.assertTrue(result["followup_budget_exhausted"])
        self.assertEqual(len(list(self.db.query(kind="TwinWorkerJob").fetch())), 2)

    def test_history_resumes_past_twenty_pages_on_followup_jobs(self):
        original = self.reader
        requested = []
        target = "historical_fills" if self.venue == "kalshi" else "fills"

        def paged(venue, operation, parameters, **kwargs):
            if operation == target:
                page = int(parameters.get("cursor" if venue == "kalshi" else "next_cursor", "0"))
                requested.append(page)
                row = dict(self.fill)
                if page < 24:
                    row["order_id" if venue == "kalshi" else "id"] = "unrelated"
                    if venue == "polymarket":
                        row["taker_order_id"] = "unrelated-order"
                cursor = str(page + 1) if page < 24 else None
                if venue == "kalshi":
                    return {"data": {"fills": [row]}, "next_cursor": cursor}
                return {
                    "data": {"data": [row], "next_cursor": cursor or "LTE="},
                    "next_cursor": cursor,
                }
            if venue == "kalshi" and operation == "fills":
                return {"data": {"fills": []}, "next_cursor": None}
            return original(venue, operation, parameters, **kwargs)

        self.reader = paged
        root = WorkerJob(
            "long-history",
            self.scope.id,
            WorkerJobKind.RECOVERY,
            {"command_id": self.command.id, "account_epoch_id": "1"},
            NOW + timedelta(minutes=30),
            created_at=NOW,
        )
        self.jobs.add(root)
        current = self.jobs.claim(root.id, worker_id="history", now=NOW, lease_seconds=120)
        for _ in range(17):
            result = self.operation()(current)
            if result["status"] == "complete":
                break
            self.assertEqual(result["reason"], "native_history_incomplete")
            self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))
            self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))
            child = self.jobs.get(result["followup_job_id"])
            self.time = child.not_before
            current = self.jobs.claim(
                child.id, worker_id="history-next", now=self.time, lease_seconds=120
            )
        else:
            self.fail("bounded continuation never completed")
        self.assertEqual(result["filled_quantity"], "2")
        self.assertEqual(requested, list(range(25)))
        self.assertEqual(list(self.db.query(kind="TwinNativeHistoryChunk").fetch()), [])

    def test_changed_order_frame_discards_history_and_stages_a_coherent_retry(self):
        original = self.reader
        seen_fills, changed = False, False

        def changes(venue, operation, parameters, **kwargs):
            nonlocal seen_fills, changed
            response = original(venue, operation, parameters, **kwargs)
            if operation == "fills":
                seen_fills = True
            if operation == "order" and seen_fills and not changed:
                changed = True
                status = "canceled" if venue == "kalshi" else "CANCELED"
                self.order["status"] = status
                row = response["data"]["order"] if venue == "kalshi" else response["data"]
                row["status"] = status
            return response

        self.reader = changes
        result = self.operation()(self.job)
        self.assertEqual(result["status"], "pending")
        self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))
        self.assertEqual(list(self.db.query(kind="TwinNativeHistoryChunk").fetch()), [])
        child = self.jobs.get(result["followup_job_id"])
        self.time = child.not_before
        current = self.jobs.claim(child.id, worker_id="coherent", now=self.time, lease_seconds=60)
        self.assertEqual(self.operation()(current)["status"], "complete")

    def test_failed_complete_capture_refreshes_when_upstream_corrects_its_price(self):
        field = "yes_price_dollars" if self.venue == "kalshi" else "price"
        self.fill[field] = ".9"
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.fill[field] = ".4"
        self.time += timedelta(seconds=1)
        self.assertEqual(self.operation()(self.job)["status"], "complete")
        self.assertEqual(
            self.lifecycle.load(self.scope.id, self.command.id).fills[0].price, Decimal(".4")
        )


class PolymarketNativeLifecycleTests(NativeLifecycleTests):
    venue = "polymarket"

    def test_taker_multi_maker_notional_is_checked_against_native_trade_price(self):
        self.fill["maker_orders"] = [
            {
                "order_id": "maker-" + str(i),
                "asset_id": "token-1",
                "side": "SELL",
                "matched_amount": "1",
                "price": price,
                "maker_address": "0x" + "b" * 40,
            }
            for i, price in enumerate((".3", ".5"))
        ]
        result = self.operation()(self.job)
        self.assertEqual(result["filled_quantity"], "2")
        self.assertEqual(
            self.lifecycle.load(self.scope.id, self.command.id).fills[0].price, Decimal(".4")
        )

    def test_taker_inconsistent_maker_notional_cannot_publish_inventory(self):
        self.fill["maker_orders"] = [
            {
                "order_id": "maker-1",
                "asset_id": "token-1",
                "side": "SELL",
                "matched_amount": "2",
                "price": ".3",
            },
        ]
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))

    def test_explicit_native_taker_fee_is_preserved_for_a_fee_bearing_trade(self):
        self.fill.update(fee_rate_bps="100", fee=".02")
        self.operation()(self.job)
        self.assertEqual(
            self.lifecycle.load(self.scope.id, self.command.id).fills[0].fee, Decimal(".02")
        )

    def test_unlisted_owned_trade_prevents_zero_fill_cancellation_release(self):
        self.order.update(status="CANCELED", size_matched="0", associate_trades=[])
        result = self.operation()(self.job)
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["reason"], "native_history_incomplete")
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))
        self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))

    def test_maker_complementary_leg_uses_owned_economics_not_parent_trade_size(self):
        self.order.update(size_matched=".5", status="LIVE")
        self.fill.update(
            taker_order_id="another-order",
            size="50",
            asset_id="another-token",
            trader_side="MAKER",
            maker_orders=[
                {
                    "order_id": self.order_id,
                    "asset_id": "token-1",
                    "maker_address": self.scope.venue_account_ref,
                    "side": "BUY",
                    "matched_amount": ".5",
                    "price": ".4",
                    "fee_rate_bps": "100",
                    "fee": ".01",
                }
            ],
        )
        result = self.operation()(self.job)
        self.assertEqual(result["filled_quantity"], "0.5")
        self.assertEqual(result["command_state"], "partially_filled")
        fill = self.lifecycle.load(self.scope.id, self.command.id).fills[0]
        self.assertEqual(fill.price, Decimal(".4"))
        self.assertEqual(fill.fee, Decimal(".01"))

    def test_pending_trade_is_not_a_fill_or_final_cancellation(self):
        self.fill["status"] = "MINED"
        self.order["status"] = "CANCELED"
        self.operation()(self.job)
        projection = self.lifecycle.load(self.scope.id, self.command.id)
        self.assertEqual(projection.filled_quantity, Decimal("0"))
        self.assertFalse(projection.terminal_evidence_complete)
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal("1"))

    def test_pending_cancel_can_later_reconcile_a_confirmed_fill(self):
        self.fill["status"] = "MINED"
        self.order["status"] = "CANCELED"
        self.assertEqual(self.operation()(self.job)["command_state"], "acknowledged")
        self.fill["status"] = "CONFIRMED"
        self.time += timedelta(seconds=1)
        result = self.operation()(self.job)
        self.assertEqual(result["command_state"], "filled")
        self.assertEqual(result["filled_quantity"], "2")
        self.assertEqual(self.store.recovery_attention(self.scope.id), ())

    def test_taker_fill_does_not_require_counterparty_maker_wallet_to_be_ours(self):
        self.fill["maker_address"] = "0x" + "b" * 40
        result = self.operation()(self.job)
        self.assertEqual(result["filled_quantity"], "2")

    def test_maker_fill_still_requires_our_wallet(self):
        self.fill.update(
            taker_order_id="counterparty-order",
            trader_side="MAKER",
            maker_orders=[
                {
                    "order_id": self.order_id,
                    "asset_id": "token-1",
                    "side": "BUY",
                    "maker_address": "0x" + "b" * 40,
                    "matched_amount": "2",
                    "price": ".4",
                    "fee_rate_bps": "0",
                }
            ],
        )
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))

    def test_nonzero_fee_rate_without_actual_fee_is_not_invented(self):
        self.fill["fee_rate_bps"] = "100"
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.lifecycle.load(self.scope.id, self.command.id))

    def test_one_thousand_associated_trades_are_batched_and_resumed_without_rereads(self):
        original = self.reader
        trades = [{**self.fill, "id": "trade-" + str(i), "size": ".002"} for i in range(1000)]
        self.order["associate_trades"] = [row["id"] for row in trades]
        requested = []

        def batched(venue, operation, parameters, **kwargs):
            if operation == "fills":
                offset = int(parameters.get("next_cursor", "0"))
                requested.append(offset)
                next_cursor = str(offset + 100) if offset + 100 < len(trades) else "LTE="
                return {
                    "data": {"data": trades[offset : offset + 100], "next_cursor": next_cursor},
                    "next_cursor": None if next_cursor == "LTE=" else next_cursor,
                }
            return original(venue, operation, parameters, **kwargs)

        self.reader = batched
        first = self.operation()(self.job)
        self.assertEqual(first["status"], "pending")
        child = self.jobs.get(first["followup_job_id"])
        self.time = child.not_before
        current = self.jobs.claim(
            child.id, worker_id="trade-history", now=self.time, lease_seconds=60
        )
        result = self.operation()(current)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(Decimal(result["filled_quantity"]), Decimal("2"))
        self.assertEqual(requested, list(range(0, 1000, 100)))
        self.assertEqual(len(self.lifecycle.load(self.scope.id, self.command.id).fills), 1000)
