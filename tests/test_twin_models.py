from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from analyzing_llm_rationale.twin import (
    AccountScope,
    CommandState,
    Completeness,
    Forecast,
    Instrument,
    MarketCursor,
    MarketSnapshot,
    PassDecision,
    Proposal,
    ProposalAction,
    RejectionReason,
    SchemaValidationError,
    TradeIntent,
    can_transition_command,
    canonical_instrument_id,
)

NOW = datetime(2025, 1, 3, tzinfo=timezone.utc)
CREATED = datetime(2025, 1, 1, tzinfo=timezone.utc)
EXPIRES = datetime(2025, 1, 2, tzinfo=timezone.utc)


def scope() -> AccountScope:
    return AccountScope(
        id="account-kalshi-demo",
        owner_id="owner-001",
        venue="kalshi",
        venue_account_ref="account-opaque-ref",
        environment="demo",
        collateral_asset="USD",
        connection_ref="connection-001",
        account_epoch=1,
        created_at=CREATED,
    )


def intent(**updates) -> TradeIntent:
    values = {
        "id": "intent-001",
        "account_scope_id": "account-kalshi-demo",
        "account_epoch": 1,
        "instrument_id": "kalshi:demo:KXTEST",
        "action": ProposalAction.BUY_YES,
        "quantity": Decimal("2.0000"),
        "limit_price": Decimal("0.4100"),
        "time_in_force": "IOC",
        "forecast_id": "forecast-001",
        "exit_reason": None,
        "policy_version": "policy-v1",
        "strategy_version": "foresea-edge-v1",
        "market_version": "market-v1",
        "fee_allowance": Decimal("0.0100"),
        "slippage_allowance": Decimal("0.0050"),
        "expires_at": EXPIRES,
        "created_at": CREATED,
    }
    values.update(updates)
    return TradeIntent(**values)


class TwinModelTests(unittest.TestCase):
    def test_trade_intent_round_trip_uses_decimal_strings_and_hash(self):
        record = intent(presentation_text="Visible rationale")

        stored = record.to_storage()
        restored = TradeIntent.from_storage(stored)

        self.assertEqual(stored["quantity"], "2.0000")
        self.assertEqual(stored["limit_price"], "0.4100")
        self.assertEqual(restored, record)
        self.assertEqual(restored.intent_hash, record.intent_hash)

    def test_trade_intent_fixture_round_trips(self):
        path = Path(__file__).parent / "fixtures" / "twin" / "records" / "trade_intent_v1.json"
        record = TradeIntent.from_storage(json.loads(path.read_text(encoding="utf-8")))

        self.assertEqual(record.action, ProposalAction.BUY_YES)
        self.assertEqual(record.quantity, Decimal("2.0000"))
        self.assertEqual(record.to_storage()["intent_hash"], record.intent_hash)

    def test_non_finite_and_unsupported_precision_are_rejected(self):
        for value in ("NaN", "Infinity", "-Infinity", "0.123456789"):
            with self.subTest(value=value):
                with self.assertRaises(SchemaValidationError):
                    intent(limit_price=value)
        with self.assertRaises(SchemaValidationError):
            intent(quantity="-1")

    def test_naive_and_future_timestamps_are_rejected(self):
        with self.assertRaises(SchemaValidationError):
            scope().__class__(
                id="account-2",
                owner_id="owner-001",
                venue="kalshi",
                venue_account_ref="account-opaque-ref",
                environment="demo",
                collateral_asset="USD",
                connection_ref="connection-001",
                account_epoch=1,
                created_at=datetime(2025, 1, 1),
            )
        with self.assertRaises(SchemaValidationError):
            intent(created_at=datetime.now(timezone.utc) + timedelta(days=1), expires_at=datetime.now(timezone.utc) + timedelta(days=2))

    def test_future_expiry_and_resolution_are_valid_scheduled_times(self):
        future = datetime.now(timezone.utc) + timedelta(days=2)
        scheduled = intent(expires_at=future)
        self.assertEqual(scheduled.expires_at, future)

    def test_instrument_id_is_execution_identity_not_display_alias(self):
        identifier = canonical_instrument_id(
            venue="polymarket",
            environment="demo",
            condition_id="condition-001",
            venue_instrument_id="market-001",
        )
        instrument = Instrument(
            id=identifier,
            venue="polymarket",
            environment="demo",
            venue_instrument_id="market-001",
            condition_id="condition-001",
            yes_token_id="yes-token-001",
            no_token_id="no-token-001",
            settlement_spec_hash="settlement-v1",
            category="politics",
            event_id="event-001",
            cluster_id="cluster-001",
            tick_size=Decimal("0.01"),
            min_quantity=Decimal("1"),
            fee_version="fees-v1",
            capability_version="capabilities-v1",
            status="open",
            close_at=CREATED,
            resolution_at=EXPIRES,
            created_at=CREATED,
            display_title="Same display title",
            display_slug="same-title",
        )

        self.assertEqual(instrument.id, "polymarket:demo:condition-001:market-001")
        with self.assertRaises(SchemaValidationError):
            Instrument(**{**instrument.__dict__, "id": "same-title"})

    def test_canonical_instrument_identity_distinguishes_environment_and_condition(self):
        one = canonical_instrument_id(
            venue="polymarket", environment="demo", condition_id="condition-a", venue_instrument_id="market-1"
        )
        two = canonical_instrument_id(
            venue="polymarket", environment="live", condition_id="condition-a", venue_instrument_id="market-1"
        )
        three = canonical_instrument_id(
            venue="polymarket", environment="demo", condition_id="condition-b", venue_instrument_id="market-1"
        )
        self.assertEqual(len({one, two, three}), 3)

    def test_intent_hash_changes_for_execution_or_authorization_fields_only(self):
        record = intent(presentation_text="First presentation")
        self.assertEqual(record.intent_hash, intent(presentation_text="Second presentation").intent_hash)
        self.assertNotEqual(record.intent_hash, intent(quantity=Decimal("2.1")).intent_hash)
        self.assertNotEqual(record.intent_hash, intent(policy_version="policy-v2").intent_hash)
        self.assertNotEqual(record.intent_hash, intent(account_epoch=2).intent_hash)

    def test_unknown_or_secret_like_trade_intent_fields_are_rejected(self):
        payload = intent().to_storage()
        payload["execute"] = True
        with self.assertRaisesRegex(SchemaValidationError, "execute"):
            TradeIntent.from_storage(payload)
        payload = intent().to_storage()
        payload["credentials"] = {"private_key": "secret"}
        with self.assertRaisesRegex(SchemaValidationError, "credentials"):
            TradeIntent.from_storage(payload)

    def test_intent_rejects_mutated_serialized_hash(self):
        payload = intent().to_storage()
        payload["quantity"] = "3.0000"
        with self.assertRaisesRegex(SchemaValidationError, "intent_hash"):
            TradeIntent.from_storage(payload)

    def test_intent_requires_one_forecast_or_deterministic_exit_reason(self):
        with self.assertRaises(SchemaValidationError):
            intent(forecast_id=None, exit_reason=None)
        with self.assertRaises(SchemaValidationError):
            intent(forecast_id="forecast-001", exit_reason="policy-expiry")

    def test_authorization_context_binds_account_epoch_and_versions(self):
        record = intent()
        record.assert_authorization_context(
            scope(), policy_version="policy-v1", strategy_version="foresea-edge-v1", market_version="market-v1"
        )
        with self.assertRaisesRegex(SchemaValidationError, "account binding"):
            record.assert_authorization_context(
                AccountScope(**{**scope().__dict__, "account_epoch": 2}),
                policy_version="policy-v1",
                strategy_version="foresea-edge-v1",
                market_version="market-v1",
            )
        with self.assertRaisesRegex(SchemaValidationError, "policy_version"):
            record.assert_authorization_context(
                scope(), policy_version="policy-v2", strategy_version="foresea-edge-v1", market_version="market-v1"
            )

    def test_command_transitions_are_separate_from_display_and_terminal_states_do_not_regress(self):
        self.assertTrue(can_transition_command(CommandState.PROPOSED, CommandState.VALIDATED))
        self.assertTrue(can_transition_command(CommandState.SUBMISSION_UNKNOWN, CommandState.ACKNOWLEDGED))
        self.assertTrue(can_transition_command(CommandState.CANCEL_REQUESTED, CommandState.FILLED))
        self.assertFalse(can_transition_command(CommandState.FILLED, CommandState.PARTIALLY_FILLED))
        self.assertFalse(can_transition_command(CommandState.CANCELLED, CommandState.SUBMITTING))

    def test_forecast_proposal_and_market_records_reject_incomplete_semantics(self):
        forecast = Forecast(
            id="forecast-001",
            instrument_id="kalshi:demo:KXTEST",
            p_yes_raw=Decimal("0.6"),
            p_yes_calibrated=None,
            calibration_status="insufficient",
            uncertainty_low=None,
            uncertainty_high=None,
            evidence_ids=("evidence-001",),
            as_of=CREATED,
            expires_at=EXPIRES,
            model_hash="model-v1",
            prompt_hash="prompt-v1",
            strategy_hash="strategy-v1",
            calibration_version="calibration-v1",
            prospective_provenance="prospective-v1",
            created_at=CREATED,
        )
        self.assertEqual(forecast.p_yes_raw, Decimal("0.6"))
        with self.assertRaises(SchemaValidationError):
            Proposal(
                id="proposal-001",
                forecast_id="forecast-001",
                market_snapshot_id="snapshot-001",
                action=ProposalAction.PASS,
                reason_codes=(),
                citation_ids=(),
                preferred_limit=None,
                pass_decision=None,
                created_at=CREATED,
            )
        proposal = Proposal(
            id="proposal-001",
            forecast_id="forecast-001",
            market_snapshot_id="snapshot-001",
            action=ProposalAction.PASS,
            reason_codes=(RejectionReason.PASS_NO_EDGE,),
            citation_ids=("citation-001",),
            preferred_limit=None,
            pass_decision=PassDecision(RejectionReason.PASS_NO_EDGE, "Costs exceed edge"),
            created_at=CREATED,
        )
        self.assertEqual(proposal.pass_decision.reason, RejectionReason.PASS_NO_EDGE)
        self.assertEqual(Proposal.from_storage(proposal.to_storage()), proposal)
        proposal_payload = proposal.to_storage()
        proposal_payload["credentials"] = {"private_key": "secret"}
        with self.assertRaisesRegex(SchemaValidationError, "credentials"):
            Proposal.from_storage(proposal_payload)
        proposal_payload = proposal.to_storage()
        proposal_payload["execute"] = True
        with self.assertRaisesRegex(SchemaValidationError, "execute"):
            Proposal.from_storage(proposal_payload)
        cursor = MarketCursor("venue-api", None, Completeness.COMPLETE, CREATED)
        self.assertEqual(cursor.completeness, Completeness.COMPLETE)
        snapshot = MarketSnapshot(
            id="snapshot-001",
            instrument_id="kalshi:demo:KXTEST",
            venue_at=CREATED,
            received_at=CREATED,
            sequence=1,
            source="venue-api",
            complete=Completeness.COMPLETE,
            stale_after_seconds=10,
            yes_bid=Decimal("0.4"),
            yes_ask=Decimal("0.41"),
            no_bid=None,
            no_ask=None,
            fee_version="fees-v1",
            created_at=CREATED,
        )
        self.assertEqual(snapshot.complete, Completeness.COMPLETE)
        with self.assertRaises(SchemaValidationError):
            MarketSnapshot(**{**snapshot.__dict__, "yes_bid": Decimal("0.5")})

    def test_schema_version_migration_is_rejected(self):
        with self.assertRaisesRegex(SchemaValidationError, "schema version"):
            intent(schema_version=2)


if __name__ == "__main__":
    unittest.main()
