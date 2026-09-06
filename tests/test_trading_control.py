from __future__ import annotations

import unittest

from analyzing_llm_rationale.trading_control import (
    CallbackSavedRunStore,
    ConfirmedManualOrderService,
    GuardrailViolation,
    claim_for_submission,
    claim_saved_run,
    submit_confirmed_manual_order,
    validate_pre_submission_policy,
)


class TradingControlTests(unittest.TestCase):
    def test_claim_transitions_only_awaiting_run_without_mutating_source(self):
        source = {"id": "run-1", "status": "awaiting_approval", "preview": {"old": True}}
        preview = {"estimated_notional": 1.25, "normalized_order": {"ticker": "KXTEST"}}

        claimed, acquired = claim_for_submission(source, preview, approved_at="2026-09-06T00:00:00+00:00")

        self.assertTrue(acquired)
        self.assertEqual(claimed["status"], "submitting")
        self.assertEqual(claimed["estimated_notional"], 1.25)
        self.assertEqual(source["status"], "awaiting_approval")
        preview["normalized_order"]["ticker"] = "CHANGED"
        self.assertEqual(claimed["preview"]["normalized_order"]["ticker"], "KXTEST")

    def test_claim_refuses_terminal_or_in_progress_run(self):
        claimed, acquired = claim_for_submission(
            {"id": "run-1", "status": "submitting"}, {}, approved_at="2026-09-06T00:00:00+00:00"
        )
        self.assertFalse(acquired)
        self.assertEqual(claimed["status"], "submitting")

    def test_stored_claim_uses_injected_store_and_clock(self):
        records = {"run": {"id": "run", "status": "awaiting_approval"}}
        store = CallbackSavedRunStore(
            load=lambda: records["run"],
            save=lambda record: records.__setitem__("run", dict(record)),
        )

        claimed, acquired = claim_saved_run(
            store,
            {"estimated_notional": 1.25},
            clock=lambda: "2026-09-06T00:00:00+00:00",
        )

        self.assertTrue(acquired)
        self.assertEqual(claimed["approved_at"], "2026-09-06T00:00:00+00:00")
        self.assertEqual(records["run"]["status"], "submitting")

    def test_submit_boundary_requires_injected_adapter_and_preserves_payload(self):
        observed = {}

        def place_order(payload, *, user_id, creds):
            observed.update(payload=payload, user_id=user_id, creds=creds)
            return {"submitted": True}

        result = submit_confirmed_manual_order(
            place_order,
            {"platform": "kalshi", "ticker": "KXTEST"},
            user_id="user-1",
            credentials={"kalshi_api_key_id": "opaque"},
            confirmation="PLACE REAL ORDER",
        )

        self.assertTrue(result["submitted"])
        self.assertEqual(observed["payload"]["execute"], True)
        self.assertEqual(observed["payload"]["confirmation"], "PLACE REAL ORDER")
        self.assertEqual(observed["user_id"], "user-1")

    def test_execution_service_resolves_credentials_and_submits_through_adapter(self):
        observed = {}

        def resolver(user_id, venue):
            observed["resolved"] = (user_id, venue)
            return {"connection": "opaque"}

        def adapter(payload, *, user_id, creds):
            observed["submitted"] = (payload, user_id, creds)
            return {"status": "submitted"}

        service = ConfirmedManualOrderService(resolver, adapter)
        credentials = service.resolve_credentials(user_id="user-1", venue="kalshi")
        result = service.submit(
            {"platform": "kalshi"},
            user_id="user-1",
            credentials=credentials,
            confirmation="PLACE REAL ORDER",
        )

        self.assertEqual(observed["resolved"], ("user-1", "kalshi"))
        self.assertTrue(observed["submitted"][0]["execute"])
        self.assertEqual(result["status"], "submitted")

    def test_pre_submission_policy_rejects_paused_duplicate_and_excess_notional(self):
        policy = {"paused": False, "max_order_notional": 2.0, "cooldown_seconds": 60}
        validate_pre_submission_policy(policy, estimated_notional=2.0, duplicate=False)
        for updates, notional, duplicate, expected in (
            ({"paused": True}, 1.0, False, "user_paused"),
            ({}, 0.0, False, "invalid_notional"),
            ({}, 2.01, False, "max_order_notional"),
            ({}, 1.0, True, "duplicate_cooldown"),
        ):
            with self.subTest(expected=expected):
                with self.assertRaises(GuardrailViolation) as raised:
                    validate_pre_submission_policy(
                        {**policy, **updates}, estimated_notional=notional, duplicate=duplicate
                    )
                self.assertEqual(raised.exception.code, expected)


if __name__ == "__main__":
    unittest.main()
