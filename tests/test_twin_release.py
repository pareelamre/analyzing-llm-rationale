from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from scripts.verify_twin_release import verify_repository_contract

from analyzing_llm_rationale.twin.execution import ExecutionBlocked, submit_claimed_command
from analyzing_llm_rationale.twin.release import (
    REQUIRED_G0_CHECKS,
    ReleaseReadinessError,
    build_shadow_release_artifact,
    validate_shadow_release_artifact,
)
from tests.test_twin_execution import reserved

NOW = datetime(2026, 9, 9, 9, 0, tzinfo=timezone.utc)
CODE_HASH = "a" * 64
CONFIG_HASH = "b" * 64
FIXTURE = Path(__file__).parent / "fixtures" / "twin" / "release_readiness_v1.json"


def artifact(**updates):
    checks = {name: True for name in REQUIRED_G0_CHECKS}
    checks.update(updates.pop("checks", {}))
    values = {
        "code_hash": CODE_HASH,
        "config_hash": CONFIG_HASH,
        "generated_at": NOW,
        "expires_at": NOW + timedelta(days=1),
        "checks": checks,
        **updates,
    }
    return build_shadow_release_artifact(**values)


class TwinReleaseTests(unittest.TestCase):
    def test_repository_release_contract_is_complete(self):
        result = verify_repository_contract()
        self.assertEqual(result["status"], "pass", result["failed"])

    def test_sample_g0_artifact_is_exact_and_valid(self):
        expected = artifact()
        sample = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(sample, expected)
        validate_shadow_release_artifact(
            sample, now=NOW + timedelta(hours=1),
            expected_code_hash=CODE_HASH, expected_config_hash=CONFIG_HASH,
        )

    def test_stale_forged_mismatched_and_live_shaped_evidence_fails_closed(self):
        valid = artifact()
        cases = []
        stale = artifact(expires_at=NOW + timedelta(seconds=1))
        cases.append((stale, NOW + timedelta(seconds=1), CODE_HASH, CONFIG_HASH))
        forged = deepcopy(valid)
        forged["checks"]["network_deny"] = False
        cases.append((forged, NOW, CODE_HASH, CONFIG_HASH))
        live = deepcopy(valid)
        live["live_eligible"] = True
        cases.append((live, NOW, CODE_HASH, CONFIG_HASH))
        cases.append((valid, NOW, "c" * 64, CONFIG_HASH))
        for item, now, code_hash, config_hash in cases:
            with self.subTest(item=item), self.assertRaises(ReleaseReadinessError):
                validate_shadow_release_artifact(
                    item, now=now, expected_code_hash=code_hash,
                    expected_config_hash=config_hash,
                )

    def test_incomplete_check_set_cannot_claim_g0(self):
        checks = {name: True for name in REQUIRED_G0_CHECKS}
        checks.pop("datastore_emulator")
        with self.assertRaisesRegex(ReleaseReadinessError, "incomplete"):
            build_shadow_release_artifact(
                code_hash=CODE_HASH, config_hash=CONFIG_HASH,
                generated_at=NOW, expires_at=NOW + timedelta(days=1),
                checks=checks,
            )

    def test_credentials_cannot_bypass_network_deny_gate(self):
        store, _, intent, command, claim, context = reserved(
            environment="live", autonomous=False,
        )
        writes = []
        denied = replace(context, runtime_live_enabled=False)
        fake_credentials = {
            "KALSHI_API_KEY_ID": "looks-configured",
            "KALSHI_PRIVATE_KEY": "looks-configured",
            "POLYMARKET_PRIVATE_KEY": "looks-configured",
        }
        with patch.dict("os.environ", fake_credentials, clear=False), self.assertRaises(ExecutionBlocked):
            submit_claimed_command(
                store, command=command, intent=intent, claim=claim, context=denied,
                now=datetime(2025, 1, 1, tzinfo=timezone.utc),
                submit=lambda _: writes.append("network-write"),
            )
        self.assertEqual(writes, [])

    def test_rollback_rebuild_preserves_manual_command_and_reservation(self):
        store, scope, intent, command, _, _ = reserved(autonomous=False)
        reservation = store.reservation(scope.id, command.reservation_id)
        rebuilt = store.rebuild_projection(scope.id)
        self.assertEqual(store.command_for_intent(intent).id, command.id)
        self.assertEqual(store.reservation(scope.id, reservation.id), reservation)
        self.assertGreaterEqual(rebuilt.revision, 1)


if __name__ == "__main__":
    unittest.main()
