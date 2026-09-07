from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.twin_replay import main as replay_main

from analyzing_llm_rationale.twin.evaluation import (
    ReadinessArtifactError,
    ReplayPolicy,
    evaluate_replay,
    readiness_artifact,
    validate_readiness_artifact,
)
from tests.test_twin_replay import dataset, row

NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)
SPLIT = NOW - timedelta(days=10)
CODE_HASH = "a" * 64


def policy(*, minimum: int = 1) -> ReplayPolicy:
    return ReplayPolicy(
        SPLIT, NOW, min_calibration_records=minimum, min_test_records=minimum,
        fee_fraction=.01, slippage=.01,
    )


def useful_dataset() -> dict:
    return dataset([
        row("train", forecast_at=SPLIT - timedelta(days=3), outcome_at=SPLIT - timedelta(days=1)),
        row("test-win", forecast_at=SPLIT + timedelta(days=1), outcome_at=NOW - timedelta(days=2), outcome=1),
        row("test-loss", forecast_at=SPLIT + timedelta(days=2), outcome_at=NOW - timedelta(days=1), outcome=0),
    ])


class TwinEvaluationTests(unittest.TestCase):
    def test_same_frozen_outcomes_create_same_artifact(self):
        first = readiness_artifact([{"id": "one", "pnl": "1"}], code_hash="code", config_hash="config")
        second = readiness_artifact([{"id": "one", "pnl": "1"}], code_hash="code", config_hash="config")
        self.assertEqual(first["artifact_hash"], second["artifact_hash"])

        replay_one = evaluate_replay(useful_dataset(), policy=policy(), code_hash=CODE_HASH)
        replay_two = evaluate_replay(useful_dataset(), policy=policy(), code_hash=CODE_HASH)
        self.assertEqual(replay_one, replay_two)

    def test_report_reuses_forecast_metrics_and_includes_costs_baselines_and_stress(self):
        artifact = evaluate_replay(useful_dataset(), policy=policy(), code_hash=CODE_HASH)
        report = artifact["out_of_sample"]
        self.assertIn("model_brier", report["strategy"]["forecast_metrics"])
        self.assertIn("market_baseline_brier", report)
        self.assertIn("fixed_policy", report)
        self.assertIn("net_pnl_after_costs", report["strategy"])
        self.assertIn("max_drawdown", report["strategy"]["portfolio"])
        self.assertIn("turnover", report["strategy"])
        self.assertIn("abstention", report["strategy"])
        self.assertEqual(set(artifact["stress"]), {"double_fees", "double_slippage", "missing_quotes", "correlated_losses"})
        self.assertFalse(artifact["live_eligible"])

    def test_insufficient_history_is_explicit(self):
        artifact = evaluate_replay(useful_dataset(), policy=policy(minimum=30), code_hash=CODE_HASH)
        self.assertEqual(artifact["status"], "insufficient_evidence")
        self.assertEqual(artifact["gates"]["calibration_depth"]["status"], "insufficient")
        self.assertEqual(artifact["gates"]["test_depth"]["status"], "insufficient")

    def test_late_arriving_outcome_does_not_change_frozen_decisions(self):
        original = useful_dataset()
        late = row("late", forecast_at=SPLIT + timedelta(days=3), outcome_at=NOW + timedelta(days=1))
        extended = deepcopy(original)
        extended["records"].append(late)
        first = evaluate_replay(original, policy=policy(), code_hash=CODE_HASH)
        second = evaluate_replay(extended, policy=policy(), code_hash=CODE_HASH)
        self.assertEqual(first["out_of_sample"], second["out_of_sample"])
        self.assertEqual(second["evidence"]["excluded"]["future_outcome"], 1)

    def test_validator_rejects_tampering_staleness_and_live_claims(self):
        artifact = evaluate_replay(useful_dataset(), policy=policy(), code_hash=CODE_HASH)
        validate_readiness_artifact(artifact, now=NOW, max_age_seconds=60)
        tampered = deepcopy(artifact)
        tampered["status"] = "ready_for_live"
        with self.assertRaisesRegex(ReadinessArtifactError, "hash"):
            validate_readiness_artifact(tampered, now=NOW, max_age_seconds=60)
        with self.assertRaisesRegex(ReadinessArtifactError, "stale"):
            validate_readiness_artifact(artifact, now=NOW + timedelta(seconds=61), max_age_seconds=60)
        live = deepcopy(artifact)
        live["live_eligible"] = True
        payload = dict(live)
        payload.pop("artifact_hash")
        from analyzing_llm_rationale.twin.replay import canonical_hash
        live["artifact_hash"] = canonical_hash(payload)
        with self.assertRaisesRegex(ReadinessArtifactError, "cannot authorize live"):
            validate_readiness_artifact(live, now=NOW, max_age_seconds=60)

    def test_cli_writes_reproducible_atomic_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path, config_path = root / "dataset.json", root / "config.yaml"
            first_path, second_path = root / "first.json", root / "second.json"
            dataset_path.write_text(json.dumps(useful_dataset()), encoding="utf-8")
            config_path.write_text(
                "replay:\n"
                f"  split_at: '{SPLIT.isoformat()}'\n"
                f"  evaluation_as_of: '{NOW.isoformat()}'\n"
                "  min_calibration_records: 1\n"
                "  min_test_records: 1\n",
                encoding="utf-8",
            )
            self.assertEqual(replay_main(["--dataset", str(dataset_path), "--config", str(config_path), "--output", str(first_path)]), 0)
            self.assertEqual(replay_main(["--dataset", str(dataset_path), "--config", str(config_path), "--output", str(second_path)]), 0)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertFalse(first_path.with_suffix(".json.tmp").exists())

    def test_checked_in_baseline_reproduces_from_documented_inputs(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            generated = Path(tmp) / "baseline.json"
            self.assertEqual(replay_main([
                "--dataset", str(repository / "tests/fixtures/twin/replay_dataset_v1.json"),
                "--config", str(repository / "configs/twin.yaml"),
                "--output", str(generated),
            ]), 0)
            self.assertEqual(
                json.loads(generated.read_text(encoding="utf-8")),
                json.loads((repository / "docs/autonomous-twin/REPLAY_BASELINE.json").read_text(encoding="utf-8")),
            )


if __name__ == "__main__":
    unittest.main()
