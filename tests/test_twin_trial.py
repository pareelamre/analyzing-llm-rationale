from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analyzing_llm_rationale.twin.trial import TrialEvidenceError, build_trial_report

ROOT = Path(__file__).resolve().parents[1]
START = datetime(2026, 9, 12, tzinfo=timezone.utc)
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def observation(day: int, **updates):
    value = {
        "id": f"observation-{day}",
        "observed_at": (START + timedelta(days=day, hours=1)).isoformat(),
        "code_hash": DIGEST_A, "config_hash": DIGEST_B,
        "complete_market_snapshots": 3, "decisions": 3,
        "simulated_commands": 1, "duplicate_commands": 0,
        "unexplained_divergences": 0, "stale_exposure_attempts": 0,
        "actual_cost_usd": "0", "uncertain_cost_usd": "0",
    }
    value.update(updates)
    return value


def evidence(*, days=0, operational=True):
    return {
        "schema_version": 1,
        "release": {
            "code_hash": DIGEST_A, "config_hash": DIGEST_B,
            "image_digest": DIGEST_C, "started_at": START.isoformat(),
        },
        "collection_operational": operational,
        "observations": [observation(day) for day in range(days)],
        "drills": {
            name: {"status": "pass", "evidence_id": f"drill-{name}"}
            for name in (
                "provider_outage", "duplicate_task", "cancel_fill_race", "kill_restart",
            )
        },
        "strategy_evidence": {
            "independent_resolved_markets": 100,
            "completed_shadow_trades": 30,
            "baseline_skill_lower_bound": "0.01",
            "conservative_net_result": "1.00",
        },
        "blockers": [],
    }


class TwinTrialTests(unittest.TestCase):
    def test_seven_clean_consecutive_days_pass_g1_and_positive_samples_pass_g2(self):
        report = build_trial_report(
            evidence(days=7), as_of=START + timedelta(days=7),
        )
        self.assertEqual(report["g1"]["status"], "pass")
        self.assertEqual(report["g2"]["status"], "pass")
        self.assertFalse(report["live_eligible"])
        self.assertEqual(len(report["artifact_hash"]), 64)

    def test_missing_days_collect_and_mechanics_failures_are_ineligible(self):
        collecting = build_trial_report(
            evidence(days=3), as_of=START + timedelta(days=3),
        )
        self.assertEqual(collecting["g1"]["status"], "collecting")
        self.assertEqual(
            collecting["g1"]["next_measurement_at"],
            (START + timedelta(days=3)).isoformat(),
        )
        broken = evidence(days=7)
        broken["observations"][-1]["unexplained_divergences"] = 1
        report = build_trial_report(broken, as_of=START + timedelta(days=7))
        self.assertEqual(report["g1"]["status"], "ineligible")

    def test_nonoperational_collection_and_explicit_blocker_are_blocked(self):
        item = evidence(operational=False)
        item["blockers"] = ["No cycle producer is deployed."]
        report = build_trial_report(item, as_of=START + timedelta(days=1))
        self.assertEqual(report["g1"]["status"], "blocked")

    def test_insufficient_or_nonpositive_strategy_evidence_cannot_pass(self):
        item = evidence(days=7)
        item["strategy_evidence"]["completed_shadow_trades"] = 29
        self.assertEqual(
            build_trial_report(item, as_of=START + timedelta(days=7))["g2"]["status"],
            "collecting",
        )
        item["strategy_evidence"]["completed_shadow_trades"] = 30
        item["strategy_evidence"]["conservative_net_result"] = "0"
        self.assertEqual(
            build_trial_report(item, as_of=START + timedelta(days=7))["g2"]["status"],
            "ineligible",
        )

    def test_mismatched_release_duplicate_ids_and_incomplete_drills_fail_closed(self):
        cases = []
        mismatch = evidence(days=1)
        mismatch["observations"][0]["code_hash"] = "d" * 64
        cases.append(mismatch)
        duplicate = evidence(days=2)
        duplicate["observations"][1]["id"] = duplicate["observations"][0]["id"]
        cases.append(duplicate)
        drills = evidence()
        drills["drills"].pop("kill_restart")
        cases.append(drills)
        for item in cases:
            with self.subTest(item=item), self.assertRaises(TrialEvidenceError):
                build_trial_report(item, as_of=START + timedelta(days=7))

    def test_cli_writes_reproducible_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evidence.json"
            source.write_text(json.dumps(evidence(days=2)), encoding="utf-8")
            outputs = []
            for suffix in ("one", "two"):
                json_output = root / f"{suffix}.json"
                markdown_output = root / f"{suffix}.md"
                completed = subprocess.run([
                    sys.executable, str(ROOT / "scripts" / "twin_trial_report.py"),
                    "--evidence", str(source),
                    "--as-of", (START + timedelta(days=2)).isoformat(),
                    "--json-output", str(json_output),
                    "--markdown-output", str(markdown_output),
                ], cwd=ROOT, capture_output=True, text=True, check=True)
                self.assertIn('"g1": "collecting"', completed.stdout)
                outputs.append((json_output.read_bytes(), markdown_output.read_bytes()))
            self.assertEqual(outputs[0], outputs[1])
            self.assertIn(b"live_eligible", outputs[0][0])
            self.assertIn(b"never authorizes live trading", outputs[0][1])


if __name__ == "__main__":
    unittest.main()
