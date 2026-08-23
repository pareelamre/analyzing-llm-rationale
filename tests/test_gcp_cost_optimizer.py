from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gcp_cost_optimizer as optimizer


class TestGCPCostOptimizer(unittest.TestCase):
    """Test GCP cost optimizer definitions and runner."""

    def test_get_optimization_actions(self):
        actions = optimizer.get_optimization_actions()
        self.assertGreaterEqual(len(actions), 4)
        for act in actions:
            self.assertIn("service", act)
            self.assertIn("name", act)
            self.assertIn("command", act)
            if act.get("policy_file"):
                self.assertTrue(Path(act["policy_file"]).exists(), f"Missing {act['policy_file']}")

    def test_policy_json_validity(self):
        actions = optimizer.get_optimization_actions()
        for act in actions:
            filepath = act.get("policy_file")
            if filepath and filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    policy_data = json.load(f)
                    self.assertTrue(isinstance(policy_data, (dict, list)))

    def test_audit_status(self):
        report = optimizer.audit_status()
        self.assertIn("project_id", report)
        self.assertIn("policies", report)
        self.assertIn("recommendations", report)
        self.assertGreaterEqual(len(report["policies"]), 4)

    def test_run_command_dry_run(self):
        res = optimizer.run_command(["gcloud", "version"], dry_run=True)
        self.assertEqual(res["status"], "skipped (dry-run)")
        self.assertEqual(res["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
