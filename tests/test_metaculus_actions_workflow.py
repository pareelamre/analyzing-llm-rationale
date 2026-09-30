"""Guard the hosted tournament scheduler's safety-critical configuration."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "metaculus-futureeval.yml"


class MetaculusActionsWorkflowTests(unittest.TestCase):
    def test_four_isolated_profiles_and_write_ahead_audit(self):
        data = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        triggers = data["on"]
        self.assertIn("schedule", triggers)
        self.assertIn("workflow_dispatch", triggers)
        self.assertNotIn("pull_request", triggers)
        self.assertEqual(data["permissions"], {"contents": "read"})
        self.assertEqual(data["concurrency"]["cancel-in-progress"], "false")

        job = data["jobs"]["forecast"]
        self.assertEqual(job["strategy"]["fail-fast"], "false")
        profiles = job["strategy"]["matrix"]["include"]
        self.assertEqual(
            {row["profile"] for row in profiles},
            {"qwen-primary", "gemma-secondary", "glm-secondary", "deepseek-secondary"},
        )
        self.assertEqual(len({row["token_secret"] for row in profiles}), 4)
        self.assertEqual(len({row["username_secret"] for row in profiles}), 4)
        self.assertEqual(len({row["audit_file"] for row in profiles}), 4)
        for row in profiles:
            self.assertEqual(row["token_env"], row["token_secret"])
            self.assertEqual(row["username_env"], row["username_secret"])
        self.assertIn("matrix.profile", job["env"]["METACULUS_AUDIT_GCS_URI"])
        self.assertIn("matrix.audit_file", job["env"]["AUDIT_PATH"])
        self.assertEqual(job["timeout-minutes"], "12")
        self.assertIn("workflow_dispatch", job["if"])
        steps = job["steps"]
        names = [step.get("name", "") for step in steps]
        self.assertLess(names.index("Restore write-ahead audit"), names.index("Forecast and submit"))
        restore = steps[names.index("Restore write-ahead audit")]["run"]
        submit = steps[names.index("Forecast and submit")]
        self.assertIn("metaculus_audit_storage restore", restore)
        self.assertIn("--submit", submit["run"])
        self.assertIn("--confirm-submit", submit["run"])
        self.assertIn("secrets[matrix.token_secret]", submit["env"]["BOT_TOKEN"])
        self.assertIn("secrets[matrix.username_secret]", submit["env"]["BOT_USERNAME"])
        self.assertIn("github.event.repository.default_branch", submit["env"]["LIVE_SUBMIT"])
        self.assertIn("vars.METACULUS_BOTS_ENABLED", submit["env"]["LIVE_SUBMIT"])
        self.assertIn('if [[ "$LIVE_SUBMIT" == "true" ]]', submit["run"])
        self.assertIn("--max-questions 1", submit["run"])
        install = steps[names.index("Install hosted bot dependencies")]["run"]
        self.assertIn("--require-hashes", install)
        self.assertIn("-z \"$BOT_TOKEN\"", submit["run"])
        self.assertIn("-z \"$BOT_USERNAME\"", submit["run"])
        self.assertIn("-z \"$SCADS_AI_API_KEY\"", submit["run"])


if __name__ == "__main__":
    unittest.main()
