"""Guard the hosted tournament scheduler's safety-critical configuration."""

from __future__ import annotations

import shlex
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "metaculus-futureeval.yml"


class MetaculusActionsWorkflowTests(unittest.TestCase):
    def test_hosted_cycle_enables_guarded_refresh_not_manual_force(self):
        from analyzing_llm_rationale.cli import build_parser

        data = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        step = next(s for s in data["jobs"]["forecast"]["steps"] if s.get("name") == "Forecast and submit")
        args = shlex.split(step["run"])
        self.assertIn("--refresh-forecasted", args)
        self.assertNotIn("--include-forecasted", args)
        parsed = build_parser().parse_args(["forecast-metaculus", "--refresh-forecasted"])
        self.assertTrue(parsed.refresh_forecasted)
        self.assertFalse(parsed.include_forecasted)

    def test_hosted_question_limit_covers_two_simultaneous_questions(self):
        from analyzing_llm_rationale.metaculus_bot import ForecastCycleConfig, run_forecast_cycle
        from test_metaculus_bot import FakeProvider

        data = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        step = next(s for s in data["jobs"]["forecast"]["steps"] if s.get("name") == "Forecast and submit")
        args = shlex.split(step["run"])
        maximum = int(args[args.index("--max-questions") + 1])
        self.assertEqual(maximum, 5)

        class Client:
            def list_open_posts(self, *args, **kwargs):
                return [{"id": 12}, {"id": 13}]

            def get_post(self, post_id):
                return {"id": post_id, "question": {"id": post_id + 100, "type": "binary", "my_forecasts": {"latest": None}}}

        result = run_forecast_cycle(
            Client(), FakeProvider('{"probability_yes": 0.7}'),
            ForecastCycleConfig(max_questions=maximum),
        )
        self.assertEqual((result.examined, result.forecasted, result.failed), (2, 2, 0))
        with patch.object(Client, "list_open_posts", return_value=[{"id": n} for n in range(12, 18)]):
            capped = run_forecast_cycle(
                Client(), FakeProvider('{"probability_yes": 0.7}'),
                ForecastCycleConfig(max_questions=maximum),
            )
        self.assertEqual((capped.examined, capped.forecasted, capped.failed), (5, 5, 0))

    def test_four_isolated_profiles_and_write_ahead_audit(self):
        data = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        triggers = data["on"]
        self.assertIn("schedule", triggers)
        self.assertIn("workflow_dispatch", triggers)
        self.assertEqual(triggers["workflow_dispatch"]["inputs"]["submit"]["type"], "boolean")
        self.assertEqual(triggers["workflow_dispatch"]["inputs"]["submit"]["default"], "false")
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
        self.assertEqual(job["timeout-minutes"], "65")
        self.assertGreaterEqual(int(job["timeout-minutes"]), 5 * 12 + 5)
        self.assertIn("workflow_dispatch", job["if"])
        self.assertIn("inputs.submit != true", job["if"])
        self.assertTrue(job["if"].startswith("vars.METACULUS_BOTS_ENABLED == 'true' || ("))
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
        self.assertIn("inputs.submit == true", submit["env"]["LIVE_SUBMIT"])
        self.assertIn('if [[ "$LIVE_SUBMIT" == "true" ]]', submit["run"])
        self.assertIn("--max-questions 5", submit["run"])
        install = steps[names.index("Install hosted bot dependencies")]["run"]
        self.assertIn("--require-hashes", install)
        self.assertIn("-z \"$BOT_TOKEN\"", submit["run"])
        self.assertIn("-z \"$BOT_USERNAME\"", submit["run"])
        self.assertIn("-z \"$SCADS_AI_API_KEY\"", submit["run"])


if __name__ == "__main__":
    unittest.main()
