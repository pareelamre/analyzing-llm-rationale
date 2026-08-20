"""Text assertions against the agent-trading CI workflow YAML files --
mirrors tests/test_track_record_tick.py's approach to locking in workflow
plumbing without actually running GitHub Actions."""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.config import load_model_configs  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS_DIR = _ROOT / ".github" / "workflows"
_REUSABLE_PATH = _WORKFLOWS_DIR / "_agent-trading-tick-reusable.yml"
_CRON_RE = re.compile(r'cron:\s*"(\d+),(\d+),(\d+),(\d+) \* \* \* \*"')


def _chat_capable_models() -> list[str]:
    models = load_model_configs(_ROOT / "configs" / "models.yaml")
    return sorted(name for name, cfg in models.items() if cfg.chat_interface_enabled)


class AgentTradingReusableWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = _REUSABLE_PATH.read_text(encoding="utf-8")

    def test_is_a_workflow_call_target(self):
        self.assertIn("workflow_call:", self.workflow)
        self.assertIn("model:", self.workflow)
        self.assertIn("required: true", self.workflow)

    def test_shadow_mode_is_hard_pinned_not_an_input(self):
        self.assertIn("FORESEA_AGENT_PLACE_TRADE_MODE: shadow", self.workflow)
        # It must be a literal, not derived from an input/secret/var.
        self.assertNotIn("FORESEA_AGENT_PLACE_TRADE_MODE: ${{", self.workflow)

    def test_never_enables_real_trading(self):
        # The explanatory header comment mentions these gates by name (that's
        # fine); what must never appear is an actual env assignment for them.
        self.assertNotIn("FORESEA_ENABLE_TRADING:", self.workflow)
        self.assertNotIn("FORESEA_ENABLE_BYO_TRADING:", self.workflow)

    def test_uses_per_model_gcs_objects_not_a_shared_file(self):
        self.assertIn("agent_trading_store__${MODEL}.sqlite", self.workflow)
        self.assertIn("agent_trading_notes__${MODEL}.json", self.workflow)

    def test_concurrency_uses_the_explicit_scads_lane(self):
        # Two provider lanes cap concurrent SCADS requests while retaining
        # independent account objects. A model's wrapper always supplies the
        # same lane, so a second run for that model is also serialized rather
        # than racing its own GCS upload.
        self.assertIn("concurrency:", self.workflow)
        self.assertIn("lane:", self.workflow)
        self.assertIn("group: agent-trading-scads-lane-${{ inputs.lane }}", self.workflow)
        self.assertIn("cancel-in-progress: false", self.workflow)

    def test_download_is_tolerant_of_a_missing_object(self):
        self.assertIn("|| echo", self.workflow)

    def test_upload_runs_even_if_the_tick_step_failed(self):
        upload_section = self.workflow.split("Upload this model's shadow account store", 1)[1]
        self.assertIn("if: always()", upload_section)

    def test_installs_the_serve_extra_and_authenticates_to_gcp(self):
        self.assertIn('pip install --quiet -e ".[serve,pipeline]"', self.workflow)
        self.assertIn("google-github-actions/auth@v2", self.workflow)
        self.assertIn("google-github-actions/setup-gcloud@v2", self.workflow)

    def test_runs_the_driver_script(self):
        self.assertIn("python scripts/agent_trading_tick.py", self.workflow)


class AgentTradingPerModelWorkflowTests(unittest.TestCase):
    def test_every_chat_capable_scads_model_has_its_own_workflow_file(self):
        models = _chat_capable_models()
        self.assertEqual(len(models), 10)
        for model in models:
            path = _WORKFLOWS_DIR / f"agent-trading-tick-{model}.yml"
            self.assertTrue(path.exists(), f"missing workflow file for {model}: {path}")

    def test_no_extra_agent_trading_workflow_files_beyond_the_expected_roster(self):
        models = set(_chat_capable_models())
        found = {
            p.stem.removeprefix("agent-trading-tick-")
            for p in _WORKFLOWS_DIR.glob("agent-trading-tick-*.yml")
        }
        self.assertEqual(found, models)

    def test_each_wrapper_calls_the_reusable_workflow_with_its_own_model(self):
        for model in _chat_capable_models():
            path = _WORKFLOWS_DIR / f"agent-trading-tick-{model}.yml"
            text = path.read_text(encoding="utf-8")
            self.assertIn("uses: ./.github/workflows/_agent-trading-tick-reusable.yml", text)
            self.assertIn(f"model: {model}", text)
            self.assertRegex(text, r'lane:\s*"[01]"')
            self.assertIn("secrets: inherit", text)
            self.assertRegex(text, _CRON_RE, f"no recognizable cron stagger in {path}")
            self.assertIn("workflow_dispatch:", text)

    def test_cron_offsets_are_staggered_across_models_to_avoid_a_rate_limit_thundering_herd(self):
        # Regression test: all 10 wrapper workflows originally shared the
        # identical "*/15 * * * *" cron, so every 15-minute tick fired all 10
        # SCADS calls at the same wall-clock minute -- observed live to cause
        # 429 RateLimitErrors on roughly half the fleet during manual
        # dispatch verification. Each model now gets its own minute offset
        # within the 15-minute window, still ticking every 15 minutes.
        offsets: dict[str, int] = {}
        for model in _chat_capable_models():
            path = _WORKFLOWS_DIR / f"agent-trading-tick-{model}.yml"
            text = path.read_text(encoding="utf-8")
            match = _CRON_RE.search(text)
            self.assertIsNotNone(match, f"no recognizable cron stagger in {path}")
            minutes = [int(g) for g in match.groups()]
            self.assertEqual(
                minutes, [minutes[0], minutes[0] + 15, minutes[0] + 30, minutes[0] + 45],
                f"{model}'s cron minutes aren't a 15-minute-cadence stagger: {minutes}",
            )
            offsets[model] = minutes[0]
        self.assertEqual(
            len(set(offsets.values())), len(offsets),
            f"two or more models share a cron offset, defeating the stagger: {offsets}",
        )


if __name__ == "__main__":
    unittest.main()
