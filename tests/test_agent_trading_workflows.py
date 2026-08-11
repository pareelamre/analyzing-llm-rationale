"""Text assertions against the agent-trading CI workflow YAML files --
mirrors tests/test_track_record_tick.py's approach to locking in workflow
plumbing without actually running GitHub Actions."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.config import load_model_configs  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS_DIR = _ROOT / ".github" / "workflows"
_REUSABLE_PATH = _WORKFLOWS_DIR / "_agent-trading-tick-reusable.yml"


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
        # No shared-object concurrency lock -- each model's objects are its own.
        self.assertNotIn("concurrency:", self.workflow)

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
            self.assertIn("secrets: inherit", text)
            self.assertIn('cron: "*/15 * * * *"', text)
            self.assertIn("workflow_dispatch:", text)


if __name__ == "__main__":
    unittest.main()
