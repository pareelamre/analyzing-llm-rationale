"""Forecast evidence ranks by meaning where the host has room (EVIDENCE_EMBEDDINGS=1)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale.cli import evidence_embeddings_enabled  # noqa: E402


class EvidenceEmbeddingsTests(unittest.TestCase):
    def test_off_unless_asked_for(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(evidence_embeddings_enabled())
        for value, expected in (("1", True), (" 1 ", True), ("0", False), ("true", False)):
            with mock.patch.dict(os.environ, {"EVIDENCE_EMBEDDINGS": value}):
                self.assertEqual(evidence_embeddings_enabled(), expected, value)

    def test_the_forecast_job_turns_it_on(self):
        workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "track-record-forecast.yml").read_text())
        steps = workflow["jobs"]["forecast"]["steps"]
        forecast = next(s for s in steps if s.get("name") == "Advance model forecast snapshots")
        self.assertEqual(forecast["env"]["EVIDENCE_EMBEDDINGS"], "1")
        install = next(s for s in steps if s.get("name") == "Install deps")
        self.assertIn("pipeline", install["run"])  # sentence-transformers comes with the pipeline extra


if __name__ == "__main__":
    unittest.main()
