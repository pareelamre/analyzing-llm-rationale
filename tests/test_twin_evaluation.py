import unittest

from analyzing_llm_rationale.twin.evaluation import readiness_artifact


class TwinEvaluationTests(unittest.TestCase):
    def test_same_frozen_outcomes_create_same_artifact(self):
        first = readiness_artifact([{"id": "one", "pnl": "1"}], code_hash="code", config_hash="config")
        second = readiness_artifact([{"id": "one", "pnl": "1"}], code_hash="code", config_hash="config")
        self.assertEqual(first["artifact_hash"], second["artifact_hash"])
