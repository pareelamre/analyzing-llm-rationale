import unittest

from analyzing_llm_rationale.twin.recovery import recovery_action


class TwinRecoveryTests(unittest.TestCase):
    def test_unknown_submission_never_retries_without_reconciliation(self):
        self.assertEqual(recovery_action("submission_unknown", None), "pause_and_reconcile")
        self.assertEqual(recovery_action("submission_unknown", True), "reconcile")
        self.assertEqual(recovery_action("filled", None), "terminal")
