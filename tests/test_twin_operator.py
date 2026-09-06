import unittest

from analyzing_llm_rationale.twin.operator import operator_status


class TwinOperatorTests(unittest.TestCase):
    def test_unknown_commands_pause_new_exposure(self):
        self.assertFalse(operator_status(mandate_active=True, paused=False, unknown_commands=1)["new_exposure_allowed"])
