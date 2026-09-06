import unittest

from analyzing_llm_rationale.twin.release import release_ready


class TwinReleaseTests(unittest.TestCase):
    def test_all_release_conditions_are_required(self):
        self.assertTrue(release_ready(health=True, readiness_artifact=True, unknown_commands=0, owner_authorized=True))
        self.assertFalse(release_ready(health=True, readiness_artifact=True, unknown_commands=1, owner_authorized=True))
