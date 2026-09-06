import unittest
from datetime import datetime, timedelta, timezone

from analyzing_llm_rationale.twin.replay import ReplayValidationError, causal_events


class TwinReplayTests(unittest.TestCase):
    def test_late_evidence_never_changes_earlier_replay(self):
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        rows = [{"id": "first", "observed_at": now}, {"id": "late", "observed_at": now + timedelta(seconds=1)}, {"id": "first", "observed_at": now}]
        self.assertEqual([item["id"] for item in causal_events(rows, as_of=now)], ["first"])

    def test_future_occurrence_and_conflicting_duplicate_are_not_replayed(self):
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(causal_events([{"id": "future", "observed_at": now, "occurred_at": now + timedelta(seconds=1)}], as_of=now), [])
        with self.assertRaises(ReplayValidationError):
            causal_events([
                {"id": "same", "observed_at": now, "value": 1},
                {"id": "same", "observed_at": now, "value": 2},
            ], as_of=now)
