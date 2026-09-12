import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from analyzing_llm_rationale.twin.cycle_runtime import (
    InMemoryStrategyRunStore,
    StrategyRun,
    StrategyRunError,
    StrategyRunPhase,
)
from analyzing_llm_rationale.twin.models import Completeness, Instrument, MarketSnapshot
from analyzing_llm_rationale.twin.strategy import StrategyCandidate

NOW = datetime(2026, 9, 12, 20, tzinfo=timezone.utc)


def candidate() -> StrategyCandidate:
    instrument = Instrument(
        "kalshi:live:KXTEST", "kalshi", "live", "KXTEST", None, None, None,
        "settlement-v1", "politics", "event-v1", "cluster-v1",
        Decimal(".01"), Decimal("1"), "fee-v1", "kalshi-v2-limit", "open",
        NOW + timedelta(days=2), NOW + timedelta(days=2), NOW,
    )
    snapshot = MarketSnapshot(
        "snapshot-v1", instrument.id, NOW, NOW, 1, "kalshi-rest",
        Completeness.COMPLETE, 30, Decimal(".4"), Decimal(".41"),
        Decimal(".58"), Decimal(".59"), "fee-v1", NOW,
    )
    return StrategyCandidate(
        instrument, snapshot, Decimal("10"), Decimal("12"),
        Decimal(".01"), Decimal(".01"), ({"id": "history-v1"},),
    )


def queued() -> StrategyRun:
    return StrategyRun(
        "strategy-cycle:abc", "shadow-scope:shadow-account-v1", 1,
        "foresea-edge-shadow-v1", NOW,
    )


class StrategyCycleRuntimeTests(unittest.TestCase):
    def test_round_trip_preserves_exact_candidate_and_phase_identity(self):
        pending = queued().advance(
            StrategyRunPhase.RESEARCH_PENDING, now=NOW + timedelta(seconds=1),
            candidates=(candidate(),), research_job_ids=("research-job-v1",),
        )
        restored = StrategyRun.from_storage(pending.to_storage())
        self.assertEqual(restored, pending)

    def test_store_fences_transitions_and_reuses_exact_initial_delivery(self):
        store = InMemoryStrategyRunStore()
        initial = store.create(queued())
        self.assertEqual(store.create(queued()), initial)
        pending = initial.advance(
            StrategyRunPhase.RESEARCH_PENDING, now=NOW + timedelta(seconds=1),
            candidates=(candidate(),), research_job_ids=("research-job-v1",),
        )
        store.save(pending, expected_revision=0)
        self.assertEqual(store.create(queued()), pending)
        with self.assertRaisesRegex(StrategyRunError, "revision conflict"):
            store.save(pending.advance(
                StrategyRunPhase.READY, now=NOW + timedelta(seconds=2),
            ), expected_revision=0)

    def test_terminal_and_invalid_phase_transitions_fail_closed(self):
        with self.assertRaisesRegex(StrategyRunError, "one job per captured candidate"):
            queued().advance(
                StrategyRunPhase.RESEARCH_PENDING, now=NOW,
                candidates=(candidate(),), research_job_ids=(),
            )
        blocked = queued().advance(
            StrategyRunPhase.BLOCKED, now=NOW, reason="market_unavailable",
        )
        with self.assertRaisesRegex(StrategyRunError, "cannot transition"):
            blocked.advance(
                StrategyRunPhase.COMPLETE, now=NOW, reason="complete",
            )

    def test_identity_changes_and_conflicting_duplicate_creation_are_rejected(self):
        store = InMemoryStrategyRunStore()
        initial = store.create(queued())
        with self.assertRaisesRegex(StrategyRunError, "different work"):
            store.create(StrategyRun(
                initial.id, initial.account_scope_id, 2,
                initial.config_release_id, initial.observed_at,
            ))
        pending = initial.advance(
            StrategyRunPhase.RESEARCH_PENDING, now=NOW,
            candidates=(candidate(),), research_job_ids=("research-job-v1",),
        )
        changed = StrategyRun(
            pending.id, pending.account_scope_id, 2, pending.config_release_id,
            pending.observed_at, pending.phase, pending.candidates,
            pending.research_job_ids, pending.reason, pending.revision,
            pending.updated_at,
        )
        with self.assertRaisesRegex(StrategyRunError, "immutable identity"):
            store.save(changed, expected_revision=0)


if __name__ == "__main__":
    unittest.main()
