import unittest

from analyzing_llm_rationale.twin.strategy import ForeseaEdgeStrategy


class TwinStrategyTests(unittest.TestCase):
    def test_cycle_is_idempotent_and_never_submits_when_gated(self):
        strategy, calls = ForeseaEdgeStrategy(), []
        cycle = strategy.run("cycle-001", reconcile=lambda: True, research=lambda: True, risk=lambda: True, submit_shadow=lambda: calls.append("submit"))
        self.assertEqual(cycle.decision, "SHADOW_SUBMITTED")
        strategy.run("cycle-001", reconcile=lambda: True, research=lambda: True, risk=lambda: True, submit_shadow=lambda: calls.append("again"))
        self.assertEqual(calls, ["submit"])
        self.assertEqual(strategy.run("cycle-002", reconcile=lambda: False, research=lambda: True, risk=lambda: True, submit_shadow=lambda: calls.append("bad")).decision, "HOLD")

    def test_exit_maintenance_runs_before_provider_bound_research(self):
        strategy, calls = ForeseaEdgeStrategy(), []
        cycle = strategy.run(
            "cycle-003", reconcile=lambda: True, evaluate_exits=lambda: calls.append("exits") or True,
            research=lambda: False, risk=lambda: True, submit_shadow=lambda: calls.append("submit"),
        )
        self.assertEqual(cycle.decision, "PASS")
        self.assertTrue(cycle.exits_evaluated)
        self.assertEqual(calls, ["exits"])
