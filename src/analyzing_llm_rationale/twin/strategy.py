"""Stable, shadow-only orchestration for one autonomous twin cycle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class StrategyCycle:
    key: str
    decision: str
    reason: str
    exits_evaluated: bool = False


class ForeseaEdgeStrategy:
    def __init__(self) -> None:
        self._finished: dict[str, StrategyCycle] = {}

    def run(
        self, key: str, *, reconcile: Callable[[], bool], research: Callable[[], bool],
        risk: Callable[[], bool], submit_shadow: Callable[[], None],
        evaluate_exits: Optional[Callable[[], bool]] = None,
    ) -> StrategyCycle:
        """Run maintenance before new research; a provider outage never skips exits."""
        if not key.strip():
            raise ValueError("strategy cycle key is required")
        if key in self._finished:
            return self._finished[key]
        if not reconcile():
            cycle = StrategyCycle(key, "HOLD", "account_incomplete")
        elif evaluate_exits is not None and not evaluate_exits():
            cycle = StrategyCycle(key, "HOLD", "exit_evaluation_failed", True)
        elif not research():
            cycle = StrategyCycle(key, "PASS", "research_unavailable", evaluate_exits is not None)
        elif not risk():
            cycle = StrategyCycle(key, "PASS", "risk_rejected", evaluate_exits is not None)
        else:
            submit_shadow()
            cycle = StrategyCycle(key, "SHADOW_SUBMITTED", "eligible", evaluate_exits is not None)
        self._finished[key] = cycle
        return cycle
