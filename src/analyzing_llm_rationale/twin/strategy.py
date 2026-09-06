"""Stable, shadow-only orchestration for one autonomous twin cycle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class StrategyCycle:
    key: str
    decision: str
    reason: str


class ForeseaEdgeStrategy:
    def __init__(self) -> None:
        self._finished: dict[str, StrategyCycle] = {}

    def run(self, key: str, *, reconcile: Callable[[], bool], research: Callable[[], bool], risk: Callable[[], bool], submit_shadow: Callable[[], None]) -> StrategyCycle:
        if key in self._finished:
            return self._finished[key]
        if not reconcile():
            cycle = StrategyCycle(key, "HOLD", "account_incomplete")
        elif not research():
            cycle = StrategyCycle(key, "PASS", "research_unavailable")
        elif not risk():
            cycle = StrategyCycle(key, "PASS", "risk_rejected")
        else:
            submit_shadow()
            cycle = StrategyCycle(key, "SHADOW_SUBMITTED", "eligible")
        self._finished[key] = cycle
        return cycle
