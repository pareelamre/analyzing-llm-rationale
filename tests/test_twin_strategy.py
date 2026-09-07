from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from analyzing_llm_rationale.twin import (
    AccountScope,
    AccountSnapshot,
    CommandState,
    Completeness,
    Forecast,
    Instrument,
    MarketSnapshot,
    Proposal,
    ProposalAction,
    RiskExposure,
    RiskLimits,
)
from analyzing_llm_rationale.twin.budget import BudgetExceeded
from analyzing_llm_rationale.twin.research_gateway import ResearchProvenance, ResearchResult
from analyzing_llm_rationale.twin.simulator import CapturedBook, DepthLevel, ShadowVenue
from analyzing_llm_rationale.twin.store import AccountProjection, ExecutionCommand
from analyzing_llm_rationale.twin.strategy import (
    CandidateMemory,
    ForeseaEdgeStrategy,
    HeldPosition,
    InMemoryStrategyStore,
    StrategyAccountState,
    StrategyCandidate,
    StrategyCycle,
    StrategyPolicy,
    load_strategy_policy,
    strategy_cycle_key,
)

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
INSTRUMENT_ID = "kalshi:demo:KXTEST"
MODEL_HASH = "a" * 64
PROMPT_HASH = "b" * 64


def scope() -> AccountScope:
    return AccountScope(
        "shadow-scope:shadow-account-001", "owner-001", "kalshi", "shadow-account-001",
        "demo", "USD", "connection-001", 1, NOW - timedelta(days=1),
    )


def instrument(*, settlement_hash: str = "settlement-v1") -> Instrument:
    return Instrument(
        INSTRUMENT_ID, "kalshi", "demo", "KXTEST", None, None, None,
        settlement_hash, "politics", "event-001", "cluster-001",
        Decimal(".01"), Decimal("1"), "fee-v1", "cap-v1", "open",
        NOW + timedelta(days=30), NOW + timedelta(days=31), NOW - timedelta(days=1),
    )


def snapshot(*, suffix: str = "001", at: datetime = NOW, yes_ask: str = ".40") -> MarketSnapshot:
    return MarketSnapshot(
        f"snapshot-{suffix}", INSTRUMENT_ID, at - timedelta(seconds=2), at - timedelta(seconds=1),
        int(suffix), "captured-rest", Completeness.COMPLETE, 10,
        Decimal(yes_ask) - Decimal(".01"), Decimal(yes_ask), Decimal(".59"), Decimal(".60"),
        "fee-v1", at - timedelta(seconds=1),
    )


def calibration_rows() -> tuple[dict, ...]:
    return tuple({
        "id": f"forecast-history-{index:02d}", "instrument_id": f"instrument-{index:02d}",
        "cluster_id": f"cluster-{index:02d}", "probability": ".74", "outcome": 1,
        "forecast_at": (NOW - timedelta(days=10)).isoformat(),
        "resolved_at": (NOW - timedelta(days=2)).isoformat(),
        "model_hash": MODEL_HASH, "prompt_hash": PROMPT_HASH, "category_family": "politics",
    } for index in range(30))


def candidate(*, suffix: str = "001", at: datetime = NOW, yes_ask: str = ".40") -> StrategyCandidate:
    return StrategyCandidate(
        instrument(), snapshot(suffix=suffix, at=at, yes_ask=yes_ask),
        Decimal("10"), Decimal("10"), Decimal(".01"), Decimal(".01"), calibration_rows(),
    )


def account_snapshot(*, cash: Decimal = Decimal("100"), at: datetime = NOW) -> AccountSnapshot:
    return AccountSnapshot(
        scope().id, 1, at - timedelta(seconds=1), Completeness.COMPLETE,
        cash, cash, Decimal("0"), cash, (), Decimal("0"), Decimal("0"), cash,
        (), (), (), (), (), False, (),
    )


def state(*, at: datetime = NOW, cash: Decimal = Decimal("100"), positions=(), exposures=()) -> StrategyAccountState:
    return StrategyAccountState(
        account_snapshot(cash=cash, at=at), AccountProjection(scope().id, 1, cash, Decimal("100"), revision=3),
        tuple(positions), tuple(exposures), Decimal("0"), Decimal("0"), cash, cash,
    )


def policy(*, config_version: str = "policy-v1", cooldown: int = 3600) -> StrategyPolicy:
    return StrategyPolicy(
        config_version, RiskLimits(
            Decimal(".1"), Decimal("10"), Decimal("20"), Decimal("30"), Decimal(".2"),
            Decimal("40"), Decimal("10"), Decimal("10"),
        ),
        candidate_cooldown_seconds=cooldown,
    )


def forecast_for(item: StrategyCandidate, *, p_yes: str = ".75", low: str = ".65", high: str = ".85") -> Forecast:
    return Forecast(
        "forecast-new", item.instrument.id, Decimal(p_yes), Decimal(p_yes), "calibrated",
        Decimal(low), Decimal(high), ("evidence-1", "evidence-2"), NOW,
        NOW + timedelta(hours=1), MODEL_HASH, PROMPT_HASH, "strategy-hash",
        "calibration_v1", "prospective-hash", NOW,
    )


def research_result(item: StrategyCandidate, *, p_yes: str = ".75") -> ResearchResult:
    forecast = Forecast(
        "forecast-new", item.instrument.id, Decimal(p_yes), None, "uncalibrated",
        Decimal(".65"), Decimal(".85"), ("evidence-1", "evidence-2"), NOW,
        NOW + timedelta(hours=1), MODEL_HASH, PROMPT_HASH, "strategy-hash",
        "insufficient", "prospective-hash", NOW,
    )
    proposal = Proposal(
        "proposal-new", forecast.id, item.snapshot.id, ProposalAction.HOLD,
        (), forecast.evidence_ids, None, None, NOW,
    )
    provenance = ResearchProvenance(
        "c" * 64, "d" * 64, PROMPT_HASH, MODEL_HASH, "bounded uncertainty",
        ("evidence-1",), ("evidence-2",),
    )
    return ResearchResult(forecast, proposal, provenance, "e" * 64)


def strategy(*, store=None, selected_policy=None) -> ForeseaEdgeStrategy:
    return ForeseaEdgeStrategy(store=store or InMemoryStrategyStore(), policy=selected_policy or policy())


class TwinStrategyTests(unittest.TestCase):
    def test_legacy_cycle_is_idempotent_and_maintenance_precedes_research(self):
        legacy, calls = ForeseaEdgeStrategy(), []
        cycle = legacy.run(
            "cycle-001", reconcile=lambda: True,
            evaluate_exits=lambda: calls.append("exits") or True,
            research=lambda: calls.append("research") or True,
            risk=lambda: True, submit_shadow=lambda: calls.append("submit"),
        )
        self.assertEqual(cycle.decision, "SHADOW_SUBMITTED")
        legacy.run(
            "cycle-001", reconcile=lambda: True, research=lambda: True,
            risk=lambda: True, submit_shadow=lambda: calls.append("again"),
        )
        self.assertEqual(calls, ["exits", "research", "submit"])

    def test_entry_cycle_emits_one_attributable_intent_and_reuses_finished_result(self):
        engine = strategy()
        calls = []
        result = engine.run_cycle(
            scope=scope(), now=NOW, reconcile=lambda: calls.append("reconcile") or state(),
            load_position_market=lambda _position: None,
            discover=lambda: (candidate(suffix="002"), candidate()),
            research=lambda item: calls.append(item.snapshot.id) or research_result(item),
        )
        self.assertEqual(result.decision, "INTENT")
        self.assertEqual(result.intent.action, ProposalAction.BUY_YES)
        self.assertEqual(result.intent.market_version, "snapshot-002")
        self.assertIsNotNone(result.risk_result.reservation_preconditions)
        self.assertEqual([step.stage for step in result.steps], [
            "reconcile", "exit", "research", "calibration", "risk",
        ])
        repeat = engine.run_cycle(
            scope=scope(), now=NOW, reconcile=lambda: calls.append("bad"),
            load_position_market=lambda _position: None, discover=lambda: (),
            research=lambda _item: research_result(candidate()),
        )
        self.assertEqual(repeat, result)
        self.assertEqual(calls, ["reconcile", "snapshot-002"])
        self.assertEqual(StrategyCycle.from_storage(result.to_storage()), result)

    def test_provider_outage_budget_exhaustion_and_empty_discovery_are_safe_decisions(self):
        unavailable = strategy().run_cycle(
            scope=scope(), now=NOW, reconcile=lambda: state(), load_position_market=lambda _position: None,
            discover=lambda: (candidate(),), research=lambda _item: (_ for _ in ()).throw(RuntimeError("offline")),
        )
        self.assertEqual((unavailable.decision, unavailable.reason), ("PASS", "research_unavailable"))
        exhausted = strategy().run_cycle(
            scope=scope(), now=NOW, reconcile=lambda: state(), load_position_market=lambda _position: None,
            discover=lambda: (candidate(),), research=lambda _item: (_ for _ in ()).throw(BudgetExceeded("limit")),
        )
        self.assertEqual((exhausted.decision, exhausted.reason), ("PASS", "budget_exhausted"))
        empty = strategy().run_cycle(
            scope=scope(), now=NOW, reconcile=lambda: state(), load_position_market=lambda _position: None,
            discover=lambda: (), research=lambda _item: research_result(candidate()),
        )
        self.assertEqual((empty.decision, empty.reason), ("HOLD", "no_changed_candidates"))

    def test_cooldown_requires_new_version_and_material_change_or_elapsed_time(self):
        store = InMemoryStrategyStore()
        first = strategy(store=store)
        first.run_cycle(
            scope=scope(), now=NOW, reconcile=lambda: state(), load_position_market=lambda _position: None,
            discover=lambda: (candidate(),), research=lambda item: research_result(item),
        )
        within = NOW + timedelta(minutes=5)
        unchanged = strategy(store=store).run_cycle(
            scope=scope(), now=within, reconcile=lambda: state(at=within),
            load_position_market=lambda _position: None,
            discover=lambda: (candidate(suffix="002", at=within, yes_ask=".405"),),
            research=lambda _item: self.fail("cooldown should skip immaterial candidate"),
        )
        self.assertEqual(unchanged.reason, "no_changed_candidates")
        changed = strategy(store=store).run_cycle(
            scope=scope(), now=within + timedelta(minutes=5), reconcile=lambda: state(at=within + timedelta(minutes=5)),
            load_position_market=lambda _position: None,
            discover=lambda: (candidate(suffix="003", at=within + timedelta(minutes=5), yes_ask=".42"),),
            research=lambda item: research_result(item),
        )
        self.assertEqual(changed.decision, "INTENT")

    def test_full_buy_partial_fill_revised_forecast_close_and_settlement(self):
        entry_engine = strategy()
        entry_candidate = candidate()
        entry = entry_engine.run_cycle(
            scope=scope(), now=NOW, reconcile=lambda: state(), load_position_market=lambda _position: None,
            discover=lambda: (entry_candidate,), research=lambda item: research_result(item),
        )
        venue = ShadowVenue(account_id="shadow-account-001", seed=4, starting_cash=Decimal("100"))
        entry_book = CapturedBook(
            entry_candidate.snapshot, "yes", (DepthLevel(Decimal(".40"), Decimal("2")),),
        )
        preview = venue.preview(entry.intent, entry.risk_result, entry_candidate.instrument, entry_book, now=NOW)
        command = ExecutionCommand(
            "command-entry", scope().id, entry.intent.id, entry.intent.intent_hash,
            CommandState.SUBMITTING, "reservation-entry", "client-entry", NOW,
            request_fingerprint="request-entry",
        )
        ack = venue.submit(command, preview, now=NOW)
        entry_order_id = ack["acknowledgement"]["venue_order_id"]
        self.assertEqual(venue.status(entry_order_id).status, "partial")

        close_at = NOW + timedelta(minutes=5)
        close_candidate = candidate(suffix="002", at=close_at)
        holding = HeldPosition(
            entry_candidate.instrument, "yes", Decimal("2"), NOW, NOW + timedelta(hours=1),
            entry_candidate.instrument.settlement_spec_hash, "policy-v1",
            latest_forecast=forecast_for(close_candidate, p_yes=".25", low=".15", high=".35"),
        )
        shadow_account = venue.account(received_at=close_at - timedelta(seconds=1))
        close_state = StrategyAccountState(
            shadow_account,
            AccountProjection(scope().id, 1, shadow_account.available_cash, Decimal("100"), revision=4),
            (holding,), (RiskExposure(INSTRUMENT_ID, "cluster-001", "kalshi", Decimal(".8"), "inventory"),),
            Decimal(".8"), Decimal("0"), Decimal("100"), Decimal("99.8"),
        )
        close = strategy().run_cycle(
            scope=scope(), now=close_at, reconcile=lambda: close_state,
            load_position_market=lambda _position: close_candidate, discover=lambda: (),
            research=lambda _item: self.fail("deterministic exit must not call the model"),
        )
        self.assertEqual((close.decision, close.reason), ("INTENT", "revised_forecast_invalidated"))
        self.assertEqual(close.intent.action, ProposalAction.SELL_YES)
        close_book = CapturedBook(
            close_candidate.snapshot, "yes", (DepthLevel(Decimal(".39"), Decimal("2")),),
        )
        close_preview = venue.preview(
            close.intent, close.risk_result, close_candidate.instrument, close_book, now=close_at,
        )
        close_command = ExecutionCommand(
            "command-close", scope().id, close.intent.id, close.intent.intent_hash,
            CommandState.SUBMITTING, "reservation-close", "client-close", close_at,
            request_fingerprint="request-close",
        )
        venue.submit(close_command, close_preview, now=close_at)
        self.assertEqual(venue.account(received_at=close_at).holdings, ())
        settled = venue.settle(
            entry_order_id, resolved_outcome="yes", settled_at=NOW + timedelta(days=31),
        )
        self.assertEqual(settled.settled_payout, Decimal("0"))
        self.assertEqual(len(venue.account(received_at=NOW + timedelta(days=31)).settlements), 1)

    def test_policy_change_and_settlement_rule_change_exit_without_discovery(self):
        for held, expected in (
            (HeldPosition(instrument(), "yes", Decimal("1"), NOW, NOW + timedelta(days=1), "settlement-v1", "old-policy"), "policy_changed"),
            (HeldPosition(instrument(), "yes", Decimal("1"), NOW, NOW + timedelta(days=1), "old-settlement", "policy-v1"), "settlement_rule_changed"),
        ):
            engine = strategy()
            result = engine.run_cycle(
                scope=scope(), now=NOW, reconcile=lambda held=held: state(positions=(held,)),
                load_position_market=lambda _position: candidate(),
                discover=lambda: self.fail("exit maintenance must precede discovery"),
                research=lambda _item: self.fail("exit maintenance must not research"),
            )
            self.assertEqual((result.decision, result.reason), ("INTENT", expected))

    def test_cycle_key_binds_strategy_account_bucket_and_config(self):
        first = strategy_cycle_key(scope=scope(), now=NOW, config_version="v1", bucket_seconds=300)
        self.assertEqual(first, strategy_cycle_key(scope=scope(), now=NOW + timedelta(seconds=299), config_version="v1", bucket_seconds=300))
        self.assertNotEqual(first, strategy_cycle_key(scope=scope(), now=NOW + timedelta(seconds=300), config_version="v1", bucket_seconds=300))
        self.assertNotEqual(first, strategy_cycle_key(scope=scope(), now=NOW, config_version="v2", bucket_seconds=300))

    def test_research_result_must_bind_the_selected_market_snapshot(self):
        selected = candidate()
        result = strategy().run_cycle(
            scope=scope(), now=NOW, reconcile=lambda: state(),
            load_position_market=lambda _position: None, discover=lambda: (selected,),
            research=lambda _item: research_result(candidate(suffix="002")),
        )
        self.assertEqual((result.decision, result.reason), ("PASS", "no_candidate_qualified"))
        self.assertEqual(result.steps[-1].reason, "no_candidate_qualified")
        self.assertTrue(any(step.reason == "forecast_stale_or_mismatched" for step in result.steps))

    def test_store_keeps_first_cycle_and_newest_cooldown_observation(self):
        store = InMemoryStrategyStore()
        first = StrategyCycle("stable-cycle", "HOLD", "first", created_at=NOW)
        second = StrategyCycle("stable-cycle", "PASS", "later-worker", created_at=NOW)
        self.assertTrue(store.record_cycle(first))
        self.assertFalse(store.record_cycle(second))
        self.assertEqual(store.get_cycle("stable-cycle"), first)

        current = candidate()
        newer = CandidateMemory(
            INSTRUMENT_ID, "snapshot-new", current.midpoint, current.rules_hash,
            "policy-v1", NOW + timedelta(minutes=2),
        )
        stale = CandidateMemory(
            INSTRUMENT_ID, "snapshot-stale", current.midpoint, current.rules_hash,
            "policy-v1", NOW + timedelta(minutes=1),
        )
        store.record_candidate(scope().id, newer)
        store.record_candidate(scope().id, stale)
        self.assertEqual(store.get_candidate(scope().id, INSTRUMENT_ID), newer)

    def test_repository_strategy_config_is_shadow_only_and_fail_closed(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "twin.yaml"
        loaded = load_strategy_policy(config_path)
        self.assertEqual(loaded.config_version, "foresea-edge-shadow-v1")
        self.assertEqual(loaded.max_new_positions_per_cycle, 1)
        self.assertEqual(loaded.risk_limits.max_order_cash, Decimal("0"))
        self.assertEqual(loaded.risk_limits.max_total_loss, Decimal("0"))


if __name__ == "__main__":
    unittest.main()
