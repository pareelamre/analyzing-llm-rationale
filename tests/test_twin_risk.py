from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from analyzing_llm_rationale.twin import (
    AccountSnapshot,
    CalibrationResult,
    Completeness,
    MarketSnapshot,
    RiskExposure,
    RiskLimits,
    calibrate_probability,
    evaluate_binary_candidate,
    size_binary_entry,
    size_reduce_only,
    sorted_candidate_ids,
)
from analyzing_llm_rationale.twin.store import AccountProjection

NOW = datetime(2025, 2, 1, tzinfo=timezone.utc)


def calibration(*, low: str = ".65", high: str = ".75") -> CalibrationResult:
    return CalibrationResult(
        Decimal(low), 30, "calibration-hash", mean_probability=Decimal(".7"),
        lower_bound=Decimal(low), upper_bound=Decimal(high), cutoff=NOW,
    )


def account(*, divergence: bool = False, received_at: datetime | None = None) -> AccountSnapshot:
    return AccountSnapshot(
        scope_id="scope-001", generation=4, received_at=received_at or NOW - timedelta(seconds=5),
        completeness=Completeness.COMPLETE, available_cash=Decimal("100"), total_cash=Decimal("100"),
        reserved_cash=Decimal("0"), settled_cash=Decimal("100"), holdings=(),
        position_basis=Decimal("0"), fees_paid=Decimal("0"),
        conservative_liquidation_value=Decimal("100"), positions=(), orders=(), fills=(), settlements=(),
        external_activity_ids=("manual-order",) if divergence else (), divergence=divergence,
        drift_reasons=("external_activity",) if divergence else (),
    )


def market(*, received_at: datetime | None = None) -> MarketSnapshot:
    received = received_at or NOW - timedelta(seconds=1)
    return MarketSnapshot(
        id="snapshot-001", instrument_id="instrument-001", venue_at=received - timedelta(seconds=1),
        received_at=received, sequence=9, source="venue-rest",
        complete=Completeness.COMPLETE, stale_after_seconds=10, yes_bid=Decimal(".39"),
        yes_ask=Decimal(".40"), no_bid=Decimal(".59"), no_ask=Decimal(".60"),
        fee_version="fee-v1", created_at=NOW - timedelta(seconds=1),
    )


def projection(*, cash: str = "100", reserved_cash: str = "0", reserved_loss: str = "0") -> AccountProjection:
    return AccountProjection(
        "scope-001", 1, Decimal(cash), Decimal("100"), revision=7,
        reserved_cash=Decimal(reserved_cash), reserved_max_loss=Decimal(reserved_loss),
    )


def limits(**changes: Decimal) -> RiskLimits:
    values = {
        "kelly_fraction": Decimal(".1"), "max_order_cash": Decimal("10"),
        "max_market_loss": Decimal("10"), "max_cluster_loss": Decimal("15"),
        "max_drawdown": Decimal(".2"), "max_total_loss": Decimal("20"),
        "max_trailing_additions": Decimal("10"), "max_daily_realized_loss": Decimal("5"),
    }
    values.update(changes)
    return RiskLimits(**values)


def evaluate(**changes):
    values = {
        "action": "BUY_YES", "instrument_id": "instrument-001", "cluster_id": "cluster-001",
        "venue": "kalshi", "market_snapshot": market(), "account_snapshot": account(),
        "account_projection": projection(), "exposures": (), "limits": limits(), "now": NOW,
        "calibration": calibration(), "fee_per_share": Decimal(".01"),
        "slippage_per_share": Decimal(".01"), "fee_version": "fee-v1",
        "available_depth": Decimal("100"),
        "cash_increment": Decimal(".01"), "tick_size": Decimal(".01"),
        "min_quantity": Decimal("1"), "trailing_additions": Decimal("0"),
        "realized_losses": Decimal("0"), "peak_equity": Decimal("100"),
        "current_equity": Decimal("100"),
    }
    values.update(changes)
    return evaluate_binary_candidate(**values)


class ReduceOnlyNeverGoesNegativeTests(unittest.TestCase):
    """A close can shrink to nothing, never to a buy.

    size_reduce_only promises to "clamp a verified close to existing
    inventory so it cannot flip exposure". Availability is floored at zero:

        available = max(_ZERO, held - pending)

    Without that floor an over-committed account -- pending sells exceeding
    holdings, which is what a stale or racing reservation looks like --
    returns a negative quantity. On a reduce-only order a negative quantity
    is a buy, the one thing the function exists to prevent.

    The reason field is set either way, so a caller that checks it is safe.
    This asserts the quantity itself, because the promise in the docstring
    is about the order and not about caller discipline.
    """

    def _reduce(self, held: str, requested: str, pending: str = "0", depth=None):
        return size_reduce_only(
            held_quantity=Decimal(held), requested_quantity=Decimal(requested),
            min_quantity=Decimal("1"), pending_sell_quantity=Decimal(pending),
            available_depth=None if depth is None else Decimal(depth),
        )

    def test_an_over_committed_account_closes_nothing_rather_than_buying(self):
        result = self._reduce("5", "10", pending="8")
        self.assertEqual(result.quantity, Decimal("0"))
        self.assertEqual(result.reason, "insufficient_inventory")

    def test_the_quantity_is_never_negative(self):
        """Across the states that produce a negative without the floor."""
        for held, pending in (("5", "8"), ("0", "1"), ("2", "100")):
            with self.subTest(held=held, pending=pending):
                self.assertGreaterEqual(
                    self._reduce(held, "10", pending=pending).quantity, Decimal("0"),
                )

    def test_a_close_is_capped_by_what_is_actually_held(self):
        self.assertEqual(self._reduce("3", "10").quantity, Decimal("3"))

    def test_pending_sells_are_reserved_against_the_close(self):
        self.assertEqual(self._reduce("10", "10", pending="4").quantity, Decimal("6"))

    def test_depth_caps_it_too(self):
        self.assertEqual(self._reduce("10", "10", depth="2").quantity, Decimal("2"))


class CalibrationTopBinTests(unittest.TestCase):
    """A forecast of exactly 1.0 belongs in the top bin, like any other.

    calibrate_probability picks its bin with `min(int(raw / bin_width), 9)`
    and then keeps only observations whose own bin matches. The observations
    are clamped the same way, so removing the clamp on the forecast side
    alone leaves it looking for bin 10, which nothing can be in: the sample
    collapses to zero and a certain forecast silently comes back
    uncalibrated rather than calibrated on the top bin.

    Same clamp, same reason, as the one in metrics.ece.
    """

    NOW = datetime(2026, 9, 7, tzinfo=timezone.utc)

    def _observations(self, count: int, probability: str = "0.95", outcome: int = 1):
        return [
            {
                "id": f"row-{i}", "instrument_id": f"instrument-{i}",
                "cluster_id": f"cluster-{i}", "probability": probability,
                "forecast_at": (self.NOW - timedelta(days=30)).isoformat(),
                "resolved_at": (self.NOW - timedelta(days=1)).isoformat(),
                "outcome": outcome,
            }
            for i in range(count)
        ]

    def _calibrate(self, raw: str, side: str = "YES"):
        return calibrate_probability(
            Decimal(raw), self._observations(40), as_of=self.NOW, outcome_side=side,
        )

    def test_total_certainty_is_calibrated_not_discarded(self):
        result = self._calibrate("1.0")
        self.assertIsNone(result.reason)
        self.assertEqual(result.sample_size, 40)
        self.assertIsNotNone(result.probability)

    def test_it_lands_in_the_same_bin_as_the_next_forecast_down(self):
        """0.95 and 1.0 are both top-bin, so they calibrate identically."""
        self.assertEqual(self._calibrate("1.0").probability,
                         self._calibrate("0.95").probability)

    def test_the_conservative_bound_is_still_below_the_raw_forecast(self):
        """The point of calibrating a 100% claim is that it comes back under
        100%."""
        result = self._calibrate("1.0")
        self.assertLess(result.probability, Decimal("1"))

    def test_a_past_forecast_of_certainty_is_kept_in_the_sample(self):
        """The observation side is clamped too, and for a sharper reason.

        Without it, every historical forecast of exactly 1.0 falls into a
        bin nothing queries, so calibration silently drops the most
        confident forecasts on record -- the ones it exists to correct.
        """
        certain = self._observations(40, probability="1.0", outcome=1)
        result = calibrate_probability(
            Decimal("0.95"), certain, as_of=self.NOW, outcome_side="YES",
        )
        self.assertIsNone(result.reason)
        self.assertEqual(result.sample_size, 40)

    def test_certain_forecasts_that_were_wrong_pull_the_bound_down(self):
        """The correction only happens if those rows are in the sample."""
        right = calibrate_probability(
            Decimal("0.95"), self._observations(40, probability="1.0", outcome=1),
            as_of=self.NOW, outcome_side="YES",
        )
        half_wrong = calibrate_probability(
            Decimal("0.95"),
            self._observations(20, probability="1.0", outcome=1)
            + [dict(row, id=f"w-{i}", instrument_id=f"wi-{i}", cluster_id=f"wc-{i}")
               for i, row in enumerate(self._observations(20, probability="1.0", outcome=0))],
            as_of=self.NOW, outcome_side="YES",
        )
        self.assertLess(half_wrong.probability, right.probability)

    def test_the_no_side_still_inverts_the_upper_bound(self):
        no_side = self._calibrate("1.0", side="NO")
        self.assertEqual(no_side.probability, Decimal("1") - no_side.upper_bound)


class DrawdownHaltBoundaryTests(unittest.TestCase):
    """An account exactly at its drawdown limit must stop.

    Both gates read `drawdown >= limits.max_drawdown`. Making either
    exclusive lets an account sitting precisely on its limit keep trading,
    and nothing failed: with max_drawdown 0.20 and a drawdown of exactly
    0.20, size_binary_entry went from refusing to sizing 208 contracts.

    FORESEA_ENABLE_BYO_TRADING is true on the deployed service, so these
    caps govern real connected accounts rather than the shadow book.
    """

    def _entry(self, drawdown: str, max_drawdown: str = "0.20"):
        return size_binary_entry(
            probability=Decimal("0.60"), ask=Decimal("0.40"),
            fee_per_share=Decimal("0"), slippage_per_share=Decimal("0"),
            available_cash=Decimal("1000"), current_market_loss=Decimal("0"),
            current_cluster_loss=Decimal("0"), drawdown=Decimal(drawdown),
            tick_size=Decimal("0.01"), min_quantity=Decimal("1"),
            limits=RiskLimits(
                kelly_fraction=Decimal("0.25"), max_order_cash=Decimal("100"),
                max_market_loss=Decimal("500"), max_cluster_loss=Decimal("500"),
                max_drawdown=Decimal(max_drawdown),
            ),
        )

    def test_exactly_at_the_limit_refuses_to_size(self):
        result = self._entry("0.20")
        self.assertEqual(result.quantity, Decimal("0"))
        self.assertEqual(result.reason, "drawdown_limit")

    def test_just_under_the_limit_still_trades(self):
        self.assertGreater(self._entry("0.19").quantity, Decimal("0"))

    def test_past_the_limit_refuses(self):
        self.assertEqual(self._entry("0.21").reason, "drawdown_limit")

    def test_the_boundary_is_the_configured_limit_not_a_constant(self):
        """Moving the cap moves the halt with it."""
        self.assertEqual(self._entry("0.10", max_drawdown="0.10").reason, "drawdown_limit")
        self.assertGreater(self._entry("0.10", max_drawdown="0.50").quantity, Decimal("0"))

    def test_the_candidate_evaluator_halts_at_the_same_boundary(self):
        """peak 100 against equity 80 is a drawdown of exactly 0.20.

        This pins the behaviour, not the line. evaluate_binary_candidate
        has its own `drawdown >= max_drawdown` check, but it is redundant
        here: the call reaches size_binary_entry, whose gate refuses first.
        Making the evaluator's copy exclusive changes nothing observable
        through this path, so no test can distinguish it -- what matters,
        that an account exactly at its limit stops, is asserted either way.
        """
        at_limit = evaluate(peak_equity=Decimal("100"), current_equity=Decimal("80"))
        self.assertEqual(at_limit.reason, "drawdown_limit")

        under = evaluate(peak_equity=Decimal("100"), current_equity=Decimal("81"))
        self.assertNotEqual(under.reason, "drawdown_limit")

    def test_the_daily_loss_halt_is_also_inclusive(self):
        at_limit = evaluate(realized_losses=Decimal("5"))
        self.assertEqual(at_limit.reason, "realized_loss_limit")

        under = evaluate(realized_losses=Decimal("4.99"))
        self.assertNotEqual(under.reason, "realized_loss_limit")


class TwinRiskTests(unittest.TestCase):
    def test_costs_depth_limits_and_currency_rounding_bind_size(self):
        result = size_binary_entry(
            probability=Decimal(".70"), ask=Decimal(".50"), fee_per_share=Decimal(".013"),
            slippage_per_share=Decimal(".011"), available_cash=Decimal("100"),
            current_market_loss=Decimal("0"), current_cluster_loss=Decimal("0"),
            drawdown=Decimal("0"), tick_size=Decimal(".01"), min_quantity=Decimal("1"),
            available_depth=Decimal("3.5"), cash_increment=Decimal(".01"),
            limits=RiskLimits(Decimal("1"), Decimal("10"), Decimal("20"), Decimal("20"), Decimal(".2")),
        )
        self.assertEqual(result.quantity, Decimal("3"))
        self.assertEqual(result.cash, Decimal("1.58"))
        self.assertEqual(result.cash_delta, Decimal("-1.58"))

    def test_fixed_bin_calibration_is_prospective_stratified_and_independent(self):
        rows = []
        for index in range(30):
            rows.append({
                "id": f"forecast-{index:02d}", "instrument_id": f"instrument-{index:02d}",
                "cluster_id": f"cluster-{index:02d}", "probability": ".64",
                "outcome": 1 if index < 24 else 0, "forecast_at": (NOW - timedelta(days=10)).isoformat(),
                "resolved_at": (NOW - timedelta(days=2)).isoformat(), "model_hash": "model-a",
                "prompt_hash": "prompt-a", "category_family": "politics",
            })
        rows.extend([
            {**rows[0], "id": "later-duplicate", "outcome": 0, "forecast_at": (NOW - timedelta(days=9)).isoformat()},
            {**rows[1], "id": "future-resolution", "resolved_at": (NOW + timedelta(days=1)).isoformat()},
            {**rows[2], "id": "wrong-model", "model_hash": "model-b"},
            {**rows[3], "id": "outcome-known-first", "forecast_at": (NOW - timedelta(days=1)).isoformat()},
        ])
        result = calibrate_probability(
            Decimal(".61"), list(reversed(rows)), as_of=NOW, model_hash="model-a",
            prompt_hash="prompt-a", category_family="politics",
        )
        repeat = calibrate_probability(
            Decimal(".61"), rows, as_of=NOW, model_hash="model-a",
            prompt_hash="prompt-a", category_family="politics",
        )
        self.assertEqual(result, repeat)
        self.assertEqual(result.sample_size, 30)
        self.assertEqual(result.mean_probability, Decimal("25") / Decimal("32"))
        self.assertEqual(result.training_ids[0], "forecast-00")
        self.assertGreater(result.probability, Decimal(".6"))
        self.assertEqual(result.calibration_version, "calibration_v1")

    def test_calibration_blocks_small_samples_and_no_side_uses_upper_bound(self):
        rows = [{
            "id": f"f-{index}", "instrument_id": f"i-{index}", "cluster_id": f"c-{index}",
            "probability": ".24", "outcome": 0, "forecast_at": (NOW - timedelta(days=3)).isoformat(),
            "resolved_at": (NOW - timedelta(days=1)).isoformat(),
        } for index in range(30)]
        self.assertEqual(
            calibrate_probability(Decimal(".24"), rows[:29], as_of=NOW).reason,
            "insufficient_calibration_sample",
        )
        no_result = calibrate_probability(Decimal(".24"), rows, as_of=NOW, outcome_side="NO")
        self.assertEqual(no_result.probability, Decimal("1") - no_result.upper_bound)

    def test_yes_and_no_entry_use_conservative_interval_and_capture_versions(self):
        yes = evaluate()
        no = evaluate(action="BUY_NO", calibration=calibration(low=".10", high=".30"))
        self.assertGreater(yes.quantity, 0)
        self.assertGreater(no.quantity, 0)
        self.assertEqual(yes.expected_account_revision, 7)
        self.assertEqual(yes.account_generation, 4)
        self.assertEqual(yes.market_snapshot_id, "snapshot-001")
        self.assertEqual(yes.calibration_hash, "calibration-hash")
        self.assertEqual(yes.reservation_preconditions.market_version, "snapshot-001")

    def test_inventory_orders_and_reservations_bind_market_cluster_and_total_caps(self):
        exposures = (
            RiskExposure("instrument-001", "cluster-001", "kalshi", Decimal("4"), "inventory"),
            RiskExposure("instrument-002", "cluster-001", "polymarket", Decimal("7"), "open_order"),
            RiskExposure("instrument-003", "cluster-002", "kalshi", Decimal("2"), "local_reservation"),
        )
        result = evaluate(
            exposures=exposures, account_projection=projection(reserved_cash="2", reserved_loss="2"),
            limits=limits(max_cluster_loss=Decimal("12"), max_total_loss=Decimal("14")),
        )
        self.assertEqual(result.quantity, Decimal("2"))
        self.assertLessEqual(result.max_loss, Decimal("1"))

    def test_reduce_only_accounts_for_pending_sells_partial_fills_and_depth(self):
        result = evaluate(
            action="SELL_YES", calibration=None, requested_quantity=Decimal("10"),
            held_quantity=Decimal("5"), pending_sell_quantity=Decimal("2"),
            available_depth=Decimal("2.5"),
        )
        self.assertEqual(result.quantity, Decimal("2"))
        self.assertEqual(result.cash, Decimal("0"))
        self.assertEqual(result.cash_delta, Decimal(".74"))
        self.assertEqual(
            size_reduce_only(
                held_quantity=Decimal("3"), pending_sell_quantity=Decimal("3"),
                requested_quantity=Decimal("1"), min_quantity=Decimal("1"),
            ).reason,
            "insufficient_inventory",
        )

    def test_external_activity_staleness_and_incomplete_portfolio_block(self):
        self.assertEqual(evaluate(account_snapshot=account(divergence=True)).reason, "account_reconciliation_blocked")
        self.assertEqual(
            evaluate(market_snapshot=market(received_at=NOW - timedelta(seconds=11))).reason,
            "stale_market_snapshot",
        )
        self.assertEqual(evaluate(exposures=None).reason, "incomplete_portfolio_state")
        self.assertEqual(
            evaluate(exposures=(), account_projection=projection(reserved_loss="1", reserved_cash="1")).reason,
            "incomplete_portfolio_state",
        )

    def test_missing_calibration_cost_depth_cash_and_uncertainty_block(self):
        self.assertEqual(evaluate(calibration=None).reason, "missing_calibration")
        self.assertEqual(
            evaluate(calibration=CalibrationResult(Decimal(".6"), 30, "hash")).reason,
            "missing_calibration",
        )
        self.assertEqual(evaluate(available_depth=None).reason, "missing_cost_or_depth")
        self.assertEqual(evaluate(fee_per_share=None).reason, "missing_cost_or_depth")
        self.assertEqual(evaluate(fee_version="fee-v0").reason, "stale_fee_schedule")
        self.assertEqual(
            evaluate(
                account_projection=projection(cash="0"),
                account_snapshot=replace(account(), available_cash=Decimal("0")),
            ).reason,
            "minimum_size_exceeds_cap",
        )

    def test_trailing_realized_and_peak_equity_limits_are_separate(self):
        self.assertEqual(evaluate(trailing_additions=Decimal("10")).reason, "trailing_additions_limit")
        self.assertEqual(evaluate(realized_losses=Decimal("5")).reason, "realized_loss_limit")
        self.assertEqual(evaluate(current_equity=Decimal("80")).reason, "drawdown_limit")
        self.assertEqual(evaluate(peak_equity=Decimal("0"), current_equity=Decimal("0")).reason, "zero_bankroll")

    def test_minimum_order_and_deterministic_priority_are_stable(self):
        self.assertEqual(evaluate(min_quantity=Decimal("1000")).reason, "minimum_size_exceeds_cap")
        self.assertEqual(
            sorted_candidate_ids([
                {"id": "b", "net_edge": ".1"}, {"id": "a", "net_edge": ".1"},
                {"id": "c", "net_edge": ".2"},
            ]),
            ("c", "a", "b"),
        )


if __name__ == "__main__":
    unittest.main()
