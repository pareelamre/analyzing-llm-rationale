"""Time-unit conversions that nothing was holding.

Four places convert a timedelta into a scalar horizon. Three divide by
86400 for days, one by 60 for minutes. The divisor is the kind of
constant that reads as obviously right and is therefore never checked --
and two of the four were not checked. Against the whole 1,986-test suite:

    market_data._within_close_window   86400.0 -> 3600.0   SURVIVED
    crypto_kalshi._minutes_to_close       60.0 -> 3600.0   SURVIVED

(track_record_live._lead_time_days was already caught. radar._days_until
has no production caller.)

Neither is cosmetic. _within_close_window decides which markets enter the
track record at all; _minutes_to_close feeds the time-to-expiry of the
Kalshi BTC volatility model, whose sigma is estimated per minute.
"""

import unittest
from datetime import datetime, timedelta, timezone

from analyzing_llm_rationale import crypto_kalshi
from analyzing_llm_rationale.market_data import _within_close_window

UTC = timezone.utc


class MinutesToCloseTests(unittest.TestCase):
    """The divisor is 60: the result is minutes, not hours or seconds."""

    NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)

    def setUp(self):
        self._real_now = crypto_kalshi._now
        crypto_kalshi._now = lambda: self.NOW
        self.addCleanup(setattr, crypto_kalshi, "_now", self._real_now)

    def test_ninety_minutes_out_reads_as_ninety(self):
        got = crypto_kalshi._minutes_to_close("2026-09-08T13:30:00Z")
        self.assertAlmostEqual(got, 90.0)
        self.assertNotAlmostEqual(got, 1.5)  # hours
        self.assertNotAlmostEqual(got, 5400.0)  # seconds

    def test_a_close_time_already_passed_is_negative(self):
        """The resolution sweep only calls the API once this goes <= 0."""
        self.assertAlmostEqual(crypto_kalshi._minutes_to_close("2026-09-08T11:45:00Z"), -15.0)

    def test_a_naive_close_time_is_not_silently_shifted(self):
        self.assertAlmostEqual(crypto_kalshi._minutes_to_close("2026-09-08T13:30:00+00:00"), 90.0)

    def test_an_unparseable_close_time_is_none(self):
        self.assertIsNone(crypto_kalshi._minutes_to_close("soon"))
        self.assertIsNone(crypto_kalshi._minutes_to_close(None))


class SigmaAndHorizonShareAUnitTests(unittest.TestCase):
    """Why the divisor matters: sigma is per-minute, so t must be minutes.

    model_prob_above computes sigma_t = sigma_min * sqrt(minutes). Passing
    hours would understate the diffusion by sqrt(60) ~ 7.75x, which does
    not merely nudge the probability -- it collapses it, and the collapsed
    figure is what the edge is computed against.
    """

    SPOT, STRIKE, SIGMA_PER_MIN = 100.0, 101.0, 0.001

    def _p(self, horizon):
        return crypto_kalshi.model_prob_above(self.SPOT, self.STRIKE, self.SIGMA_PER_MIN, horizon)

    def test_the_wrong_unit_does_not_merely_shift_the_probability(self):
        in_minutes = self._p(90.0)
        in_hours = self._p(1.5)
        self.assertGreater(in_minutes, 0.10)
        self.assertLess(in_hours, 1e-10)

    def test_a_longer_horizon_pulls_toward_a_coin_flip(self):
        """Sanity on the direction, so the numbers above are not accidental."""
        self.assertLess(self._p(10.0), self._p(90.0))
        self.assertLess(self._p(90.0), self._p(100000.0))

    def test_a_non_positive_horizon_has_no_probability(self):
        self.assertIsNone(self._p(0.0))
        self.assertIsNone(self._p(-5.0))


class CloseWindowTests(unittest.TestCase):
    """The divisor is 86400: the bounds are days.

    This gate decides which markets are worth tracking -- not same-day
    noise, not multi-decade markets that never score. Reading the lead in
    hours would make every bound 24x tighter and quietly starve the track
    record.

    Not pinned here: `lead < min_days` -> `lead <= min_days`. lead is
    derived from the wall clock, so landing exactly on a bound is not
    reachable and the two forms cannot be told apart. That survivor is
    equivalent rather than a gap.
    """

    @staticmethod
    def _closing_in(hours):
        return (datetime.now(UTC) + timedelta(hours=hours)).isoformat()

    def test_a_market_a_day_and_a_half_out_sits_in_the_one_to_two_day_band(self):
        thirty_six_hours = self._closing_in(36)
        self.assertTrue(_within_close_window(thirty_six_hours, 1.0, 2.0))
        self.assertFalse(_within_close_window(thirty_six_hours, 2.0, 3.0))
        self.assertFalse(_within_close_window(thirty_six_hours, None, 1.0))

    def test_each_bound_can_be_disabled_on_its_own(self):
        two_days = self._closing_in(48)
        self.assertTrue(_within_close_window(two_days, None, 5.0))
        self.assertTrue(_within_close_window(two_days, 1.0, None))
        self.assertFalse(_within_close_window(two_days, 5.0, None))

    def test_no_bounds_at_all_keeps_everything(self):
        self.assertTrue(_within_close_window(self._closing_in(9000), None, None))
        self.assertTrue(_within_close_window(None, None, None))
        self.assertTrue(_within_close_window("nonsense", None, None))

    def test_an_unusable_close_time_is_kept_only_when_there_is_no_max(self):
        for bad in (None, "", "whenever"):
            self.assertTrue(_within_close_window(bad, 1.0, None), bad)
            self.assertFalse(_within_close_window(bad, None, 30.0), bad)

    def test_a_naive_close_time_is_read_as_utc(self):
        naive = (datetime.now(UTC) + timedelta(hours=36)).replace(tzinfo=None).isoformat()
        self.assertTrue(_within_close_window(naive, 1.0, 2.0))

    def test_a_market_already_closed_falls_below_any_positive_minimum(self):
        self.assertFalse(_within_close_window(self._closing_in(-24), 0.0, 30.0))


if __name__ == "__main__":
    unittest.main()
