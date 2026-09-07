"""The bounds on the risk policy's environment variables.

_risk_guard_policy validates five settings and raises on each. None of it
was pinned: loosening any of the five comparisons passed the whole suite.

Two of the five fail in the direction that matters. A concentration limit
of 0, an account value of 0 or a per-cycle spend of 0 stop the agent
trading, which is loud enough to notice. But max_drawdown_limit = 1.0 and
daily_risk_limit_pct > 1 do the opposite: they leave the agent trading
with the guard switched off, because a 100% drawdown means the account is
already gone and no daily loss can exceed the whole account. A typo in a
Cloud Run env var would disable the protection and look like nothing
happened.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import benchmark_tools  # noqa: E402

#: Values that keep every *other* setting valid while one is varied.
_SANE = {
    "FORESEA_AGENT_ACCOUNT_VALUE": "10000",
    "FORESEA_AGENT_CONCENTRATION_LIMIT": "0.15",
    "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "0.20",
    "FORESEA_AGENT_MAX_DRAWDOWN_LIMIT": "0.50",
    "FORESEA_AGENT_DAILY_RISK_LIMIT_PCT": "0.30",
}


def _policy_with(**overrides):
    env = {**_SANE, **{k: str(v) for k, v in overrides.items()}}
    with mock.patch.dict(os.environ, env, clear=False):
        return benchmark_tools._risk_guard_policy()


class RejectedSettingsTests(unittest.TestCase):
    """Each bound, given the value just outside it."""

    CASES = (
        ("FORESEA_AGENT_ACCOUNT_VALUE", "0", "account value"),
        ("FORESEA_AGENT_ACCOUNT_VALUE", "-1", "negative account value"),
        ("FORESEA_AGENT_CONCENTRATION_LIMIT", "0", "concentration of nothing"),
        ("FORESEA_AGENT_CONCENTRATION_LIMIT", "1.5", "concentration above the account"),
        ("FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT", "0", "no spend allowed"),
        ("FORESEA_AGENT_MAX_DRAWDOWN_LIMIT", "0", "drawdown limit of zero"),
        ("FORESEA_AGENT_MAX_DRAWDOWN_LIMIT", "1", "drawdown limit of the whole account"),
        ("FORESEA_AGENT_DAILY_RISK_LIMIT_PCT", "0", "daily risk of zero"),
        ("FORESEA_AGENT_DAILY_RISK_LIMIT_PCT", "1.5", "daily risk above the account"),
    )

    def test_each_out_of_range_setting_raises(self):
        for name, value, description in self.CASES:
            with self.subTest(setting=name, value=value, is_=description):
                with self.assertRaises(ValueError) as ctx:
                    _policy_with(**{name: value})
                self.assertIn(name, str(ctx.exception))


class AcceptedSettingsTests(unittest.TestCase):
    def test_the_defaults_are_inside_their_own_bounds(self):
        """A default that its own validator rejects would be unbootable."""
        policy = _policy_with()
        self.assertEqual(policy.account_value, 10000.0)
        self.assertEqual(policy.concentration_limit, 0.15)
        self.assertEqual(policy.max_drawdown_limit, 0.50)
        # The two stored as amounts, against the 10,000 account above.
        self.assertEqual(policy.per_cycle_spend_limit, 2000.0)
        self.assertEqual(policy.daily_risk_limit, 3000.0)

    def test_the_inclusive_ends_are_accepted(self):
        """concentration and daily risk are `<= 1`; drawdown is `< 1`."""
        self.assertEqual(
            _policy_with(FORESEA_AGENT_CONCENTRATION_LIMIT="1").concentration_limit, 1.0,
        )
        # daily_risk_limit and per_cycle_spend_limit are stored as amounts,
        # account_value * pct, not as the percentage that was validated.
        self.assertEqual(
            _policy_with(FORESEA_AGENT_DAILY_RISK_LIMIT_PCT="1").daily_risk_limit,
            10000.0,
        )
        self.assertAlmostEqual(
            _policy_with(FORESEA_AGENT_MAX_DRAWDOWN_LIMIT="0.999").max_drawdown_limit,
            0.999,
        )


class TheDangerousDirectionTests(unittest.TestCase):
    """Rejecting a limit that silently switches a guard off.

    The other bounds fail loudly -- the agent stops trading. These two
    leave it trading with no guard, which is why they are called out.
    """

    def test_a_drawdown_limit_of_one_is_not_a_limit(self):
        """100% drawdown means the account is gone; nothing can exceed it."""
        with self.assertRaises(ValueError):
            _policy_with(FORESEA_AGENT_MAX_DRAWDOWN_LIMIT="1.0")

    def test_a_daily_risk_limit_above_the_account_is_not_a_limit(self):
        with self.assertRaises(ValueError):
            _policy_with(FORESEA_AGENT_DAILY_RISK_LIMIT_PCT="2")


if __name__ == "__main__":
    unittest.main()
