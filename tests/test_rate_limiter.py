"""Direct tests for the rate limiter, including the Redis fallback.

The limiter was previously exercised only through endpoint tests, which
patch `_calls` and never reach the Redis path at all. That path is where
the interesting behaviour is: when Redis fails the limiter falls back to
an in-process window, which counts per instance. With N instances the
effective limit becomes N times the configured one, and before this it
happened silently.

Falling back is correct -- a limiter that raises when Redis blinks would
take the service down to protect it. Doing it quietly is the problem.
"""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.server_security import RateLimiter  # noqa: E402


class _FlakyRedis:
    """Counts calls and raises while `broken` is set."""

    def __init__(self):
        self.broken = True
        self.calls = 0

    def pipeline(self):
        self.calls += 1
        if self.broken:
            raise ConnectionError("redis is gone")
        return self

    # the pipeline surface the limiter drives
    def incr(self, *a, **k): return self
    def expire(self, *a, **k): return self
    def execute(self): return [1]


def _limiter_with(redis, calls=2, period=60):
    limiter = RateLimiter(calls=calls, period=period)
    limiter._redis = redis
    return limiter


class RedisFallbackTests(unittest.TestCase):
    def test_a_redis_failure_still_serves_the_request(self):
        """Fail open to the local window, never raise at the caller."""
        limiter = _limiter_with(_FlakyRedis())
        self.assertTrue(limiter.is_allowed("ip-1"))

    def test_the_fallback_still_limits(self):
        """Degraded is not unlimited."""
        limiter = _limiter_with(_FlakyRedis(), calls=2)
        self.assertTrue(limiter.is_allowed("ip-1"))
        self.assertTrue(limiter.is_allowed("ip-1"))
        self.assertFalse(limiter.is_allowed("ip-1"))

    def test_the_degradation_is_logged(self):
        limiter = _limiter_with(_FlakyRedis())
        with self.assertLogs("analyzing_llm_rationale.server_security", level="WARNING") as caught:
            limiter.is_allowed("ip-1")
        joined = "\n".join(caught.output)
        self.assertIn("per-instance", joined)
        self.assertIn("ConnectionError", joined)

    def test_it_logs_once_per_outage_not_once_per_request(self):
        """This runs on every request; per-call logging buries the cause."""
        limiter = _limiter_with(_FlakyRedis())
        with self.assertLogs("analyzing_llm_rationale.server_security", level="WARNING") as caught:
            for _ in range(50):
                limiter.is_allowed("ip-1")
        self.assertEqual(len(caught.output), 1)

    def test_recovery_is_logged_and_redis_is_used_again(self):
        redis = _FlakyRedis()
        limiter = _limiter_with(redis)
        with self.assertLogs("analyzing_llm_rationale.server_security", level="WARNING"):
            limiter.is_allowed("ip-1")

        redis.broken = False
        with self.assertLogs("analyzing_llm_rationale.server_security", level="INFO") as caught:
            self.assertTrue(limiter.is_allowed("ip-1"))
        self.assertIn("Redis again", "\n".join(caught.output))

    def test_a_second_outage_after_recovery_warns_again(self):
        """The latch must reset, or a repeat outage goes unreported."""
        redis = _FlakyRedis()
        limiter = _limiter_with(redis)
        with self.assertLogs("analyzing_llm_rationale.server_security", level="WARNING"):
            limiter.is_allowed("k")
        redis.broken = False
        limiter.is_allowed("k")
        redis.broken = True
        with self.assertLogs("analyzing_llm_rationale.server_security", level="WARNING") as caught:
            limiter.is_allowed("k")
        self.assertEqual(len(caught.output), 1)


class HealthyRedisTests(unittest.TestCase):
    def test_a_working_redis_is_used_and_says_nothing(self):
        redis = _FlakyRedis()
        redis.broken = False
        limiter = _limiter_with(redis)
        logger = logging.getLogger("analyzing_llm_rationale.server_security")
        with self.assertNoLogs(logger, level="INFO"):
            self.assertTrue(limiter.is_allowed("ip-1"))
        self.assertEqual(redis.calls, 1)

    def test_without_redis_the_local_window_is_used(self):
        limiter = RateLimiter(calls=1, period=60)
        limiter._redis = None
        self.assertTrue(limiter.is_allowed("ip-1"))
        self.assertFalse(limiter.is_allowed("ip-1"))

    def test_keys_are_limited_independently(self):
        limiter = RateLimiter(calls=1, period=60)
        limiter._redis = None
        self.assertTrue(limiter.is_allowed("ip-1"))
        self.assertTrue(limiter.is_allowed("ip-2"))
        self.assertFalse(limiter.is_allowed("ip-1"))


if __name__ == "__main__":
    unittest.main()
