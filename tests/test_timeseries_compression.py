import unittest
from datetime import datetime, timedelta, timezone

from analyzing_llm_rationale.timeseries_compression import downsample_lttb, downsample_time_decay


class TimeSeriesCompressionTests(unittest.TestCase):
    def test_downsample_lttb_reduces_points_and_preserves_bounds(self):
        # Generate noisy wave
        data = [(float(i), math_sin(i * 0.1)) for i in range(100)]
        sampled = downsample_lttb(data, 20)
        self.assertEqual(len(sampled), 20)
        self.assertEqual(sampled[0], data[0])
        self.assertEqual(sampled[-1], data[-1])

    def test_downsample_lttb_handles_small_data(self):
        data = [(1.0, 2.0), (2.0, 3.0)]
        sampled = downsample_lttb(data, 10)
        self.assertEqual(sampled, data)

    def test_downsample_time_decay_preserves_recent_and_thins_old(self):
        now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
        now_ts = now.timestamp()

        records = []
        # 100 points in the last 2 hours (should all be kept)
        for i in range(100):
            t = (now - timedelta(minutes=i)).isoformat()
            records.append({"timestamp": t, "value": i})

        # 1000 points 10 days ago (1 per minute -> should be thinned to 1 per 15 min)
        base_10d = now - timedelta(days=10)
        for i in range(1000):
            t = (base_10d + timedelta(minutes=i)).isoformat()
            records.append({"timestamp": t, "value": 1000 + i})

        # 500 points 60 days ago (1 per hour -> should be thinned to 1 per day)
        base_60d = now - timedelta(days=60)
        for i in range(500):
            t = (base_60d + timedelta(hours=i)).isoformat()
            records.append({"timestamp": t, "value": 5000 + i})

        thinned = downsample_time_decay(records, now_ts=now_ts)
        self.assertLess(len(thinned), len(records))
        self.assertGreater(len(thinned), 100)


def math_sin(x):
    import math
    return math.sin(x)


if __name__ == "__main__":
    unittest.main()
