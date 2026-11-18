"""The long view on commodity pairs: position, analogues, episodes and their stories (made-up prices)."""
from __future__ import annotations

import json
import re
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import commodity_pairs as cp  # noqa: E402
from analyzing_llm_rationale import fund_orchestrator as orch  # noqa: E402
from analyzing_llm_rationale import pair_history as ph  # noqa: E402

ZINC_LEAD = next(p for p in cp.PAIRS if p["key"] == "zinc-lead")


def _months(n, start=(2000, 1)):
    y, m = start
    out = []
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _monthly(n=200, spike_at=None, seed=3, size=0.5, decay=0.7):
    """Lead flat with noise; zinc tied to it, with an optional spike in the ratio that fades."""
    rng = np.random.default_rng(seed)
    gap = rng.normal(0, 0.03, n)
    if spike_at is not None:
        for k in range(spike_at, n):
            gap[k] += size * decay ** (k - spike_at)
    lead = 2000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    zinc = lead * 1.3 * np.exp(gap)
    months = _months(n)
    return {"zinc": list(zip(months, zinc)), "lead": list(zip(months, lead))}, months


class FetchTests(unittest.TestCase):
    def test_monthly_prices_drop_the_current_month_and_average_daily_oil(self):
        class Resp:
            status_code = 200

            def __init__(self, text):
                self.text = text

        def http_get(url, params=None, **_):
            sid = params["id"]
            if sid in ph.IMF_MONTHLY.values():
                return Resp("observation_date,v\n2026-09-01,10\n2026-10-01,11\n2026-11-01,12\n")
            return Resp("observation_date,v\n2026-10-01,80\n2026-10-02,.\n2026-10-05,90\n2026-11-02,95\n")

        monthly = ph.fetch_monthly(date(2026, 11, 17), http_get=http_get)
        self.assertEqual(monthly["copper"], [("2026-09", 10.0), ("2026-10", 11.0)])
        self.assertEqual(monthly["crudeoil"], [("2026-10", 85.0)])


class PositionTests(unittest.TestCase):
    def test_today_carries_the_last_month_forward_by_the_daily_move(self):
        months = ["2026-08", "2026-09"]
        r = np.log([1.2, 1.5])
        days = [date(2026, 9, 1), date(2026, 9, 2), date(2026, 10, 1)]
        daily = np.log([1.0, 1.0, 1.1])  # another market's level: only its move since September counts
        self.assertAlmostEqual(ph.now_log_ratio(months, r, days, daily), np.log(1.5 * 1.1), places=9)
        self.assertIsNone(ph.now_log_ratio(months, r, [date(2026, 11, 2)], np.log([1.0])))

    def test_a_spike_is_one_episode_with_what_followed(self):
        monthly, months = _monthly(spike_at=150)
        m, r = ph.monthly_ratio(ZINC_LEAD, monthly)
        episodes = ph.detect_episodes(m, r)
        spike = [e for e in episodes if e["from"] <= months[150] <= e["to"]]
        self.assertEqual(len(spike), 1)
        e = spike[0]
        self.assertEqual((e["side"], e["peak"]), ("high", months[150]))
        self.assertLess(e["after_12m"], -0.2)
        self.assertLessEqual(e["months_to_normal"], 12)
        self.assertTrue(e["settled"])

    def test_analogues_count_only_months_this_far_out_on_the_same_side(self):
        monthly, _ = _monthly(spike_at=150)
        m, r = ph.monthly_ratio(ZINC_LEAD, monthly)
        self.assertIsNone(ph.analogues(m, r, 1.0))
        high = ph.analogues(m, r, 2.5)
        self.assertGreater(high["months"], 0)
        self.assertGreater(high["back_12m"], 0.5)


class StoryTests(unittest.TestCase):
    STORIES = [{"side": "high", "from": "2012-01", "to": "2013-12", "driver": "temporary",
                "cause": "A made-up mine flood.", "ended": "Back within a year.", "sources": ["Acme"]},
               {"side": "high", "from": "2016-01", "to": None, "driver": "lasting",
                "cause": "A made-up new process.", "ended": None, "sources": ["Acme"]}]

    def test_episodes_take_their_story_and_ended_ones_are_tallied_by_cause(self):
        monthly, months = _monthly(spike_at=150)  # 2012-07
        m, r = ph.monthly_ratio(ZINC_LEAD, monthly)
        joined = ph.join_stories(ph.detect_episodes(m, r), self.STORIES)
        spike = next(e for e in joined if e["peak"] == months[150])
        self.assertEqual(spike["story"]["driver"], "temporary")
        record = ph.driver_record({"zinc-lead": joined})
        self.assertEqual(record["temporary"], {"episodes": 1, "back": 1})
        self.assertNotIn("lasting", record)

    def test_a_pair_far_out_today_carries_the_running_story(self):
        monthly, months = _monthly(n=200, spike_at=197, size=0.6, decay=1.0)  # still high at the end
        days = [date(2016, 8, 1), date(2016, 8, 2)]
        last = monthly["zinc"][-1][1] / monthly["lead"][-1][1]
        ctx = ph.pair_context(ZINC_LEAD, monthly, days, np.log([last, last]), self.STORIES)
        self.assertTrue(ctx["extreme"])
        self.assertGreater(ctx["percentile"], 0.95)
        self.assertEqual(ctx["story"]["driver"], "lasting")
        self.assertIsNotNone(ctx["analogues"])

    def test_no_context_without_ten_years_of_months(self):
        monthly, _ = _monthly(n=100)
        self.assertIsNone(ph.pair_context(ZINC_LEAD, monthly, [date(2008, 4, 1)], np.log([1.3]), []))


class EpisodeFileTests(unittest.TestCase):
    def test_every_story_is_dated_sourced_and_names_its_cause(self):
        stories = ph.load_stories()
        self.assertTrue(stories)
        month = re.compile(r"^\d{4}-\d{2}$")
        for key, rows in stories.items():
            self.assertIn(key, {p["key"] for p in cp.PAIRS})
            for s in rows:
                self.assertIn(s["side"], ("high", "low"))
                self.assertIn(s["driver"], ("temporary", "lasting", "mixed"))
                self.assertRegex(s["from"], month)
                self.assertTrue(s["to"] is None or (month.match(s["to"]) and s["from"] <= s["to"]))
                self.assertTrue(s["cause"] and s["sources"])
                self.assertEqual(s["to"] is None, s["ended"] is None)


class TrackerAndFundTests(unittest.TestCase):
    LONG = {"pairs": {"gold-silver": {"z": 2.1, "percentile": 0.97, "window_months": 120, "source": "made up",
                                      "story": {"driver": "temporary", "cause": "A made-up squeeze. More detail."},
                                      "analogues": {"back_12m": 0.7}, "episodes": []}},
            "driver_record": {"temporary": {"episodes": 9, "back": 8}}}

    def test_the_tracker_attaches_the_long_view_and_which_way_a_bet_points(self):
        self.assertEqual(cp.tide(2.5, {"z": 1.8}), "toward")
        self.assertEqual(cp.tide(-2.5, {"z": 1.8}), "away")
        # Every trade is labelled (TIDE_Z 0, picked on 2007-2016 alone): copper/zinc on 2026-11-18,
        # cheap at z -3.0 over 60 sessions but 0.83 above its 10-year average, bets away.
        self.assertEqual(cp.tide(-3.0, {"z": 0.83}), "away")
        self.assertEqual(cp.tide(2.5, {"z": 0.4}), "toward")
        self.assertEqual(cp.tide(-2.5, {"z": 0.0}), "neutral")
        self.assertIsNone(cp.tide(None, {"z": 1.8}))

        # Silver a random walk; gold tied to it, with a late jump that makes gold rich.
        rng = np.random.default_rng(1)
        days = [date(2025, 6, 2) + timedelta(days=i) for i in range(400)]
        days = [d for d in days if d.weekday() < 5][:200]
        ls = np.log(30) + np.cumsum(rng.normal(0, 0.015, 200))
        gap = np.zeros(200)
        for t in range(1, 200):
            gap[t] = 0.8 * gap[t - 1] + rng.normal(0, 0.01)
        gap[199:] += 0.08
        lg = np.log(2400) + 0.9 * (ls - ls[0]) + gap
        series = {("gold", "india"): list(zip(days, np.exp(lg))), ("silver", "india"): list(zip(days, np.exp(ls)))}
        now = datetime(2026, 3, 9, tzinfo=timezone.utc)
        payload, _ = cp.run(series, [], now, long_run=self.LONG)
        row = next(p for p in payload["pairs"] if p["key"] == "gold-silver")
        self.assertGreater(row["z"], cp.ENTRY_Z)
        self.assertEqual(row["long_run"]["tide"], "toward")  # gold rich now and over ten years
        self.assertEqual(payload["long_run"]["driver_record"]["temporary"]["back"], 8)
        label = payload["long_run"]["tide"]
        self.assertEqual(label["away_size"], cp.TIDE_AWAY_SIZE)
        self.assertGreater(label["backtest"]["toward"]["avg_return"], label["backtest"]["away"]["avg_return"])
        self.assertEqual(set(label["live"]), {"toward", "away"})
        bare, _ = cp.run(series, [], now)
        self.assertIsNone(bare["long_run"])

    def test_a_new_live_signal_keeps_its_label_and_the_label_is_scored_when_signals_close(self):
        row = {"key": "gold-silver", "_hist": [{"date": date(2026, 3, 9), "i": 0, "z": 2.4, "z_ratio": 2.0,
                                                "beta": 1.0, "spread": 0.3}],
               "_la": np.array([0.0]), "_lb": np.array([0.0]), "_days": [date(2026, 3, 9)]}
        ledger: list[dict] = []
        cp.update_ledger(ledger, [row], datetime(2026, 3, 9, tzinfo=timezone.utc), tides={"gold-silver": "away"})
        self.assertEqual(ledger[0]["tide"], "away")
        unlabelled: list[dict] = []
        cp.update_ledger(unlabelled, [row], datetime(2026, 3, 9, tzinfo=timezone.utc))
        self.assertNotIn("tide", unlabelled[0])
        closed = [{**ledger[0], "status": "reverted", "spread_return": 0.01},
                  {**ledger[0], "tide": "toward", "status": "timed_out", "spread_return": -0.02}]
        self.assertEqual(cp.summarize([e for e in closed if e.get("tide") == "away"])["win_rate"], 1.0)

    def test_a_fund_pair_trade_carries_the_long_view_as_a_fact(self):
        fact = orch._long_run_fact({**self.LONG["pairs"]["gold-silver"], "tide": "away"},
                                   self.LONG["driver_record"])
        self.assertEqual(fact["label"], "long-run context")
        self.assertIn("97th percentile of the last 10 years", fact["value"])
        self.assertIn("away from that 10-year average", fact["value"])
        self.assertIn("cause (temporary): A made-up squeeze", fact["value"])
        self.assertIn("temporary causes 8 of 9", fact["value"])
        self.assertIn("70% of the time", fact["value"])
        self.assertIsNone(orch._long_run_fact(None, {}))
        self.assertEqual(json.loads(json.dumps(fact)), fact)


if __name__ == "__main__":
    unittest.main()
