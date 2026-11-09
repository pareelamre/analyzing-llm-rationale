from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fund_finbert_eval", _ROOT / "scripts" / "fund_finbert_eval.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules["fund_finbert_eval"] = ev
_spec.loader.exec_module(ev)


def _rows(n_days: int, predictive: bool, seed: int = 1) -> list[dict]:
    rng = np.random.default_rng(seed)
    start = date(2026, 1, 1)
    out = []
    for i in range(n_days):
        ret = float(rng.normal(0, 0.01))
        signal = float(np.sign(ret) * 0.4 + rng.normal(0, 0.05)) if predictive else float(rng.normal(0, 0.4))
        out.append({"instrument": "Nifty 50", "day": start + timedelta(days=i), "signal": signal, "headlines": 9,
                    "ret": ret, "cost": 0.0005})
    return out


class DailyRowsTests(unittest.TestCase):
    def test_only_headlines_seen_before_the_close_count_and_five_are_needed(self):
        d0, d1, d2 = date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4)
        prices = {"Nifty 50": [(d0, 100.0), (d1, 101.0), (d2, 99.0)]}
        close_d1 = datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc)
        items = [(close_d1 - timedelta(hours=h), f"good {h}") for h in (1, 2, 3, 4, 5)]
        # After the close: must not reach day d1's signal however strong it is.
        items.append((close_d1 + timedelta(minutes=1), "terrible after close"))
        scores = {"india_equity": {t: 1.0 for _, t in items}}
        scores["india_equity"]["terrible after close"] = -100.0
        rows = ev.daily_rows(prices, {"india_equity": sorted(items)}, scores)
        by_day = {r["day"]: r for r in rows if r["instrument"] == "Nifty 50"}
        self.assertEqual(by_day[d1]["headlines"], 5)
        self.assertAlmostEqual(by_day[d1]["signal"], 1.0)
        self.assertAlmostEqual(by_day[d1]["ret"], 99.0 / 101.0 - 1.0)
        # Day d0 had no headlines in its window: no row.
        self.assertNotIn(d0, by_day)


class EvaluateTests(unittest.TestCase):
    def test_a_truly_predictive_signal_passes_every_check(self):
        result = ev.evaluate(_rows(400, predictive=True))
        self.assertEqual(result["decision"], "GO", result["checks"])

    def test_noise_fails(self):
        result = ev.evaluate(_rows(400, predictive=False))
        self.assertEqual(result["decision"], "NO-GO")
        self.assertFalse(result["checks"]["correlation_above_zero"])

    def test_too_small_a_sample_fails_even_when_the_signal_is_perfect(self):
        result = ev.evaluate(_rows(200, predictive=True))
        self.assertFalse(result["checks"]["enough_test_days"])
        self.assertEqual(result["decision"], "NO-GO")

    def test_costs_can_turn_a_real_correlation_into_no_go(self):
        rows = _rows(400, predictive=True)
        for r in rows:
            r["cost"] = 0.05  # 5% a round trip swamps a 1% daily move
        self.assertFalse(ev.evaluate(rows)["checks"]["rule_net_above_zero"])


class FredTests(unittest.TestCase):
    def test_a_timeout_is_retried_before_giving_up(self):
        import requests

        calls = []

        class Resp:
            text = "observation_date,SP500\n2026-11-02,6800.5\n2026-11-03,.\n2026-11-04,6790.1\n"

            def raise_for_status(self):
                return None

        def http_get(url, **kwargs):
            calls.append(url)
            if len(calls) < 3:
                raise requests.ReadTimeout("slow")
            return Resp()

        rows = ev.fred_closes("SP500", date(2026, 11, 1), http_get=http_get, sleep=lambda s: None)
        self.assertEqual(len(calls), 3)
        self.assertEqual(rows, [(date(2026, 11, 2), 6800.5), (date(2026, 11, 4), 6790.1)])

        def always_down(url, **kwargs):
            raise requests.ReadTimeout("down")

        with self.assertRaises(requests.ReadTimeout):
            ev.fred_closes("SP500", date(2026, 11, 1), http_get=always_down, sleep=lambda s: None)


class RunTests(unittest.TestCase):
    def test_run_writes_a_report_per_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            news = Path(tmp) / "news"
            news.mkdir()
            seen = datetime(2026, 11, 3, 9, 0, tzinfo=timezone.utc)
            (news / "india_equity.jsonl").write_text("\n".join(
                json.dumps({"seen": (seen - timedelta(minutes=i)).isoformat(), "title": f"t{i}"}) for i in range(6)))
            prices = {"Nifty 50": [(date(2026, 11, 3), 100.0), (date(2026, 11, 4), 101.0)]}
            report = ev.run(news, Path(tmp) / "out" / "eval", {"fake": lambda titles: [0.5] * len(titles)},
                            prices=prices)
            self.assertEqual(report["headlines"], {"india_equity": 6})
            self.assertEqual(report["models"]["fake"]["decision"], "NO-GO")
            self.assertTrue((Path(tmp) / "out" / "eval.json").is_file())


if __name__ == "__main__":
    unittest.main()
