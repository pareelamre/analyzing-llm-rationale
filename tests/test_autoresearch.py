from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.autoresearch import (  # noqa: E402
    AutoresearchConfig,
    Score,
    is_improvement,
    run_autoresearch_once,
    score_results,
)


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, temperature, max_tokens):
        self.calls.append(messages)
        return self.responses.pop(0)


class AutoresearchTests(unittest.TestCase):
    def test_score_results_computes_forecasting_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "dataset.json"
            results = root / "results.json"
            dataset.write_text(
                json.dumps(
                    [
                        {"id": 1, "answer": "Yes"},
                        {"id": 2, "answer": "No"},
                    ]
                ),
                encoding="utf-8",
            )
            results.write_text(
                json.dumps(
                    [
                        {"id": 1, "predicted_answer": "Yes", "confidence": 0.8},
                        {"id": 2, "predicted_answer": "No", "confidence": 0.7},
                    ]
                ),
                encoding="utf-8",
            )

            score = score_results(results, dataset)

            self.assertEqual(score.n_scored, 2)
            self.assertEqual(score.n_missing, 0)
            self.assertAlmostEqual(score.accuracy or 0.0, 1.0)
            self.assertAlmostEqual(score.brier_score or 0.0, 0.065)

    def test_is_improvement_respects_metric_direction(self):
        baseline = Score(n_scored=10, n_missing=0, accuracy=0.7, brier_score=0.2, ece=0.1)
        better_brier = Score(n_scored=10, n_missing=0, accuracy=0.7, brier_score=0.18, ece=0.1)
        worse_brier = Score(n_scored=10, n_missing=0, accuracy=0.7, brier_score=0.22, ece=0.1)
        better_accuracy = Score(n_scored=10, n_missing=0, accuracy=0.8, brier_score=0.2, ece=0.1)

        self.assertTrue(is_improvement(better_brier, baseline, metric="brier_score"))
        self.assertFalse(is_improvement(worse_brier, baseline, metric="brier_score"))
        self.assertTrue(is_improvement(better_accuracy, baseline, metric="accuracy"))

    def test_run_autoresearch_once_writes_log_and_score(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset = root / "dataset.json"
            system = root / "system.txt"
            candidate = root / "candidate_prompt.txt"
            output_dir = root / "analysis" / "autoresearch"
            baseline = root / "baseline.json"

            dataset.write_text(
                json.dumps(
                    [
                        {
                            "id": 1,
                            "question": "Will X happen?",
                            "answer": "Yes",
                            "news_articles": [],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            system.write_text("Return JSON.", encoding="utf-8")
            candidate.write_text("[question]\nReturn strict JSON.", encoding="utf-8")
            baseline.write_text(
                json.dumps([{"id": 1, "predicted_answer": "Yes", "confidence": 0.6}]),
                encoding="utf-8",
            )
            provider = FakeProvider(
                [json.dumps({"predicted_answer": "Yes", "confidence": 0.8, "rationale": "Likely."})]
            )

            report = run_autoresearch_once(
                AutoresearchConfig(
                    input_path=dataset,
                    system_prompt_path=system,
                    candidate_prompt_path=candidate,
                    output_dir=output_dir,
                    baseline_results_path=baseline,
                    max_records=1,
                    metric="brier_score",
                ),
                provider,
            )

            self.assertEqual(len(provider.calls), 1)
            self.assertTrue(report["improved"])
            self.assertTrue(Path(str(report["results_path"])).exists())
            self.assertTrue((Path(str(report["results_path"])).parent / "score.json").exists())
            self.assertTrue((output_dir / "experiments.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
