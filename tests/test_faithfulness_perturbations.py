from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_mod = load_script("build_faithfulness_perturbation_dataset", "scripts/build_faithfulness_perturbation_dataset.py")
analyze_mod = load_script("analyze_faithfulness_perturbations", "scripts/analyze_faithfulness_perturbations.py")


class FaithfulnessPerturbationTests(unittest.TestCase):
    def sample_record(self):
        return {
            "id": 12,
            "question": "Will France pass the law by 2025?",
            "answer": "yes",
            "description": "France is debating a law.",
            "resolution_criteria": "Resolve yes if France passes the law by 2025.",
            "gnews_query": "Will France pass the law by 2025?",
            "news_articles": [
                {
                    "title": "France law update",
                    "summary_llm": "France passed the law in parliament. Analysts expect implementation soon.",
                    "summary": "France passed the law in parliament.",
                    "text": "France passed the law in parliament. Analysts expect implementation soon.",
                }
            ],
        }

    def test_mask_evidence_removes_sentence_overlapping_rationale(self):
        baseline = {
            "id": 12,
            "predicted_answer": "Yes",
            "confidence": 0.8,
            "rationale": "France passed the law in parliament, so the event looks likely.",
        }

        masked = build_mod.mask_evidence(self.sample_record(), baseline)

        self.assertIsNotNone(masked)
        self.assertNotIn("France passed the law in parliament.", masked["news_articles"][0]["summary_llm"])
        self.assertEqual(masked["perturbation_detail"]["masked_field"], "summary_llm")

    def test_build_rows_assigns_unique_ids_and_metadata(self):
        baseline = {
            12: {
                "id": 12,
                "predicted_answer": "Yes",
                "confidence": 0.8,
                "rationale": "France passed the law in parliament, so the event looks likely.",
            }
        }

        rows = build_mod.build_rows(
            [self.sample_record()],
            baseline,
            ["evidence_masking", "contradiction", "actor_date_swap", "criterion_swap"],
            max_records=0,
            seed=1,
        )

        self.assertEqual({row["original_id"] for row in rows}, {12})
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        self.assertIn("contradiction", {row["perturbation_type"] for row in rows})

    def test_p_yes_uses_answer_direction(self):
        self.assertEqual(analyze_mod.p_yes({"predicted_answer": "Yes", "confidence": 0.7}), 0.7)
        self.assertAlmostEqual(analyze_mod.p_yes({"predicted_answer": "No", "confidence": 0.7}), 0.3)

    def test_paired_rows_joins_by_original_id_metadata(self):
        perturbation_input = [{"id": 1201, "original_id": 12, "perturbation_type": "evidence_masking"}]
        baseline = [{"id": 12, "predicted_answer": "Yes", "confidence": 0.8}]
        perturbation_results = [{"id": 1201, "predicted_answer": "No", "confidence": 0.6}]

        pairs = analyze_mod.paired_rows(perturbation_input, baseline, perturbation_results)

        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["baseline_p_yes"], 0.8)
        self.assertAlmostEqual(pairs[0]["perturbed_p_yes"], 0.4)
        self.assertEqual(pairs[0]["answer_flipped"], 1)


if __name__ == "__main__":
    unittest.main()
