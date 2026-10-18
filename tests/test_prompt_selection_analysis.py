"""Behavior contracts for saved-forecast prompt selection; no model calls."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prompt_selection_analysis.py"
spec = importlib.util.spec_from_file_location("prompt_selection_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analysis
spec.loader.exec_module(analysis)


class SelectionTests(unittest.TestCase):
    def test_folds_are_order_independent_and_question_disjoint(self):
        forward = analysis.assign_folds(list(range(12)), 3, 42)
        self.assertEqual(forward, analysis.assign_folds(list(reversed(range(12))), 3, 42))
        self.assertEqual(sorted(forward.values()), [0]*4 + [1]*4 + [2]*4)

    def test_test_labels_never_choose_the_test_fold_prompt(self):
        targets = {0: 0, 1: 0, 2: 1, 3: 1}
        probabilities = {"v0": {q: 0.1 for q in targets},
                         "v1": {q: 0.9 for q in targets}}
        folds = {0: 0, 1: 0, 2: 1, 3: 1}
        report = analysis.evaluate(probabilities, targets, folds, "v0")
        changed = analysis.evaluate(probabilities, {**targets, 0: 1, 1: 1}, folds, "v0")
        for before, after in zip(report["predictions"][:2], changed["predictions"][:2]):
            self.assertEqual(before["selected_variant"], "v1")
            self.assertEqual(before["selected_variant"], after["selected_variant"])
            self.assertEqual(before["selected_probability"], after["selected_probability"])

    def test_all_strategies_use_the_intersection(self):
        report = analysis.evaluate({"v0": {0: 0.1, 1: 0.8, 2: 0.4},
                                    "v1": {0: 0.3, 1: 0.7}},
                                   {0: 0, 1: 1, 2: 0}, {0: 0, 1: 1, 2: 0}, "v0")
        self.assertEqual(report["common_question_ids"], [0, 1])
        self.assertEqual(report["coverage_by_variant"], {"v0": 3, "v1": 2})
        self.assertAlmostEqual(report["ensemble_brier"], 0.05125)
        self.assertAlmostEqual(report["mean_member_brier"] - report["ensemble_brier"],
                               report["mean_probability_disagreement"])

    def test_deterministic_ties_and_no_empty_training_folds(self):
        probabilities = {"v1": {0: .5, 1: .5}, "v0": {0: .5, 1: .5}}
        report = analysis.evaluate(probabilities, {0: 0, 1: 1}, {0: 0, 1: 1}, "v0")
        self.assertTrue(all(p["selected_variant"] == "v0" for p in report["predictions"]))
        with self.assertRaises(ValueError):
            analysis.evaluate(probabilities, {0: 0, 1: 1}, {0: 0, 1: 0}, "v0")

    def test_confidence_readings_and_invalid_values(self):
        self.assertAlmostEqual(analysis.probability_yes("no", .35, "selected"), .65)
        self.assertAlmostEqual(analysis.probability_yes("no", .35, "direct"), .35)
        for value in (True, None, float("nan"), 1.01, -0.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                analysis.probability_yes("yes", value, "selected")
        with self.assertRaises(ValueError):
            analysis.probability_yes("yes", .4, "guess")

    def test_duplicate_and_unknown_ids_reject_instead_of_overwriting(self):
        for rows in ([{"id": 1}, {"id": 1}], [{"id": True}], [{"id": 2}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                analysis.score_rows(rows, {1: 0}, "selected")

    def test_bad_predictions_are_excluded_and_missing_ids_counted(self):
        result = analysis.score_rows([{"id": 1, "predicted_answer": "yes", "confidence": .7},
                                      {"id": 2, "predicted_answer": "maybe", "confidence": .2}],
                                     {1: 1, 2: 0, 3: 1}, "selected")
        self.assertEqual(result, {1: .7})

    def test_changed_input_hash_and_existing_output_reject(self):
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            source = base / "input.json"
            source.write_text("[]")
            digest = analysis.sha(source)
            self.assertEqual(analysis.read_verified(base, "input.json", {"input.json": digest}), [])
            source.write_text("{}")
            with self.assertRaises(ValueError):
                analysis.read_verified(base, "input.json", {"input.json": digest})
            with self.assertRaises(ValueError):
                analysis.read_verified(base, "../input.json", {"../input.json": digest})
            out = base / "report.json"
            analysis.write_report(out, {"value": 1})
            with self.assertRaises(FileExistsError):
                analysis.write_report(out, {"value": 2})
            self.assertEqual(json.loads(out.read_text()), {"value": 1})

    def test_complete_report_hashes_all_cells_and_reports_both_readings(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            hashes = {}
            dataset = [{"id": 1, "answer": "yes"}, {"id": 2, "answer": "no"}]
            benchmark = root / analysis.BENCHMARK
            benchmark.write_text(json.dumps(dataset))
            hashes[analysis.BENCHMARK] = analysis.sha(benchmark)
            for model, temperature in analysis.MODEL_TEMPERATURE.items():
                for variant in range(9):
                    relative = f"results/{model}/temperature_{temperature}/results_variant{variant}_test.json"
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps([{"id": q["id"], "predicted_answer": "no", "confidence": .2}
                                                for q in dataset]))
                    hashes[relative] = analysis.sha(path)
            reference = root / "reference.json"
            reference.write_text(json.dumps({"input_sha256": hashes}))
            report = analysis.build_report(root, reference, 2)
            self.assertEqual(report["input_sha256"], hashes)
            self.assertEqual(len(report["conditions"]), 6)
            self.assertEqual({c["reading"] for c in report["conditions"]}, {"selected", "direct"})
            self.assertTrue(all(c["common_question_ids"] == [1, 2] for c in report["conditions"]))
            benchmark.write_text("[]")
            with self.assertRaises(ValueError):
                analysis.build_report(root, reference, 2)

    def test_scoring_uses_verified_bytes_when_source_changes_after_read(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as name:
            for root in (Path(name), Path(name).relative_to(Path.cwd())):
                with self.subTest(root=str(root)):
                    source = (root / "input.json").resolve()
                    source.write_text('[{"value": 1}]')
                    digest = analysis.sha(source)
                    read_bytes = Path.read_bytes

                    def changing_read(path, read_bytes=read_bytes, source=source):
                        payload = read_bytes(path)
                        if path == source:
                            path.write_text('[{"value": 2}]')
                        return payload

                    with patch.object(Path, "read_bytes", changing_read):
                        result = analysis.read_verified(root, "input.json", {"input.json": digest})
                    self.assertEqual(result, [{"value": 1}])
                    self.assertEqual(json.loads(source.read_text()), [{"value": 2}])


if __name__ == "__main__":
    unittest.main()
