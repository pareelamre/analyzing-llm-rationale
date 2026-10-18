"""Baseline-only calibration fitting and immutable saved-table contracts."""
import csv
import gzip
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/calibration_transfer_analysis.py"
spec = importlib.util.spec_from_file_location("calibration_transfer_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analysis
spec.loader.exec_module(analysis)


def rows():
    return [{"id": i, "fold": i % 2, "target": i // 2, "raw": .1 if i < 2 else .9,
             "isotonic": float(i // 2)} for i in range(4)]


class TransferTests(unittest.TestCase):
    def test_transfer_excludes_heldout_outcomes(self):
        baseline = rows()
        variant = [{**r, "raw": .3 if r["target"] == 0 else .7} for r in baseline]
        predicted = analysis.apply_transfer(baseline, variant, analysis.fit_baseline(baseline)[0])
        changed = [{**r, "target": 1-r["target"] if r["fold"] == 0 else r["target"]} for r in baseline]
        changed_variant = [{**v, "target": b["target"]} for v, b in zip(variant, changed)]
        new = analysis.apply_transfer(changed, changed_variant, analysis.fit_baseline(changed)[0])
        for q in (0, 2):
            self.assertEqual(predicted[q], new[q])

    def test_baseline_reproduction_and_changed_reference_reject(self):
        baseline = rows()
        predictions = analysis.apply_transfer(baseline, baseline, analysis.fit_baseline(baseline)[0])
        self.assertEqual(analysis.check_reproduction(baseline, predictions), 0)
        with self.assertRaises(ValueError):
            analysis.check_reproduction([{**r, "isotonic": .5} for r in baseline], predictions)

    def test_missing_duplicate_ids_changed_labels_and_folds_reject(self):
        baseline = rows()
        models = analysis.fit_baseline(baseline)[0]
        variants = [baseline[:-1], baseline+[baseline[0]],
                    [{**r, "target": 1-r["target"]} for r in baseline],
                    [{**r, "fold": 1-r["fold"]} for r in baseline]]
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                analysis.apply_transfer(baseline, variant, models)

    def test_bad_probabilities_and_one_fold_reject(self):
        for value in (float("nan"), -1, 2, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                analysis.fit_baseline([{**r, "raw": value} for r in rows()])
        with self.assertRaises(ValueError):
            analysis.fit_baseline([{**r, "fold": 0} for r in rows()])

    def test_transfer_scores_are_paired_and_finite(self):
        baseline = rows()
        predictions = analysis.apply_transfer(baseline, baseline, analysis.fit_baseline(baseline)[0])
        scores = analysis.summarize(baseline, predictions)
        self.assertEqual(scores["n"], 4)
        self.assertAlmostEqual(scores["raw_brier"], .01)
        self.assertEqual(scores["transfer_brier"], 0)
        self.assertEqual(scores["separate_brier"], 0)

    def test_reader_requires_pinned_bytes_complete_family_and_original_folds(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            fold_buffer = "id,fold\n"+"".join(f"{q},{q % 5}\n" for q in range(10))
            (root / "question_folds.csv").write_bytes(fold_buffer.encode())
            records = [{"model": m, "reading": reading, "variant": v, "id": q,
                        "target": q % 2, "fold": q % 5, "raw": .5, "isotonic": .5}
                       for m in analysis.MODELS for reading in ("selected", "direct")
                       for v in range(9) for q in range(10)]

            def save(items):
                buffer = io.StringIO()
                writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(items)
                (root / "oof_predictions.csv.gz").write_bytes(gzip.compress(buffer.getvalue().encode()))
                hashes = {f: analysis.sha_bytes((root / f).read_bytes())
                          for f in ("oof_predictions.csv.gz", "question_folds.csv")}
                (root / "manifest.json").write_text(json.dumps({"output_sha256": hashes}))

            save(records)
            self.assertEqual(len(analysis.read_saved(root)[0]), 54)
            with (root / "oof_predictions.csv.gz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                analysis.read_saved(root)
            save(records[:-1])
            with self.assertRaises(ValueError):
                analysis.read_saved(root)
            save([{**r, "fold": 4 if r["id"] == 0 else r["fold"]} for r in records])
            with self.assertRaises(ValueError):
                analysis.read_saved(root)


if __name__ == "__main__":
    unittest.main()
