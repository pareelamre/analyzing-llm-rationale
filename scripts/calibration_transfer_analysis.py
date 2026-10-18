"""Transfer training-fold V0 isotonic calibration to saved prompt forecasts."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import sys
from pathlib import Path
from statistics import fmean

MODELS = ("GPT-OSS-120B", "Qwen2.5-7b-instruct", "Qwen3-32B")


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def validate_rows(rows):
    ids = [r["id"] for r in rows]
    if not ids or any(type(q) is not int for q in ids) or len(ids) != len(set(ids)):
        raise ValueError("Nonempty unique integer question IDs required")
    for r in rows:
        if type(r["target"]) is not int or r["target"] not in (0, 1) or type(r["fold"]) is not int:
            raise ValueError("Binary targets and fixed integer folds required")
        for key in ("raw", "isotonic"):
            p = r[key]
            if isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("Finite probabilities in [0,1] required")
    return {r["id"]: r for r in rows}


def fit_baseline(rows):
    from sklearn.isotonic import IsotonicRegression

    validate_rows(rows)
    folds = sorted({r["fold"] for r in rows})
    if len(folds) < 2:
        raise ValueError("At least two nonempty question folds required")
    models, trace = {}, []
    for fold in folds:
        train = sorted((r for r in rows if r["fold"] != fold), key=lambda r: r["id"])
        test = sorted((r for r in rows if r["fold"] == fold), key=lambda r: r["id"])
        fitted = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0, y_max=1)
        fitted.fit([r["raw"] for r in train], [r["target"] for r in train])
        models[fold] = fitted
        trace.append({"fold": fold, "train_ids": [r["id"] for r in train],
                      "test_ids": [r["id"] for r in test],
                      "x_thresholds": fitted.X_thresholds_.tolist(),
                      "y_thresholds": fitted.y_thresholds_.tolist()})
    return models, trace


def apply_transfer(baseline, variant, models):
    old, new = validate_rows(baseline), validate_rows(variant)
    if old.keys() != new.keys() or any((old[q]["target"], old[q]["fold"]) !=
                                     (new[q]["target"], new[q]["fold"]) for q in old):
        raise ValueError("Baseline and prompt must have identical IDs, targets and folds")
    if set(models) != {r["fold"] for r in variant}:
        raise ValueError("Calibrator fold mismatch")
    predictions = {}
    for fold, fitted in models.items():
        test = sorted((r for r in variant if r["fold"] == fold), key=lambda r: r["id"])
        values = fitted.predict([r["raw"] for r in test])
        predictions.update({r["id"]: float(p) for r, p in zip(test, values)})
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in predictions.values()):
        raise ArithmeticError("Invalid transferred probabilities")
    return predictions


def check_reproduction(baseline, predictions):
    if set(predictions) != {r["id"] for r in baseline}:
        raise ValueError("Reproduction cohort mismatch")
    error = max(abs(predictions[r["id"]]-r["isotonic"]) for r in baseline)
    if not math.isfinite(error) or error > 1e-12:
        raise ValueError(f"V0 stored calibration reproduction failed: {error}")
    return error


def summarize(rows, predictions):
    if set(predictions) != {r["id"] for r in rows}:
        raise ValueError("Summary cohort mismatch")
    scores = {name+"_brier": fmean((getter(r)-r["target"])**2 for r in rows)
              for name, getter in (("raw", lambda r: r["raw"]),
                                   ("transfer", lambda r: predictions[r["id"]]),
                                   ("separate", lambda r: r["isotonic"]))}
    return {"n": len(rows), **scores,
            "transfer_minus_raw": scores["transfer_brier"]-scores["raw_brier"],
            "transfer_minus_separate": scores["transfer_brier"]-scores["separate_brier"]}


def read_saved(directory):
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    verified = {}
    buffers = {}
    for name in ("oof_predictions.csv.gz", "question_folds.csv"):
        payload = (directory / name).read_bytes()
        digest = sha_bytes(payload)
        if digest != manifest["output_sha256"][name]:
            raise ValueError(f"Archived artifact hash mismatch: {name}")
        verified[name], buffers[name] = digest, payload
    fold_rows = list(csv.DictReader(io.StringIO(buffers["question_folds.csv"].decode("utf-8-sig"))))
    folds = {int(r["id"]): int(r["fold"]) for r in fold_rows}
    if len(folds) != len(fold_rows) or not folds or set(folds.values()) != set(range(5)):
        raise ValueError("Duplicate IDs or invalid original five folds")
    rows = csv.DictReader(io.StringIO(gzip.decompress(buffers["oof_predictions.csv.gz"]).decode("utf-8-sig")))
    expected = {(m, reading, v) for m in MODELS for reading in ("selected", "direct") for v in range(9)}
    groups, targets = {}, {}
    for row in rows:
        key = row["model"], row["reading"], int(row["variant"])
        if key not in expected:
            raise ValueError("Unexpected model/reading/variant in archived OOF table")
        value = {k: int(row[k]) for k in ("id", "target", "fold")}
        value.update({k: float(row[k]) for k in ("raw", "isotonic")})
        q = value["id"]
        if q not in folds or value["fold"] != folds[q] or (q in targets and targets[q] != value["target"]):
            raise ValueError("Archived target/fold disagreement")
        targets[q] = value["target"]
        groups.setdefault(key, []).append(value)
    if set(groups) != expected:
        raise ValueError("Incomplete54-condition archived OOF family")
    for group in groups.values():
        if set(validate_rows(group)) != set(folds):
            raise ValueError("Incomplete archived question cohort")
    return groups, {"manifest.json": sha_bytes(manifest_bytes), **verified}


def build_report(directory):
    import numpy
    import sklearn

    groups, hashes = read_saved(directory)
    conditions, summaries, traces, predictions, reproduction = [], [], [], [], []
    for model in MODELS:
        for reading in ("selected", "direct"):
            baseline = groups[model, reading, 0]
            fitted, trace = fit_baseline(baseline)
            traces.extend({"model": model, "reading": reading, **r} for r in trace)
            context = []
            for variant in range(9):
                rows = groups[model, reading, variant]
                transferred = apply_transfer(baseline, rows, fitted)
                if variant == 0:
                    reproduction.append(check_reproduction(baseline, transferred))
                result = {"model": model, "reading": reading, "variant": variant,
                          **summarize(rows, transferred)}
                conditions.append(result)
                if variant:
                    context.append(result)
                predictions.append({"model": model, "reading": reading, "variant": variant,
                                    "probabilities": {str(q): transferred[q] for q in sorted(transferred)}})
            summaries.append({"model": model, "reading": reading, "variant_n": 8,
                              **{name: fmean(c[name] for c in context) for name in
                                 ("raw_brier", "transfer_brier", "separate_brier",
                                  "transfer_minus_raw", "transfer_minus_separate")}})
    return {"schema": "v0_isotonic_transfer_v1", "llm_calls": 0, "calibration_fit_n": 30,
            "input_sha256": hashes, "script_sha256": sha_bytes(Path(__file__).read_bytes()),
            "versions": {"python": sys.version, "numpy": numpy.__version__, "sklearn": sklearn.__version__},
            "settings": {"folds": "original stored five folds; no reseeding",
                         "method": "increasing isotonic, output0..1, out_of_bounds=clip",
                         "fit_source": "V0 only, other four question folds",
                         "aggregation": "equal weights over V1..V8; per-model/reading",
                         "uncertainty": "descriptive only; no intervals/pvalues",
                         "baseline_reproduction_tolerance": 1e-12},
            "limitations": ["retrospective evidence; no prospective claim", "related events may span folds",
                            "conditional archived-confidence readings", "export precision16significant digits"],
            "max_v0_reproduction_error": max(reproduction), "conditions": conditions,
            "summaries": summaries, "fold_models": traces, "transferred_predictions": predictions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = build_report(args.calibration_directory)
    payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(json.dumps({"max_v0_reproduction_error": report["max_v0_reproduction_error"],
                      "summaries": report["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
