"""Offline, question-held-out prompt selection on hash-verified saved forecasts.

Reports conditional confidence readings separately. No inference, retrieval,
calibration fitting or prospective-performance claim is made by this analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from statistics import fmean

BENCHMARK = "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
MODEL_TEMPERATURE = {"GPT-OSS-120B": "00", "Qwen2.5-7b-instruct": "000", "Qwen3-32B": "0"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_verified(root, relative, hashes):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Input outside root: {relative}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != hashes.get(relative):
        raise ValueError(f"Input outside root or SHA256 mismatch: {relative}")
    return json.loads(payload.decode("utf-8-sig"), object_pairs_hook=unique_keys)


def probability_yes(answer, confidence, reading):
    answer = str(answer).strip().lower()
    if answer not in {"yes", "no"} or reading not in {"selected", "direct"}:
        raise ValueError("Explicit selected/direct probability reading and Yes/No answer required")
    if isinstance(confidence, bool) or confidence is None:
        raise ValueError("Confidence must be a finite probability")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Confidence must be a finite probability")
    return confidence if reading == "direct" or answer == "yes" else 1-confidence


def score_rows(rows, targets, reading):
    probabilities, seen = {}, set()
    for row in rows:
        qid = row.get("id")
        if type(qid) is not int or qid not in targets or qid in seen:
            raise ValueError("Forecast IDs must be unique known integers")
        seen.add(qid)
        try:
            probabilities[qid] = probability_yes(row.get("predicted_answer"), row.get("confidence"), reading)
        except (ValueError, TypeError):
            continue
    return probabilities


def assign_folds(question_ids, n_folds=5, seed=42):
    ids = sorted(question_ids)
    if (type(n_folds) is not int or not 2 <= n_folds <= len(ids)
            or len(ids) != len(set(ids)) or any(type(q) is not int for q in ids)):
        raise ValueError("Unique integer IDs and 2..N folds required")
    random.Random(seed).shuffle(ids)
    return {qid: index % n_folds for index, qid in enumerate(ids)}


def evaluate(probabilities, targets, folds, baseline):
    if baseline not in probabilities or len(probabilities) < 2:
        raise ValueError("At least two variants including the baseline required")
    names = sorted(probabilities)
    ids = sorted(set(targets).intersection(*(set(p) for p in probabilities.values())))
    if len(ids) < 2 or any(q not in folds for q in ids):
        raise ValueError("At least two common questions with fixed fold membership required")
    if any(type(targets[q]) is not int or targets[q] not in (0, 1) for q in ids):
        raise ValueError("Binary integer targets required")
    if any(isinstance(p[q], bool) or not math.isfinite(p[q]) or not 0 <= p[q] <= 1
           for p in probabilities.values() for q in ids):
        raise ValueError("Finite probability vectors required")
    present_folds = sorted({folds[q] for q in ids})
    if len(present_folds) < 2:
        raise ValueError("At least two nonempty folds required")
    predictions, selections = [], []
    for fold in present_folds:
        train = [q for q in ids if folds[q] != fold]
        test = [q for q in ids if folds[q] == fold]
        selected = min(names, key=lambda name: (
            fmean((probabilities[name][q]-targets[q])**2 for q in train), name))
        selections.append({"fold": fold, "selected_variant": selected,
                           "train_n": len(train), "test_n": len(test)})
        for q in test:
            predictions.append({"question_id": q, "fold": fold, "target": targets[q],
                                "selected_variant": selected,
                                "baseline_probability": probabilities[baseline][q],
                                "selected_probability": probabilities[selected][q],
                                "ensemble_probability": fmean(probabilities[name][q] for name in names)})
    predictions.sort(key=lambda p: p["question_id"])
    scores = {strategy+"_brier": fmean((p[strategy+"_probability"]-p["target"])**2
                                      for p in predictions)
              for strategy in ("baseline", "selected", "ensemble")}
    member_scores = {name: fmean((probabilities[name][q]-targets[q])**2 for q in ids) for name in names}
    disagreement = fmean((probabilities[name][p["question_id"]]-p["ensemble_probability"])**2
                         for p in predictions for name in names)
    return {"n_expected": len(targets), "common_n": len(ids), "common_question_ids": ids,
            "coverage_by_variant": {name: len(probabilities[name]) for name in names},
            "variant_brier_on_common_cohort": member_scores, **scores,
            "selected_minus_baseline": scores["selected_brier"]-scores["baseline_brier"],
            "ensemble_minus_baseline": scores["ensemble_brier"]-scores["baseline_brier"],
            "mean_member_brier": fmean(member_scores.values()),
            "mean_probability_disagreement": disagreement,
            "fold_selections": selections, "predictions": predictions}


def build_report(root, reference, n_folds=5, seed=42):
    manifest_bytes = reference.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"), object_pairs_hook=unique_keys)
    hashes = manifest["input_sha256"]
    dataset = read_verified(root, BENCHMARK, hashes)
    targets = {}
    for row in dataset:
        qid, answer = row.get("id"), str(row.get("answer")).strip().lower()
        if type(qid) is not int or qid in targets or answer not in {"yes", "no"}:
            raise ValueError("Benchmark IDs/labels must be unique integers and Yes/No")
        targets[qid] = int(answer == "yes")
    folds = assign_folds(targets, n_folds, seed)
    inputs, conditions = {BENCHMARK: hashes[BENCHMARK]}, []
    for model, temperature in MODEL_TEMPERATURE.items():
        raw, baseline = {}, None
        for variant in range(9):
            prefix = f"results/{model}/temperature_{temperature}/results_variant{variant}_"
            matches = [name for name in hashes if name.startswith(prefix) and name.endswith(".json")
                       and "/" not in name[len(prefix):]]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one archived T=0 input: {prefix}")
            relative = matches[0]
            rows = read_verified(root, relative, hashes)
            if not isinstance(rows, list):
                raise ValueError("Archived forecast file must contain a list")
            name = Path(relative).stem.removeprefix("results_")
            raw[name] = rows
            if variant == 0:
                baseline = name
            inputs[relative] = hashes[relative]
        for reading in ("selected", "direct"):
            probabilities = {name: score_rows(rows, targets, reading) for name, rows in raw.items()}
            conditions.append({"model": model, "reading": reading,
                               **evaluate(probabilities, targets, folds, baseline)})
    return {"schema": "offline_prompt_selection_v1", "model_calls": 0,
            "settings": {"temperature": 0, "n_folds": n_folds, "seed": seed,
                         "selection": "lowest training-question Brier; variant-name tie break",
                         "ensemble": "equal probability mean of all nine variants",
                         "cohort": "intersection of valid questions across nine variants per model",
                         "uncertainty": "descriptive point estimates; no significance tests or intervals",
                         "deployment_cost": "selection requires nine saved training forecasts per question; ensemble requires nine inference responses per new question, baseline/selected each require one"},
            "limitations": ["Retrospective evidence access; not a prospective forecast test",
                            "Random question folds do not separate event families or time",
                            "Both archived-confidence readings are conditional interpretations",
                            "One generation per question and condition; no run-variability estimate"],
            "reference_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "script_sha256": sha(Path(__file__)), "python": sys.version,
            "input_sha256": inputs,
            "question_folds": {str(q): folds[q] for q in sorted(folds)}, "conditions": conditions}


def write_report(path, report):
    # Serialize before opening so validation failures cannot leave a partial report.
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = build_report(args.data_root.resolve(), args.reference_manifest, args.folds, args.seed)
    write_report(args.output, report)
    print(json.dumps([{key: row[key] for key in
                       ("model", "reading", "common_n", "baseline_brier", "selected_brier", "ensemble_brier")}
                      for row in report["conditions"]], indent=2))


if __name__ == "__main__":
    main()
