from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class Example:
    predicted_answer: str
    confidence: float
    target: int

    @property
    def predicted_label(self) -> int:
        return 1 if self.predicted_answer == "yes" else 0

    @property
    def p_yes(self) -> float:
        return self.confidence if self.predicted_label == 1 else 1.0 - self.confidence

    @property
    def correct(self) -> int:
        return int(self.predicted_label == self.target)


def load_targets(dataset_path: Path) -> dict[int, int]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    targets: dict[int, int] = {}
    for row in payload:
        answer = str(row["answer"]).strip().lower()
        if answer not in {"yes", "no"}:
            continue
        targets[int(row["id"])] = 1 if answer == "yes" else 0
    return targets


def normalize_answer(value: object) -> Optional[str]:
    if value is None:
        return None
    answer = str(value).strip().lower()
    if answer in {"yes", "no"}:
        return answer
    return None


def normalize_confidence(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(confidence) or math.isinf(confidence):
        return None
    if not (0.0 <= confidence <= 1.0):
        return None
    return confidence


def iter_examples(
    rows: Iterable[dict], targets: dict[int, int]
) -> Tuple[list[Example], int]:
    examples: list[Example] = []
    missing = 0
    for row in rows:
        rid = row.get("id")
        if rid not in targets:
            missing += 1
            continue
        answer = normalize_answer(row.get("predicted_answer"))
        confidence = normalize_confidence(row.get("confidence"))
        if answer is None or confidence is None:
            missing += 1
            continue
        examples.append(Example(answer, confidence, targets[rid]))
    return examples, missing


def accuracy(examples: list[Example]) -> float:
    return sum(ex.correct for ex in examples) / len(examples)


def brier_score(examples: list[Example]) -> float:
    return sum((ex.p_yes - ex.target) ** 2 for ex in examples) / len(examples)


def ece(examples: list[Example], bins: int = 10) -> float:
    total = len(examples)
    if total == 0:
        return float("nan")

    bin_counts = [0] * bins
    bin_confidence = [0.0] * bins
    bin_accuracy = [0.0] * bins

    for ex in examples:
        idx = min(int(ex.confidence * bins), bins - 1)
        bin_counts[idx] += 1
        bin_confidence[idx] += ex.confidence
        bin_accuracy[idx] += ex.correct

    total_error = 0.0
    for count, conf_sum, acc_sum in zip(bin_counts, bin_confidence, bin_accuracy):
        if count == 0:
            continue
        avg_conf = conf_sum / count
        avg_acc = acc_sum / count
        total_error += (count / total) * abs(avg_acc - avg_conf)
    return total_error
