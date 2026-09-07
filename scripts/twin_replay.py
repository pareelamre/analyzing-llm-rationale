"""Generate a deterministic autonomous-twin replay/readiness artifact."""
from __future__ import annotations

import argparse
import json
import os
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale.config import load_yaml  # noqa: E402
from analyzing_llm_rationale.twin.evaluation import evaluate_dataset  # noqa: E402


def _code_hash() -> str:
    digest = sha256()
    for relative in (
        "src/analyzing_llm_rationale/twin/replay.py",
        "src/analyzing_llm_rationale/twin/evaluation.py",
        "src/analyzing_llm_rationale/forecast_evaluation.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    config_document = load_yaml(args.config)
    config = config_document.get("replay")
    if not isinstance(config, dict):
        parser.error("config must contain a replay object")
    artifact = evaluate_dataset(dataset, config, code_hash=_code_hash())
    _write_atomic(args.output, artifact)
    print(json.dumps({
        "output": str(args.output), "status": artifact["status"],
        "artifact_hash": artifact["artifact_hash"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
