#!/usr/bin/env python3
"""Compatibility wrapper for downloading the local Qwen model."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from analyzing_llm_rationale.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["download-model"]))
