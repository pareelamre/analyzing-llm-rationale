#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from analyzing_llm_rationale.cli import main as cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main(["verify-results", *sys.argv[1:]]))
