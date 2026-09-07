#!/usr/bin/env python3
"""Check that a test actually fails when you break the code it covers.

A green test suite says the code passes its tests. It does not say the tests
would notice if the code were wrong, and in this repository they twice did
not: `test_place_trade_polymarket_settles_open_positions_before_new_cycle`
asserted a Kalshi settlement fee on Polymarket, so the venue-blind fee rate
it should have caught was instead locked in by it.

The check is: break the line on purpose, run the test, expect red. Two
things make that lie, and both were hit while auditing this repo.

1. The mutation silently does not apply. A substring anchor matches a
   different line, or matches nothing after a quoting slip, and the run
   reports OK -- which reads as "the test does not pin this" when nothing
   was ever mutated. Worse, an anchor can match *two* lines when one is a
   prefix of the other at different indentation, which is why this works on
   line numbers and asserts the file changed.

2. Stale bytecode. Rewriting a file inside a loop can land inside the same
   filesystem mtime tick, so CPython reuses the cached .pyc and runs the
   code you just replaced. That produces a pass on mutated source, or a
   failure on restored source. Caches are cleared around every run and the
   child runs with -B.

Usage:

    python scripts/mutation_check.py src/pkg/mod.py 42 'old' 'new' tests.test_mod

Exit status is 0 when the mutation was caught (the test failed), 1 when it
survived, and 2 when the mutation could not be applied -- which is not a
result about the test at all.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
from typing import List, Tuple

CAUGHT, SURVIVED, NOT_APPLIED = 0, 1, 2


def clear_bytecode(root: pathlib.Path) -> None:
    """Remove __pycache__ so a same-tick rewrite cannot be masked."""
    for cache in root.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def run_tests(module: str, root: pathlib.Path) -> Tuple[bool, str]:
    """Run one unittest module. Returns (passed, summary)."""
    clear_bytecode(root)
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", module],
        capture_output=True, text=True, cwd=root,
    )
    summary = " ".join(
        line for line in (proc.stderr or "").splitlines()
        if line.startswith(("Ran ", "OK", "FAILED", "ERROR"))
    )
    return proc.returncode == 0, summary or "(no summary)"


def apply_mutation(lines: List[str], lineno: int, old: str, new: str) -> List[str]:
    """Replace `old` with `new` on one line. Raises if it would be a no-op."""
    index = lineno - 1
    if not 0 <= index < len(lines):
        raise ValueError(f"line {lineno} is outside the file (1..{len(lines)})")
    if old not in lines[index]:
        raise ValueError(f"line {lineno} does not contain {old!r}: {lines[index].strip()!r}")
    mutated = list(lines)
    mutated[index] = lines[index].replace(old, new)
    if mutated[index] == lines[index]:
        raise ValueError("mutation is a no-op -- old and new produce the same line")
    return mutated


def check(target: pathlib.Path, lineno: int, old: str, new: str,
          module: str, root: pathlib.Path) -> int:
    original = target.read_text(encoding="utf-8")
    try:
        mutated = apply_mutation(original.splitlines(keepends=True), lineno, old, new)
    except ValueError as exc:
        print(f"mutation not applied: {exc}", file=sys.stderr)
        return NOT_APPLIED

    passed_before, before = run_tests(module, root)
    if not passed_before:
        print(f"baseline is already failing, nothing to learn: {before}", file=sys.stderr)
        return NOT_APPLIED

    target.write_text("".join(mutated), encoding="utf-8")
    if target.read_text(encoding="utf-8") == original:
        target.write_text(original, encoding="utf-8")
        print("mutation not applied: file unchanged after write", file=sys.stderr)
        return NOT_APPLIED
    try:
        passed_after, after = run_tests(module, root)
    finally:
        target.write_text(original, encoding="utf-8")
        clear_bytecode(root)

    print(f"baseline : {before}")
    print(f"mutated  : {after}")
    if passed_after:
        print(f"SURVIVED -- {module} does not pin {target}:{lineno}")
        return SURVIVED
    print(f"CAUGHT -- {module} fails when {target}:{lineno} is broken")
    return CAUGHT


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file")
    parser.add_argument("line", type=int)
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("test_module")
    parser.add_argument("--root", default=".", help="repository root to run from")
    args = parser.parse_args(argv)
    root = pathlib.Path(args.root).resolve()
    return check(root / args.file, args.line, args.old, args.new, args.test_module, root)


if __name__ == "__main__":
    raise SystemExit(main())
