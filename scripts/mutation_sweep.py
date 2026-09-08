#!/usr/bin/env python3
"""Find the lines in a module that its tests would not notice were wrong.

scripts/mutation_check.py answers that for one line you already suspect.
This asks it of every line in a module, which is a different question --
you find out what you were not suspicious of.

That matters here because this codebase's failure mode is not exceptions,
it is plausible-looking defaults: a cap written as min() that could be
max(), a divisor of 86400 that could be 3600, a guard that reads as
obviously right and is therefore never checked. Those survive review for
the same reason they survive testing, so the search has to be mechanical.

Two things it will not tell you.

First, a survivor is not automatically a bug. Many are *equivalent
mutations* -- the two forms cannot be told apart because the input that
would separate them is unreachable. `lead < min_days` and `lead <=
min_days` differ only when a wall-clock float lands exactly on a bound.
`market_brier is not None and model_brier is not None` differs from `or`
only when exactly one is None, which cannot happen because both are None
together or neither is. Read every survivor before writing a test for it;
the useful output of a sweep is a shortlist, not a defect list.

Second, this runs one test module. A line that survives its own module is
often caught by another -- of three survivors found this way in
portfolio_optimizer, two were held by the full suite and only one was a
real gap. Confirm a survivor against the whole suite before acting.

Usage:

    python scripts/mutation_sweep.py src/pkg/mod.py tests.test_mod

Add --start/--end to sweep a region, and --rule to narrow the operators
tried. Output lists survivors as file:line with the substitution, ready
to paste into mutation_check.py.
"""

from __future__ import annotations

import argparse
import io
import pathlib
import re
import subprocess
import sys
import tokenize

#: name -> (pattern, replacement). Each is a substitution that changes
#: behaviour without changing whether the line parses.
RULES: dict[str, tuple[str, str]] = {
    "ge-gt": (r">=", ">"),
    "le-lt": (r"<=", "<"),
    "gt-ge": (r"(?<![<>=!])>(?!=)", ">="),
    "lt-le": (r"(?<![<>=!])<(?!=)", "<="),
    "max-min": (r"\bmax\(", "min("),
    "min-max": (r"\bmin\(", "max("),
    "and-or": (r"\band\b", "or"),
    "or-and": (r"\bor\b", "and"),
    "plus-minus": (r" \+ ", " - "),
    "minus-plus": (r" - ", " + "),
}

CAUGHT, SURVIVED, NOT_APPLIED = 0, 1, 2


def code_lines(path: pathlib.Path) -> set[int]:
    """Line numbers holding real code, not comments or string contents.

    A regex sweep over raw text mutates the inside of docstrings and log
    messages, which produces noise rather than findings. Tokenising is the
    only reliable way to tell an operator from the same characters inside
    a literal.
    """
    source = path.read_text(encoding="utf-8")
    skip: set[int] = set()
    keep: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            start, end = token.start[0], token.end[0]
            if token.type in (tokenize.STRING, tokenize.COMMENT):
                skip.update(range(start, end + 1))
            elif token.type == tokenize.OP:
                keep.add(start)
            elif token.type == tokenize.NAME and token.string in ("max", "min", "and", "or"):
                keep.add(start)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return set()
    return keep - skip


def candidates(path: pathlib.Path, rules: list[str], start: int, end: int):
    """One mutation per eligible line -- the first rule that matches."""
    eligible = code_lines(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, text in enumerate(lines, 1):
        if number not in eligible or not (start <= number <= end):
            continue
        for name in rules:
            pattern, replacement = RULES[name]
            if re.search(pattern, text):
                mutated = re.sub(pattern, replacement, text, count=1)
                if mutated != text:
                    yield number, name, text.strip(), mutated.strip()
                break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="module to mutate, e.g. src/pkg/mod.py")
    parser.add_argument("test_module", help="test module to run, e.g. tests.test_mod")
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--start", type=int, default=1, help="first line to consider")
    parser.add_argument("--end", type=int, default=10**9, help="last line to consider")
    parser.add_argument(
        "--rule", action="append", choices=sorted(RULES), default=None,
        help="restrict to these substitutions (repeatable; default is all)",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    path = root / args.file
    if not path.is_file():
        print(f"no such file: {path}", file=sys.stderr)
        return 2

    rules = args.rule or list(RULES)
    work = list(candidates(path, rules, args.start, args.end))
    if not work:
        print(f"no candidate lines in {args.file}")
        return 0

    print(f"{len(work)} candidate lines in {args.file}, running {args.test_module}")
    survivors, unapplied = [], 0
    for index, (number, rule, old, new) in enumerate(work, 1):
        result = subprocess.run(
            [
                sys.executable, str(root / "scripts" / "mutation_check.py"),
                args.file, str(number), old, new, args.test_module,
                "--root", str(root),
            ],
            cwd=root, capture_output=True, text=True,
        )
        if result.returncode == SURVIVED:
            survivors.append((number, rule, old, new))
            print(f"  [{index}/{len(work)}] {args.file}:{number} SURVIVED ({rule})")
        elif result.returncode not in (CAUGHT, SURVIVED):
            unapplied += 1

    print(
        f"\n{len(survivors)} survived, {len(work) - len(survivors) - unapplied} caught, "
        f"{unapplied} not applied"
    )
    for number, rule, old, new in survivors:
        print(f"\n{args.file}:{number}  ({rule})")
        print(f"    {old}")
        print(f" -> {new}")
    if survivors:
        print(
            "\nRead each of these before writing a test. Some are equivalent "
            "mutations, and some are caught by other test modules -- confirm "
            "against the full suite first."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
