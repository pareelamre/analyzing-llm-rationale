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

A sweep rewrites the module in place, hundreds of times. Two
consequences follow, and both bit while this was being written.

Do not run anything else against the repository while a sweep is going.
A test run started alongside one imports whichever half-mutated state
the file happened to be in, and reports failures that belong to the
sweep rather than to the code.

And an interrupted sweep can leave the last mutation applied, because
the restore never ran. That is worse than a crash: the source looks
fine, and the bad line reaches a commit. So this refuses to start when
the target file already has uncommitted changes -- which is exactly what
a previous interrupted run leaves behind -- and restores the file on
SIGINT and SIGTERM as well as on a normal exit.

Usage:

    python scripts/mutation_sweep.py src/pkg/mod.py tests.test_mod

Add --start/--end to sweep a region, and --rule to narrow the operators
tried. Output lists survivors as file:line with the substitution, ready
to paste into mutation_check.py.
"""

from __future__ import annotations

import argparse
import atexit
import io
import pathlib
import re
import signal
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


def literal_spans(path: pathlib.Path) -> dict[int, list[tuple[int, int]]]:
    """Column ranges on each line that are literal text, not code.

    A regex sweep over raw text mutates the inside of docstrings and log
    messages, which is noise rather than a finding. Tokenising is the only
    way to tell an operator from the same characters inside a literal.

    Skipping whole lines is not enough, and Python 3.12 is why. It no
    longer emits an f-string as a single STRING token: the literal text
    arrives as FSTRING_MIDDLE with the interpolations around it as
    ordinary tokens. So a line like

        strategy = f"BUY YES @ {p:.1f}% + BUY NO @ {k:.1f}%"

    genuinely contains code, a line-level filter keeps it, and the regex
    then finds the first `+` -- which is prose. Two junk survivors came
    out of an arbitrage_scanner sweep that way.

    This records where the prose is, and a match is rejected when it
    overlaps any of it. Recording where the *code* is does not work: a
    pattern like " + " or "max(" spans a token boundary, so it fits
    inside no single code span.
    """
    source = path.read_text(encoding="utf-8")
    spans: dict[int, list[tuple[int, int]]] = {}
    kinds = {tokenize.STRING, tokenize.COMMENT}
    # FSTRING_MIDDLE exists from 3.12; before that an f-string is one
    # STRING token and is already covered.
    kinds.add(getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING))
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type not in kinds:
                continue
            if token.start[0] == token.end[0]:
                spans.setdefault(token.start[0], []).append(
                    (token.start[1], token.end[1])
                )
            else:
                # A triple-quoted block: every line it covers is prose.
                for line in range(token.start[0], token.end[0] + 1):
                    spans.setdefault(line, []).append((0, 10**6))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return {}
    return spans


def parses(path: pathlib.Path) -> bool:
    """Whether the file tokenises at all. An unparseable file has no findings."""
    try:
        for _ in tokenize.generate_tokens(
            io.StringIO(path.read_text(encoding="utf-8")).readline
        ):
            pass
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return False
    return True


def _first_match_in_code(pattern, text, prose):
    """The first match of `pattern` that does not run through literal text."""
    for match in re.finditer(pattern, text):
        if not any(match.start() < hi and lo < match.end() for lo, hi in prose):
            return match
    return None


def candidates(path: pathlib.Path, rules: list[str], start: int, end: int):
    """One mutation per eligible line -- the first rule that matches."""
    if not parses(path):
        return
    prose = literal_spans(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, text in enumerate(lines, 1):
        if not (start <= number <= end):
            continue
        for name in rules:
            pattern, replacement = RULES[name]
            match = _first_match_in_code(pattern, text, prose.get(number, ()))
            if match is None:
                continue
            mutated = (
                text[: match.start()]
                + match.expand(replacement)
                + text[match.end():]
            )
            if mutated != text:
                yield number, name, text.strip(), mutated.strip()
            break


def guard_the_file(path: pathlib.Path) -> None:
    """Put the module back however the sweep ends.

    mutation_check restores after each mutation, so a sweep that runs to
    completion is clean. A sweep that is killed is not: the file stays as
    the interrupted iteration left it, and nothing says so. SIGKILL still
    cannot be caught, which is why has_local_changes exists as well.
    """
    original = path.read_bytes()

    def restore(*_args) -> None:
        if path.read_bytes() != original:
            path.write_bytes(original)
            print(f"restored {path}", file=sys.stderr)

    atexit.register(restore)
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous = signal.getsignal(sig)

        def handler(signum, frame, _previous=previous):
            restore()
            if callable(_previous):
                _previous(signum, frame)
            raise SystemExit(130)

        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass    # not the main thread, or unsupported on this platform


def has_local_changes(root: pathlib.Path, relpath: str) -> bool:
    """True when git reports the target dirty -- often an earlier sweep."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", relpath],
            cwd=root, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False        # not a checkout, or no git: nothing to compare
    return result.returncode == 0 and bool(result.stdout.strip())


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
    parser.add_argument(
        "--force", action="store_true",
        help="sweep even though the target file has uncommitted changes",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    path = root / args.file
    if not path.is_file():
        print(f"no such file: {path}", file=sys.stderr)
        return 2

    if has_local_changes(root, args.file) and not args.force:
        print(
            f"{args.file} has uncommitted changes.",
            "A sweep rewrites it in place, so it would restore to the modified",
            "state, and any edit made while it runs would be lost.",
            "An interrupted sweep leaves the file looking exactly like this",
            "and may have left a mutation applied -- read the diff before --force.",
            sep=chr(10), file=sys.stderr,
        )
        return 2

    guard_the_file(path)
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
