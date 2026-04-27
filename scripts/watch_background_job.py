#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a background job and restart it if it disappears or stalls."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--match-pattern", required=True)
    parser.add_argument("--restart-cmd", required=True)
    parser.add_argument("--progress-root", type=Path, required=True)
    parser.add_argument("--check-interval", type=int, default=60)
    parser.add_argument("--stale-seconds", type=int, default=900)
    parser.add_argument("--startup-grace-seconds", type=int, default=300)
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def find_matching_pids(pattern: str) -> list[int]:
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_str, *_ = stripped.split(maxsplit=1)
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def newest_mtime(root: Path) -> float:
    newest = 0.0
    if root.is_file():
        return root.stat().st_mtime
    if not root.exists():
        return newest
    for path in root.rglob("*"):
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
    return newest


def kill_pids(pids: list[int]) -> None:
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    time.sleep(5)
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def restart_job(cwd: Path, restart_cmd: str) -> None:
    shell_cmd = f"cd {shlex.quote(str(cwd))} && {restart_cmd}"
    subprocess.run(
        ["bash", "-lc", shell_cmd],
        check=True,
    )


def main() -> None:
    args = parse_args()
    last_restart_at = 0.0
    log(
        f"watchdog started for {args.name}; pattern={args.match_pattern!r}; "
        f"progress_root={str(args.progress_root)!r}"
    )
    while True:
        now = time.time()
        pids = find_matching_pids(args.match_pattern)
        progress_mtime = newest_mtime(args.progress_root)
        progress_age = float("inf") if progress_mtime <= 0 else now - progress_mtime
        recently_restarted = (now - last_restart_at) < args.startup_grace_seconds
        stale = progress_age > args.stale_seconds and not recently_restarted

        if not pids:
            log(f"{args.name}: process missing; restarting")
            restart_job(args.cwd, args.restart_cmd)
            last_restart_at = time.time()
        elif stale:
            log(
                f"{args.name}: progress stale for {int(progress_age)}s; "
                f"restarting pids={pids}"
            )
            kill_pids(pids)
            restart_job(args.cwd, args.restart_cmd)
            last_restart_at = time.time()

        time.sleep(args.check_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
