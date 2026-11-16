#!/usr/bin/env python3
"""CLM-8B against Open-Jev 2B as the rationale checker (docs/fund.md, pre-registered 2026-11-16).

The checker reads an on-device forecast's rationale and says whether it concludes the event
is likely (scripts/fund_rationale_check.py). This scores both checkers on the same cases:
the 7 known forecasts of 2026-11-08 (4 inverted) and every clear on-device forecast made in
the week after the confidence fix, labelled by the side its number takes.

Usage:
  fund_rationale_checker_eval.py cases --store STORE.duckdb --out cases.json
  fund_rationale_checker_eval.py score --cases cases.json --checker jev2b --out scores-jev2b.json
  fund_rationale_checker_eval.py score --cases cases.json --checker clm --shard 0/6 \
      --emb-url http://127.0.0.1:8090/v1/embeddings --head CLM_v0.1-8B.pt --out scores-clm-0.json
  fund_rationale_checker_eval.py report --cases cases.json --scores scores-*.json \
      --out reports/fund_rationale_checker_eval.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fund_rationale_check", ROOT / "scripts" / "fund_rationale_check.py")
check = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("fund_rationale_check", check)
_spec.loader.exec_module(check)

KNOWN_PATH = ROOT / "tests" / "fixtures" / "rationale_check_cases.json"
REPORT_PATH = ROOT / "reports" / "fund_rationale_checker_eval.json"
#: Live forecasts from after the confidence fix (#746) to the day the test was registered.
START = datetime(2026, 11, 9, tzinfo=timezone.utc)
END = datetime(2026, 11, 16, tzinfo=timezone.utc)
#: A number is clear at this distance from the middle or more (0.30 or less, 0.70 or more).
CLEAR_NUMBER = 0.30
MIN_LIVE, MIN_MARKETS = 40, 15
MAX_MEDIAN_SECONDS = 60.0
DRAWS, SEED = 10_000, 0
CHECKERS = ("jev2b", "clm")
NAMES = {"jev2b": "Open-Jev 2B", "clm": "CLM-8B"}


def _when(ts) -> datetime:
    """A snapshot time as an aware datetime, whether the store gives a datetime or text."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    text = str(ts).strip().replace(" ", "T")
    if re.search(r"[+-]\d\d$", text):
        text += ":00"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    when = datetime.fromisoformat(text)
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def live_cases(store: Path, *, start: datetime = START, end: datetime = END) -> list[dict]:
    """Clear on-device forecasts with a rationale in [start, end), one per market and rationale."""
    import duckdb
    con = duckdb.connect(str(store), read_only=True)
    rows = con.execute("""
        SELECT key, platform, ident, question, model_probability, snapshot_ts, rationale
        FROM forecast_snapshot
        WHERE model = ? AND rationale IS NOT NULL AND length(trim(rationale)) > 0
          AND model_probability IS NOT NULL
    """, [check.FORECAST_MODEL]).fetchall()
    con.close()
    out, seen = [], set()
    for key, platform, ident, question, p, ts, rationale in sorted(rows, key=lambda r: (_when(r[5]), str(r[0]))):
        when = _when(ts)
        p = float(p)
        if not (start <= when < end) or CLEAR_NUMBER < p < 1 - CLEAR_NUMBER:
            continue
        market = f"{str(platform or 'kalshi').lower()}:{ident}"
        text = str(rationale).strip()
        if (market, text) in seen:
            continue
        seen.add((market, text))
        out.append({"id": f"live:{key}", "kind": "live", "market": market, "question": question or ident,
                    "forecast": p, "label": 1 if p >= 0.5 else 0, "forecasted_at": when.isoformat(),
                    "rationale": text[:check.MAX_RATIONALE_CHARS]})
    return out


def known_cases(path: Path = KNOWN_PATH) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    return [{"id": f"known:{c['ident']}", "kind": "known", "market": c["ident"], "question": c["question"],
             "forecast": float(c["forecast"]), "expected": c["expected"],
             "rationale": c["rationale"][:check.MAX_RATIONALE_CHARS]} for c in cases]


def clm_checker(emb_url: str, head: str) -> Callable[[str, str], float]:
    """P(the rationale concludes Yes) from CLM-8B's noul answer, with the same state and
    question the Open-Jev job uses. The encoder is the llama.cpp server at ``emb_url``."""
    from clm.embedder import Embedder
    from clm.engine import Engine
    engine = Engine(embedder=Embedder(url=emb_url, model="qwen3-8b", max_tokens=None, batch=1),
                    checkpoint=head, device="cpu", action_cache="0")

    def ask(question: str, rationale: str) -> float:
        out = engine.answer({"forecast_question": question, "rationale": rationale},
                            {"concludes_yes": {"type": "noul", "instructions": check.QUESTION}})
        return float(out["answers"]["concludes_yes"]["noul"])
    return ask


def shard_of(cases: list[dict], shard: str) -> list[dict]:
    i, n = (int(x) for x in shard.split("/"))
    return cases[i::n]


def score(cases: list[dict], ask: Callable[[str, str], float], checker: str, out_path: Path,
          load_seconds: float = 0.0) -> dict:
    """Ask the checker about every case, saving after each so a timeout keeps what was done."""
    result = {"checker": checker, "load_seconds": round(load_seconds, 2), "scores": {}}
    for case in cases:
        t0 = time.perf_counter()
        p_yes = ask(case["question"], case["rationale"])
        seconds = time.perf_counter() - t0
        result["scores"][case["id"]] = {"p_yes": round(float(p_yes), 6), "seconds": round(seconds, 3)}
        print(f"{checker:5} {seconds:6.1f}s p_yes {p_yes:.3f} forecast {case['forecast']:.2f} | {case['id'][:70]}",
              flush=True)
        out_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    out_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def merge_scores(paths: list[Path]) -> dict[str, dict[str, dict]]:
    merged: dict[str, dict[str, dict]] = {c: {} for c in CHECKERS}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        merged.setdefault(data["checker"], {}).update(data["scores"])
    return merged


def market_bootstrap(diffs: list[tuple[str, float]], *, draws: int = DRAWS, seed: int = SEED) -> tuple[float, float]:
    """95% interval of the mean difference, resampling markets (each keeps all its cases)."""
    by_market: dict[str, list[float]] = {}
    for market, d in diffs:
        by_market.setdefault(market, []).append(d)
    groups = [(sum(v), len(v)) for v in by_market.values()]
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        pick = [groups[rng.randrange(len(groups))] for _ in groups]
        means.append(sum(s for s, _ in pick) / sum(n for _, n in pick))
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def report(cases: list[dict], scores: dict[str, dict[str, dict]]) -> dict:
    known = [c for c in cases if c["kind"] == "known"]
    live = [c for c in cases if c["kind"] == "live" and all(c["id"] in scores.get(k, {}) for k in CHECKERS)]
    missing = sum(1 for c in cases if c["kind"] == "live") - len(live)
    per: dict[str, dict] = {}
    for k in CHECKERS:
        s = scores.get(k, {})
        known_got = [(c, check.verdict(c["forecast"], s[c["id"]]["p_yes"])) for c in known if c["id"] in s]
        live_verdicts = [check.verdict(c["forecast"], s[c["id"]]["p_yes"]) for c in live]
        seconds = [v["seconds"] for v in s.values()]
        per[k] = {
            "name": NAMES[k],
            "brier": (statistics.fmean((s[c["id"]]["p_yes"] - c["label"]) ** 2 for c in live) if live else None),
            "known": {
                "scored": len(known_got),
                "right": sum(got == c["expected"] for c, got in known_got),
                "inversions_caught": sum(got == "contradicts" for c, got in known_got if c["expected"] == "contradicts"),
                "inversions": sum(c["expected"] == "contradicts" for c in known),
                "false_flags": sum(got == "contradicts" for c, got in known_got if c["expected"] == "agrees"),
                "correct_forecasts": sum(c["expected"] == "agrees" for c in known),
            },
            "live_false_flags": live_verdicts.count("contradicts"),
            "live_unclear_share": (live_verdicts.count("unclear") / len(live)) if live else None,
            "median_seconds": statistics.median(seconds) if seconds else None,
        }
    diffs = [(c["market"], (scores["clm"][c["id"]]["p_yes"] - c["label"]) ** 2
              - (scores["jev2b"][c["id"]]["p_yes"] - c["label"]) ** 2) for c in live]
    markets = len({c["market"] for c in live})
    lo, hi = market_bootstrap(diffs) if diffs else (None, None)
    agree = (sum(check.verdict(c["forecast"], scores["clm"][c["id"]]["p_yes"])
                 == check.verdict(c["forecast"], scores["jev2b"][c["id"]]["p_yes"]) for c in live) / len(live)
             if live else None)
    clm, jev = per["clm"], per["jev2b"]
    tests = {
        "clm_brier_better": hi is not None and hi < 0,
        "known_cases_no_worse": (clm["known"]["scored"] == len(known) and jev["known"]["scored"] == len(known)
                                 and clm["known"]["inversions_caught"] >= jev["known"]["inversions_caught"]
                                 and clm["known"]["false_flags"] <= jev["known"]["false_flags"]),
        "live_false_flags_no_more": clm["live_false_flags"] <= jev["live_false_flags"],
        "fast_enough": clm["median_seconds"] is not None and clm["median_seconds"] <= MAX_MEDIAN_SECONDS,
    }
    if len(live) < MIN_LIVE or markets < MIN_MARKETS:
        verdict = "KEEP (too few to tell)"
    elif all(tests.values()):
        verdict = "SWITCH"
    else:
        verdict = "KEEP"
    return {
        "verdict": verdict,
        "rule": "docs/fund.md, Rationale checker: CLM-8B against Open-Jev 2B (pre-registered 2026-11-16)",
        "live_cases": len(live), "live_markets": markets, "live_cases_missing_a_score": missing,
        "brier_difference_clm_minus_jev2b": (statistics.fmean(d for _, d in diffs) if diffs else None),
        "brier_difference_ci95": [lo, hi],
        "tests": tests,
        "checkers": per,
        "description_only": {"verdicts_agree_share": agree},
        "cases": [{"id": c["id"], "kind": c["kind"], "market": c["market"], "forecast": round(c["forecast"], 4),
                   **({"label": c["label"]} if c["kind"] == "live" else {"expected": c["expected"]}),
                   **{f"p_yes_{k}": scores.get(k, {}).get(c["id"], {}).get("p_yes") for k in CHECKERS}}
                  for c in cases],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--store", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--cases", type=Path, required=True)
    s.add_argument("--checker", choices=CHECKERS, required=True)
    s.add_argument("--shard", default="0/1")
    s.add_argument("--emb-url", default="http://127.0.0.1:8090/v1/embeddings")
    s.add_argument("--head")
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--cases", type=Path, required=True)
    r.add_argument("--scores", type=Path, nargs="+", required=True)
    r.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    if args.cmd == "cases":
        cases = known_cases() + live_cases(args.store)
        args.out.write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        live = [x for x in cases if x["kind"] == "live"]
        print(f"known={len(cases) - len(live)} live={len(live)} markets={len({x['market'] for x in live})} "
              f"yes_labels={sum(x['label'] for x in live)}")
        return 0
    if args.cmd == "score":
        cases = shard_of(json.loads(args.cases.read_text(encoding="utf-8")), args.shard)
        t0 = time.perf_counter()
        if args.checker == "clm":
            if not args.head:
                parser.error("--head is required for clm")
            ask = clm_checker(args.emb_url, args.head)
            for q, r_ in (("Will it rain in Paris tomorrow?", "Every forecast service expects heavy rain in Paris "
                                                              "all day tomorrow, so rain is very likely."),
                          ("Will it rain in Paris tomorrow?", "A strong high-pressure system keeps Paris dry and "
                                                              "sunny tomorrow; rain is very unlikely.")):
                print(f"sanity p_yes {ask(q, r_):.3f} | {r_[:60]}", flush=True)
        else:
            ask = check.open_jev_checker(device="cpu")
        score(cases, ask, args.checker, args.out, load_seconds=time.perf_counter() - t0)
        return 0
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    result = report(cases, merge_scores(args.scores))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
