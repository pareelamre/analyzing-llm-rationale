#!/usr/bin/env python3
"""Hand the retired SCADS agents' open positions to the on-device agent -- run once.

Reads each retired agent's shadow store, merges their open positions by market and side,
prices them at today's bid (what they could be sold for), and books them into a new account
for the on-device model worth $10,000 in total: the positions at today's bid, the rest cash.
So the new account's return measures only its own decisions from here on.

A position in profit at today's bid moves over. A losing one moves only if the on-device
model, asked about that market now, still favours that side at today's price. A market with
no bid, or one that has closed, stays behind. Every decision goes into a JSON report.

Each transfer is booked as a ``trade`` with ``mode="transfer"`` and no fee, so cash, the
equity curve, settlement and the audit replay treat it like any opening trade; its metadata
names the agents it came from and their original entry prices.

Env:
  SELF_HOSTED_LLM_URL      the on-device model's chat-completions URL
  SELF_HOSTED_LLM_API_KEY  its key
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import benchmark_tools, market_data  # noqa: E402
from analyzing_llm_rationale.accounting import MarketQuote  # noqa: E402

RETIRED = ("deepseek-v4-flash", "gemma-4-26b-a4b-it", "glm-5-3", "glm-5-3-flash", "gpt-oss-120b",
           "llama-3.3-70b-instruct", "minimax-m3", "qwen3-8-27b")
ON_DEVICE = "qwen3.5-9b-selfhosted"
ACCOUNT_VALUE = 10_000.0
CYCLE_ID = "transfer-2026-11-08"
MODEL_TIMEOUT_S = 300

Quote = Callable[[str, str], Dict[str, Any]]
Judge = Callable[[Dict[str, Any], str, float], Optional[float]]


def retired_positions(store_dir: Path) -> Dict[tuple, Dict[str, Any]]:
    """Open positions across the retired stores, merged by (platform, ticker, side)."""
    merged: Dict[tuple, Dict[str, Any]] = {}
    for model in RETIRED:
        path = store_dir / model / "store.sqlite"
        if not path.exists():
            continue
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM agent_positions WHERE quantity > ?", (benchmark_tools.MIN_POSITION_QUANTITY,))
        for row in rows:
            key = (row["platform"], row["ticker"], row["side"])
            entry = merged.setdefault(key, {"quantity": 0.0, "cost_basis": 0.0, "holders": []})
            entry["quantity"] += float(row["quantity"])
            entry["cost_basis"] += float(row["cost_basis"])
            entry["holders"].append({"agent": model, "quantity": float(row["quantity"]),
                                     "avg_entry_price": float(row["avg_entry_price"])})
        conn.close()
    return merged


def fetch_quote(platform: str, ticker: str) -> Dict[str, Any]:
    quote = market_data.fetch_polymarket(slug=ticker) if platform == "polymarket" else market_data.fetch_kalshi(ticker)
    return {**quote, "platform": platform, "ident": ticker}


def _probability(text: str) -> Optional[float]:
    """The last number in [0, 1] in a reply (after any reasoning block), or None."""
    tail = text.rsplit("</think>", 1)[-1]
    for raw in reversed(re.findall(r"(?<![\d.])(?:0?\.\d+|0|1(?:\.0+)?)(?![\d.])", tail)):
        value = float(raw)
        if 0.0 <= value <= 1.0:
            return value
    return None


def ask_on_device_model(quote: Dict[str, Any], side: str, bid: float) -> Optional[float]:
    """The on-device model's probability that the market resolves YES, or None if it gave none."""
    url, key = os.environ.get("SELF_HOSTED_LLM_URL", ""), os.environ.get("SELF_HOSTED_LLM_API_KEY", "")
    if not url or not key:
        return None
    rules = str(quote.get("resolution_criteria") or quote.get("description") or "")[:1500]
    yes_price = quote.get("probability")
    price_line = f" The market prices YES at about {float(yes_price):.2f}." if isinstance(yes_price, (int, float)) else ""
    prompt = (
        f"Market: {quote.get('question')}\nResolution rules: {rules}\n"
        f"Today's date: {date.today().isoformat()}.{price_line}\n"
        "What is the probability that this market resolves YES? Reply with one number between 0 and 1 and nothing else."
    )
    try:
        resp = requests.post(url, timeout=MODEL_TIMEOUT_S, headers={"Authorization": f"Bearer {key}"}, json={
            "model": "mlx-community/Qwen3.5-9B-4bit", "max_tokens": 400, "temperature": 0,
            "messages": [{"role": "system", "content": "You are a careful forecaster. Answer with a probability only."},
                         {"role": "user", "content": prompt}],
        })
        resp.raise_for_status()
        return _probability(str(resp.json()["choices"][0]["message"]["content"] or ""))
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return None


def plan_transfer(positions: Dict[tuple, Dict[str, Any]], *, quote: Quote = fetch_quote,
                  judge: Judge = ask_on_device_model) -> List[Dict[str, Any]]:
    """One decision per merged position: keep (with its price) or leave, and why."""
    decisions = []
    for (platform, ticker, side), held in sorted(positions.items()):
        row = {"platform": platform, "ticker": ticker, "side": side, "quantity": round(held["quantity"], 6),
               "original_cost": round(held["cost_basis"], 4), "holders": held["holders"]}
        try:
            q = quote(platform, ticker)
        except Exception as exc:  # noqa: BLE001 - a missing market stays behind, with the reason
            decisions.append({**row, "keep": False, "reason": f"no quote ({type(exc).__name__})"})
            continue
        bid = MarketQuote.from_mapping(q).bid(side)
        row["question"] = q.get("question")
        if not isinstance(bid, (int, float)) or not math.isfinite(bid) or bid <= 0:
            decisions.append({**row, "keep": False, "reason": "no bid: unpriced or closed"})
            continue
        value = held["quantity"] * bid
        row.update({"bid": round(bid, 4), "value": round(value, 4), "pnl": round(value - held["cost_basis"], 4)})
        if value > held["cost_basis"]:
            decisions.append({**row, "keep": True, "reason": "in profit at today's bid"})
            continue
        p_yes = judge(q, side, bid)
        if p_yes is None:
            decisions.append({**row, "keep": False, "reason": "losing; the on-device model gave no probability"})
            continue
        p_side = p_yes if side.upper() == "YES" else 1 - p_yes
        keep = p_side > bid
        decisions.append({**row, "keep": keep, "model_p_yes": round(p_yes, 4),
                          "reason": f"losing; the model gives {side} {p_side:.0%} against a {bid:.0%} price"
                                    + ("" if keep else ", so it is left behind")})
    return decisions


def book_transfer(store_path: Path, decisions: List[Dict[str, Any]], *, model: str = ON_DEVICE) -> Dict[str, Any]:
    """Create the on-device account with the kept positions. Refuses to touch an existing account."""
    kept = [d for d in decisions if d["keep"]]
    invested = sum(d["value"] for d in kept)
    if invested >= ACCOUNT_VALUE:
        raise SystemExit(f"Kept positions are worth {invested:.2f}, more than the {ACCOUNT_VALUE:.0f} account")
    conn = sqlite3.connect(store_path)
    conn.row_factory = sqlite3.Row
    benchmark_tools._ensure_account_schema(conn)
    if conn.execute("SELECT 1 FROM agent_accounts WHERE agent_id = ?", (model,)).fetchone():
        raise SystemExit(f"{model} already has an account in {store_path}; the transfer runs only once")
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            "INSERT INTO agent_accounts (agent_id, starting_cash, cash, realized_pnl, fees_paid, settlement_fees_paid,"
            " high_watermark, updated_at) VALUES (?, ?, ?, 0, 0, 0, ?, ?)",
            (model, ACCOUNT_VALUE, ACCOUNT_VALUE - invested, ACCOUNT_VALUE, now),
        )
        for d in kept:
            benchmark_tools._upsert_position(conn, agent_id=model, platform=d["platform"], ticker=d["ticker"],
                                             side=d["side"], quantity=d["quantity"], cost_basis=d["value"])
            benchmark_tools._record_account_action(
                conn, agent_id=model, action_type="trade", cycle_id=CYCLE_ID, mode="transfer", submitted=False,
                platform=d["platform"], ticker=d["ticker"], side=d["side"], price=d["bid"], quantity=d["quantity"],
                notional=d["value"], fee=0.0, cash_required=d["value"], cash_delta=-d["value"], outcome="open",
                metadata={"transfer": {"from": d["holders"], "original_cost": d["original_cost"], "reason": d["reason"]}},
            )
    conn.close()
    return {"account_value": ACCOUNT_VALUE, "positions": len(kept), "invested": round(invested, 2),
            "cash": round(ACCOUNT_VALUE - invested, 2)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retired-dir", type=Path, required=True, help="Folder with <model>/store.sqlite per retired agent")
    parser.add_argument("--out", type=Path, required=True, help="The on-device agent's store to create")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Decide and report, but book nothing")
    args = parser.parse_args(argv)

    decisions = plan_transfer(retired_positions(args.retired_dir))
    summary = None if args.dry_run else book_transfer(args.out, decisions)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "model": ON_DEVICE, "dry_run": args.dry_run,
              "summary": summary, "decisions": decisions}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for d in decisions:
        print(f"{'KEEP ' if d['keep'] else 'LEAVE'} {d['platform']:10s} {d['ticker'][:50]:50s} {d['side']:3s} "
              f"qty {d['quantity']:>10.2f} bid {d.get('bid', '-')!s:>6} pnl {d.get('pnl', '-')!s:>9}  {d['reason']}")
    print("summary:", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
