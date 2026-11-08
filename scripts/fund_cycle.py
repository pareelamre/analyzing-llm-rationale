#!/usr/bin/env python3
"""Run one Foresea Fund cycle and publish static/fund_live.json (docs/fund.md).

Every run values the fund at current prices and republishes the page's payload. Trading
happens only inside FUND_RUN_HOURS_IST and at least FUND_MIN_HOURS_BETWEEN_CYCLES after the
last trading cycle (or with FUND_ANY_HOUR=1).

The first run opens the fund from the on-device agent's account (FUND_SEED_AGENT_STORE):
its cash, and each open position at the agent's own entry price, so the fund starts at the
same $10,000 the agent did and the agent's track continues inside the fund.

Inputs: the Edge board (static/mark_to_market_live.json), the pairs tracker
(static/commodity_pairs_live.json), the fund universe (static/fund_universe.json, rebuilt when
older than FUND_UNIVERSE_MAX_AGE_DAYS), live quotes, and USD/INR. Trade notes are worded by the
on-device model (FUND_LLM_URL + SELF_HOSTED_LLM_API_KEY) when it is reachable; otherwise each
note is the plain fact sheet.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import fund_ledger as ledger  # noqa: E402
from analyzing_llm_rationale import fund_notes, market_data  # noqa: E402
from analyzing_llm_rationale import fund_orchestrator as orch  # noqa: E402
from analyzing_llm_rationale import fund_universe as fu  # noqa: E402

FUND_ID = "foresea-fund"
IST = timezone(timedelta(hours=5, minutes=30))
STORE_PATH = Path(os.environ.get("FUND_STORE_PATH", ROOT / "data" / "fund_store.sqlite"))
PAYLOAD_PATH = ROOT / "static" / "fund_live.json"
UNIVERSE_PATH = ROOT / "static" / "fund_universe.json"
BOARD_PATH = ROOT / "static" / "mark_to_market_live.json"
PAIRS_PATH = ROOT / "static" / "commodity_pairs_live.json"
FX_URLS = (
    "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json",
    "https://latest.currency-api.pages.dev/v1/currencies/usd.json",
)
SEED_AGENT_ID = "qwen3.5-9b-selfhosted"
UNIVERSE_MAX_AGE_DAYS = int(os.environ.get("FUND_UNIVERSE_MAX_AGE_DAYS", "30"))
RUN_HOURS = {int(h) for h in os.environ.get("FUND_RUN_HOURS_IST", "1,2,13,14").split(",") if h.strip()}
MIN_HOURS_BETWEEN = float(os.environ.get("FUND_MIN_HOURS_BETWEEN_CYCLES", "10"))
RECENT_TRADES = 40


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def usd_inr() -> float:
    for url in FX_URLS:
        try:
            rate = float(requests.get(url, timeout=20).json()["usd"]["inr"])
            if rate > 0:
                return rate
        except (requests.RequestException, ValueError, KeyError, TypeError):
            continue
    raise SystemExit("no USD/INR rate; not valuing rupee positions at a guess")


def load_universe(today: date, fx: float) -> dict:
    cached = _load_json(UNIVERSE_PATH)
    try:
        fresh = (today - date.fromisoformat(cached["as_of"])).days < UNIVERSE_MAX_AGE_DAYS
    except (KeyError, ValueError):
        fresh = False
    if not fresh:
        cached = fu.universe_payload(fu.build_universe(today, fx))
        UNIVERSE_PATH.write_text(json.dumps(cached, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cached


def seed_from_agent(conn: sqlite3.Connection, agent_store: Path, now: str) -> dict:
    """Open the fund with the agent's cash and open positions at the agent's entry prices."""
    agent = sqlite3.connect(agent_store)
    agent.row_factory = sqlite3.Row
    account = agent.execute("SELECT * FROM agent_accounts WHERE agent_id = ?", (SEED_AGENT_ID,)).fetchone()
    if account is None:
        raise SystemExit(f"{agent_store} has no {SEED_AGENT_ID} account to seed the fund from")
    held = list(agent.execute("SELECT * FROM agent_positions WHERE agent_id = ? AND quantity > 0", (SEED_AGENT_ID,)))
    invested = sum(float(p["cost_basis"]) for p in held)
    ledger.open_fund(conn, FUND_ID, float(account["starting_cash"]), cash=float(account["cash"]) + invested, now=now)
    for p in held:
        platform = str(p["platform"] or "kalshi").lower()
        instrument = orch.event_instrument({"platform": platform, "ident": p["ticker"]})
        ledger.buy(conn, FUND_ID, instrument, str(p["side"]).upper(), float(p["quantity"]),
                   float(p["cost_basis"]) / float(p["quantity"]), attribution={"agent_transfer": 1.0},
                   action="transfer", cycle_id="fund-seed", now=now,
                   note=f"Carried over from the on-device agent's account at its entry price ({p['ticker']}).",
                   note_source="facts")
    return {"positions": len(held), "cash": float(account["cash"]), "invested": invested}


def event_quote(raw: dict) -> dict:
    """Bid and ask for both sides; Polymarket only quotes Yes, so No is its mirror."""
    yb, ya = raw.get("yes_bid"), raw.get("yes_ask")
    nb = raw.get("no_bid") if raw.get("no_bid") is not None else (1.0 - ya if ya is not None else None)
    na = raw.get("no_ask") if raw.get("no_ask") is not None else (1.0 - yb if yb is not None else None)
    return {"yes_bid": yb, "yes_ask": ya, "no_bid": nb, "no_ask": na}


def fetch_quotes(instrument_ids: set[str], universe: dict, today: date) -> dict[str, dict]:
    quotes: dict[str, dict] = {}
    perps = None
    known = {i["id"]: i for i in universe.get("indices", [])}
    for pair in universe.get("pairs", []):
        known[pair["a"]["id"]] = pair["a"]
        known[pair["b"]["id"]] = pair["b"]
    for iid in sorted(instrument_ids):
        venue, _, key = iid.partition(":")
        try:
            if venue == "kalshi":
                quotes[iid] = event_quote(market_data.fetch_kalshi(key))
            elif venue == "polymarket":
                quotes[iid] = event_quote(market_data.fetch_polymarket(slug=key))
            elif venue == "kalshi_perp":
                if perps is None:
                    perps = {m.get("ticker"): m for m in fu.kalshi_margin_markets()}
                if key in perps:
                    quotes[iid] = fu.kalshi_quote(perps[key])
            elif iid in known:
                candles = fu.daily_candles(known[iid]["key"], today - timedelta(days=7), today)
                if candles:
                    last = float(sorted(candles, key=lambda c: c[0])[-1][4])
                    quotes[iid] = {"bid": last, "ask": last, "last": last}
        except (market_data.MarketDataError, fu.FeedError, requests.RequestException, ValueError) as exc:
            print(f"  no quote for {iid}: {exc}", file=sys.stderr)
    return quotes


def note_writer() -> Optional[Callable[[str], str]]:
    url, key = os.environ.get("FUND_LLM_URL", "").strip(), os.environ.get("SELF_HOSTED_LLM_API_KEY", "").strip()
    if not url or not key:
        return None
    model = os.environ.get("FUND_LLM_MODEL", "mlx-community/Qwen3.5-9B-4bit")

    def call(prompt: str) -> str:
        resp = requests.post(url, timeout=180, headers={"Authorization": f"Bearer {key}"},
                             json={"model": model, "max_tokens": 220, "temperature": 0.0,
                                   "messages": [{"role": "user", "content": prompt}]})
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    return call


def _last_cycle(conn: sqlite3.Connection) -> Optional[datetime]:
    row = conn.execute("SELECT MAX(ts) FROM fund_trades WHERE fund_id = ? AND cycle_id LIKE 'fund-%'"
                       " AND cycle_id != 'fund-seed'", (FUND_ID,)).fetchone()
    return datetime.fromisoformat(row[0]) if row and row[0] else None


def trading_window(conn: sqlite3.Connection, now: datetime) -> tuple[bool, str]:
    if os.environ.get("FUND_ANY_HOUR") == "1":
        return True, "forced"
    hour = now.astimezone(IST).hour
    if hour not in RUN_HOURS:
        return False, f"{now.astimezone(IST):%H:%M} IST is outside the trading hours"
    last = _last_cycle(conn)
    if last and (now - last).total_seconds() < MIN_HOURS_BETWEEN * 3600:
        return False, f"last trading cycle {last:%H:%M} UTC, under {MIN_HOURS_BETWEEN:g} hours ago"
    return True, "in window"


def _sessions_since(stamp: str, today: date) -> int:
    start = datetime.fromisoformat(stamp).date()
    return sum(1 for d in range((today - start).days) if (start + timedelta(days=d + 1)).weekday() < 5)


def run(now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(IST).date()
    stamp = now.isoformat()
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(STORE_PATH)
    ledger.ensure_schema(conn)
    seeded = None
    if conn.execute("SELECT 1 FROM fund_account WHERE fund_id = ?", (FUND_ID,)).fetchone() is None:
        agent_store = Path(os.environ.get("FUND_SEED_AGENT_STORE", ""))
        if not agent_store.is_file():
            raise SystemExit("no fund yet and no FUND_SEED_AGENT_STORE to open it from")
        seeded = seed_from_agent(conn, agent_store, stamp)
        conn.commit()

    fx = usd_inr()
    rates = {"USD": 1.0, "INR": fx}
    universe = load_universe(today, fx)
    board = (_load_json(BOARD_PATH).get("edge_board") or [])
    tracker = _load_json(PAIRS_PATH)
    signals, skipped_events = orch.event_signals(board, now)
    pair_signals, skipped_pairs = orch.pair_signals(tracker, universe.get("pairs", []))
    signals += pair_signals
    forecasts = {orch.event_instrument(r)["id"]: r for r in board}

    held_ids = {p["instrument_id"] for p in ledger.positions(conn, FUND_ID)}
    hedges = {p["b"]["underlying"]: p["b"] for p in universe.get("pairs", []) if p["b"]["venue"] == "kalshi_perp"}
    wanted = held_ids | {s.instrument["id"] for s in signals} | {h["id"] for h in hedges.values()}
    quotes = fetch_quotes(wanted, universe, today)
    snapshot = ledger.nav(conn, FUND_ID, quotes, rates)

    can_trade, why = trading_window(conn, now)
    executed, passed = [], []
    if can_trade:
        held_pairs = {}
        for pos in snapshot["positions"]:
            if pos.get("pair_key"):
                held_pairs[pos["pair_key"]] = {"sessions": _sessions_since(pos["opened_at"], today)}
        pair_rows = {p.get("key"): p for p in tracker.get("pairs") or []}
        result = orch.plan(snapshot, signals, quotes, hedges, forecasts=forecasts, held_pairs=held_pairs,
                           pair_rows=pair_rows)
        passed = result["passed"]
        writer = note_writer()
        cycle_id = f"fund-{now:%Y%m%dT%H%M}"
        for order in result["orders"]:
            body = dataclasses.asdict(order)
            note = fund_notes.write_note(body, writer)
            rate = rates.get(order.instrument.get("currency", "USD"), 1.0)
            try:
                if order.action == "buy":
                    ledger.buy(conn, FUND_ID, order.instrument, order.side, order.quantity, order.price, fx=rate,
                               attribution=order.attribution, purpose=order.purpose, pair_key=order.pair_key,
                               cycle_id=cycle_id, now=stamp, **note)
                else:
                    held = next(p for p in snapshot["positions"]
                                if p["instrument_id"] == order.instrument["id"] and p["side"] == order.side)
                    ledger.sell(conn, FUND_ID, order.instrument["id"], order.side, order.quantity, order.price,
                                fx=rates.get(held.get("currency", "USD"), rate), purpose=order.purpose,
                                pair_key=order.pair_key, cycle_id=cycle_id, now=stamp, **note)
                executed.append({**body, **note})
            except (ledger.LedgerError, StopIteration) as exc:
                passed.append({"instrument": order.instrument["id"], "reason": str(exc)})
        conn.commit()
        snapshot = ledger.nav(conn, FUND_ID, quotes, rates)

    ledger.record_nav(conn, snapshot, now=stamp)
    conn.commit()
    payload = build_payload(conn, snapshot, universe, now=now, window=why, traded=can_trade, executed=executed,
                            passed=passed, skipped=skipped_events + skipped_pairs, signals=signals, seeded=seeded)
    PAYLOAD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    conn.close()
    return payload


def build_payload(conn: sqlite3.Connection, snapshot: dict, universe: dict, *, now: datetime, window: str,
                  traded: bool, executed: list, passed: list, skipped: list, signals: list,
                  seeded: Optional[dict]) -> dict:
    trades = [dict(r) for r in conn.execute(
        "SELECT ts, cycle_id, instrument_id, side, action, purpose, pair_key, quantity, price, notional_usd,"
        " realized_pnl_usd, attribution_json, note, note_source, evidence_json FROM fund_trades WHERE fund_id = ?"
        " ORDER BY ts DESC LIMIT ?", (FUND_ID, RECENT_TRADES))]
    for t in trades:
        t["attribution"] = json.loads(t.pop("attribution_json") or "{}")
        t["evidence"] = json.loads(t.pop("evidence_json") or "[]")
    curve = [{"ts": r[0], "nav_usd": r[1]} for r in conn.execute(
        "SELECT ts, nav_usd FROM fund_nav WHERE fund_id = ? ORDER BY ts", (FUND_ID,))]
    analysts = sorted({s.analyst for s in signals} | {"event_model", "pairs_model"})
    return {
        "generated_at": now.isoformat(), "fund_id": FUND_ID, "paper": True,
        "nav_usd": snapshot["nav_usd"], "cash_usd": snapshot["cash_usd"], "starting_value": snapshot["starting_value"],
        "return_pct": round((snapshot["nav_usd"] / snapshot["starting_value"] - 1.0) * 100, 3),
        "realized_pnl_usd": snapshot["realized_pnl_usd"], "positions": snapshot["positions"],
        "by_asset_class": snapshot["by_asset_class"], "net_exposure_by_underlying": snapshot["net_exposure_by_underlying"],
        "attribution": {"unrealized": snapshot["unrealized_pnl_by_analyst"],
                        "realized": snapshot["realized_pnl_by_analyst"]},
        "curve": curve, "trades": trades,
        "cycle": {"traded": traded, "window": window, "orders": len(executed), "passed": passed,
                  "skipped": skipped, "signals": len(signals), "seeded": seeded},
        "analysts": analysts,
        "limits": {"position": orch.POSITION_CAP, "asset_class": orch.CLASS_CAP, "underlying": orch.UNDERLYING_CAP,
                   "cash_floor": orch.CASH_FLOOR, "kelly_fraction": orch.KELLY_FRACTION},
        "universe": {"as_of": universe.get("as_of"), "indices": [i["label"] for i in universe.get("indices", [])],
                     "pairs": [p["label"] for p in universe.get("pairs", [])]},
    }


if __name__ == "__main__":
    out = run()
    print(json.dumps({k: out[k] for k in ("nav_usd", "cash_usd", "return_pct", "cycle")}, indent=2, default=str))
