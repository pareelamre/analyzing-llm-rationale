#!/usr/bin/env python3
"""Run one Foresea Fund cycle and publish static/fund_live.json (docs/fund.md).

Every run values the fund at current prices and republishes the page's payload. Trading
happens only inside FUND_RUN_HOURS_IST and at least FUND_MIN_HOURS_BETWEEN_CYCLES after the
last trading cycle (or with FUND_ANY_HOUR=1), and each venue only while it is open, on a
price from its session under way (fund_hours: NSE and BSE derivatives 09:15-15:30 IST, MCX
09:00-23:30/23:55 IST, on their trading days; Kalshi and Polymarket at any hour).

Owner, 2026-11-23: trade at the start of the exchanges' trading window and check once at its
end. So the day has three slots: the night run (01:00-02:59 IST, after the nightly forecast:
prediction markets and Kalshi perps), the open (09:00-10:59 IST: everything, NSE and MCX at
that session's first prices), and the end-of-day check (FUND_CHECK_HOURS_IST, 14:00-15:59 IST:
exits, owner trims and hedges only, no new positions).

The first run opens the fund from the on-device agent's account (FUND_SEED_AGENT_STORE):
its cash, and each open position at the agent's own entry price, so the fund starts at the
same $10,000 the agent did and the agent's track continues inside the fund.

Inputs: the Edge board (static/mark_to_market_live.json), the pairs tracker
(static/commodity_pairs_live.json), the index model (static/index_forecast_live.json, for the
options analyst), the fund universe (static/fund_universe.json, rebuilt when older than
FUND_UNIVERSE_MAX_AGE_DAYS), live quotes, and USD/INR. Option prices come from the live order
book when UPSTOX_ANALYTICS_TOKEN is set and from Upstox's open last trades otherwise
(fund_options). Prediction markets trade on the blend analyst's odds (market and model,
configs/fund_blend.json), not the raw model, and a forecast the rationale checker says
contradicts its own rationale never reaches the blend (guard_board); the weather,
decided-ranges, options and VIX fade analysts are retired (RETIRED, docs/fund.md). Option structures past expiry are settled
at the index's close, and Kalshi positions at their market's result, every run. Kalshi buys pay
Kalshi's taker fee. Trade
notes are worded by the on-device model (FUND_LLM_URL + SELF_HOSTED_LLM_API_KEY) when it is
reachable; otherwise each note is the plain fact sheet. When Foresea's knowledge base is
checked out (KNOWLEDGE_DIR), each commodity pair trade also carries what the newest report's
price forecasts imply for the pair, and whether that agrees with the trade (fund_knowledge;
evidence only, sizes unchanged).
"""
from __future__ import annotations

import dataclasses
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import fund_hours as fh  # noqa: E402
from analyzing_llm_rationale import fund_knowledge as fk  # noqa: E402
from analyzing_llm_rationale import fund_ledger as ledger  # noqa: E402
from analyzing_llm_rationale import (  # noqa: E402
    fund_notes,
    fund_risk,  # noqa: E402
    market_data,
)
from analyzing_llm_rationale import fund_options as fo  # noqa: E402
from analyzing_llm_rationale import fund_orchestrator as orch  # noqa: E402
from analyzing_llm_rationale import fund_scorecard as fsc  # noqa: E402
from analyzing_llm_rationale import fund_universe as fu  # noqa: E402
from analyzing_llm_rationale import fund_weather as fw  # noqa: E402

FUND_ID = "foresea-fund"
IST = timezone(timedelta(hours=5, minutes=30))
STORE_PATH = Path(os.environ.get("FUND_STORE_PATH", ROOT / "data" / "fund_store.sqlite"))
PAYLOAD_PATH = ROOT / "static" / "fund_live.json"
UNIVERSE_PATH = ROOT / "static" / "fund_universe.json"
BOARD_PATH = ROOT / "static" / "mark_to_market_live.json"
PAIRS_PATH = ROOT / "static" / "commodity_pairs_live.json"
INDEX_FORECAST_PATH = ROOT / "static" / "index_forecast_live.json"
PUT_SLEEVE_REPORT_PATH = ROOT / "reports" / "fund_ndx_put_eval.json"
#: Bank Nifty's own test (docs/fund.md, "Bank Nifty put spreads"), on top of the sleeve's.
BANKNIFTY_SLEEVE_REPORT_PATH = ROOT / "reports" / "fund_banknifty_put_eval.json"
#: The put premium sleeve's analyst (docs/fund.md, "Put premium sleeve").
PUT_SLEEVE = "put_premium"
BLEND_PATH = ROOT / "configs" / "fund_blend.json"
#: Owner instructions carried out at the next trading cycle (trims to a risk limit).
OWNER_ACTIONS_PATH = ROOT / "configs" / "fund_owner_actions.json"
CORRECTIONS_PATH = ROOT / "configs" / "fund_corrections.json"
#: Trades alone (at the marks the cycle used) should barely move the fund's value: fees and the
#: gap between a fill and its mark. A bigger move is an accounting error, and the cycle trades nothing.
ACCOUNTING_TOLERANCE = float(os.environ.get("FUND_ACCOUNTING_TOLERANCE", "0.02"))
CHECKS_PATH = ROOT / "static" / "fund_rationale_checks.json"
#: Rationale checks older than this no longer describe a live forecast.
CHECKS_MAX_AGE_H = 48
#: The rationale checker is a live guard (docs/fund.md, "Rationale checker goes live", owner 2026-11-16).
CHECKER_LABEL = "Open-Jev 2B (Qwen3.5-2B base)"
GUARD_REASON = "its forecast points away from its own rationale (rationale checker), so the market prices it alone"
#: Retired by the owner on 2026-11-15 (docs/fund.md, "Shadow cleanup"): each failed its
#: pre-registered test or could never reach a verdict. They no longer run, record or trade.
RETIRED = (
    {"analyst": "event_model", "reason": "the on-device model's raw odds trail the market on Foresea's record; "
                                         "prediction markets now trade on the blend of market and model"},
    {"analyst": "weather_model", "reason": "lost money in every weather test: Kalshi and Polymarket daily highs, "
                                           "and Kalshi's hourly markets"},
    {"analyst": "decided_ranges", "reason": "none of 1,141 already-decided ranges could be bought, so its "
                                            "100-fill verdict could never come"},
    {"analyst": "options_model", "reason": "index straddles and condors failed their backtest; the put side "
                                           "trades as the put premium sleeve"},
    {"analyst": "vix_fade", "reason": "about 12 signals a year put its 50-trade verdict years away, and no "
                                      "same-night VIX price exists to trade it"},
)
FX_URLS = (
    "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json",
    "https://latest.currency-api.pages.dev/v1/currencies/usd.json",
)
SEED_AGENT_ID = "qwen3.5-9b-selfhosted"
UNIVERSE_MAX_AGE_DAYS = int(os.environ.get("FUND_UNIVERSE_MAX_AGE_DAYS", "30"))
RUN_HOURS = {int(h) for h in os.environ.get("FUND_RUN_HOURS_IST", "1,2,9,10,14,15").split(",") if h.strip()}
#: The end-of-day check's hours: exits and risk only (plan(entries=False)).
CHECK_HOURS = {int(h) for h in os.environ.get("FUND_CHECK_HOURS_IST", "14,15").split(",") if h.strip()}
#: Keeps each slot to one trading cycle; the slots themselves are at least four hours apart.
MIN_HOURS_BETWEEN = float(os.environ.get("FUND_MIN_HOURS_BETWEEN_CYCLES", "3"))
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


def _legs(universe: dict) -> list[dict]:
    return list(universe.get("indices") or []) + [p[s] for p in universe.get("pairs") or [] for s in ("a", "b")]


def contract_expiries(universe: dict) -> dict[str, str]:
    """Upstox key -> expiry of every contract this universe or an earlier one listed (kept 120 days),
    so a held contract's expiry is known after the universe moves on to the next one."""
    out = dict(universe.get("expiries") or {})
    out.update({leg["key"]: leg["expiry"] for leg in _legs(universe) if leg.get("key") and leg.get("expiry")})
    return out


def load_universe(today: date, fx: float) -> dict:
    """The cached universe, rebuilt every UNIVERSE_MAX_AGE_DAYS and whenever one of its futures is
    inside fu.ROLL_DAYS of expiry, so the fund trades and rolls into the next contract in time."""
    cached = _load_json(UNIVERSE_PATH)
    try:
        fresh = (today - date.fromisoformat(cached["as_of"])).days < UNIVERSE_MAX_AGE_DAYS
        fresh = fresh and not any(leg.get("expiry") and date.fromisoformat(leg["expiry"]) < today + timedelta(days=fu.ROLL_DAYS)
                                  for leg in _legs(cached))
    except (KeyError, ValueError):
        fresh = False
    if not fresh:
        old = contract_expiries(cached) if cached else {}
        cached = fu.universe_payload(fu.build_universe(today, fx))
        horizon = (today - timedelta(days=120)).isoformat()
        cached["expiries"] = {k: v for k, v in {**old, **contract_expiries(cached)}.items() if v >= horizon}
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
        try:
            quote = (market_data.fetch_polymarket(slug=p["ticker"]) if platform == "polymarket"
                     else market_data.fetch_kalshi(p["ticker"]))
        except (market_data.MarketDataError, requests.RequestException):
            quote = {}
        instrument = orch.event_instrument({"platform": platform, "ident": p["ticker"],
                                            "question": quote.get("question")})
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


def fetch_quotes(instrument_ids: set[str], universe: dict, now: datetime,
                 keys: Optional[dict[str, str]] = None) -> dict[str, dict]:
    """Quotes for the fund's instruments. An Upstox-listed contract (MCX, an index) is priced at
    its latest trade: this session's 30-minute bar while it trades, else its last daily close,
    with the time of that price (``at``), so the fund never trades on a stale close
    (fund_hours). ``keys`` prices a held future on its own contract until it is rolled. A
    Kalshi perp's quote carries whether its market is open."""
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
                    schedule = perps[key].get("schedule") or {}
                    quotes[iid] = {**fu.kalshi_quote(perps[key]), "open": schedule.get("is_open", True) is not False}
            elif iid in known or iid in (keys or {}):
                candle = fo.latest_candle((keys or {}).get(iid) or known[iid]["key"], now, http_get=requests.get)
                if candle is not None and candle[1] > 0:
                    at, last, _volume = candle
                    quotes[iid] = {"bid": last, "ask": last, "last": last, "at": at.isoformat(), "source": "last trade"}
        except (market_data.MarketDataError, fu.FeedError, requests.RequestException, ValueError) as exc:
            print(f"  no quote for {iid}: {exc}", file=sys.stderr)
    return quotes


def roll_contracts(conn: sqlite3.Connection, universe: dict, quotes: dict, now: datetime, stamp: str,
                   rates: dict, calendar) -> tuple[list[dict], list[dict]]:
    """Move held futures onto the universe's current contract before theirs expires.

    Each is sold at its own contract's price from this session (its last close once it has
    expired) and bought again on the new contract for the same rupee value: same side, pair,
    attribution and opening date. A pair's legs roll together or wait together. ``quotes`` then
    price the position on its new contract. Returns (trades, positions that wait, with why).
    """
    current = {leg["id"]: leg for leg in _legs(universe) if leg.get("venue") in fh.SESSIONS and leg.get("key")}
    expiries = contract_expiries(universe)
    today = now.astimezone(IST).date()
    due = [dict(p) for p in ledger.positions(conn, FUND_ID) if p["kind"] == "linear"
           and p["instrument_id"] in current and p.get("key") and p["key"] != current[p["instrument_id"]]["key"]]
    groups: dict[str, list[dict]] = {}
    for p in due:
        groups.setdefault(p.get("pair_key") or p["instrument_id"], []).append(p)
    done, waiting = [], []
    for _group, held in groups.items():
        legs, why = [], None
        for p in held:
            new = current[p["instrument_id"]]
            venue = p["instrument_id"].split(":", 1)[0]
            old_q, new_q = (fo.latest_candle(k, now, http_get=requests.get) for k in (p["key"], new["key"]))
            if new_q is None or not fh.live({"at": new_q[0].isoformat()}, venue, now, calendar):
                why = f"waiting to roll: no price from this session yet on {new.get('contract') or new['key']}"
                break
            expired = (expiries.get(p["key"]) or "9999") < today.isoformat()
            if old_q is None or not (expired or fh.live({"at": old_q[0].isoformat()}, venue, now, calendar)):
                why = "waiting to roll: no price from this session yet on the contract held"
                break
            legs.append((p, new, old_q, new_q))
        if why:
            waiting += [{"instrument": p["instrument_id"], "label": p["label"], "reason": why} for p in held]
            continue
        for p, new, old_q, new_q in legs:
            fx = rates.get(p["currency"], 1.0)
            old_price, new_price = float(old_q[1]), float(new_q[1])
            note = (f"Rolled from {p['key']} to {new.get('contract') or new['key']} before expiry "
                    f"({p['side'].lower()} the same rupee value: {old_price:,.2f} -> {new_price:,.2f}).")
            ledger.sell(conn, FUND_ID, p["instrument_id"], p["side"], p["quantity"], old_price, fx=fx,
                        purpose="roll", pair_key=p.get("pair_key"), cycle_id=f"fund-{now:%Y%m%dT%H%M}",
                        note=note, now=stamp)
            quantity = p["quantity"] * old_price / new_price
            ledger.buy(conn, FUND_ID, new, p["side"], quantity, new_price, fx=fx,
                       attribution=json.loads(p["attribution_json"]), purpose="roll", pair_key=p.get("pair_key"),
                       cycle_id=f"fund-{now:%Y%m%dT%H%M}", note=note, action="open", now=stamp)
            conn.execute("UPDATE fund_positions SET opened_at = ? WHERE fund_id = ? AND instrument_id = ? AND side = ?",
                         (p["opened_at"], FUND_ID, p["instrument_id"], p["side"]))
            quotes[p["instrument_id"]] = {"bid": new_price, "ask": new_price, "last": new_price,
                                          "at": new_q[0].isoformat(), "source": "last trade"}
            done.append({"instrument": {"id": p["instrument_id"], "label": p["label"]}, "side": p["side"],
                         "action": "roll", "purpose": "roll", "quantity": quantity, "price": new_price,
                         "pair_key": p.get("pair_key"), "reason": note})
    return done, waiting


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
    if is_check(now):
        return True, "end-of-day check: exits and risk only, no new positions"
    return True, "in window"


def is_check(now: datetime) -> bool:
    """Whether this run is the end-of-day check, which opens no new positions."""
    return os.environ.get("FUND_ANY_HOUR") != "1" and now.astimezone(IST).hour in CHECK_HOURS


def settle_expired(conn: sqlite3.Connection, specs: dict[str, dict], now: datetime, stamp: str,
                   rates: dict[str, float]) -> list[dict]:
    """Book every option structure past expiry at its value at the index's close that day."""
    settled = []
    held = {p["instrument_id"]: p for p in ledger.positions(conn, FUND_ID)}
    for iid, spec in list(specs.items()):
        pos = held.get(iid)
        if pos is None or not fo.expired(spec, now):
            continue
        level = fo.settlement_level(spec)
        if level is None:
            continue
        value = fo.structure_of(spec).payoff(level)
        note = (f"Expired: {spec.get('label') or iid}. {spec.get('index')} closed at {level:,.2f} on "
                f"{spec['expiry']}, so the structure paid ₹{value:,.2f} per unit.")
        ledger.sell(conn, FUND_ID, iid, pos["side"], pos["quantity"], value, fx=rates.get(pos["currency"], 1.0),
                    purpose="expiry", action="expire", cycle_id=f"expiry-{spec['expiry']}", now=stamp, note=note,
                    note_source="facts", evidence=[{"source": "Upstox daily close", "figure": f"{spec.get('index')}: {level:,.2f}"}])
        settled.append({"instrument": iid, "label": spec.get("label"), "level": level, "value": value})
        del specs[iid]
    return settled


def settle_events(conn: sqlite3.Connection, now: datetime, stamp: str) -> list[dict]:
    """Book every Kalshi position whose market has settled: a dollar a contract for the side
    that won, nothing for the other."""
    settled = []
    for pos in ledger.positions(conn, FUND_ID):
        venue, _, ticker = pos["instrument_id"].partition(":")
        if pos["kind"] != "binary" or venue != "kalshi":
            continue
        try:
            result = market_data.fetch_kalshi_result(ticker)
        except (market_data.MarketDataError, requests.RequestException, ValueError):
            continue
        if result is None:
            continue
        value = 1.0 if pos["side"].lower() == result else 0.0
        ledger.sell(conn, FUND_ID, pos["instrument_id"], pos["side"], pos["quantity"], value, purpose="settlement",
                    action="settle", cycle_id=f"settle-{now:%Y%m%d}", now=stamp, note_source="facts",
                    note=f"Settled: {pos.get('label') or ticker} resolved {result.capitalize()}, so {pos['side']} "
                         f"paid ${value:.2f} a contract.",
                    evidence=[{"source": "Kalshi", "figure": f"result: {result}"}])
        settled.append({"instrument": pos["instrument_id"], "label": pos.get("label"), "result": result,
                        "side": pos["side"], "value": value})
    return settled


def _instant(stamp: object) -> Optional[datetime]:
    try:
        then = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return then if then.tzinfo else then.replace(tzinfo=timezone.utc)


def guard_board(board: list) -> tuple[list, dict]:
    """The Edge board without the forecasts the rationale checker says contradict their own
    rationale, and the guard's summary (docs/fund.md, "Rationale checker goes live").

    A check counts only for the forecast it read (same market, same forecast time), so a newer
    forecast is never dropped for an older one's rationale. A forecast not yet checked is used
    as it is and counted as unchecked.
    """
    checks = {}
    for c in _load_json(CHECKS_PATH).get("checks") or []:
        when = _instant(c.get("forecasted_at"))
        if when is not None:
            checks[(c.get("instrument_id"), when)] = c
    kept, dropped = [], []
    unchecked = 0
    for row in board:
        when = _instant(row.get("forecasted_at"))
        check = checks.get((orch.event_instrument(row)["id"], when)) if when is not None else None
        if check is None:
            unchecked += 1
            kept.append(row)
        elif check.get("verdict") == "contradicts":
            dropped.append({k: row.get(k) for k in ("platform", "ident", "question", "model_probability",
                                                   "market_bid", "market_ask", "market_probability",
                                                   "forecasted_at")} | {"rationale_yes": check.get("rationale_yes")})
        else:
            kept.append(row)
    return kept, {"mode": "live", "kind": "checker", "checker": CHECKER_LABEL, "board": len(board),
                  "checked": len(board) - unchecked, "unchecked": unchecked, "dropped": dropped}


def prediction_market_signals(board: list, now: datetime) -> tuple[list, list, dict, dict]:
    """Signals, skips and exit forecasts for prediction markets, all from the blend analyst's odds,
    and the rationale guard's summary.

    The raw on-device model trails the market on Foresea's record, so it no longer trades
    (owner, 2026-11-15); with no blend parameters, nothing trades in prediction markets. A
    forecast that contradicts its own rationale never reaches the blend (owner, 2026-11-16).
    """
    board, guard = guard_board(board)
    dropped = [{"instrument": orch.event_instrument(r)["id"], "label": str(r.get("question") or r.get("ident")),
                "reason": GUARD_REASON} for r in guard["dropped"]]
    params = _load_json(BLEND_PATH)
    if not params.get("coef"):
        return [], dropped + [{"instrument": "event", "label": "Prediction markets",
                               "reason": "no blend parameters"}], {}, guard
    priced = orch.blend_board(board, params)
    signals, skipped = orch.event_signals(priced, now, analyst="blend_model", model_label="the blend analyst")
    return signals, dropped + skipped, {orch.event_instrument(r)["id"]: r for r in priced}, guard


def put_sleeve_mode() -> str:
    """The put premium sleeve trades only while its out-of-sample check says GO (docs/fund.md)."""
    return "GO" if (_load_json(PUT_SLEEVE_REPORT_PATH).get("put_spread") or {}).get("verdict") == "GO" else "OFF"


def put_sleeve_modes() -> dict[str, str]:
    """Each index the sleeve may sell on: Nifty while the sleeve's check says GO, Bank Nifty while
    its own test says GO as well."""
    base = put_sleeve_mode()
    bank = (_load_json(BANKNIFTY_SLEEVE_REPORT_PATH).get("put_spread") or {}).get("verdict") == "GO"
    return {"NIFTY": base, "BANKNIFTY": base if bank else "OFF"}


def sleeve_symbol(instrument_id: str) -> str:
    """The index of a sleeve spread, from its id ``nse_fo:{symbol}:...``."""
    parts = instrument_id.split(":")
    return parts[1] if len(parts) > 1 else ""


def options_cycle(index_forecast: dict, now: datetime, quote_fn, price_source: str, held: dict, *,
                  sleeve: Sequence[str] = ()) -> dict:
    """The options analyst's signals, values and quotes, and the put premium sleeve's spread on each
    index in ``sleeve``, or why there were none."""
    empty = {"signals": [], "skipped": [], "values": {}, "quotes": {}, "sleeve": []}
    if not index_forecast.get("indices"):
        return empty
    try:
        rows = fu.upstox_options("NSE")
    except (fu.FeedError, requests.RequestException, ValueError) as exc:
        return {**empty, "skipped": [{"instrument": "nse_fo", "label": "Index options", "reason": f"no option list: {exc}"}]}
    out = fo.option_signals(index_forecast, rows, now, quote_fn=quote_fn, price_source=price_source, held=held)
    out["sleeve"] = [fo.put_spread_signal(index_forecast, rows, now, quote_fn=quote_fn, price_source=price_source,
                                          symbol=symbol) for symbol in sleeve]
    return out


def _scale(card: dict, analyst: str) -> float:
    """The scorecard's size multiple for ``analyst`` (1 when it is not scored)."""
    return next((float(r["scale"]) for r in card.get("strategies") or [] if r["analyst"] == analyst), 1.0)


def held_pair_entries(positions: list[dict], universe_pairs: list[dict], today: date) -> dict[str, dict]:
    """Each held pair: sessions held, and the side it holds on the price ratio (+1 when the
    first leg is long or the second short), which relative-value exits need."""
    first_legs = {p["key"]: p["a"]["id"] for p in universe_pairs}
    out: dict[str, dict] = {}
    for pos in positions:
        if not pos.get("pair_key"):
            continue
        entry = out.setdefault(pos["pair_key"], {"sessions": _sessions_since(pos["opened_at"], today)})
        if pos["pair_key"] in first_legs and pos.get("side") in ("LONG", "SHORT"):
            on_first = pos["instrument_id"] == first_legs[pos["pair_key"]]
            entry.setdefault("side", (1 if pos["side"] == "LONG" else -1) * (1 if on_first else -1))
    return out


def _sessions_since(stamp: str, today: date) -> int:
    start = datetime.fromisoformat(stamp).date()
    return sum(1 for d in range((today - start).days) if (start + timedelta(days=d + 1)).weekday() < 5)


def accounting_check(value_before: float, value_after: float) -> Optional[str]:
    """Why a cycle's trades must be undone, or None. Trading at the cycle's own marks barely moves
    the fund's value (fees, fills off the mark); a bigger move is a booking error, such as a rupee
    sale counted in dollars (2026-11-25)."""
    if abs(value_after - value_before) <= ACCOUNTING_TOLERANCE * max(value_before, 1.0):
        return None
    return (f"accounting check: this cycle's trades would move the fund's value from ${value_before:,.2f} "
            f"to ${value_after:,.2f}; nothing was traded")


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
    # One-time repairs of trades booked wrongly (configs/fund_corrections.json), each applied once.
    for fix in _load_json(CORRECTIONS_PATH).get("corrections") or []:
        done = ledger.reprice_trade(conn, FUND_ID, fix, now=stamp)
        if done:
            print(f"  corrected {done['instrument_id']} at {done['trade_ts']}: ${done['delta_usd']:+,.2f}", file=sys.stderr)
    conn.commit()

    fx = usd_inr()
    rates = {"USD": 1.0, "INR": fx}
    option_specs = {p["instrument_id"]: ledger.instrument_spec(conn, FUND_ID, p["instrument_id"]) or {}
                    for p in ledger.positions(conn, FUND_ID) if p["kind"] == "option"}
    settled = settle_expired(conn, option_specs, now, stamp, rates)
    settled_events = settle_events(conn, now, stamp)
    conn.commit()
    # The scorecard sizes every analyst by its proven edge, counting positions closed just now.
    card = fsc.scorecard(conn, FUND_ID, ROOT)
    quote_fn, price_source = fo.quote_source(os.environ.get("UPSTOX_ANALYTICS_TOKEN", "").strip() or None, now=now)
    # Each venue trades only while it is open, on a price from that session (docs/fund.md, "Trading hours").
    calendar = fh.fetch_calendar(today)
    hours = {venue: fh.is_open(venue, now, calendar) for venue in fh.SESSIONS}
    universe = load_universe(today, fx)
    board = (_load_json(BOARD_PATH).get("edge_board") or [])
    tracker = _load_json(PAIRS_PATH)
    signals, skipped_events, forecasts, guard = prediction_market_signals(board, now)
    pair_signals, skipped_pairs = orch.pair_signals(tracker, universe.get("pairs", []))
    # What the knowledge base's newest price forecasts imply for each pair: evidence, not sizing.
    knowledge_views = fk.annotate(pair_signals, universe.get("pairs", []), today)
    signals += pair_signals

    held_ids = {p["instrument_id"] for p in ledger.positions(conn, FUND_ID)}
    hedges = {p["b"]["underlying"]: p["b"] for p in universe.get("pairs", []) if p["b"]["venue"] == "kalshi_perp"}
    wanted = held_ids | {s.instrument["id"] for s in signals} | {h["id"] for h in hedges.values()}
    held_keys = {p["instrument_id"]: p["key"] for p in map(dict, ledger.positions(conn, FUND_ID))
                 if p["kind"] == "linear" and p.get("key")}
    quotes = fetch_quotes(wanted, universe, now, keys=held_keys)
    quotes.update(fo.structure_quotes({k: v for k, v in option_specs.items() if v.get("legs")}, quote_fn))
    snapshot = ledger.nav(conn, FUND_ID, quotes, rates)

    can_trade, why = trading_window(conn, now)
    executed, passed = [], []
    rolled, roll_waiting = [], []
    if can_trade:
        # Futures near expiry move to the next contract first (fund_universe.ROLL_DAYS).
        rolled, roll_waiting = roll_contracts(conn, universe, quotes, now, stamp, rates, calendar)
        if rolled:
            conn.commit()
            snapshot = ledger.nav(conn, FUND_ID, quotes, rates)
    options = {"signals": [], "skipped": [], "values": {}, "quotes": {}, "sleeve": []}
    option_skips: list = []
    sleeve_modes = put_sleeve_modes()
    # The sleeve holds one spread per index at a time, to expiry: its spreads never take the
    # options analyst's early exits, which the tested rule did not have.
    sleeve_ids = {p["instrument_id"] for p in snapshot["positions"] if PUT_SLEEVE in (p.get("attribution") or {})}
    sleeve_wanted = [sym for sym in fo.SLEEVE_SYMBOLS
                     if sleeve_modes.get(sym) == "GO" and sym not in {sleeve_symbol(i) for i in sleeve_ids}]
    if can_trade:
        nse_open, nse_closed = hours["nse_fo"]
        if nse_open:
            options = options_cycle(_load_json(INDEX_FORECAST_PATH), now, quote_fn, price_source, option_specs,
                                    sleeve=sleeve_wanted)
            quotes.update(options["quotes"])
        else:
            option_skips = [{"instrument": f"nse_fo:{sym}", "label": f"{fo.SLEEVE_NAMES[sym]} put premium",
                             "reason": nse_closed} for sym in sleeve_wanted]
        # The options analyst is retired: only a failed option list and the sleeve's own skips are reported.
        option_skips += [s for s in options["skipped"] if s.get("instrument") == "nse_fo"]
        for spread in options.get("sleeve") or []:
            if spread["signal"]:
                signals.append(spread["signal"])
                quotes.update(spread["quotes"])
            else:
                option_skips.append(spread["skipped"])
        held_pairs = held_pair_entries(snapshot["positions"], universe.get("pairs", []), today)
        tracker_rows = {p.get("key"): p for p in tracker.get("pairs") or []}
        pair_rows = {key: tracker_rows.get(orch.tracker_key(key)) or {} for key in held_pairs}
        rv_rule = (tracker.get("relative_value") or {}).get("rule") if tracker.get("relative_value") else None
        signals, unfunded = fsc.apply(signals, card)
        option_skips += unfunded
        signals, closed = fh.gate(signals, quotes, now, calendar)
        option_skips += closed
        result = orch.plan(snapshot, signals, quotes, hedges, forecasts=forecasts, held_pairs=held_pairs,
                           pair_rows=pair_rows, rates=rates, rv_rule=rv_rule, entries=not is_check(now),
                           option_values={k: v for k, v in options["values"].items() if k not in sleeve_ids},
                           owner_trims=_load_json(OWNER_ACTIONS_PATH).get("trims") or [],
                           owner_holds=_load_json(OWNER_ACTIONS_PATH).get("holds") or [],
                           risk_trims=os.environ.get("FUND_RISK_TRIMS", "1") == "1")
        # Exits and hedges wait for their venue too; a pair's legs go or stay together.
        orders, closed = fh.gate(result["orders"], quotes, now, calendar)
        passed = result["passed"] + closed
        writer = note_writer()
        cycle_id = f"fund-{now:%Y%m%dT%H%M}"
        value_before = ledger.nav(conn, FUND_ID, quotes, rates)["nav_usd"]
        conn.execute("SAVEPOINT fund_orders")
        for order in orders:
            body = dataclasses.asdict(order)
            note = fund_notes.write_note(body, writer)
            rate = rates.get(order.instrument.get("currency", "USD"), 1.0)
            try:
                if order.action == "buy":
                    fee = fw.kalshi_fee(order.price) * order.quantity if order.instrument.get("venue") == "kalshi" else 0.0
                    ledger.buy(conn, FUND_ID, order.instrument, order.side, order.quantity, order.price, fx=rate,
                               fee_usd=fee, attribution=order.attribution, purpose=order.purpose,
                               pair_key=order.pair_key, cycle_id=cycle_id, now=stamp, **note)
                else:
                    held = next(p for p in snapshot["positions"]
                                if p["instrument_id"] == order.instrument["id"] and p["side"] == order.side)
                    ledger.sell(conn, FUND_ID, order.instrument["id"], order.side, order.quantity, order.price,
                                fx=rates.get(held.get("currency", "USD"), rate), purpose=order.purpose,
                                pair_key=order.pair_key, cycle_id=cycle_id, now=stamp, **note)
                executed.append({**body, **note})
            except (ledger.LedgerError, StopIteration) as exc:
                passed.append({"instrument": order.instrument["id"], "reason": str(exc)})
        why_not = accounting_check(value_before, ledger.nav(conn, FUND_ID, quotes, rates)["nav_usd"])
        if why_not:
            conn.execute("ROLLBACK TO fund_orders")
            print(f"  {why_not}", file=sys.stderr)
            passed += [{"instrument": o["instrument"]["id"], "label": o["instrument"].get("label"), "reason": why_not}
                       for o in executed]
            executed = []
        conn.execute("RELEASE fund_orders")
        conn.commit()
        snapshot = ledger.nav(conn, FUND_ID, quotes, rates)

    guard = record_guard(conn, guard, now, stamp)
    ledger.record_nav(conn, snapshot, now=stamp)
    conn.commit()
    executed, passed = rolled + executed, roll_waiting + passed
    payload = build_payload(conn, snapshot, universe, now=now, window=why, traded=can_trade, executed=executed,
                            passed=passed, skipped=skipped_events + skipped_pairs + option_skips,
                            signals=signals, seeded=seeded)
    payload["shadow"] = {}  # no shadow analysts left: the checker went live on 2026-11-16
    payload["guard"] = guard
    payload["hours"] = {"holiday_list": calendar is not None,
                        "venues": {venue: {"open": ok, "why": why_not} for venue, (ok, why_not) in hours.items()}}
    payload["scorecard"] = card
    payload["knowledge"] = {"pairs": knowledge_views}
    payload["retired"] = [{**r, "since": "2026-11-15"} for r in RETIRED]
    blend = _load_json(BLEND_PATH)
    payload["prediction_markets"] = {"analyst": "blend_model", "test": blend.get("time_split_test"),
                                     "trained_on": blend.get("trained_on")}
    payload["settled"] = settled_events
    payload["options"] = {"mode": "OFF", "put_sleeve": {"mode": sleeve_modes["NIFTY"], "modes": sleeve_modes,
                                                        "holding": sorted(sleeve_ids)},
                          "prices": price_source, "settled": settled,
                          "indices": [name for name, _key in fo.INDEX_OPTIONS.values()],
                          "risk_per_structure": orch.OPTION_RISK * _scale(card, PUT_SLEEVE),
                          "class_cap": orch.OPTION_CLASS_CAP}
    # Each position's worst case and alerts against the last published board (fund_risk).
    payload["risk"] = fund_risk.review(snapshot, previous=_load_json(PAYLOAD_PATH))
    payload["owner_actions"] = _load_json(OWNER_ACTIONS_PATH).get("trims") or []
    payload["corrections"] = ledger.corrections(conn, FUND_ID)
    PAYLOAD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    conn.close()
    return payload


def record_guard(conn: sqlite3.Connection, guard: dict, now: datetime, stamp: str) -> dict:
    """Keep the rationale guard's record and return its summary for the page.

    Each recent check is recorded (analyst rationale_check), and so is every forecast the guard
    dropped this cycle, with the market's price then (analyst rationale_dropped), so the drops
    can be graded as their markets resolve (docs/fund.md, "Rationale checker goes live").
    """
    checks = [c for c in (_load_json(CHECKS_PATH).get("checks") or [])
              if (orch._age_hours(c.get("forecasted_at"), now) or 1e9) <= CHECKS_MAX_AGE_H]
    if checks:
        rows = [{"platform": c["instrument_id"].split(":", 1)[0], "ident": c["instrument_id"].split(":", 1)[1],
                 "question": c.get("question"), "model_probability": c["rationale_yes"]} for c in checks]
        ledger.record_predictions(conn, FUND_ID, "rationale_check", "live", rows, [], now=stamp)
    dropped = [r for r in guard.get("dropped") or [] if r.get("model_probability") is not None]
    if dropped:
        ledger.record_predictions(conn, FUND_ID, "rationale_dropped", "live", dropped, [], now=stamp)
    days = conn.execute("SELECT COUNT(DISTINCT day) FROM analyst_predictions WHERE fund_id = ?"
                        " AND analyst = 'rationale_check'", (FUND_ID,)).fetchone()[0]
    return {**guard, "markets": len(checks), "days_recorded": days,
            "dropped": [{"label": r.get("question") or r.get("ident"), "forecast": r.get("model_probability"),
                         "rationale_yes": r.get("rationale_yes"), "forecasted_at": r.get("forecasted_at")}
                        for r in guard.get("dropped") or []],
            "flags": [{"label": c.get("question"), "forecast": c["forecast"], "rationale_yes": c["rationale_yes"],
                       "forecasted_at": c.get("forecasted_at")} for c in checks if c.get("verdict") == "contradicts"]}


def build_payload(conn: sqlite3.Connection, snapshot: dict, universe: dict, *, now: datetime, window: str,
                  traded: bool, executed: list, passed: list, skipped: list, signals: list,
                  seeded: Optional[dict]) -> dict:
    trades = [dict(r) for r in conn.execute(
        "SELECT ts, cycle_id, instrument_id, label, side, action, purpose, pair_key, quantity, price, notional_usd,"
        " realized_pnl_usd, attribution_json, note, note_source, evidence_json FROM fund_trades WHERE fund_id = ?"
        " ORDER BY ts DESC LIMIT ?", (FUND_ID, RECENT_TRADES))]
    for t in trades:
        t["attribution"] = json.loads(t.pop("attribution_json") or "{}")
        t["evidence"] = json.loads(t.pop("evidence_json") or "[]")
    curve = [{"ts": r[0], "nav_usd": r[1]} for r in conn.execute(
        "SELECT ts, nav_usd FROM fund_nav WHERE fund_id = ? ORDER BY ts", (FUND_ID,))]
    analysts = sorted({s.analyst for s in signals} | {"blend_model", "pairs_model", PUT_SLEEVE})
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
