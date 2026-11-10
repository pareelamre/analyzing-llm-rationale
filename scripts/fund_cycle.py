#!/usr/bin/env python3
"""Run one Foresea Fund cycle and publish static/fund_live.json (docs/fund.md).

Every run values the fund at current prices and republishes the page's payload. Trading
happens only inside FUND_RUN_HOURS_IST and at least FUND_MIN_HOURS_BETWEEN_CYCLES after the
last trading cycle (or with FUND_ANY_HOUR=1).

The first run opens the fund from the on-device agent's account (FUND_SEED_AGENT_STORE):
its cash, and each open position at the agent's own entry price, so the fund starts at the
same $10,000 the agent did and the agent's track continues inside the fund.

Inputs: the Edge board (static/mark_to_market_live.json), the pairs tracker
(static/commodity_pairs_live.json), the index model (static/index_forecast_live.json, for the
options analyst), the fund universe (static/fund_universe.json, rebuilt when older than
FUND_UNIVERSE_MAX_AGE_DAYS), live quotes, and USD/INR. Option prices come from the live order
book when UPSTOX_ANALYTICS_TOKEN is set and from Upstox's open last trades otherwise
(fund_options). The weather analyst (fund_weather) prices Kalshi's daily high-temperature
markets; it trades only if its pre-registered test (reports/fund_weather_eval.json) said GO,
and otherwise records its predictions in shadow mode. Option structures past expiry are settled
at the index's close, and Kalshi positions at their market's result, every run. Kalshi buys pay
Kalshi's taker fee. Trade
notes are worded by the on-device model (FUND_LLM_URL + SELF_HOSTED_LLM_API_KEY) when it is
reachable; otherwise each note is the plain fact sheet.
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
from analyzing_llm_rationale import fund_options as fo  # noqa: E402
from analyzing_llm_rationale import fund_orchestrator as orch  # noqa: E402
from analyzing_llm_rationale import fund_pm_live as fpl  # noqa: E402
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
WEATHER_REPORT_PATH = ROOT / "reports" / "fund_weather_eval.json"
OPTIONS_REPORT_PATH = ROOT / "reports" / "fund_nifty_premium_eval.json"
BLEND_PATH = ROOT / "configs" / "fund_blend.json"
CHECKS_PATH = ROOT / "static" / "fund_rationale_checks.json"
#: Rationale checks older than this no longer describe a live forecast.
CHECKS_MAX_AGE_H = 48
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


def weather_mode() -> tuple[str, dict]:
    """GO only when the pre-registered test said so; SHADOW otherwise (docs/fund.md)."""
    report = _load_json(WEATHER_REPORT_PATH)
    return ("GO" if report.get("verdict") == "GO" else "SHADOW"), report


def options_mode() -> tuple[str, dict]:
    """The options analyst trades only while its backtest's live-rule test says GO (docs/fund.md)."""
    report = _load_json(OPTIONS_REPORT_PATH)
    test = report.get("test_a_live_rule") or {}
    return ("GO" if test.get("verdict") == "GO" else "SHADOW"), test


def options_cycle(index_forecast: dict, now: datetime, quote_fn, price_source: str, held: dict) -> dict:
    """The options analyst's signals, values and quotes, or why it had none."""
    if not index_forecast.get("indices"):
        return {"signals": [], "skipped": [], "values": {}, "quotes": {}}
    try:
        rows = fu.upstox_options("NSE")
    except (fu.FeedError, requests.RequestException, ValueError) as exc:
        return {"signals": [], "values": {}, "quotes": {},
                "skipped": [{"instrument": "nse_fo", "label": "Index options", "reason": f"no option list: {exc}"}]}
    return fo.option_signals(index_forecast, rows, now, quote_fn=quote_fn, price_source=price_source, held=held)


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
    option_specs = {p["instrument_id"]: ledger.instrument_spec(conn, FUND_ID, p["instrument_id"]) or {}
                    for p in ledger.positions(conn, FUND_ID) if p["kind"] == "option"}
    settled = settle_expired(conn, option_specs, now, stamp, rates)
    settled_events = settle_events(conn, now, stamp)
    conn.commit()
    quote_fn, price_source = fo.quote_source(os.environ.get("UPSTOX_ANALYTICS_TOKEN", "").strip() or None, now=now)
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
    quotes.update(fo.structure_quotes({k: v for k, v in option_specs.items() if v.get("legs")}, quote_fn))
    snapshot = ledger.nav(conn, FUND_ID, quotes, rates)

    can_trade, why = trading_window(conn, now)
    executed, passed = [], []
    options = {"signals": [], "skipped": [], "values": {}, "quotes": {}}
    weather = {"signals": [], "priced": [], "skipped": []}
    mode, weather_report = weather_mode()
    option_mode, option_test = options_mode()
    if can_trade:
        options = options_cycle(_load_json(INDEX_FORECAST_PATH), now, quote_fn, price_source, option_specs)
        if option_mode == "GO":
            signals += options["signals"]
        quotes.update(options["quotes"])
        weather = fw.weather_signals(now)
        if mode == "GO":
            signals += weather["signals"]
        held_pairs = {}
        for pos in snapshot["positions"]:
            if pos.get("pair_key"):
                held_pairs[pos["pair_key"]] = {"sessions": _sessions_since(pos["opened_at"], today)}
        pair_rows = {p.get("key"): p for p in tracker.get("pairs") or []}
        result = orch.plan(snapshot, signals, quotes, hedges, forecasts=forecasts, held_pairs=held_pairs,
                           pair_rows=pair_rows, option_values=options["values"], rates=rates)
        passed = result["passed"]
        writer = note_writer()
        cycle_id = f"fund-{now:%Y%m%dT%H%M}"
        for order in result["orders"]:
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
        conn.commit()
        snapshot = ledger.nav(conn, FUND_ID, quotes, rates)

    shadow = record_shadow(conn, board, now, stamp)
    if option_mode != "GO":
        shadow["options_model"] = {
            "mode": "shadow", "kind": "options", "checked": can_trade,
            "test": {k: option_test.get(k) for k in ("verdict", "trades", "first", "last", "mean_return",
                                                     "mean_return_ci", "win_share")},
            "would_trade": [{"label": s.instrument.get("label"), "side": s.side, "price": round(s.price, 2),
                             "edge": round(s.edge, 4)} for s in options["signals"]],
        }
    # The live Polymarket decided-ranges test records what the order book offered at every run
    # (docs/fund.md); it never trades, and a failed pass leaves the rest of the cycle alone.
    try:
        shadow[fpl.ANALYST] = fpl.run(conn, FUND_ID, now)
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"  live decided-ranges pass failed: {exc}", file=sys.stderr)
    if mode != "GO" and weather["priced"]:
        ledger.record_predictions(conn, FUND_ID, "weather_model", "shadow", weather["priced"], weather["signals"],
                                  now=stamp)
    days = conn.execute("SELECT COUNT(DISTINCT day), COUNT(*) FROM analyst_predictions WHERE fund_id = ?"
                        " AND analyst = 'weather_model'", (FUND_ID,)).fetchone()
    if mode != "GO":
        pooled = weather_report.get("pooled") or {}
        shadow["weather_model"] = {
            "mode": "shadow", "kind": "weather", "markets": len(weather["priced"]), "days_recorded": days[0],
            "predictions_recorded": days[1],
            "test": {"verdict": weather_report.get("verdict"), "trades": pooled.get("trades"),
                     "mean_return": pooled.get("mean_return"), "mean_return_ci": pooled.get("mean_return_ci"),
                     "brier_skill": pooled.get("brier_skill"), "window": weather_report.get("window")},
            "would_trade": [{"label": s.instrument.get("label"), "side": s.side, "price": s.price,
                             "edge": round(s.edge, 4)} for s in weather["signals"]],
        }
    ledger.record_nav(conn, snapshot, now=stamp)
    conn.commit()
    payload = build_payload(conn, snapshot, universe, now=now, window=why, traded=can_trade, executed=executed,
                            passed=passed, skipped=skipped_events + skipped_pairs + options["skipped"] + weather["skipped"],
                            signals=signals, seeded=seeded)
    payload["shadow"] = shadow
    payload["settled"] = settled_events
    payload["weather"] = {"mode": mode, "cities": len(fw.CITIES), "verdict_window": weather_report.get("window")}
    payload["options"] = {"mode": option_mode, "prices": price_source, "settled": settled,
                          "indices": [name for name, _key in fo.INDEX_OPTIONS.values()],
                          "risk_per_structure": orch.OPTION_RISK, "class_cap": orch.OPTION_CLASS_CAP}
    PAYLOAD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    conn.close()
    return payload


def record_shadow(conn: sqlite3.Connection, board: list, now: datetime, stamp: str) -> dict:
    """Shadow analysts: record what each would predict and trade, without trading it."""
    shadow: dict = {}
    checks = [c for c in (_load_json(CHECKS_PATH).get("checks") or [])
              if (orch._age_hours(c.get("forecasted_at"), now) or 1e9) <= CHECKS_MAX_AGE_H]
    if checks:
        rows = [{"platform": c["instrument_id"].split(":", 1)[0], "ident": c["instrument_id"].split(":", 1)[1],
                 "question": c.get("question"), "model_probability": c["rationale_yes"]} for c in checks]
        ledger.record_predictions(conn, FUND_ID, "rationale_check", "shadow", rows, [], now=stamp)
        shadow["rationale_check"] = {
            "mode": "shadow", "kind": "checker", "markets": len(checks),
            "days_recorded": conn.execute("SELECT COUNT(DISTINCT day) FROM analyst_predictions WHERE fund_id = ?"
                                          " AND analyst = 'rationale_check'", (FUND_ID,)).fetchone()[0],
            "checker": "Open-Jev 2B (Qwen3.5-2B base)",
            "flags": [{"label": c.get("question"), "forecast": c["forecast"], "rationale_yes": c["rationale_yes"],
                       "forecasted_at": c.get("forecasted_at")} for c in checks if c.get("verdict") == "contradicts"],
        }
    params = _load_json(BLEND_PATH)
    if not params.get("coef"):
        return shadow
    rows = orch.blend_board(board, params)
    would, _ = orch.event_signals(rows, now, analyst="blend_model", model_label="the blend analyst")
    ledger.record_predictions(conn, FUND_ID, "blend_model", "shadow", rows, would, now=stamp)
    days = conn.execute("SELECT COUNT(DISTINCT day), COUNT(*) FROM analyst_predictions WHERE fund_id = ?"
                        " AND analyst = 'blend_model'", (FUND_ID,)).fetchone()
    return {**shadow, "blend_model": {
        "mode": "shadow", "markets": len(rows), "days_recorded": days[0], "predictions_recorded": days[1],
        "test": params.get("time_split_test"), "trained_on": params.get("trained_on"),
        "would_trade": [{"label": s.instrument.get("label"), "side": s.side, "price": s.price,
                         "edge": round(s.edge, 4)} for s in would],
        "predictions": [{"label": r.get("question"), "blend": r["model_probability"],
                         "model": r.get("raw_model_probability"), "market": r.get("market_probability")}
                        for r in rows],
    }}


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
    analysts = sorted({s.analyst for s in signals} | {"event_model", "pairs_model", "options_model", "weather_model"})
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
