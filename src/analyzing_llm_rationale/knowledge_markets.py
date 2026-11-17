"""Exchange data for the knowledge base: what futures markets price and how traders are positioned
(owner, 2026-11-17: "add comex, and use exchanges as well").

Readers in the same shape as ``knowledge_tables`` (no model; checked before writing):

* ``cftc-cot``: the CFTC's weekly Commitments of Traders for COMEX (gold, silver, copper) and
  NYMEX (WTI crude, Henry Hub gas, RBOB gasoline, ULSD, platinum, palladium): money managers' net
  position, its share of open interest, and the change over one and four weeks. CME's own site
  (COMEX, NYMEX) does not answer our crawler, so positioning comes from the regulator's public
  API, which is US government data.
* ``shfe-futures``: Shanghai Futures Exchange settlement prices for copper, aluminium, zinc,
  nickel, gold and silver: the most traded contract, December contracts and the farthest month
  listed, from the exchange's public daily data. A futures price is what the market pays today for delivery
  later, not a forecast; claims say so.

Exchanges checked on 2026-11-17 that refuse the crawler: CME (no answer), ICE data (403), MCX (403),
NSE (no answer), LBMA prices (403). LME allows crawling, but its prices are licensed data, so it is
left out.

Each edition is weekly. ``keep`` limits how many editions stay in the base: positioning keeps only
the latest (each claim already carries the one- and four-week change), futures keep four weeks so
the newest price shows beside the last few.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any, Mapping, Optional
from urllib.parse import urlencode

from . import knowledge
from . import knowledge_tables as tables
from . import knowledge_watch as watch

ENTITIES = {
    "comex": {"name": "COMEX", "type": "organization", "aliases": ["Commodity Exchange Inc", "Commodity Exchange"]},
    "nymex": {"name": "NYMEX", "type": "organization", "aliases": ["New York Mercantile Exchange"]},
    "cme_group": {"name": "CME Group", "type": "organization", "aliases": ["CME"]},
    "cftc": {"name": "CFTC", "type": "organization",
             "aliases": ["Commodity Futures Trading Commission", "Commitments of Traders", "COT report"]},
    "shfe": {"name": "Shanghai Futures Exchange", "type": "organization", "aliases": ["SHFE"]},
    "futures_positioning": {"name": "Futures positioning", "type": "metric",
                            "aliases": ["speculative positioning", "money managers", "managed money", "net long",
                                        "net short", "hedge fund positioning"]},
    "palladium": {"name": "Palladium", "type": "commodity", "aliases": []},
    "diesel": {"name": "Diesel", "type": "commodity", "aliases": ["ULSD", "heating oil", "gasoil"]},
}

# --------------------------------------------------------------------------------------------
# CFTC Commitments of Traders (disaggregated, futures only)
# --------------------------------------------------------------------------------------------

COT_API = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
COT_SCENARIO = "CFTC Commitments of Traders"
#: code -> (exchange, market as the claim names it, contract size, entities)
COT_MARKETS = {
    "088691": ("COMEX", "gold", "100 troy ounces", ["comex", "gold"]),
    "084691": ("COMEX", "silver", "5,000 troy ounces", ["comex", "silver"]),
    "085692": ("COMEX", "copper", "25,000 pounds", ["comex", "copper"]),
    "076651": ("NYMEX", "platinum", "50 troy ounces", ["nymex", "platinum"]),
    "075651": ("NYMEX", "palladium", "100 troy ounces", ["nymex", "palladium"]),
    "067651": ("NYMEX", "WTI crude oil", "1,000 barrels", ["nymex", "crude_oil"]),
    "023651": ("NYMEX", "Henry Hub natural gas", "10,000 MMBtu", ["nymex", "natural_gas"]),
    "111659": ("NYMEX", "RBOB gasoline", "42,000 gallons", ["nymex", "gasoline"]),
    "022651": ("NYMEX", "NY Harbor ULSD (diesel)", "42,000 gallons", ["nymex", "diesel"]),
}


def _cot_rows(crawler: watch.Crawler, weeks: int) -> list[dict]:
    codes = ",".join(f"'{c}'" for c in COT_MARKETS)
    query = urlencode({"$select": "cftc_contract_market_code,report_date_as_yyyy_mm_dd,open_interest_all,"
                                  "m_money_positions_long_all,m_money_positions_short_all",
                       "$where": f"cftc_contract_market_code in({codes})",
                       "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": str(len(COT_MARKETS) * weeks)})
    try:
        rows = json.loads(tables._fetch(crawler, f"{COT_API}?{query}").text())
    except ValueError as exc:
        raise tables.TableError(f"unexpected CFTC response ({exc})") from exc
    if not isinstance(rows, list):
        raise tables.TableError("unexpected CFTC response (not a list)")
    return rows


def _cot_ident(entry: Mapping[str, Any], crawler: watch.Crawler, corpus: knowledge.Corpus):
    rows = _cot_rows(crawler, 1)
    if not rows:
        raise tables.TableError("the CFTC API returned no reports")
    week = str(rows[0]["report_date_as_yyyy_mm_dd"])[:10]
    return f"cftc-{week}-cot", {"week": week}


def _net(row: Mapping[str, Any]) -> tuple[int, int]:
    long_, short = tables._num(row.get("m_money_positions_long_all")), tables._num(row.get("m_money_positions_short_all"))
    oi = tables._num(row.get("open_interest_all"))
    if long_ is None or short is None or not oi:
        raise tables.TableError(f"CFTC row {row.get('cftc_contract_market_code')} lacks positions")
    net = int(long_ - short)
    if abs(net) > oi:
        raise tables.TableError(f"CFTC {row.get('cftc_contract_market_code')}: net {net} exceeds open interest {oi:.0f}")
    return net, int(oi)


def _signed(n: int) -> str:
    return f"up {n:,}" if n > 0 else f"down {-n:,}" if n < 0 else "unchanged"


def _cot_read(info: Mapping[str, Any], crawler: watch.Crawler, today: date, corpus: knowledge.Corpus) -> tables.Edition:
    week = info["week"]
    sid = f"cftc-{week}-cot"
    by_code: dict[str, dict[str, dict]] = {}
    for row in _cot_rows(crawler, 6):
        by_code.setdefault(row["cftc_contract_market_code"], {})[str(row["report_date_as_yyyy_mm_dd"])[:10]] = row
    label = datetime.strptime(week, "%Y-%m-%d").strftime("%-d %B %Y")
    claims: list[dict] = []
    for code, (exchange, market, size, entities) in COT_MARKETS.items():
        history = by_code.get(code) or {}
        if week not in history:
            continue
        net, oi = _net(history[week])
        dates = sorted(history, reverse=True)
        notes = []
        for back, words in ((1, "a week earlier"), (4, "four weeks earlier")):
            if len(dates) > back:
                notes.append(f"{_signed(net - _net(history[dates[back]])[0])} from {words}")
        side = "net long" if net >= 0 else "net short"
        text = (f"In the week to {label}, money managers held a {side} position of {abs(net):,} {exchange} "
                f"{market} futures contracts ({size} each), {abs(net) / oi * 100:.1f}% of open interest"
                + (f"; that is {' and '.join(notes)}." if notes else "."))
        claims.append(tables._claim(sid, len(claims) + 1, kind="fact", text=text,
                                    metric=f"{exchange} {market} money-manager net position", value=net,
                                    unit="contracts", geo="US", horizon=week, scenario=COT_SCENARIO,
                                    entities=["cftc", "futures_positioning", *entities], section=f"CFTC {code}"))
    source = {"id": sid, "title": f"Commitments of Traders, week to {label} (COMEX and NYMEX)", "publisher": "CFTC",
              "published": _cot_release(week).isoformat(), "url": "https://www.cftc.gov/MarketReports/CommitmentsofTraders",
              "licence": "US government data (public domain).", "kind": "table", "series": "cftc-cot",
              "topics": ["futures positioning", "gold", "silver", "copper", "crude oil", "natural gas"],
              "summary": "Money managers' net futures positions on COMEX and NYMEX from the CFTC's disaggregated "
                         "Commitments of Traders report (futures only).",
              "extracted_by": "knowledge_markets (no model) from the CFTC public reporting API", "added": today.isoformat()}
    return tables.Edition(source, claims, None, entities=ENTITIES)


def _cot_release(week: str) -> date:
    """Positions as of Tuesday are released on the Friday after."""
    return date.fromisoformat(week) + timedelta(days=3)


# --------------------------------------------------------------------------------------------
# Shanghai Futures Exchange daily settlements
# --------------------------------------------------------------------------------------------

SHFE_DATA = "https://www.shfe.com.cn/data/tradedata/future/dailydata/kx{day}.dat"
SHFE_SCENARIO = "SHFE futures settlement"
#: product id -> (name, unit, words for the unit, entities)
SHFE_PRODUCTS = {
    "cu_f": ("copper", "CNY/t", "yuan a tonne", ["copper"]),
    "al_f": ("aluminium", "CNY/t", "yuan a tonne", ["aluminium"]),
    "zn_f": ("zinc", "CNY/t", "yuan a tonne", ["zinc"]),
    "ni_f": ("nickel", "CNY/t", "yuan a tonne", ["nickel"]),
    "au_f": ("gold", "CNY/g", "yuan a gram", ["gold"]),
    "ag_f": ("silver", "CNY/kg", "yuan a kilogram", ["silver"]),
}


def _shfe_day(crawler: watch.Crawler, day: date) -> Optional[list[dict]]:
    resp = crawler.fetch(SHFE_DATA.format(day=day.strftime("%Y%m%d")), max_bytes=watch.PAGE_BYTES)
    if resp.status == 404:
        return None  # no trading that day
    if not resp.ok:
        raise tables.TableError(f"SHFE data for {day}: {resp.error or f'HTTP {resp.status}'}")
    try:
        rows = json.loads(resp.text()).get("o_curinstrument")
    except (ValueError, AttributeError):
        return None  # a holiday answers with a page, not data
    return rows if isinstance(rows, list) and rows else None


def _last_friday(today: date) -> date:
    """The latest Friday before today (today's settlement may not be out yet)."""
    back = (today.weekday() - 4) % 7 or 7
    return today - timedelta(days=back)


#: Today's date; tests replace it.
TODAY = date.today


def _shfe_ident(entry: Mapping[str, Any], crawler: watch.Crawler, corpus: knowledge.Corpus):
    friday = _last_friday(TODAY())
    for back in range(5):  # the week's last trading day, if Friday was a holiday
        day = friday - timedelta(days=back)
        rows = _shfe_day(crawler, day)
        if rows:
            return f"shfe-{day.isoformat()}-futures", {"day": day, "rows": rows}
    raise tables.TableError(f"no SHFE settlement data in the week to {friday}")


def _delivery(month: str) -> Optional[str]:
    """SHFE's 'YYMM' delivery month as YYYY-MM."""
    month = str(month or "").strip()
    if len(month) != 4 or not month.isdigit() or not 1 <= int(month[2:]) <= 12:
        return None
    return f"20{month[:2]}-{month[2:]}"


def _shfe_read(info: Mapping[str, Any], crawler: watch.Crawler, today: date, corpus: knowledge.Corpus) -> tables.Edition:
    day: date = info["day"]
    sid = f"shfe-{day.isoformat()}-futures"
    label = day.strftime("%-d %B %Y")
    claims: list[dict] = []
    for product, (name, unit, words, entities) in SHFE_PRODUCTS.items():
        contracts = []
        for row in info["rows"]:
            if str(row.get("PRODUCTID") or "").strip() != product:
                continue
            month = _delivery(row.get("DELIVERYMONTH"))
            settle = tables._num(row.get("SETTLEMENTPRICE"))
            oi = tables._num(row.get("OPENINTEREST")) or 0
            if month and settle and settle > 0:
                contracts.append((month, settle, oi))
        if not contracts:
            continue
        active = max(contracts, key=lambda c: c[2])[0]
        farthest = max(contracts)[0]
        for month, settle, _oi in sorted(contracts):
            # A few points per metal, so one exchange does not crowd a question's evidence.
            if month not in (active, farthest) and not month.endswith("-12"):
                continue
            when = datetime.strptime(month, "%Y-%m").strftime("%B %Y")
            price = f"{settle:,.2f}".rstrip("0").rstrip(".") if unit == "CNY/g" else f"{settle:,.0f}"
            note = "; the most traded contract" if month == active else ""
            claims.append(tables._claim(
                sid, len(claims) + 1, kind="estimate", metric=f"SHFE {name} futures price", value=settle, unit=unit,
                geo="China", horizon=month, scenario=SHFE_SCENARIO, entities=["shfe", *entities, "china"],
                section=f"SHFE {product}",
                text=f"On {label}, Shanghai Futures Exchange {name} for delivery in {when} settled at "
                     f"{price} {words} (a futures price, not a forecast{note})."))
    source = {"id": sid, "title": f"Shanghai Futures Exchange settlement prices, {label}", "publisher": "SHFE",
              "published": day.isoformat(), "url": "https://www.shfe.com.cn/eng/",
              "licence": "Exchange daily data published on its public website.", "kind": "table",
              "series": "shfe-futures", "topics": ["copper", "aluminium", "zinc", "nickel", "gold", "silver", "futures"],
              "summary": "Settlement prices of Shanghai Futures Exchange base and precious metal futures: the most "
                         "traded contract, December contracts and the farthest month listed.",
              "extracted_by": "knowledge_markets (no model) from the SHFE daily data file", "added": today.isoformat()}
    return tables.Edition(source, claims, None, entities=ENTITIES)


READERS = {
    "cftc-cot": tables.Reader(_cot_ident, _cot_read, COT_SCENARIO, 7, keep=1),
    "shfe-futures": tables.Reader(_shfe_ident, _shfe_read, SHFE_SCENARIO, 10, keep=4),
}
