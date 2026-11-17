"""Turn publishers' forecast tables into knowledge-base claims, with no model in the loop.

Phase 3 of growing the base (docs/knowledge.md). For publishers on the watch list that put their
forecasts in a table, a small reader takes the current edition straight from the source file and
writes one source folder of claims in our own words, each with a metric, value, unit, place, year
and a fixed scenario name per publisher. Because the scenario name does not change between
editions, a new edition supersedes the last one on the same metric and year ("Earlier: ..."),
while other publishers' numbers stay beside it as rival estimates.

Nothing is written unless the edition passes checks: enough rows were found, every value is in a
plausible range, and no value moved implausibly far from the publisher's previous edition. A
failed check leaves the base untouched and says why: a changed table layout must never put wrong
numbers in front of the forecasts or the fund.

Readers (all standard library, so the scheduled job installs nothing):
  eia-steo         EIA Short-Term Energy Outlook, from STEO_m.xlsx (monthly)
  fed-sep          Federal Reserve Summary of Economic Projections, web table (quarterly)
  ecb-projections  ECB staff projections, the comparison table on the projections page (quarterly)
  imf-weo          IMF World Economic Outlook, from the IMF DataMapper API (twice a year)
"""
from __future__ import annotations

import calendar
import html
import io
import json
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from . import knowledge
from . import knowledge_watch as watch


class TableError(Exception):
    """An edition that cannot be read safely; the message says why."""


@dataclass
class Edition:
    source: dict
    claims: list[dict]
    #: The watch list's key for this edition, so its inbox row can be marked as read.
    watch_key: Optional[str] = None


#: Entities the readers link to, added to entities.jsonl when the base lacks them.
ENTITIES = {
    "federal_reserve": {"name": "Federal Reserve", "type": "organization",
                        "aliases": ["Fed", "FOMC", "Federal Open Market Committee"]},
    "federal_funds_rate": {"name": "Federal funds rate", "type": "metric",
                           "aliases": ["fed funds rate", "fed funds", "Fed rate", "Fed rates", "US interest rates"]},
    "unemployment": {"name": "Unemployment", "type": "metric", "aliases": ["unemployment rate", "jobless rate"]},
    "ecb": {"name": "European Central Bank", "type": "organization", "aliases": ["ECB"]},
    "gasoline": {"name": "Gasoline", "type": "commodity",
                 "aliases": ["gasoline price", "gasoline prices", "petrol", "pump prices"]},
}

#: Plausible values per unit; anything outside means the reader misread the table.
BOUNDS = {"$/bbl": (5, 400), "$/mmbtu": (0.3, 60), "$/gal": (0.5, 15), "cents/kWh": (2, 80),
          "million b/d": (0.5, 200), "bcf/d": (10, 300), "%": (-40, 150)}
#: How far a value may move from the publisher's previous edition before a person must look.
MAX_MOVE_REL = 0.6
MAX_MOVE_PP = 8.0
UNIT_WORDS = {"$/bbl": "a barrel", "$/mmbtu": "per million Btu", "$/gal": "a gallon"}


def _num(text: Any) -> Optional[float]:
    if isinstance(text, (int, float)):
        return float(text)
    clean = str(text or "").strip().replace("−", "-").replace("–", "-").replace(",", "")
    return float(clean) if re.fullmatch(r"-?\d+(?:\.\d+)?", clean) else None


def fmt(value: float, unit: str) -> str:
    """A value as the claim text says it."""
    if unit == "$/bbl":
        text = f"{value:,.1f}"
        return f"${text[:-2] if text.endswith('.0') else text} {UNIT_WORDS[unit]}"
    if unit in ("$/mmbtu", "$/gal"):
        return f"${value:.2f} {UNIT_WORDS[unit]}"
    if unit == "cents/kWh":
        return f"{value:.1f} cents per kilowatthour"
    if unit == "million b/d":
        return f"{value:.2f} million barrels a day"
    if unit == "bcf/d":
        return f"{value:.1f} billion cubic feet a day"
    return f"{value:.1f}%"


def _round(value: float, unit: str) -> float:
    if unit == "$/bbl":
        return round(value, 1)
    if unit in ("cents/kWh", "bcf/d", "%"):
        return round(value, 1)
    return round(value, 2)


def _claim(sid: str, n: int, *, text: str, metric: str, value: float, unit: str, geo: str,
           horizon: Optional[str], scenario: str, entities: Sequence[str], section: str,
           kind: str = "projection") -> dict:
    claim = {"id": f"{sid}:c{n:03d}", "kind": kind, "section": section, "text": text,
             "entities": list(entities), "metric": metric, "value": _round(value, unit), "unit": unit,
             "geo": geo, "scenario": scenario}
    if horizon:
        claim["horizon"] = horizon
    return claim


def _fetch(crawler: watch.Crawler, url: str, *, max_bytes: int = watch.PAGE_BYTES) -> watch.Response:
    resp = crawler.fetch(url, max_bytes=max_bytes)
    if not resp.ok:
        raise TableError(f"could not fetch {url}: {resp.error or f'HTTP {resp.status}'}")
    return resp


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def html_tables(page: str) -> list[list[list[str]]]:
    """Every table on a page as rows of cell texts (header and data cells alike)."""
    out = []
    for table in re.findall(r"<table\b.*?</table>", page, re.S | re.I):
        rows = []
        for row in re.findall(r"<tr\b.*?</tr>", table, re.S | re.I):
            cells = re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", row, re.S | re.I)
            rows.append([" ".join(html.unescape(re.sub(r"<[^>]+>", " ", c)).split()) for c in cells])
        out.append(rows)
    return out


# --------------------------------------------------------------------------------------------
# A spreadsheet reader small enough to need no library
# --------------------------------------------------------------------------------------------

_XL = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


class Workbook:
    """Cell values of an .xlsx file: strings, floats, or None."""

    def __init__(self, data: bytes):
        try:
            self.zip = zipfile.ZipFile(io.BytesIO(data))
            names = set(self.zip.namelist())
            self.shared = []
            if "xl/sharedStrings.xml" in names:
                for si in ET.fromstring(self.zip.read("xl/sharedStrings.xml")).iter(f"{_XL}si"):
                    self.shared.append("".join(t.text or "" for t in si.iter(f"{_XL}t")))
            rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(self.zip.read("xl/_rels/workbook.xml.rels"))}
            self.sheets = {}
            for sheet in ET.fromstring(self.zip.read("xl/workbook.xml")).iter(f"{_XL}sheet"):
                target = rels[sheet.get(f"{_REL}id")].lstrip("/")
                self.sheets[sheet.get("name")] = target if target.startswith("xl/") else f"xl/{target}"
        except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
            raise TableError(f"not a readable spreadsheet ({exc})") from exc

    def rows(self, name: str) -> list[list[Any]]:
        if name not in self.sheets:
            raise TableError(f"the spreadsheet has no sheet {name!r}")
        out = []
        for row in ET.fromstring(self.zip.read(self.sheets[name])).iter(f"{_XL}row"):
            cells: dict[int, Any] = {}
            for cell in row.iter(f"{_XL}c"):
                col = 0
                for ch in re.match(r"[A-Z]+", cell.get("r") or "A").group(0):
                    col = col * 26 + ord(ch) - 64
                kind, v = cell.get("t"), cell.find(f"{_XL}v")
                if kind == "inlineStr":
                    value: Any = "".join(t.text or "" for t in cell.iter(f"{_XL}t"))
                elif v is None or v.text is None:
                    continue
                elif kind == "s":
                    value = self.shared[int(v.text)]
                elif kind in ("str", "b", "e"):
                    value = v.text
                else:
                    value = _num(v.text)
                cells[col - 1] = value
            out.append([cells.get(i) for i in range(max(cells) + 1)] if cells else [])
        return out


# --------------------------------------------------------------------------------------------
# EIA Short-Term Energy Outlook
# --------------------------------------------------------------------------------------------

STEO_XLSX = "https://www.eia.gov/outlooks/steo/xls/STEO_m.xlsx"
STEO_SCENARIO = "EIA STEO baseline"
#: (series code, metric, unit, place, what the text calls it, entities, also a December claim)
STEO_SERIES = [
    ("BREPUUS", "Brent crude price", "$/bbl", "global", "the Brent crude spot price", ["crude_oil"], True),
    ("WTIPUUS", "WTI crude price", "$/bbl", "US", "the WTI crude spot price", ["crude_oil", "us"], True),
    ("NGHHUUS", "US natural gas price", "$/mmbtu", "US", "the Henry Hub natural gas spot price",
     ["natural_gas", "us"], True),
    ("MGRARUS_$", "US retail gasoline price", "$/gal", "US", "the US retail price of regular gasoline",
     ["gasoline", "us"], True),
    ("DSRTUUS_$", "US retail diesel price", "$/gal", "US", "the US retail price of on-highway diesel", ["us"], False),
    ("ESRCUUS", "US residential electricity price", "cents/kWh", "US", "the US residential electricity price",
     ["electricity_prices", "us"], False),
    ("COPRPUS", "US crude oil production", "million b/d", "US", "US crude oil production", ["crude_oil", "us"], False),
    ("NGPRPUS", "US dry natural gas production", "bcf/d", "US", "US dry natural gas production",
     ["natural_gas", "us"], False),
    ("papr_world", "world liquid fuels production", "million b/d", "global",
     "world production of crude oil and other liquids", ["crude_oil"], False),
    ("patc_world", "world liquid fuels consumption", "million b/d", "global",
     "world consumption of crude oil and other liquids", ["crude_oil"], False),
]


def _steo_ident(entry: Mapping[str, Any], crawler: watch.Crawler, corpus: knowledge.Corpus):
    key = entry.get("latest")
    if not key:
        return None
    try:
        released = datetime.strptime(key, "%B %d, %Y").date()
    except ValueError as exc:
        raise TableError(f"unexpected STEO release date {key!r}") from exc
    return f"eia-{released:%Y-%m}-steo", {"released": released, "key": key}


def _steo_read(info: Mapping[str, Any], crawler: watch.Crawler, today: date, corpus: knowledge.Corpus) -> Edition:
    released: date = info["released"]
    sid = f"eia-{released:%Y-%m}-steo"
    book = Workbook(_fetch(crawler, STEO_XLSX, max_bytes=watch.FILE_BYTES).body)
    dates = book.rows("Dates")
    label = next((str(r[3]).strip() for r in dates if len(r) > 3 and str(r[0] or "").startswith("Forecast Month")), "")
    if label != f"{released:%B %Y}":
        raise TableError(f"STEO_m.xlsx is the {label or 'unknown'} edition, the watch list expects {released:%B %Y}")
    last_hist = next((int(r[3]) for r in dates if len(r) > 3 and str(r[0] or "").startswith("Last Historical Month")), 0)
    months_row = next((r for r in dates if len(r) > 2 and str(r[1] or "").startswith("AAAA_DATEX")), None)
    if not last_hist or months_row is None:
        raise TableError("STEO_m.xlsx Dates sheet changed: no last historical month or month row")
    months = [int(m) for m in months_row[2:] if m is not None]
    series: dict[str, list[Any]] = {}
    for name in book.sheets:
        if not re.fullmatch(r"\d+[a-z]?(\(\d\))?tab", name):
            continue
        for row in book.rows(name):
            if len(row) > 2 and isinstance(row[0], str) and row[0] not in series:
                series[row[0]] = row[2:2 + len(months)]
    month_label, hist_label = f"{released:%B %Y}", f"{calendar.month_name[last_hist % 100]} {last_hist // 100}"
    claims: list[dict] = []
    for code, metric, unit, geo, what, entities, december in STEO_SERIES:
        values = series.get(code)
        if values is None:
            continue
        by_month = {m: _num(v) for m, v in zip(months, values)}
        for year in sorted({m // 100 for m in months if m > last_hist}):
            year_values = [by_month.get(year * 100 + m) for m in range(1, 13)]
            if any(v is None for v in year_values):
                continue
            mean = sum(year_values) / 12
            partly = last_hist // 100 == year
            claims.append(_claim(
                sid, len(claims) + 1, metric=metric, value=mean, unit=unit, geo=geo, horizon=str(year),
                scenario=STEO_SCENARIO, entities=["eia", *entities], section="STEO forecast tables",
                text=f"EIA's Short-Term Energy Outlook ({month_label}) puts {what} at an average "
                     f"{fmt(_round(mean, unit), unit)} in {year}"
                     + (f" (actual data through {hist_label}, forecast after)." if partly else ".")))
            dec = by_month.get(year * 100 + 12)
            if december and dec is not None and year * 100 + 12 > last_hist:
                claims.append(_claim(
                    sid, len(claims) + 1, metric=f"{metric} (December average)", value=dec, unit=unit, geo=geo,
                    horizon=f"{year}-12", scenario=STEO_SCENARIO, entities=["eia", *entities],
                    section="STEO forecast tables",
                    text=f"EIA's Short-Term Energy Outlook ({month_label}) expects {what} to average "
                         f"{fmt(_round(dec, unit), unit)} in December {year}."))
    last_year = max(m // 100 for m in months)
    source = {"id": sid, "title": f"Short-Term Energy Outlook ({month_label})", "publisher": "US EIA",
              "published": released.isoformat(), "url": "https://www.eia.gov/outlooks/steo/",
              "licence": "US government work (public domain).", "kind": "table",
              "topics": ["oil prices", "natural gas prices", "gasoline", "US energy production", "world oil balance"],
              "summary": f"EIA's monthly forecast of energy prices, US production and world oil supply and demand "
                         f"through December {last_year}, read from its STEO_m.xlsx tables.",
              "extracted_by": "knowledge_tables (no model) from STEO_m.xlsx", "added": today.isoformat()}
    return Edition(source, claims, info["key"])


# --------------------------------------------------------------------------------------------
# Federal Reserve Summary of Economic Projections
# --------------------------------------------------------------------------------------------

FED_SCENARIO = "FOMC median projection"
#: Table 1 row label -> (metric, phrase with {v} and {y}, longer-run name, entities)
FED_ROWS = {
    "Change in real GDP": ("real GDP growth (Q4/Q4)", "real GDP growth of {v} in {y}, fourth quarter over fourth quarter",
                           "real GDP growth rate", ["gdp_growth", "us"]),
    "Unemployment rate": ("unemployment rate (Q4)", "an unemployment rate of {v} in the fourth quarter of {y}",
                          "unemployment rate", ["unemployment", "us"]),
    "PCE inflation": ("PCE inflation (Q4/Q4)", "PCE inflation of {v} in {y}, fourth quarter over fourth quarter",
                      "PCE inflation rate", ["inflation", "us"]),
    "Core PCE inflation": ("core PCE inflation (Q4/Q4)",
                           "core PCE inflation of {v} in {y}, fourth quarter over fourth quarter",
                           "core PCE inflation rate", ["inflation", "us"]),
    "Federal funds rate": ("federal funds rate (year-end)", "the federal funds rate at {v} at the end of {y}",
                           "federal funds rate", ["federal_funds_rate", "us"]),
}


def _fed_ident(entry: Mapping[str, Any], crawler: watch.Crawler, corpus: knowledge.Corpus):
    key = entry.get("latest")
    if not key or not re.fullmatch(r"\d{8}", key):
        return None
    released = datetime.strptime(key, "%Y%m%d").date()
    url = (entry.get("seen") or {}).get(key, {}).get("url") or \
        f"https://www.federalreserve.gov/monetarypolicy/fomcprojtabl{key}.htm"
    return f"fed-{released:%Y-%m}-sep", {"released": released, "url": url, "key": key}


def _fed_read(info: Mapping[str, Any], crawler: watch.Crawler, today: date, corpus: knowledge.Corpus) -> Edition:
    released: date = info["released"]
    sid, month_label = f"fed-{released:%Y-%m}-sep", f"{released:%B %Y}"
    table = next((t for t in html_tables(_fetch(crawler, info["url"]).text())
                  if any(r and r[0].startswith("Federal funds rate") for r in t)), None)
    if table is None:
        raise TableError("no Summary of Economic Projections table with a federal funds rate row")
    header = next((r for r in table if "Longer run" in r), None)
    if header is None:
        raise TableError("the projections table has no year header")
    columns = header[:header.index("Longer run") + 1]
    claims: list[dict] = []
    for n, row in enumerate(table):
        label = re.sub(r"\s*\d+$", "", row[0]) if row else ""
        if label not in FED_ROWS or len(row) < len(columns) + 1:
            continue
        metric, phrase, long_name, entities = FED_ROWS[label]
        before = table[n + 1] if n + 1 < len(table) else []
        prior = before[0].replace(" projection", "") if before and before[0].endswith("projection") else None
        for i, col in enumerate(columns):
            value = _num(row[1 + i])
            if value is None:
                continue
            was = _num(before[1 + i]) if prior and len(before) > 1 + i else None
            note = f" ({prior} projection: {was:.1f}%)" if was is not None else ""
            ents = ["federal_reserve", *entities]
            if col == "Longer run":
                claims.append(_claim(
                    sid, len(claims) + 1, kind="estimate", metric=f"{metric.split(' (')[0]} (longer run)",
                    value=value, unit="%", geo="US", horizon=None, scenario=FED_SCENARIO, entities=ents,
                    section="Summary of Economic Projections, Table 1",
                    text=f"The Fed's {month_label} projections (FOMC median) put the longer-run {long_name} "
                         f"at {value:.1f}%{note}."))
            elif re.fullmatch(r"\d{4}", col):
                claims.append(_claim(
                    sid, len(claims) + 1, metric=metric, value=value, unit="%", geo="US", horizon=col,
                    scenario=FED_SCENARIO, entities=ents, section="Summary of Economic Projections, Table 1",
                    text=f"The Fed's {month_label} projections (FOMC median) have "
                         f"{phrase.format(v=f'{value:.1f}%', y=col)}{note}."))
    source = {"id": sid, "title": f"FOMC Summary of Economic Projections ({month_label})",
              "publisher": "Federal Reserve", "published": released.isoformat(), "url": info["url"],
              "licence": "US government work (public domain).", "kind": "table",
              "topics": ["federal funds rate", "US inflation", "US unemployment", "US GDP growth"],
              "summary": "The median projections of FOMC participants for US growth, unemployment, PCE inflation and "
                         "the federal funds rate, read from Table 1 of the Summary of Economic Projections.",
              "extracted_by": "knowledge_tables (no model) from the SEP web table", "added": today.isoformat()}
    return Edition(source, claims, info["key"])


# --------------------------------------------------------------------------------------------
# ECB staff projections
# --------------------------------------------------------------------------------------------

ECB_SCENARIO = "ECB staff baseline"
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def _ecb_ident(entry: Mapping[str, Any], crawler: watch.Crawler, corpus: knowledge.Corpus):
    key = entry.get("latest")
    url = (entry.get("seen") or {}).get(key or "", {}).get("url")
    if not key or not url or not re.fullmatch(r"\d{6}", key):
        return None
    return f"ecb-{key[:4]}-{key[4:]}-projections", {"year": int(key[:4]), "month": int(key[4:]), "url": url, "key": key}


def _ecb_spec(group: str) -> tuple[str, str, list[str]]:
    if "excluding" in group:
        return "core HICP inflation (excl. energy and food)", "HICP inflation excluding energy and food", ["inflation"]
    if "HICP" in group:
        return "HICP inflation", "HICP inflation", ["inflation"]
    return "real GDP growth", "real GDP growth", ["gdp_growth"]


def _ecb_read(info: Mapping[str, Any], crawler: watch.Crawler, today: date, corpus: knowledge.Corpus) -> Edition:
    year, month = info["year"], info["month"]
    sid, month_label = f"ecb-{year}-{month:02d}-projections", f"{calendar.month_name[month]} {year}"
    page = _fetch(crawler, info["url"]).text()
    # The page's own publication date; else the month's last day, so a backtest never sees the
    # projections early. Dates in the text are no good: "On 7 September" was a data note there.
    meta = re.search(r'article:published_time"\s+content="(\d{4}-\d{2}-\d{2})', page)
    stated = knowledge.parse_when(meta.group(1)) if meta else None
    published = stated if stated and (stated.year, stated.month) == (year, month) else _month_end(year, month)
    table = next((t for t in html_tables(page)
                  if any(r and re.match(r"(ECB|Eurosystem) staff projections", r[0]) for r in t)), None)
    if table is None:
        raise TableError("no comparison table with an ECB staff projections row")
    groups = next(([c for c in r if re.search(r"GDP|HICP", c)] for r in table
                   if any("GDP" in c for c in r) and any("HICP" in c for c in r)), [])
    years = next(([c for c in r if c] for r in table
                  if r and all(re.fullmatch(r"\d{4}", c) for c in r if c) and len([c for c in r if c]) >= 3), [])
    row = next(r for r in table if r and re.match(r"(ECB|Eurosystem) staff projections", r[0]))
    values = [_num(c) for c in row[2:]]
    if not groups or not years or len(years) % len(groups) or len(values) != len(years):
        raise TableError(f"the ECB comparison table changed shape ({len(groups)} groups, {len(years)} years, "
                         f"{len(values)} values)")
    per = len(years) // len(groups)
    claims: list[dict] = []
    for g, group in enumerate(groups):
        metric, phrase, entities = _ecb_spec(group)
        for j in range(per):
            value, horizon = values[g * per + j], years[g * per + j]
            if value is None:
                continue
            claims.append(_claim(
                sid, len(claims) + 1, metric=metric, value=value, unit="%", geo="euro area", horizon=horizon,
                scenario=ECB_SCENARIO, entities=["ecb", "euro_area", *entities],
                section="Comparison with other forecasts",
                text=f"ECB staff projections ({month_label}) see euro area {phrase} of {value:.1f}% in {horizon}."))
    source = {"id": sid, "title": f"ECB staff macroeconomic projections for the euro area ({month_label})",
              "publisher": "ECB", "published": published.isoformat(), "url": info["url"],
              "licence": "ECB: reuse allowed with the source credited.", "kind": "table",
              "topics": ["euro area inflation", "euro area GDP growth", "ECB"],
              "summary": "ECB staff baseline projections for euro area real GDP growth, HICP inflation and HICP "
                         "inflation excluding energy and food, from the projections' comparison table.",
              "extracted_by": "knowledge_tables (no model) from the projections web page", "added": today.isoformat()}
    return Edition(source, claims, info["key"])


# --------------------------------------------------------------------------------------------
# IMF World Economic Outlook (DataMapper API)
# --------------------------------------------------------------------------------------------

IMF_API = "https://www.imf.org/external/datamapper/api/v1"
IMF_SCENARIO = "IMF WEO baseline"
#: (indicator, metric, phrase with {v}, entities)
IMF_INDICATORS = [
    ("NGDP_RPCH", "real GDP growth", "real GDP growth of {v}", ["gdp_growth"]),
    ("PCPIPCH", "headline inflation", "average consumer price inflation of {v}", ["inflation"]),
    ("LUR", "unemployment rate", "an unemployment rate of {v}", ["unemployment"]),
]
#: (DataMapper code, place, name in text, entities)
IMF_ECONOMIES = [
    ("WEOWORLD", "global", "the world", []), ("USA", "US", "the US", ["us"]), ("CHN", "China", "China", ["china"]),
    ("IND", "India", "India", ["india"]), ("EURO", "euro area", "the euro area", ["euro_area"]),
    ("JPN", "Japan", "Japan", ["japan"]), ("GBR", "UK", "the UK", ["uk"]), ("DEU", "Germany", "Germany", ["germany"]),
    ("FRA", "France", "France", ["france"]), ("RUS", "Russia", "Russia", ["russia"]),
    ("BRA", "Brazil", "Brazil", ["brazil"]), ("SAU", "Saudi Arabia", "Saudi Arabia", ["saudi_arabia"]),
    ("IDN", "Indonesia", "Indonesia", ["indonesia"]),
]


def _imf_ident(entry: Mapping[str, Any], crawler: watch.Crawler, corpus: knowledge.Corpus):
    try:
        meta = json.loads(_fetch(crawler, f"{IMF_API}/indicators").text())["indicators"]["NGDP_RPCH"]
    except (ValueError, KeyError, TypeError) as exc:
        raise TableError(f"unexpected DataMapper indicators response ({exc})") from exc
    match = re.search(rf"World Economic Outlook \(({MONTHS}) (\d{{4}})\)", str(meta.get("source") or ""))
    if not match:
        raise TableError(f"cannot tell the WEO edition from {meta.get('source')!r}")
    month, year = list(calendar.month_name).index(match.group(1)), int(match.group(2))
    # The release date from a source already in the base or the watch list, else the month's end.
    stamp = f"{year}-{month:02d}"
    known = sorted(str(s.get("published")) for s in corpus.sources.values()
                   if s.get("publisher") == "IMF" and str(s.get("published") or "").startswith(stamp)
                   and knowledge.parse_when(s.get("published")) and len(str(s.get("published"))) == 10)
    seen = sorted(k for k in (entry.get("seen") or {}) if k.startswith(f"{year}/{month:02d}/"))
    published = (date.fromisoformat(known[0]) if known else
                 datetime.strptime(seen[0], "%Y/%m/%d").date() if seen else _month_end(year, month))
    return f"imf-{stamp}-weo-data", {"year": year, "month": month, "published": published}


def _imf_read(info: Mapping[str, Any], crawler: watch.Crawler, today: date, corpus: knowledge.Corpus) -> Edition:
    year, month = info["year"], info["month"]
    sid, month_label = f"imf-{year}-{month:02d}-weo-data", f"{calendar.month_name[month]} {year}"
    claims: list[dict] = []
    for code, metric, phrase, entities in IMF_INDICATORS:
        try:
            values = json.loads(_fetch(crawler, f"{IMF_API}/{code}").text())["values"][code]
        except (ValueError, KeyError, TypeError) as exc:
            raise TableError(f"unexpected DataMapper response for {code} ({exc})") from exc
        for econ, geo, name, places in IMF_ECONOMIES:
            for y in range(year, year + 3):
                value = _num((values.get(econ) or {}).get(str(y)))
                if value is None:
                    continue
                # The WEO gives India's national accounts and prices by fiscal year (April-March).
                fiscal = f" (India's fiscal year from April {y})" if econ == "IND" and code != "LUR" else ""
                claims.append(_claim(
                    sid, len(claims) + 1, metric=metric, value=value, unit="%", geo=geo, horizon=str(y),
                    scenario=IMF_SCENARIO, entities=["imf", *entities, *places], section=f"DataMapper {code}",
                    text=f"The IMF's World Economic Outlook ({month_label}) projects "
                         f"{phrase.format(v=f'{value:.1f}%')} for {name} in {y}{fiscal}."))
    source = {"id": sid, "title": f"World Economic Outlook database ({month_label})", "publisher": "IMF",
              "published": info["published"].isoformat(), "url": "https://www.imf.org/external/datamapper/",
              "licence": "IMF data; free to use with attribution.", "kind": "table",
              "topics": ["GDP growth", "inflation", "unemployment", "world economy"],
              "summary": "IMF World Economic Outlook projections of real GDP growth, consumer price inflation and "
                         "unemployment for the world and major economies, from the IMF DataMapper.",
              "extracted_by": "knowledge_tables (no model) from the IMF DataMapper API", "added": today.isoformat()}
    return Edition(source, claims, None)


# --------------------------------------------------------------------------------------------
# Checks, writing, and the run
# --------------------------------------------------------------------------------------------

@dataclass
class Reader:
    ident: Callable[..., Any]
    read: Callable[..., Edition]
    scenario: str
    min_claims: int


READERS = {
    "eia-steo": Reader(_steo_ident, _steo_read, STEO_SCENARIO, 16),
    "fed-sep": Reader(_fed_ident, _fed_read, FED_SCENARIO, 12),
    "ecb-projections": Reader(_ecb_ident, _ecb_read, ECB_SCENARIO, 6),
    "imf-weo": Reader(_imf_ident, _imf_read, IMF_SCENARIO, 60),
}


def _key(claim: Mapping[str, Any]) -> tuple:
    return tuple(str(claim.get(f) or "").strip().lower() for f in ("metric", "geo", "horizon", "unit"))


def dedupe(claims: list[dict], corpus: knowledge.Corpus, publisher: str) -> list[dict]:
    """Drop claims the base already states: same publisher, metric, place, year, unit and value."""
    have = {(_key(c), float(c["value"])) for c in corpus.claims
            if c.get("value") is not None and corpus.sources.get(c.get("source"), {}).get("publisher") == publisher
            and _num(c.get("value")) is not None}
    kept = [c for c in claims if (_key(c), float(c["value"])) not in have]
    sid = claims[0]["id"].split(":")[0] if claims else ""
    return [{**c, "id": f"{sid}:c{n:03d}"} for n, c in enumerate(kept, 1)]


def check(claims: Sequence[Mapping[str, Any]], corpus: knowledge.Corpus, scenario: str, min_claims: int,
          found: int) -> list[str]:
    """Why an edition must not be written (empty when it is fine)."""
    problems = []
    if found < min_claims:
        problems.append(f"only {found} values found, expected at least {min_claims}: the table may have changed")
    previous: dict[tuple, tuple[str, float]] = {}
    for c in corpus.claims:
        if c.get("scenario") == scenario and _num(c.get("value")) is not None and c.get("horizon"):
            published = str(corpus.sources.get(c.get("source"), {}).get("published") or "")
            if _key(c) not in previous or published > previous[_key(c)][0]:
                previous[_key(c)] = (published, float(c["value"]))
    for c in claims:
        value, unit = float(c["value"]), c["unit"]
        lo, hi = BOUNDS.get(unit, (float("-inf"), float("inf")))
        if not lo <= value <= hi:
            problems.append(f"{c['metric']} {c.get('horizon', '')} {c.get('geo')} = {value} {unit} is outside {lo}..{hi}")
        old = previous.get(_key(c))
        if old is not None:
            moved = abs(value - old[1]) if unit == "%" else abs(value / old[1] - 1) if old[1] else 0.0
            limit = MAX_MOVE_PP if unit == "%" else MAX_MOVE_REL
            if moved > limit:
                problems.append(f"{c['metric']} {c.get('horizon')} {c.get('geo')} moved from {old[1]} to {value} "
                                f"since the {old[0]} edition")
    return problems


def write(root: Path, edition: Edition) -> Path:
    """Add the edition's source folder and any missing entities; undo both if the base breaks."""
    sid = edition.source["id"]
    folder = root / "sources" / sid
    if folder.exists():
        raise TableError(f"sources/{sid} already exists")
    entities_path = root / "entities.jsonl"
    before = entities_path.read_text(encoding="utf-8") if entities_path.exists() else ""
    have = {json.loads(line)["id"] for line in before.splitlines() if line.strip()}
    needed = sorted({e for c in edition.claims for e in c["entities"]} - have)
    missing = [e for e in needed if e not in ENTITIES]
    if missing:
        raise TableError(f"claims name entities the base does not have: {', '.join(missing)}")
    folder.mkdir(parents=True)
    try:
        (folder / "source.json").write_text(json.dumps(edition.source, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
        knowledge.write_jsonl(folder / "claims.jsonl", edition.claims)
        knowledge.write_jsonl(folder / "relations.jsonl", [])
        if needed:
            sep = "" if not before or before.endswith("\n") else "\n"
            entities_path.write_text(before + sep + "".join(json.dumps({"id": e, **ENTITIES[e]}, ensure_ascii=False) + "\n"
                                                            for e in needed), encoding="utf-8")
        errors = knowledge.load(root).errors
        if errors:
            raise TableError("the base would not load: " + "; ".join(errors[:3]))
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        entities_path.write_text(before, encoding="utf-8")
        raise
    return folder


def mark_inbox(root: Path, reader: str, watch_key: Optional[str], sid: str) -> int:
    """Mark the inbox row for this edition as read into the base; returns how many rows changed."""
    path = root / "inbox.jsonl"
    if watch_key is None or not path.exists():
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    for row in rows:
        if row.get("source") == reader and row.get("edition") == watch_key and row.get("status") != "read":
            row.update(status="read", kb_source=sid)
            changed += 1
    if changed:
        knowledge.write_jsonl(path, rows)
    return changed


def update(root: Path, crawler: watch.Crawler, *, today: date, only: Sequence[str] = (),
           dry_run: bool = False) -> list[dict[str, Any]]:
    """Read every publisher's current edition that the base does not have yet."""
    state = watch.load_json(root / "watch_state.json", {})
    corpus = knowledge.load(root)
    if corpus.errors:
        raise TableError("the base has errors: " + "; ".join(corpus.errors[:3]))
    results = []
    for rid, reader in READERS.items():
        if only and rid not in only:
            continue
        out: dict[str, Any] = {"reader": rid}
        try:
            ident = reader.ident(state.get(rid) or {}, crawler, corpus)
            if ident is None:
                out.update(status="no edition", detail="the watch list has no current edition yet")
            elif (root / "sources" / ident[0]).exists():
                out.update(status="current", source=ident[0])
            else:
                edition = reader.read(ident[1], crawler, today, corpus)
                found = len(edition.claims)
                edition.claims = dedupe(edition.claims, corpus, edition.source["publisher"])
                problems = check(edition.claims, corpus, reader.scenario, reader.min_claims, found)
                if problems:
                    raise TableError("; ".join(problems[:4]) + (f" (+{len(problems) - 4} more)" if len(problems) > 4 else ""))
                out.update(source=edition.source["id"], claims=len(edition.claims), duplicates=found - len(edition.claims))
                if dry_run:
                    out.update(status="would add", preview=[c["text"] for c in edition.claims[:3]])
                else:
                    write(root, edition)
                    mark_inbox(root, rid, edition.watch_key, edition.source["id"])
                    corpus = knowledge.load(root)
                    out.update(status="added")
        except TableError as exc:
            out.update(status="failed", detail=str(exc))
        results.append(out)
    return results


def summary(results: Sequence[Mapping[str, Any]]) -> str:
    lines = ["| Reader | Status | Source | Claims |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['reader']} | {r['status']} | {r.get('source') or r.get('detail', '')} | {r.get('claims', '')} |")
    return "\n".join(lines) + "\n"
