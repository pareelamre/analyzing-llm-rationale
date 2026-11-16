"""What Foresea's knowledge base says about the fund's commodity pairs (owner, 2026-11-17).

For each pair the newest source in the base that forecasts both legs' prices for this year and
next gives an implied price ratio for each year. If the ratio is set to rise, that source expects
leg ``a`` to gain on leg ``b``; if it falls, the reverse. The fund attaches that view to every pair
trade as evidence and records whether it agrees with the trade's side. The view does not change
sizes: it is new and unscored, so it is shown first and judged on resolved trades before it may
move money (the same path the rationale guard took, docs/fund.md).

Ratios of forecast averages are compared only within one source and one unit system, so the view
is about direction, never about the level of an MCX ratio. Adverse, downside and futures-based
scenarios are left out: the view uses each source's baseline.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Mapping, Optional, Sequence

from . import knowledge

#: The knowledge-base metric (and its geo) that forecasts each pair leg's underlying. MCX natural
#: gas follows Henry Hub; MCX crude follows WTI, for which Brent stands in as the direction.
LEG_METRICS = {
    "copper": ("copper price", "global"), "aluminium": ("aluminium price", "global"),
    "zinc": ("zinc price", "global"), "lead": ("lead price", "global"), "nickel": ("nickel price", "global"),
    "tin": ("tin price", "global"), "gold": ("gold price", "global"), "silver": ("silver price", "global"),
    "platinum": ("platinum price", "global"), "crude": ("Brent crude price", "global"),
    "natgas": ("US natural gas price", "US"),
}
NAMES = {"crude": "crude oil", "natgas": "natural gas"}
#: A forecast ratio change smaller than this is no tilt either way.
TILT = 0.01
NOT_BASELINE = re.compile(r"adverse|downside|severe|futures|high .* case|low .* case", re.I)


def _forecasts(index: knowledge.Index, metric: tuple[str, str], as_of: date) -> dict[str, dict[int, dict]]:
    """Baseline forecasts for a (metric, geo): source id -> year -> figure."""
    out: dict[str, dict[int, dict]] = {}
    for fig in index.figures(metric[0], as_of=as_of, geo=metric[1]):
        claim = fig["claim"]
        when = knowledge.parse_when(claim.get("horizon"))
        if when is None or NOT_BASELINE.search(str(claim.get("scenario") or "")):
            continue
        out.setdefault(claim["source"], {})[when.year] = fig
    return out


def pair_view(index: knowledge.Index, a: str, b: str, today: date) -> Optional[dict[str, Any]]:
    """The newest source's expected move in the a/b price ratio from this year to next, or None."""
    metric_a, metric_b = LEG_METRICS.get(a), LEG_METRICS.get(b)
    if not metric_a or not metric_b or metric_a == metric_b:
        return None
    years = (today.year, today.year + 1)
    fa, fb = _forecasts(index, metric_a, today), _forecasts(index, metric_b, today)
    usable = [sid for sid in set(fa) & set(fb) if all(y in fa[sid] and y in fb[sid] for y in years)]
    if not usable:
        return None
    sid = max(usable, key=lambda s: str(fa[s][years[0]]["published"]))
    try:
        ratios = [float(fa[sid][y]["claim"]["value"]) / float(fb[sid][y]["claim"]["value"]) for y in years]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    change = ratios[1] / ratios[0] - 1 if ratios[0] else 0.0
    favours = "a" if change > TILT else "b" if change < -TILT else None
    source = fa[sid][years[0]]["source"]
    return {"source": sid, "cite": f"{source.get('publisher')}, {knowledge.month_label(source.get('published'))}",
            "years": list(years), "ratio": [round(r, 4) for r in ratios], "change": round(change, 4),
            "favours": favours,
            "claims": sorted({fa[sid][y]["claim"]["id"] for y in years} | {fb[sid][y]["claim"]["id"] for y in years})}


def _name(underlying: str) -> str:
    return NAMES.get(underlying, underlying)


def annotate(signals: Sequence[Any], pairs: Sequence[Mapping[str, Any]], today: date, *,
             index: Optional[knowledge.Index] = None) -> dict[str, Any]:
    """Attach the base's view to each pair signal as a fact; return every pair's view for the page.

    ``signals`` are fund_orchestrator.Signal objects; each pair leg gets its own copy of the facts.
    """
    if index is None:
        try:
            index = knowledge.shared_index()
        except Exception:  # aqg: top-level boundary -- the base is optional context
            index = None
    if index is None:
        return {}
    views: dict[str, Any] = {}
    for pair in pairs:
        a, b = pair["a"].get("underlying"), pair["b"].get("underlying")
        view = pair_view(index, str(a), str(b), today)
        if view is None:
            continue
        legs = [s for s in signals if getattr(s, "pair_key", None) == pair["key"]]
        lead = next((s for s in legs if s.instrument.get("id") == pair["a"]["id"]), None)
        wanted = None if lead is None else ("a" if lead.side == "LONG" else "b")
        agrees = None if wanted is None or view["favours"] is None else view["favours"] == wanted
        y0, y1 = view["years"]
        r0, r1 = view["ratio"]
        text = (f"{view['cite']} forecasts imply {_name(a)}/{_name(b)} moving from {r0:.2f} ({y0}) to {r1:.2f} ({y1}), "
                f"{view['change'] * 100:+.1f}%")
        if wanted is not None:
            bought = _name(a) if wanted == "a" else _name(b)
            verdict = ("no clear tilt" if agrees is None else
                       f"agrees with buying {bought}" if agrees else f"against buying {bought}")
            fact = {"label": "knowledge base view", "value": f"{text}: {verdict}", "source": view["cite"]}
            for s in legs:
                s.facts = [*s.facts, fact]
        views[pair["key"]] = {**view, "traded": wanted is not None, "agrees": agrees, "text": text}
    return views
