"""Trade notes for the Foresea Fund that can only say what the numbers say (docs/fund.md).

Every order carries facts: a label, a value and the source the value came from. The note is
built from those alone. A model may put them into plain sentences, but its text is checked:
any number in it must appear among the facts, and it must not be empty or run long. A note
that fails is replaced by the plain fact sheet, and the ledger records which one was used
(note_source "model" or "facts").
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Optional, Sequence

#: A model note longer than this is rambling, not explaining.
MAX_NOTE_CHARS = 420
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

PROMPT = (
    "Explain this paper trade in at most two plain sentences, for a reader who is not a trader. "
    "Use only the facts listed below. Do not add any number, date, name or source that is not "
    "in the list, and do not predict anything beyond what the facts say.\n\nTrade: {headline}\n\nFacts:\n{facts}\n"
)


def headline(order: Mapping[str, Any]) -> str:
    verb = {"exit": "Sell", "hedge": "Hedge with"}.get(order["purpose"], "Buy")
    label = order["instrument"].get("label") or order["instrument"]["id"]
    if order["side"] in ("BULL", "BEAR", "MOVE", "RANGE"):  # an option structure's label says what it is
        return f"{verb} {label}"
    return f"{verb} {order['side']} {label}"


def fact_sheet(order: Mapping[str, Any]) -> str:
    parts = [f"{f['label']}: {f['value']} ({f['source']})" for f in order["facts"]]
    reason = f" Reason: {order['reason']}." if order.get("reason") else ""
    return f"{headline(order)}. " + "; ".join(parts) + "." + reason


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "").rstrip(".") for m in _NUMBER.finditer(text)}


def grounded(text: str, facts: Sequence[Mapping[str, Any]], *, headline_text: str = "") -> bool:
    """True when every number in ``text`` is one the facts (or the headline) state."""
    if not text or len(text) > MAX_NOTE_CHARS:
        return False
    allowed = set().union(*(_numbers(f"{f['label']} {f['value']} {f['source']}") for f in facts), _numbers(headline_text))
    return _numbers(text) <= allowed


def write_note(order: Mapping[str, Any], model: Optional[Callable[[str], str]] = None) -> dict[str, Any]:
    """The note for ``order``: the model's wording if it passes the check, else the fact sheet."""
    sheet = fact_sheet(order)
    evidence = [{"source": f["source"], "figure": f"{f['label']}: {f['value']}"} for f in order["facts"]]
    if model is not None:
        prompt = PROMPT.format(headline=headline(order),
                               facts="\n".join(f"- {f['label']}: {f['value']} ({f['source']})" for f in order["facts"]))
        try:
            text = re.sub(r"<think>.*?</think>", "", model(prompt) or "", flags=re.S).strip()
        except Exception:  # the Mac may be asleep; the fact sheet still explains the trade
            text = ""
        if grounded(text, order["facts"], headline_text=headline(order)):
            return {"note": text, "note_source": "model", "evidence": evidence}
    return {"note": sheet, "note_source": "facts", "evidence": evidence}
