"""Foresea Edge Credibility Auditor CLI.

Audits live market edge candidates to evaluate:
- Statistical edge magnitude vs. evidence depth
- Resolution rule clarity and timeline self-containment
- Market liquidity and volume spread risks
- Horizon validity (rejecting past due / expired markets)

Usage:
    # Audit live edge board:
    python scripts/audit_market_edge.py

    # Audit with higher credibility threshold:
    python scripts/audit_market_edge.py --min-credibility 0.75 --limit 10
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.edge_credibility import audit_edge_opportunity  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("foresea-audit")

DEFAULT_FORESEA_URL = "https://foresea.ink"


def fetch_and_audit_edge_board(base_url: str = DEFAULT_FORESEA_URL, limit: int = 15) -> List[Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/edge-board"
    logger.info("Fetching edge board candidates from %s...", url)
    resp = requests.get(url, params={"limit": limit}, timeout=15)
    if resp.status_code != 200:
        logger.error("Failed to fetch edge board: HTTP %d", resp.status_code)
        return []

    data = resp.json()
    raw_opps = data.get("opportunities") or data.get("edge_board") if isinstance(data, dict) else (data if isinstance(data, list) else [])
    audited = []
    for opp in raw_opps[:limit]:
        aud = audit_edge_opportunity(opp)
        merged = dict(opp)
        merged.update(aud)
        audited.append(merged)
    return audited


def print_audit_report(audited_opps: List[Dict[str, Any]], min_credibility: float = 0.0) -> None:
    print("\n" + "=" * 80)
    print("                FORESEA MARKET EDGE CREDIBILITY AUDIT REPORT")
    print("=" * 80)

    passing = [o for o in audited_opps if o.get("credibility_score", 0.0) >= min_credibility]
    print(f"Total Evaluated: {len(audited_opps)} | Passing Filter (Score >= {min_credibility:.2f}): {len(passing)}\n")

    for i, opp in enumerate(audited_opps, 1):
        q = opp.get("question") or opp.get("title") or "Unknown"
        venue = opp.get("platform", "Venue")
        m_prob = opp.get("market_probability")
        f_prob = opp.get("model_probability")
        edge = opp.get("edge") or (abs(f_prob - m_prob) if f_prob is not None and m_prob is not None else 0.0)
        grade = opp.get("credibility_grade", "C")
        score = opp.get("credibility_score", 0.0)
        flags = opp.get("credibility_flags", [])
        summary = opp.get("audit_summary", "")

        status_icon = "[PASS]" if opp.get("is_credible") else "[WARN]"

        print(f"[{i}] {status_icon} Grade {grade} ({score*100:.0f}% Credibility) | [{venue}] {q[:60]}")
        print(f"    Market: {m_prob*100:.0f}% -> Model: {f_prob*100:.0f}% (Edge: {edge*100:+.1f}%)")
        print(f"    Flags: {', '.join(flags[:4])}")
        print(f"    Verdict: {summary}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Audit Foresea Market Edge Credibility")
    parser.add_argument("--base-url", default=os.getenv("FORESEA_BASE_URL", DEFAULT_FORESEA_URL))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-credibility", type=float, default=0.60)
    args = parser.parse_args()

    opps = fetch_and_audit_edge_board(base_url=args.base_url, limit=args.limit)
    if not opps:
        logger.error("No edge opportunities found to audit.")
        sys.exit(1)

    print_audit_report(opps, min_credibility=args.min_credibility)


if __name__ == "__main__":
    main()
