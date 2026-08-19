"""Foresea Cross-Venue Arbitrage & Spread Scanner.

Detects synthetic arbitrage and statistical mispricing spreads between Polymarket and Kalshi
on identical or highly correlated underlying events (e.g. Fed rate decisions, CPI, macro, elections).

Arbitrage Mechanics:
1. Long Synthetic Arbitrage:
   Buy YES on Venue A @ P_A + Buy NO on Venue B @ P_B_no (where P_B_no = 1 - P_B).
   Cost = P_A + (1 - P_B).
   Guaranteed Payout = $1.00.
   Arbitrage Spread = 1.00 - (P_A + 1 - P_B) = P_B - P_A.
   If P_B > P_A + fees, instant risk-free / low-risk spread exists.

2. Statistical Spread:
   Absolute divergence: |P_Polymarket - P_Kalshi|.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _normalize_title(text: str) -> str:
    """Normalize question string for entity & keyword matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [w for w in text.split() if w not in {"will", "the", "a", "an", "in", "by", "to", "of", "on", "for", "is", "at", "be"}]
    return " ".join(tokens)


def _compute_keyword_overlap(text_a: str, text_b: str) -> float:
    """Calculate Jaccard overlap between key tokens."""
    tokens_a = set(_normalize_title(text_a).split())
    tokens_b = set(_normalize_title(text_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(intersection) / len(union)


def scan_cross_venue_arbitrage(
    polymarket_markets: Optional[List[Dict[str, Any]]] = None,
    kalshi_markets: Optional[List[Dict[str, Any]]] = None,
    min_spread: float = 0.03,
    min_overlap: float = 0.35,
) -> List[Dict[str, Any]]:
    """Scan and match overlapping markets across Polymarket and Kalshi."""
    if polymarket_markets is None or kalshi_markets is None:
        try:
            from analyzing_llm_rationale import market_data
            poly = market_data.list_polymarket(limit=30) if polymarket_markets is None else polymarket_markets
            kalshi = market_data.list_kalshi(limit=30) if kalshi_markets is None else kalshi_markets
        except Exception:
            poly = polymarket_markets or []
            kalshi = kalshi_markets or []
    else:
        poly = polymarket_markets
        kalshi = kalshi_markets

    opportunities: List[Dict[str, Any]] = []

    for p in poly:
        p_q = p.get("question") or p.get("title") or ""
        p_prob = p.get("probability")
        if not p_q or p_prob is None:
            continue

        for k in kalshi:
            k_q = k.get("question") or k.get("title") or ""
            k_prob = k.get("probability")
            if not k_q or k_prob is None:
                continue

            overlap = _compute_keyword_overlap(p_q, k_q)
            if overlap >= min_overlap:
                spread = abs(p_prob - k_prob)
                if spread >= min_spread:
                    # Check which venue is cheaper
                    if p_prob < k_prob:
                        buy_venue = "Polymarket"
                        buy_price = p_prob
                        sell_venue = "Kalshi"
                        sell_price = k_prob
                        strategy = f"BUY YES on Polymarket @ {p_prob*100:.1f}% + BUY NO on Kalshi @ {(1-k_prob)*100:.1f}%"
                    else:
                        buy_venue = "Kalshi"
                        buy_price = k_prob
                        sell_venue = "Polymarket"
                        sell_price = p_prob
                        strategy = f"BUY YES on Kalshi @ {k_prob*100:.1f}% + BUY NO on Polymarket @ {(1-p_prob)*100:.1f}%"

                    total_entry_cost = round(buy_price + (1.0 - sell_price), 4)
                    gross_roi_pct = round((spread / total_entry_cost) * 100, 2) if total_entry_cost > 0 else 0.0

                    opportunities.append({
                        "event_summary": p_q[:80],
                        "overlap_score": round(overlap, 2),
                        "spread": round(spread, 4),
                        "spread_pct": round(spread * 100, 2),
                        "gross_roi_pct": gross_roi_pct,
                        "strategy": strategy,
                        "polymarket": {
                            "question": p_q,
                            "probability": round(p_prob, 4),
                            "url": p.get("market_url") or "",
                            "ticker": p.get("id") or "",
                        },
                        "kalshi": {
                            "question": k_q,
                            "probability": round(k_prob, 4),
                            "url": k.get("market_url") or "",
                            "ticker": k.get("ticker") or "",
                        },
                        "executable_action": {
                            "long_venue": buy_venue,
                            "long_price": buy_price,
                            "short_venue": sell_venue,
                            "short_price_no": round(1.0 - sell_price, 4),
                            "net_cost": total_entry_cost,
                        },
                    })

    # Sort descending by spread
    opportunities.sort(key=lambda x: x["spread"], reverse=True)
    return opportunities
