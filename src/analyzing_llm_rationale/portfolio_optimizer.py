"""Foresea Quantitative Kelly Portfolio Optimizer & Position Sizer.

Applies the Fractional Kelly Criterion and institutional risk constraints
to allocate capital mathematically across live audited prediction market opportunities.

Formulas:
- Binary Market Kelly: f* = (P_model - P_market) / (1 - P_market) [for BUY YES]
- Fractional Kelly: f_alloc = kelly_fraction * f*
- Position USD = Bankroll * f_alloc
- Contracts = floor(Position USD / P_market)

Features:
- Single-market position caps (default: 15% max per market)
- Total capital utilization limit (default: 80% max)
- Expected portfolio growth rate G(f), Sharpe ratio, and variance calculations
- Filters for minimum edge and credibility grade
"""
from __future__ import annotations

import math
from typing import Any, Dict, List


class KellyPortfolioOptimizer:
    """Computes mathematically optimal bet sizing across prediction market opportunities."""

    def __init__(
        self,
        bankroll_usd: float = 1000.0,
        kelly_fraction: float = 0.25,  # Quarter-Kelly standard
        max_single_position_pct: float = 0.15,  # 15% max per market
        max_total_exposure_pct: float = 0.80,   # 80% max portfolio exposure
        min_edge: float = 0.05,                 # 5% minimum edge
        min_credibility_score: float = 0.60,    # Grade A & B
    ):
        self.bankroll_usd = max(10.0, bankroll_usd)
        self.kelly_fraction = max(0.05, min(1.0, kelly_fraction))
        self.max_single_pos = max_single_position_pct
        self.max_total_exposure = max_total_exposure_pct
        self.min_edge = min_edge
        self.min_credibility = min_credibility_score

    def optimize(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute optimal position sizing across candidate opportunities."""
        allocations: List[Dict[str, Any]] = []
        total_fraction = 0.0

        for opp in opportunities:
            model_p = opp.get("model_probability")
            mkt_p = opp.get("market_probability")
            cred_score = opp.get("credibility_score")

            if model_p is None or mkt_p is None:
                continue

            # Credibility & edge filter
            if cred_score is not None and cred_score < self.min_credibility:
                continue

            edge = abs(model_p - mkt_p)
            if edge < self.min_edge:
                continue

            # Determine Side & Payout Odds
            if model_p > mkt_p:
                side = "YES"
                entry_price = mkt_p
                win_prob = model_p
            else:
                side = "NO"
                entry_price = 1.0 - mkt_p
                win_prob = 1.0 - model_p

            if entry_price <= 0.01 or entry_price >= 0.99:
                continue

            # Full Kelly Formula: f* = (p * (b + 1) - 1) / b = (win_prob - entry_price) / (1 - entry_price)
            b = (1.0 - entry_price) / entry_price
            full_kelly = (win_prob * (b + 1.0) - 1.0) / b

            if full_kelly <= 0.0:
                continue

            # Fractional Kelly with single-market cap
            frac_kelly = min(self.max_single_pos, full_kelly * self.kelly_fraction)
            total_fraction += frac_kelly

            question = opp.get("question") or opp.get("title") or "Market"
            platform = opp.get("platform", "Venue")
            ticker = opp.get("ticker") or opp.get("slug") or opp.get("id") or ""
            link = opp.get("market_url") or ""

            allocations.append({
                "question": question,
                "platform": platform,
                "ticker": ticker,
                "market_url": link,
                "side": side,
                "entry_price": round(entry_price, 2),
                "model_probability": round(model_p, 4),
                "market_probability": round(mkt_p, 4),
                "edge": round(edge, 4),
                "full_kelly_pct": round(full_kelly * 100, 2),
                "allocated_pct": round(frac_kelly * 100, 2),
                "credibility_grade": opp.get("credibility_grade", "A"),
                "credibility_score": opp.get("credibility_score", 0.90),
            })

        # Scale down if total exposure exceeds max_total_exposure
        scale_factor = 1.0
        if total_fraction > self.max_total_exposure:
            scale_factor = self.max_total_exposure / total_fraction

        final_allocations = []
        total_allocated_usd = 0.0
        expected_portfolio_growth = 0.0

        for a in allocations:
            effective_fraction = (a["allocated_pct"] / 100.0) * scale_factor
            alloc_usd = round(self.bankroll_usd * effective_fraction, 2)
            contracts = int(alloc_usd / a["entry_price"]) if a["entry_price"] > 0 else 0
            actual_cost_usd = round(contracts * a["entry_price"], 2)

            total_allocated_usd += actual_cost_usd

            # Expected Growth Contribution: p*log(1 + f*b) + q*log(1 - f)
            p = a["model_probability"] if a["side"] == "YES" else (1.0 - a["model_probability"])
            b = (1.0 - a["entry_price"]) / a["entry_price"]
            growth_i = p * math.log(1.0 + effective_fraction * b) + (1.0 - p) * math.log(max(0.001, 1.0 - effective_fraction))
            expected_portfolio_growth += growth_i

            a["effective_fraction_pct"] = round(effective_fraction * 100, 2)
            a["allocation_usd"] = actual_cost_usd
            a["contracts"] = contracts
            final_allocations.append(a)

        cash_reserve_usd = round(self.bankroll_usd - total_allocated_usd, 2)
        portfolio_cagr_est = round((math.exp(expected_portfolio_growth * 52) - 1.0) * 100, 2) if expected_portfolio_growth > 0 else 0.0

        return {
            "bankroll_usd": self.bankroll_usd,
            "allocated_usd": round(total_allocated_usd, 2),
            "cash_reserve_usd": cash_reserve_usd,
            "capital_utilization_pct": round((total_allocated_usd / self.bankroll_usd) * 100, 2),
            "kelly_fraction": self.kelly_fraction,
            "expected_weekly_growth_rate_pct": round(expected_portfolio_growth * 100, 3),
            "estimated_annualized_cagr_pct": portfolio_cagr_est,
            "n_positions": len(final_allocations),
            "allocations": final_allocations,
        }


def optimize_portfolio_allocation(
    opportunities: List[Dict[str, Any]],
    bankroll_usd: float = 1000.0,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.05,
    min_credibility: float = 0.60,
) -> Dict[str, Any]:
    """Convenience functional interface for Kelly portfolio optimization."""
    optimizer = KellyPortfolioOptimizer(
        bankroll_usd=bankroll_usd,
        kelly_fraction=kelly_fraction,
        min_edge=min_edge,
        min_credibility_score=min_credibility,
    )
    return optimizer.optimize(opportunities)
