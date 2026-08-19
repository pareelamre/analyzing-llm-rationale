"""Foresea Official Developer CLI.

Provides unified CLI commands for interacting with Foresea's forecasting API,
multi-agent debates, Kelly portfolio optimizer, and cross-venue arbitrage scanner.

Usage:
    # 1. Calibrated Forecast
    python scripts/foresea_cli.py forecast "Will the Fed cut rates in September?"

    # 2. Bull vs. Bear Adversarial Debate
    python scripts/foresea_cli.py debate "Will Ethereum reach $5000 in 2026?" --platform Polymarket --prob 0.25

    # 3. Live Radar Desk Opportunities
    python scripts/foresea_cli.py radar --limit 5

    # 4. Cross-Venue Arbitrage Scanner
    python scripts/foresea_cli.py arbitrage --min-spread 0.03

    # 5. Kelly Portfolio Sizing
    python scripts/foresea_cli.py portfolio --bankroll 5000 --fraction 0.25

    # 6. Venue Orderbook
    python scripts/foresea_cli.py orderbook --platform kalshi --ident KXFED-25MAY

    # 7. System Health
    python scripts/foresea_cli.py health
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.sdk import Foresea  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Foresea Forecasting & Alpha Intelligence CLI")
    parser.add_argument("--url", default="https://foresea.ink", help="Foresea base URL")
    parser.add_argument("--key", default=None, help="API Key")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # forecast
    p_fc = subparsers.add_parser("forecast", help="Generate a calibrated forecast with live evidence")
    p_fc.add_argument("question", help="Forecasting question")
    p_fc.add_argument("--platform", default=None)
    p_fc.add_argument("--prob", type=float, default=None)

    # debate
    p_deb = subparsers.add_parser("debate", help="Conduct 3-agent Bull vs Bear adversarial debate")
    p_deb.add_argument("question", help="Forecasting question")
    p_deb.add_argument("--platform", default="Market")
    p_deb.add_argument("--prob", type=float, default=None)

    # radar
    p_rad = subparsers.add_parser("radar", help="List live mispriced opportunities on Radar Desk")
    p_rad.add_argument("--limit", type=int, default=10)

    # arbitrage
    p_arb = subparsers.add_parser("arbitrage", help="Scan cross-venue arbitrage between Polymarket & Kalshi")
    p_arb.add_argument("--min-spread", type=float, default=0.03)

    # portfolio
    p_port = subparsers.add_parser("portfolio", help="Calculate optimal Fractional Kelly portfolio sizing")
    p_port.add_argument("--bankroll", type=float, default=1000.0)
    p_port.add_argument("--fraction", type=float, default=0.25)
    p_port.add_argument("--min-edge", type=float, default=0.05)

    # orderbook
    p_ob = subparsers.add_parser("orderbook", help="Fetch venue orderbook")
    p_ob.add_argument("--platform", default="kalshi")
    p_ob.add_argument("--ident", required=True)

    # whale-flow
    p_wh = subparsers.add_parser("whale-flow", help="Track large block trades and institutional sentiment")
    p_wh.add_argument("--min-usd", type=float, default=250.0)
    p_wh.add_argument("--limit", type=int, default=20)

    # health
    subparsers.add_parser("health", help="Check system & venue latency health")

    args = parser.parse_args()
    client = Foresea(base_url=args.url, api_key=args.key)

    if args.command == "whale-flow":
        res = client.whale_flow(min_notional_usd=args.min_usd, limit=args.limit)
        print(f"\nWhale Sentiment: {res.get('whale_sentiment_label')} ({res.get('whale_sentiment_index_pct')}% Bullish)")
        print(f"Total Block Volume: ${res.get('total_whale_volume_usd'):,.2f} (YES: ${res.get('whale_yes_volume_usd'):,.2f} | NO: ${res.get('whale_no_volume_usd'):,.2f})\n")
        for p in res.get("top_prints", []):
            print(f"- [{p.get('platform')}] {p.get('side')} ${p.get('notional_usd'):,.2f} ({p.get('size')} contracts @ ${p.get('price'):.2f}) -> {p.get('market_title')[:55]}")

    elif args.command == "forecast":
        res = client.forecast(args.question, platform=args.platform, market_probability=args.prob)
        print(f"\n[FORECAST] Answer: {res.get('predicted_answer')} (Confidence: {res.get('confidence')})")
        print(f"Rationale:\n{res.get('rationale') or res.get('model_rationale')}\n")

    elif args.command == "debate":
        res = client.debate(args.question, platform=args.platform, market_probability=args.prob)
        print("\n=== BULL ADVOCATE (YES) ===")
        print(f"Thesis: {res['bull_agent'].get('thesis')}")
        print(f"Advocated Prob: {res['bull_agent'].get('advocated_probability')*100:.1f}%")
        print("\n=== BEAR ADVOCATE (NO) ===")
        print(f"Thesis: {res['bear_agent'].get('thesis')}")
        print(f"Advocated Prob: {res['bear_agent'].get('advocated_probability')*100:.1f}%")
        print("\n=== CHIEF RISK JUDGE ===")
        print(f"Verdict: {res['chief_risk_judge'].get('verdict')}")
        print(f"Blind Spots: {', '.join(res['chief_risk_judge'].get('blind_spots', []))}")
        print(f"Recommendation: {res.get('recommendation')}\n")

    elif args.command == "radar":
        res = client.radar(limit=args.limit)
        markets = res.get("markets", [])
        print(f"\nFound {len(markets)} Radar Opportunities:\n")
        for idx, m in enumerate(markets, 1):
            q = m.get("question", "")[:60]
            plat = m.get("platform", "Venue")
            m_prob = m.get("market_probability")
            f_prob = m.get("model_probability")
            edge = m.get("edge", 0.0)
            grade = m.get("credibility_grade", "A")
            print(f"[{idx}] [{plat}] {q} [Grade {grade}]")
            print(f"     Market: {m_prob*100:.0f}% -> Model: {f_prob*100:.0f}% (Edge: {edge*100:+.1f}%)\n")

    elif args.command == "arbitrage":
        opps = client.arbitrage(min_spread=args.min_spread)
        print(f"\nFound {len(opps)} Cross-Venue Arbitrage Opportunities:\n")
        for idx, o in enumerate(opps, 1):
            print(f"[{idx}] Spread: {o.get('spread_pct')}% (Gross ROI: {o.get('gross_roi_pct')}%)")
            print(f"     Strategy: {o.get('strategy')}")
            print(f"     Polymarket: {o['polymarket'].get('question')[:50]} @ {o['polymarket'].get('probability')*100:.1f}%")
            print(f"     Kalshi:     {o['kalshi'].get('question')[:50]} @ {o['kalshi'].get('probability')*100:.1f}%\n")

    elif args.command == "portfolio":
        res = client.optimize_portfolio(bankroll_usd=args.bankroll, kelly_fraction=args.fraction, min_edge=args.min_edge)
        print(f"\nBankroll: ${res.get('bankroll_usd'):,.2f} | Allocated: ${res.get('allocated_usd'):,.2f} ({res.get('capital_utilization_pct')}%) | Cash: ${res.get('cash_reserve_usd'):,.2f}")
        print(f"Estimated CAGR: {res.get('estimated_annualized_cagr_pct')}%\n")
        for p in res.get("allocations", []):
            print(f"- [{p.get('platform')}] {p.get('side')} x {p.get('contracts')} @ ${p.get('entry_price'):.2f} (${p.get('allocation_usd'):,.2f}, {p.get('allocated_pct')}%) -> {p.get('question')[:55]}")

    elif args.command == "orderbook":
        res = client.orderbook(platform=args.platform, ident=args.ident)
        print(json.dumps(res, indent=2))

    elif args.command == "health":
        res = client.system_health()
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
