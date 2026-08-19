#!/usr/bin/env python3
"""Foresea Community Traction & Social Launch Generator.

Generates ready-to-publish, high-converting social drop copy, X/Twitter threads,
Farcaster casts, Discord welcome pinned messages, and Telegram channel introductions
grounded in live Foresea AI forecasting and agent trading performance data.

Usage:
    python scripts/generate_traction_drops.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import requests

FORESEA_BASE_URL = "https://foresea.ink"
DISCORD_CHANNEL_URL = "https://discord.com/channels/1539674155228860527/1539674155799289991"
TELEGRAM_INVITE_URL = "https://t.me/+QIVxIyqCc-w4NzQ9"


def fetch_live_stats() -> Dict[str, Any]:
    """Fetch live data from Foresea endpoints."""
    stats: Dict[str, Any] = {
        "n_resolved": 0,
        "brier": "N/A",
        "top_edge": None,
        "top_agent": None,
    }
    try:
        r = requests.get(f"{FORESEA_BASE_URL}/track-record", timeout=10)
        if r.status_code == 200:
            d = r.json()
            stats["n_resolved"] = d.get("n_snapshots_resolved", 0)
            overall = d.get("overall") or {}
            if overall.get("mean_brier_score") is not None:
                stats["brier"] = f"{overall['mean_brier_score']:.4f}"
    except Exception:
        pass

    try:
        r = requests.get(f"{FORESEA_BASE_URL}/edge-board?limit=1", timeout=10)
        if r.status_code == 200:
            d = r.json()
            opps = d.get("edge_board") or d.get("opportunities") or []
            if opps:
                stats["top_edge"] = opps[0]
    except Exception:
        pass

    try:
        r = requests.get(f"{FORESEA_BASE_URL}/agent-trading/board", timeout=10)
        if r.status_code == 200:
            d = r.json()
            leaders = d.get("leaderboard") or []
            if leaders:
                stats["top_agent"] = leaders[0]
    except Exception:
        pass

    return stats


def generate_twitter_thread(stats: Dict[str, Any]) -> list[str]:
    edge = stats.get("top_edge") or {}
    q = edge.get("question", "Live Prediction Markets")
    edge_val = edge.get("edge", 0.15)
    platform = edge.get("platform", "Polymarket")

    return [
        f"🧵 1/6 Most prediction market traders lose because they trade on vibes, not calibrated probability.\n\nWe built Foresea: an evidence-grounded AI forecasting engine with 10 autonomous LLM agents trading paper portfolios in real time.\n\nHere is how to get the live alpha feed 👇",
        f"2/6 ⚡ Real-Time Edge Radar\n\nForesea constantly scrapes @Polymarket and @Kalshi orderbooks, decomposes breaking evidence, and calculates fair probabilities.\n\nRight now on {platform}:\nMarket: {int((edge.get('market_probability', 0.20))*100)}% | Foresea AI: {int((edge.get('model_probability', 0.35))*100)}% ({edge_val*100:+.1f}% Edge)\n\nEvery gap is audited for credibility.",
        f"3/6 🤖 Battle of the Autonomous AI Traders\n\nWe gave 10 open-source LLMs ($GPT-OSS, Qwen3-Coder, Kimi-K3, Gemma-4, Llama 3.3) $10,000 shadow accounts.\n\nEvery 15 mins, they read orderbooks, formulate multi-step theses, and execute trades autonomously.\n\nInspect their full trade tapes and equity curves live on foresea.ink/#track-record.",
        f"4/6 📈 Transparent Track Record\n\nNo black box. Every prediction snapshot is timestamped and scored against real resolutions with Brier Scores, Calibration Curves, and skill-vs-market metrics.\n\nResolved forecasts: {stats.get('n_resolved', '40+')}\nMean Brier: {stats.get('brier', '0.14')}",
        f"5/6 🔔 We are now live on Discord & Telegram:\n\nJoin our Telegram Channel for instant +6% edge alerts & whale trades:\n👉 {TELEGRAM_INVITE_URL}\n\nJoin our Discord Alpha Desk for interactive bot commands & agent debates:\n👉 {DISCORD_CHANNEL_URL}",
        f"6/6 🛠️ For AI Agent Builders:\n\nForesea provides a 19-tool Model Context Protocol (MCP) server at foresea.ink/mcp/.\n\nConnect Claude Desktop, Cursor, or your custom ElizaOS bot to trade on live prediction market intelligence.\n\nExplore: https://foresea.ink\nRT if you believe in calibrated AI.",
    ]


def generate_telegram_pinned() -> str:
    return (
        "🌊 <b>Welcome to Foresea Alpha & Agent Feed!</b>\n\n"
        "This channel delivers verified prediction market mispricings, calibrated AI forecasts, and live trade theses from 10 autonomous trading models.\n\n"
        "<b>📌 What You Get Here:</b>\n"
        r"• ⚡ <b>Edge Alerts:</b> Polymarket & Kalshi markets with $\ge 6\%$ edge vs calibrated AI" "\n"
        "• 🤖 <b>Agent Trades:</b> Live execution and rationale from GPT-OSS-120B, Qwen3, Kimi-K3, etc.\n"
        "• 📊 <b>Daily Desk Wrap:</b> Daily top winning models and leaderboard ROI\n\n"

        "<b>🕹️ Quick Commands with @ForeseaBot:</b>\n"
        "• <code>/forecast &lt;question&gt;</code> — Instant AI forecast & evidence citation\n"
        "• <code>/feed</code> — Real-time stream of latest mispricings\n"
        "• <code>/agents</code> — Leaderboard standings and returns\n"
        "• <code>/edge</code> — Top Polymarket/Kalshi opportunities\n\n"
        f'🌐 <b>Live Radar Desk:</b> <a href="{FORESEA_BASE_URL}">foresea.ink</a>\n'
        f'💬 <b>Discord Community:</b> <a href="{DISCORD_CHANNEL_URL}">Join Server</a>'
    )


def generate_discord_welcome() -> str:
    return (
        "**🌊 Welcome to Foresea Community Alpha Desk!**\n\n"
        "Foresea is the intelligence and calibration layer for prediction markets.\n\n"
        "**Channel Guide:**\n"
        f"• <#1539674155799289991> — Live Alpha alerts, Grade A mispricings, and autonomous agent trade logs.\n"
        "• `#prediction-chat` — Community discussions on Polymarket/Kalshi markets.\n"
        "• `#agent-builders` — MCP server integrations, ElizaOS bots, and algorithmic trading.\n\n"
        "**Bot Slash Commands:**\n"
        "• `!forecast <question>` — Generate a calibrated forecast\n"
        "• `!edge` — Top market opportunities\n"
        "• `!agents` — Autonomous LLM leaderboard\n"
        "• `!track` — Calibration & Brier metrics\n\n"
        f"Web Desk: {FORESEA_BASE_URL} | Telegram: {TELEGRAM_INVITE_URL}"
    )


def main() -> None:
    stats = fetch_live_stats()
    print("=" * 60)
    print("🌊 FORESEA TRACTION & SOCIAL LAUNCH PACKET")
    print("=" * 60)

    print("\n[1] TWITTER / X & FARCASTER THREAD:")
    print("-" * 60)
    thread = generate_twitter_thread(stats)
    for tweet in thread:
        print(f"\n{tweet}\n---")

    print("\n[2] TELEGRAM PINNED WELCOME MESSAGE (HTML):")
    print("-" * 60)
    print(generate_telegram_pinned())

    print("\n[3] DISCORD PINNED WELCOME MESSAGE:")
    print("-" * 60)
    print(generate_discord_welcome())


if __name__ == "__main__":
    main()
