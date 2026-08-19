#!/usr/bin/env python3
"""Example: Autonomous LangChain Agent for Prediction Markets using Foresea Tools.

Demonstrates how to bind Foresea prediction market tools (calibrated forecasting,
orderbook analysis, and edge board scanning) to a LangChain agent.

Usage:
    python examples/langchain_prediction_agent.py
"""
from __future__ import annotations

import os
import sys

from analyzing_llm_rationale.langchain_tools import ForeseaClient, get_foresea_langchain_tools


def main() -> None:
    print("=" * 60)
    print("🌊 FORESEA LANGCHAIN PREDICTION MARKET AGENT DEMO")
    print("=" * 60)

    client = ForeseaClient()

    # 1. Test Direct Forecast via Client
    question = "Will OpenAI release GPT-5 before December 2026?"
    print(f"\n[1] Running Foresea Calibrated Forecast for: '{question}'...")
    try:
        res = client.forecast(question=question)
        prob = res.get("predicted_probability")
        ans = res.get("predicted_answer")
        rationale = res.get("rationale", "")[:200]
        print(f"✅ Predicted: {ans} ({prob*100:.1f}%)")
        print(f"📝 Rationale snippet: {rationale}...")
    except Exception as e:
        print(f"⚠️ Forecast endpoint note: {e}")

    # 2. Test Edge Board Scan
    print("\n[2] Scanning Live Polymarket & Kalshi Edge Board...")
    try:
        opps = client.get_edge_board(min_edge=0.05, limit=3)
        print(f"✅ Found {len(opps)} mispriced opportunities.")
        for i, o in enumerate(opps, 1):
            print(f"   {i}. [{o.get('platform')}] {o.get('question')} ({o.get('edge', 0)*100:+.1f}% edge)")
    except Exception as e:
        print(f"⚠️ Edge board note: {e}")

    # 3. Test Live Feed
    print("\n[3] Fetching Latest Alpha Feed Stream...")
    try:
        feed = client.get_feed_latest(limit=3)
        signals = feed.get("market_edge_signals", [])
        print(f"✅ Received {len(signals)} alpha signals.")
    except Exception as e:
        print(f"⚠️ Alpha feed note: {e}")

    print("\n" + "=" * 60)
    print("🤖 LANGCHAIN BINDING SNIPPET:")
    print("=" * 60)
    print("""
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from analyzing_llm_rationale.langchain_tools import get_foresea_langchain_tools

tools = get_foresea_langchain_tools()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an autonomous prediction market quantitative researcher."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run agent:
result = executor.invoke({"input": "Find the top 2 mispriced markets on Polymarket and forecast their true odds."})
print(result["output"])
    """)


if __name__ == "__main__":
    main()
