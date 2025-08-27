import os
from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from tools.mcp import MCPClientManager
import json
from pydantic import BaseModel
import asyncio
from state import State  # Import the State class from graph.py)
from agents.analysis import (
    analyze_fisher_growth_quality,
    analyze_margins_stability,
    analyze_management_efficiency_leverage,
    analyze_fisher_valuation,
    analyze_insider_activity,
    analyze_sentiment,
)
from agents.update_preloaded_cache import update_preloaded_cache
from langsmith import traceable

mcp_manager = MCPClientManager()
tools = asyncio.run(mcp_manager.load_tools())
tools = {tool.name: tool for tool in tools}

@traceable(run_type="chain", name="phil_fisher_agent_core")
def phil_fisher_agent_core(
    state: State,
    agent_id: str = "phil_fisher_agent",
    *,
    api_key: Optional[str] = None,
) -> State:
    """
    Conversational agent core for Phil Fisher analysis.
    Uses preserved tools to fetch data if not preloaded in state.data.
    Computes analysis and updates state.analysis_data and state.analyst_signals.
    """
    end_date = state.data["end_date"]
    tickers: List[str] = state.data["tickers"]
    preloaded: Dict[str, Any] = state.data.get("preloaded", {})

    analysis_data: Dict[str, Any] = {}
    print(f"Running Phil Fisher analysis for tickers: {tickers}")
    for ticker in tickers:
        # Initialize raw results dictionary to collect fetched data
        raw_results = {}
        
        pli = preloaded.get(ticker, {}).get("financial_line_items")
        if pli is None:
            pli = asyncio.run(tools["search_line_items"].ainvoke({
                "ticker": ticker,
                "line_items": [
                    "revenue", "net_income", "earnings_per_share", "free_cash_flow",
                    "research_and_development", "operating_income", "operating_margin",
                    "gross_margin", "total_debt", "shareholders_equity",
                    "cash_and_equivalents", "ebit", "ebitda"
                ],
                "end_date": end_date,
                "period": "annual",
                "limit": 5
            }))
            raw_results["financial_line_items"] = pli
        

        mcap = preloaded.get(ticker, {}).get("market_cap")
        if mcap is None:
            mcap = asyncio.run(tools["get_market_cap"].ainvoke({
                "ticker": ticker,
                "end_date": end_date
            }))
            raw_results["market_cap"] = mcap
        # Fix: convert mcap to float if it's a string
        if isinstance(mcap, str):
            try:
                mcap = float(mcap.replace(",", ""))
            except Exception:
                try:
                    mcap_obj = json.loads(mcap)
                    if isinstance(mcap_obj, dict) and "market_cap" in mcap_obj:
                        mcap = float(mcap_obj["market_cap"])
                    else:
                        print(f"[ERROR] Could not extract market_cap from JSON: {mcap}")
                except Exception as e:
                    print(f"[ERROR] Failed to parse or extract market_cap: {e}")

        insiders = preloaded.get(ticker, {}).get("insider_trades")
        if insiders is None:
            insiders = asyncio.run(tools["get_insider_news"].ainvoke({
                "ticker": ticker,
                "end_date": end_date,
                "limit": 10
            }))
            raw_results["insider_trades"] = insiders

        news = preloaded.get(ticker, {}).get("company_news")
        if news is None:
            news = asyncio.run(tools["get_company_news"].ainvoke({
                "ticker": ticker,
                "end_date": end_date,
                "limit": 50
            }))
            raw_results["company_news"] = news
        
        if isinstance(pli, str):
            try:
                pli = json.loads(pli)
            except Exception as e:
                print(f"[ERROR] Failed to parse pli JSON: {e}")
        print(f"[DEBUG] pli type: {type(pli)}, length: {len(pli) if hasattr(pli, '__len__') else 'N/A'}")
        if pli and isinstance(pli, list) and len(pli) > 0:
            print(f"[DEBUG] pli[0] type: {type(pli[0])}, pli[0] sample: {pli[0]}")
        else:
            print(f"[DEBUG] pli is empty or not a list: {pli}")
        growth_quality = analyze_fisher_growth_quality(pli)
        margins_stability = analyze_margins_stability(pli)
        mgmt_efficiency = analyze_management_efficiency_leverage(pli)
        fisher_valuation = analyze_fisher_valuation(pli, mcap)
        insider_activity = analyze_insider_activity(insiders)
        sentiment_analysis = analyze_sentiment(news)

        total_score = (
            growth_quality["score"] * 0.30
            + margins_stability["score"] * 0.25
            + mgmt_efficiency["score"] * 0.20
            + fisher_valuation["score"] * 0.15
            + insider_activity["score"] * 0.05
            + sentiment_analysis["score"] * 0.05
        )

        if total_score >= 7.5:
            signal_lbl = "bullish"
        elif total_score <= 4.5:
            signal_lbl = "bearish"
        else:
            signal_lbl = "neutral"

        analysis_data[ticker] = {
            "signal": signal_lbl,
            "score": total_score,
            "max_score": 10,
            "growth_quality": growth_quality,
            "margins_stability": margins_stability,
            "management_efficiency": mgmt_efficiency,
            "valuation_analysis": fisher_valuation,
            "insider_activity": insider_activity,
            "sentiment_analysis": sentiment_analysis,
        }
        print(f"Analysis for {ticker}: Signal={signal_lbl}, Score={total_score:.2f}/10")
        
        # Update preloaded cache with fetched data and analysis results
        if raw_results:  # Only update if we fetched new data
            preloaded = update_preloaded_cache(ticker, raw_results, analysis_data[ticker], preloaded)
            print(f"Updated preloaded cache for {ticker}")

    # Update state fields for downstream nodes
    if state.analyst_signals is None:
        state.analyst_signals = {}
    state.analyst_signals[agent_id] = analysis_data
    state.analysis_data = analysis_data
    
    # Update the state's preloaded data with the updated cache
    state.data["preloaded"] = preloaded

    return state

@traceable(run_type="chain", name="generate_fisher_output")
def generate_fisher_output(
    *,
    ticker: str,
    analysis_data: Dict[str, Any],
    llm: Any,
    state: State
) -> State:
    """
    Generate a conversational response using ChatPromptTemplate and state.
    Updates chat history and returns the new state.
    """
    user_message = state.user_message
    chat_history = state.chat_history or []

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Phil Fisher AI agent, making investment decisions using his principles:

                1. Emphasize long-term growth potential and quality of management.
                2. Focus on companies investing in R&D for future products/services.
                3. Look for strong profitability and consistent margins.
                4. Willing to pay more for exceptional companies but still mindful of valuation.
                5. Rely on thorough research (scuttlebutt) and thorough fundamental checks.

                When providing your reasoning, be thorough and specific by:
                1. Discussing the company's growth prospects in detail with specific metrics and trends
                2. Evaluating management quality and their capital allocation decisions
                3. Highlighting R&D investments and product pipeline that could drive future growth
                4. Assessing consistency of margins and profitability metrics with precise numbers
                5. Explaining competitive advantages that could sustain growth over 3-5+ years
                6. Using Phil Fisher's methodical, growth-focused, and long-term oriented voice

                """,
            ),
            (
                "human",
                """Based on the following analysis, create a Phil Fisher-style investment signal and Response User Query.

                 Analysis Data for {ticker}:
                {analysis_data}
                USER QUERY: {user_message}
            """,
            ),
        ]
    )

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker,
        "user_message": user_message or ""
    })

    print(f"Generating response for {ticker} with user message: {user_message}")    
    # Get LLM response (as text)
    response = llm.invoke(
        prompt,
        config={
            "metadata": {
                "node": "generate_fisher_output",
                "ticker": ticker,
            }
        },
    )
    response_text = getattr(response, "content", None) or str(response)

    # Update chat history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "bot", "content": response_text})

    # Update state and return
    state.chat_history = chat_history
    state.user_message = None  # Clear user message after processing
    return state
