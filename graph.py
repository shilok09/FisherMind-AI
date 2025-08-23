import os
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from state import State
from agents.agent import phil_fisher_agent_core, generate_fisher_output
# Set LLM credentials
os.environ["OPENAI_API_KEY"] = os.getenv("GITHUB_TOKEN", "GITHUB_TOKEN")
os.environ["OPENAI_API_BASE"] = "https://models.github.ai/inference"

llm = ChatOpenAI(
    model="openai/gpt-4.1",
    temperature=0.7
)

# Node: LLM (handles conversation and decides if analysis is needed)
def llm_node(state: State) -> State:
    """
    LLM node: If user query requires analysis, set a flag to trigger analysis node.
    Otherwise, just generate a conversational response.
    """
    user_message = state.user_message or ""
    # Simple trigger: if user asks about "fisher" or "analysis", run agent
    trigger_keywords = ["fisher", "analysis", "signal", "should I invest", "growth", "valuation"]
    if any(kw in user_message.lower() for kw in trigger_keywords):
        state.data["run_analysis"] = True
    else:
        state.data["run_analysis"] = False

    # If analysis already present, generate response with it
    if state.data.get("run_analysis") and state.analysis_data:
        # Use generate_fisher_output to respond with analysis
        return generate_fisher_output(
            ticker=state.data["tickers"][0],  # or loop for multi-ticker
            analysis_data=state.analysis_data[state.data["tickers"][0]],
            llm=llm,
            state=state
        )
    elif not state.data.get("run_analysis"):
        # Just echo back or do a generic LLM response
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful financial chatbot. Answer user questions conversationally."),
                ("user", state.user_message or ""),
            ]
        )
        response = llm.invoke(prompt)
        response_text = getattr(response, "content", None) or str(response)
        state.chat_history.append({"role": "user", "content": state.user_message})
        state.chat_history.append({"role": "bot", "content": response_text})
        state.user_message = None
        return state
    else:
        # If analysis is needed but not present, pass to analysis node
        return state

# Node: Analysis (runs only if flagged by LLM node)
def analysis_node(state: State) -> State:
    return phil_fisher_agent_core(state)

# Build the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("llm", llm_node)
graph.add_node("analysis", analysis_node)

# Define edges
graph.add_edge(START, "llm")
graph.add_conditional_edges(
    "llm",
    lambda state: "analysis" if state.data.get("run_analysis") and not state.analysis_data else END,
    {"analysis": "analysis", END: END}
)
graph.add_edge("analysis", "llm")  # After analysis, return to LLM for response

# Compile the workflow
workflow = graph.compile()

# Example usage:
if __name__ == "__main__":
    # Example initial state
    state = State(
        user_message="Please provide a full Phil Fisher-style fundamental analysis for the stock AAPL. Include assessments of growth quality, margins stability, management efficiency, valuation, insider activity, and sentiment. Give detailed reasoning for each score.",
        chat_history=[],
        data={
            "end_date": "2025-8-23",
            "tickers": ["AAPL"],
            "preloaded": {},
        }
    )
    # Run the workflow
    final_state = workflow.invoke(state)
    print(final_state)

