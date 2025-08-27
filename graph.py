import os
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from state import State
from agents.agent import phil_fisher_agent_core, generate_fisher_output
from langsmith import traceable, Client
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import StateGraph, MessagesState
from tools.mcp import MCPClientManager
from langgraph.types import Command
from langchain_core.tools import tool
from typing import Annotated
import asyncio

#=====================MCP====================
#for the basic calls done by the supervior agent
# mcp_manager = MCPClientManager()
# tools = asyncio.run(mcp_manager.load_tools())
# tools = {tool.name: tool for tool in tools}


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("GITHUB_TOKEN", "YOUR GITHUB TOKEN")
os.environ["OPENAI_API_BASE"] = "https://models.github.ai/inference"


llm = ChatOpenAI(
    model="openai/gpt-4.1",
    temperature=0.7,
)

# Simplified handoff tool without complex annotations
@tool
def transfer_to_fisher_analysis():
    """Transfer to the Phil Fisher analysis agent for full stock analysis."""
    return Command(
        goto="fisher_analysis_agent",
        graph=Command.PARENT,
    )



# Tool function for Phil Fisher core analysis
@tool
def phil_fisher_core_tool(
    state: Annotated[State, InjectedState]
) -> State:
    """Run Phil Fisher core analysis for the current state."""
    print("passing state to phil_fisher_agent_core...")
    return phil_fisher_agent_core(state)

# Tool function for generating Fisher output
@tool
def generate_fisher_output_tool(
    state: Annotated[State, InjectedState]
) -> State:
    """Generate conversational output for Fisher analysis using the current state."""
    # Defensive: check for tickers and analysis_data
    ticker = None
    if state.data and "tickers" in state.data and state.data["tickers"]:
        ticker = state.data["tickers"][0]
    else:
        # Optionally, handle this error more gracefully
        raise ValueError("No ticker found in state.data['tickers']")
    
    analysis_data = {}
    if getattr(state, "analysis_data", None) and isinstance(state.analysis_data, dict):
        analysis_data = state.analysis_data.get(ticker, {})
    # else: leave as empty dict
    print("Generating Fisher output...")
    return generate_fisher_output(
        ticker=ticker,
        analysis_data=analysis_data,
        llm=llm,
        state=state
    )

# 1. Define your Phil Fisher analysis agent as a node
fisher_analysis_agent = create_react_agent(
    model=llm,
    tools=[phil_fisher_core_tool, generate_fisher_output_tool],
    prompt="You are the Phil Fisher analysis agent. Use the phil_fisher_core_tool to analyze stocks and generate_fisher_output_tool to create responses.",
    name="fisher_analysis_agent"
)

# 2. Define your supervisor agent with the handoff tool
supervisor_agent = create_react_agent(
    model=llm,
    tools=list([transfer_to_fisher_analysis]),
    prompt="You are a supervisor agent. Your name is MR.PHIL. Have conversations about stocks and finance. For full Phil Fisher stock analysis, use the transfer_to_fisher_analysis tool.",
    name="supervisor_agent"
)

# 3. Register both agents in the graph
multi_agent_graph = (
    StateGraph(MessagesState)
    .add_node("supervisor_agent", supervisor_agent)
    .add_node("fisher_analysis_agent", fisher_analysis_agent)
    .add_edge(START, "supervisor_agent")
    .compile()
)

# Example usage:
if __name__ == "__main__":
    print("Welcome to Phil Fisher AI! Type 'exit' to quit.\n")
    
    while True:
        user_input = input("YOU: ")
        if user_input.strip().lower() == "exit":
            print("AI: Goodbye!")
            break

        try:
            # Run the multi-agent graph
            response_chunks = multi_agent_graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": "1"}}
            )
            
            ai_response = ""
            for chunk in response_chunks:
                # Extract the actual response from the chunk
                if isinstance(chunk, dict):
                    for node_name, node_output in chunk.items():
                        if "messages" in node_output and node_output["messages"]:
                            last_message = node_output["messages"][-1]
                            if hasattr(last_message, 'content'):
                                content = last_message.content
                            elif isinstance(last_message, dict) and "content" in last_message:
                                content = last_message["content"]
                            else:
                                content = str(last_message)
                            
                            if content and content != ai_response:
                                print(f"AI ({node_name}): {content}")
                                ai_response = content
                                    
        except Exception as e:
            print(f"Error: {e}")
            print("AI: Sorry, I encountered an error. Please try again.")
