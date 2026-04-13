"""
graph.py
--------
LangGraph ReAct agent for KrishiRakshak.

Flow:
  agent_node → (tool_calls?) → tool_node → agent_node → ... → END

Memory: DynamoDB-backed checkpointer in production, MemorySaver for local.
"""

import logging
import os

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

BEDROCK_REGION    = "us-east-1"
CLAUDE_MODEL_ID   = "us.anthropic.claude-sonnet-4-6-20251001-v1:0"

SYSTEM_PROMPT = """You are KrishiRakshak, an AI assistant helping Indian farmers diagnose crop diseases and get treatment advice.

Follow the ReAct pattern — think step by step before each action:

Thought: reason about what to do based on user input and previous observations
Action: call the appropriate tool
Observation: analyze tool result
Thought: decide next step
... repeat until final answer, then call audio_generation_tool

TOOL USAGE RULES:
1. Image input  → image_diagnosis_tool → retriever_tool → rag_generator_tool → audio_generation_tool
2. PDF (small)  → direct_context_tool → audio_generation_tool
3. PDF (large)  → retriever_tool → rag_generator_tool → audio_generation_tool
4. Text query   → retriever_tool → if poor results: web_search_tool → rag_generator_tool → audio_generation_tool

LANGUAGE RULES:
- Detect language of user query
- Respond in the SAME language throughout
- Pass correct language code ('en' or 'hi') to audio_generation_tool"""


def _build_llm() -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model      =CLAUDE_MODEL_ID,
        region_name=BEDROCK_REGION,
        temperature=0.1,
    )


def _agent_node(state: AgentState) -> AgentState:
    llm      = _build_llm().bind_tools(ALL_TOOLS)
    response = llm.invoke(state["messages"])

    if response.content and not response.tool_calls:
        logger.info(f"[Thought] {str(response.content)[:200]}")
    if response.tool_calls:
        for call in response.tool_calls:
            logger.info(f"[Action] {call['name']} | args: {list(call['args'].keys())}")

    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_graph(use_memory: bool = True):
    """
    Compile and return the LangGraph agent.

    Args:
        use_memory: if True, use MemorySaver for conversation memory
    """
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    checkpointer = MemorySaver() if use_memory else None
    agent        = graph.compile(checkpointer=checkpointer)
    logger.info("LangGraph agent compiled")
    return agent


def run(
    user_input : str,
    session_id : str  = "default",
    file_path  : str  = None,
    agent             = None,
) -> str:
    """
    Run the agent for a single user turn.

    Args:
        user_input : user's text query
        session_id : unique per farmer session (for memory)
        file_path  : path to uploaded image or PDF (optional)
        agent      : compiled LangGraph agent (built once at startup)

    Returns:
        Final text response from agent
    """
    if agent is None:
        agent = build_graph()

    content = user_input
    if file_path:
        content += f"\n[Uploaded file: {file_path}]"

    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {
            "messages"  : [SystemMessage(content=SYSTEM_PROMPT),],
            "session_id": session_id,
            "language"  : "en",
        },
        config=config,
    )

    # Append user message and re-invoke for proper ReAct flow
    from langchain_core.messages import HumanMessage
    result = agent.invoke(
        {"messages": [HumanMessage(content=content)]},
        config=config,
    )

    return result["messages"][-1].content
