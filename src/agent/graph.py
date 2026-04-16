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
CLAUDE_MODEL_ID   = "us.anthropic.claude-sonnet-4-6"

SYSTEM_PROMPT = """You are KrishiRakshak, an AI assistant helping Indian farmers diagnose crop diseases and get treatment advice.

TOOL USAGE RULES — follow exactly, minimum tool calls:
1. ALWAYS call retriever_tool first for any crop/disease question.
2. If retriever returns empty or irrelevant results, call web_search_tool ONCE.
3. After getting context (from retriever OR web search), write the final answer yourself. Do NOT call rag_generator_tool.
4. Never call web_search_tool more than once per query.

ANSWER FORMAT:
- Plain prose only. No markdown, no bullet points, no headers.
- Under 150 words.
- Mention at least one pesticide brand available in India with dosage if relevant.
- End with one short prevention tip.
- Respond in the SAME language as the user query (Hindi → Hindi, English → English)."""


# ── LLM singleton — built once, reused across all agent loop iterations ────────
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatBedrockConverse(
            model      =CLAUDE_MODEL_ID,
            region_name=BEDROCK_REGION,
            temperature=0.1,
        ).bind_tools(ALL_TOOLS)
        logger.info(f"Agent LLM ready ({CLAUDE_MODEL_ID})")
    return _llm


def _agent_node(state: AgentState) -> AgentState:
    response = _get_llm().invoke(state["messages"])

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
    Compile and return the LangGraph ReAct agent.
    Max 4 iterations: agent → retriever → agent → (web_search) → agent → final answer.
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
    logger.info("LangGraph ReAct agent compiled")
    return agent


def run(
    user_input : str,
    session_id : str  = "default",
    file_path  : str  = None,
    agent             = None,
) -> tuple:
    """
    Run the agent for a single user turn.

    Args:
        user_input : user's text query
        session_id : unique per farmer session (for memory)
        file_path  : path to uploaded image or PDF (optional)
        agent      : compiled LangGraph agent (built once at startup)

    Returns:
        (answer: str, context_chunks: list) — chunks are from retriever_tool
        calls, used for RAG eval metrics in monitoring.
    """
    import json as _json
    from langchain_core.messages import HumanMessage, ToolMessage

    if agent is None:
        agent = build_graph()

    content = user_input
    if file_path:
        content += f"\n[Uploaded file: {file_path}]"

    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {
            "messages"  : [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)],
            "session_id": session_id,
            "language"  : "en",
        },
        config=config,
    )

    # Extract retrieved chunks from retriever_tool ToolMessages for monitoring
    context_chunks = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "retriever_tool":
            try:
                context_chunks.extend(_json.loads(msg.content))
            except Exception:
                pass

    return result["messages"][-1].content, context_chunks
