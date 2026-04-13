"""
state.py
--------
LangGraph AgentState definition.
"""

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    messages : Annotated[list, operator.add]
    session_id: str
    language  : str
