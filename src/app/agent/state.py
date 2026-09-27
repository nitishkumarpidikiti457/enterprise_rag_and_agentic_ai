"""Typed state carried through the LangGraph agent."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    provider: str | None
    history: Annotated[list[dict], operator.add]  # persisted across turns via the checkpointer
    intent: str  # docs | data | both | chat
    plan: list[str]
    doc_hits: list[dict]
    sql_result: dict | None
    answer: str
    citations: list[dict]
    steps: list[dict]  # tool/trace events for this turn (streamed to clients)
    step_count: int
    retries: int
    model: str
    extra: dict[str, Any]
