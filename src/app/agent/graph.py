"""LangGraph agent: multi-step reasoning with tool calling over documents + a SQL database.

    START -> start -> classify --docs--> retrieve ------------------> synthesize -> validate -> END
                              --data--> sql_tool --------------------^      ^           |
                              --both--> retrieve -> sql_tool --------^      +-- retry --+
                              --chat--> chat -> END

State (conversation history) is persisted per thread_id with a SQLite checkpointer.
"""

from __future__ import annotations

import re
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent import tools
from app.agent.state import AgentState
from app.config import get_settings
from app.llm import prompts
from app.llm.providers import Message
from app.llm.router import get_router
from app.rag import IDK
from app.retrieval.retriever import format_context
from app.retrieval.vectorstore import SearchHit

INTENTS = {"docs", "data", "both", "chat"}
_DATA_WORDS = re.compile(
    r"\b(sales|sold|revenue|units|inventory|stock|how many|total|average|top \d+|quarter|q[1-4]|"
    r"store[s]?|region|trend|count)\b", re.IGNORECASE)
_DOC_WORDS = re.compile(r"\b(policy|policies|procedure|manual|guideline|handbook|rule|recall|return|"
                        r"allowed|required|must|should|how do|what is)\b", re.IGNORECASE)
_CHAT_WORDS = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening))\b", re.IGNORECASE)


def heuristic_intent(q: str) -> str:
    if _CHAT_WORDS.match(q) and len(q.split()) < 6:
        return "chat"
    data, docs = bool(_DATA_WORDS.search(q)), bool(_DOC_WORDS.search(q))
    if data and docs:
        return "both"
    return "data" if data else "docs"


def _hits_to_dicts(hits: list[SearchHit]) -> list[dict]:
    return [{"chunk_id": h.chunk_id, "text": h.text, "metadata": h.metadata, "score": h.score} for h in hits]


def _dicts_to_hits(ds: list[dict]) -> list[SearchHit]:
    return [SearchHit(d["chunk_id"], d["text"], d["metadata"], d["score"]) for d in ds]


def _history_text(history: list[dict], n: int = 6) -> str:
    return "\n".join(f"{h['role']}: {h['content'][:500]}" for h in history[-n:]) or "(none)"


# ----------------------------------------------------------------------------- nodes

async def start(state: AgentState) -> dict:
    return {"steps": [], "step_count": 0, "retries": 0, "doc_hits": [], "sql_result": None,
            "citations": [], "answer": ""}


async def classify(state: AgentState) -> dict:
    q = state["question"]
    intent = ""
    try:
        resp = await get_router().complete(
            [Message("user", prompts.render("classify_intent", question=q))],
            task="classify", provider=state.get("provider"), max_tokens=5, temperature=0.0)
        intent = resp.text.strip().lower().strip(".").split()[0] if resp.text.strip() else ""
    except Exception:  # noqa: BLE001
        intent = ""
    source = "llm"
    if intent not in INTENTS:
        intent, source = heuristic_intent(q), "heuristic"
    plan = {"docs": ["vector_search", "synthesize"], "data": ["sql_query", "synthesize"],
            "both": ["vector_search", "sql_query", "synthesize"], "chat": ["respond"]}[intent]
    return {"intent": intent, "plan": plan, "step_count": state.get("step_count", 0) + 1,
            "steps": state.get("steps", []) + [{"node": "classify", "intent": intent, "by": source, "plan": plan}]}


async def retrieve_docs(state: AgentState) -> dict:
    # include the previous user turn so follow-ups ("what about returns?") keep context
    prev = [h["content"] for h in state.get("history", []) if h["role"] == "user"][-1:]
    query = " ".join(prev + [state["question"]]) if len(state["question"].split()) < 6 else state["question"]
    hits = tools.vector_search(query, top_n=5)
    return {"doc_hits": _hits_to_dicts(hits), "step_count": state.get("step_count", 0) + 1,
            "steps": state.get("steps", []) + [{"node": "tool", "tool": "vector_search", "query": query,
                                                "results": [h.citation() for h in hits]}]}


async def sql_tool(state: AgentState) -> dict:
    q = state["question"]
    schema = tools.get_schema()
    result: dict
    sql = ""
    try:
        resp = await get_router().complete(
            [Message("system", prompts.render("sql_system", schema=schema)), Message("user", q)],
            task="sql", provider=state.get("provider"), max_tokens=300, temperature=0.0)
        sql = resp.text
        result = tools.sql_query(sql)
    except Exception as e:  # noqa: BLE001 - guard errors and DB errors are reported, not raised
        result = {"sql": tools.clean_sql(sql), "error": str(e)[:200], "columns": [], "rows": []}
    return {"sql_result": result, "step_count": state.get("step_count", 0) + 1,
            "steps": state.get("steps", []) + [{"node": "tool", "tool": "sql_query", "sql": result.get("sql"),
                                                "rows": len(result.get("rows", [])), "error": result.get("error")}]}


def _synth_messages(state: AgentState, strict: bool = False) -> list[Message]:
    hits = _dicts_to_hits(state.get("doc_hits", []))
    system = prompts.render("synthesize_system")
    if strict:
        system += "\nIMPORTANT: your previous answer lacked citations. Cite every claim or say you don't know."
    user = prompts.render("synthesize_user", question=state["question"],
                          history=_history_text(state.get("history", [])),
                          context=format_context(hits) if hits else "(none)",
                          sql_result=tools.format_sql_result(state.get("sql_result")))
    return [Message("system", system), Message("user", user)]


async def synthesize(state: AgentState) -> dict:
    strict = state.get("retries", 0) > 0
    resp = await get_router().complete(_synth_messages(state, strict), task="answer",
                                       provider=state.get("provider"), use_cache=not strict)
    answer = resp.text.strip()
    if not answer or (resp.provider == "extractive" and state.get("sql_result") and not state.get("doc_hits")):
        answer = _fallback_answer(state)
    return {"answer": answer, "model": f"{resp.provider}:{resp.model}",
            "step_count": state.get("step_count", 0) + 1,
            "steps": state.get("steps", []) + [{"node": "synthesize", "provider": resp.provider}]}


def _fallback_answer(state: AgentState) -> str:
    parts = []
    hits = state.get("doc_hits", [])
    if hits:
        parts.append(f"{hits[0]['text'][:500]} [1]")
    res = state.get("sql_result")
    if res and not res.get("error") and res.get("rows"):
        parts.append("Database results [DB]:\n" + tools.format_sql_result(res))
    return "\n\n".join(parts) or IDK


def _citations(state: AgentState, answer: str) -> list[dict]:
    hits = state.get("doc_hits", [])
    cites = []
    for n in sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)}):
        if 0 < n <= len(hits):
            md = hits[n - 1]["metadata"]
            cites.append({"ref": n, "source": md.get("source"), "page": md.get("page"),
                          "section": md.get("section"), "snippet": hits[n - 1]["text"][:240]})
    if "[DB]" in answer and state.get("sql_result"):
        cites.append({"ref": "DB", "source": "sales database", "sql": state["sql_result"].get("sql")})
    return cites


async def validate(state: AgentState) -> dict:
    answer = state.get("answer", "")
    cites = _citations(state, answer)
    grounded = bool(cites) or "don't know" in answer.lower() or state.get("intent") == "chat"
    retry = (not grounded and state.get("retries", 0) < 1
             and state.get("step_count", 0) < get_settings().agent_max_steps)
    out: dict = {"citations": cites, "step_count": state.get("step_count", 0) + 1,
                 "steps": state.get("steps", []) + [{"node": "validate", "grounded": grounded, "retry": retry}]}
    if retry:
        out["retries"] = state.get("retries", 0) + 1
    else:
        out["history"] = [{"role": "user", "content": state["question"]},
                          {"role": "assistant", "content": answer}]
    return out


async def chat(state: AgentState) -> dict:
    msgs = [Message("system", prompts.render("chat_system"))]
    msgs += [Message(h["role"], h["content"]) for h in state.get("history", [])[-6:]]
    msgs.append(Message("user", state["question"]))
    resp = await get_router().complete(msgs, task="classify", provider=state.get("provider"), max_tokens=200)
    answer = resp.text.strip()
    if not answer or resp.provider == "extractive":
        answer = "Hello! I can answer questions about company documents and sales/inventory data."
    return {"answer": answer, "model": f"{resp.provider}:{resp.model}",
            "history": [{"role": "user", "content": state["question"]}, {"role": "assistant", "content": answer}],
            "steps": state.get("steps", []) + [{"node": "chat"}]}


# ----------------------------------------------------------------------------- graph

def _route_intent(state: AgentState) -> str:
    return {"docs": "retrieve", "both": "retrieve", "data": "sql_tool", "chat": "chat"}[state["intent"]]


def _after_retrieve(state: AgentState) -> str:
    return "sql_tool" if state["intent"] == "both" else "synthesize"


def _after_validate(state: AgentState) -> str:
    last = state["steps"][-1]
    return "synthesize" if last.get("retry") else END


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("start", start)
    g.add_node("classify", classify)
    g.add_node("retrieve", retrieve_docs)
    g.add_node("sql_tool", sql_tool)
    g.add_node("synthesize", synthesize)
    g.add_node("validate", validate)
    g.add_node("chat", chat)
    g.add_edge(START, "start")
    g.add_edge("start", "classify")
    g.add_conditional_edges("classify", _route_intent, ["retrieve", "sql_tool", "chat"])
    g.add_conditional_edges("retrieve", _after_retrieve, ["sql_tool", "synthesize"])
    g.add_edge("sql_tool", "synthesize")
    g.add_edge("synthesize", "validate")
    g.add_conditional_edges("validate", _after_validate, ["synthesize", END])
    g.add_edge("chat", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


_GRAPH = None


def get_graph():
    """Graph with a persistent SQLite checkpointer (conversation memory per thread_id)."""
    global _GRAPH
    if _GRAPH is None:
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            path = Path(get_settings().checkpoint_db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            _GRAPH = build_graph(AsyncSqliteSaver(aiosqlite.connect(str(path))))
        except Exception:  # noqa: BLE001 - fall back to in-memory state
            _GRAPH = build_graph(MemorySaver())
    return _GRAPH


def set_graph(graph) -> None:
    global _GRAPH
    _GRAPH = graph


async def run_agent(question: str, thread_id: str, provider: str | None = None) -> AgentState:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    return await get_graph().ainvoke({"question": question, "provider": provider}, config)


async def stream_agent(question: str, thread_id: str, provider: str | None = None):
    """Yields step events as each node finishes, then token events for the final answer."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    final: dict = {}
    seen = 0
    async for update in get_graph().astream({"question": question, "provider": provider}, config,
                                            stream_mode="values"):
        final = update
        steps = update.get("steps", [])
        for s in steps[seen:]:
            yield {"event": "step", "data": s}
        seen = len(steps)
    answer = final.get("answer", "")
    for i in range(0, len(answer), 40):
        yield {"event": "token", "data": answer[i:i + 40]}
    yield {"event": "done", "data": {"citations": final.get("citations", []), "intent": final.get("intent"),
                                     "model": final.get("model")}}

