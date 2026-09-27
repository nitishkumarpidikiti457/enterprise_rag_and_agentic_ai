import pytest

from app.agent import tools
from app.agent.graph import heuristic_intent, run_agent, stream_agent


@pytest.mark.parametrize("sql", [
    "DELETE FROM sales",
    "SELECT 1; DROP TABLE sales",
    "UPDATE products SET unit_price = 0",
    "PRAGMA table_info(sales)",
    "",
])
def test_sql_guard_blocks_writes(sql):
    with pytest.raises(tools.SQLGuardError):
        tools.validate_sql(sql)


def test_sql_guard_adds_limit_and_strips_markdown():
    assert tools.validate_sql("```sql\nSELECT * FROM products\n```").endswith("LIMIT 50")


def test_sql_query_runs_read_only():
    res = tools.sql_query("SELECT COUNT(*) AS n FROM recalls")
    assert res["rows"][0][0] == 3


def test_heuristic_intent():
    assert heuristic_intent("hello") == "chat"
    assert heuristic_intent("What were total sales in Q2?") == "data"
    assert heuristic_intent("What is the return policy?") == "docs"
    assert heuristic_intent("Which products in the recall policy had sales in Q2?") == "both"


async def test_agent_docs_question(store, fake_llm):
    state = await run_agent("How quickly must Class I recalls be pulled?", "t-docs")
    assert state["intent"] == "docs"
    assert any(s.get("tool") == "vector_search" for s in state["steps"])
    assert state["citations"][0]["source"] == "product_recall_policy.pdf"


async def test_agent_multi_source_question_cites_docs_and_db(store, fake_llm):
    state = await run_agent("Which products in the recall policy had sales last quarter?", "t-both")
    assert state["intent"] == "both"
    tools_used = [s.get("tool") for s in state["steps"] if s.get("node") == "tool"]
    assert tools_used == ["vector_search", "sql_query"]
    assert state["sql_result"]["rows"]
    refs = {c["ref"] for c in state["citations"]}
    assert "DB" in refs and 1 in refs


async def test_agent_remembers_conversation(store, fake_llm):
    await run_agent("How quickly must Class I recalls be pulled?", "t-mem")
    state = await run_agent("And Class III?", "t-mem")
    assert len(state["history"]) == 4


async def test_agent_streams_steps_and_tokens(store, fake_llm):
    events = [e async for e in stream_agent("hi", "t-stream")]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "step" and "token" in kinds and kinds[-1] == "done"
