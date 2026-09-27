"""Agent tools: vector search over documents and guarded read-only SQL over the sales DB."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from app.config import get_settings
from app.retrieval.retriever import RetrievalConfig, retrieve
from app.retrieval.vectorstore import SearchHit, get_store

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|truncate)\b",
    re.IGNORECASE,
)


class SQLGuardError(ValueError):
    pass


def vector_search(query: str, top_n: int = 5, source: str | None = None) -> list[SearchHit]:
    return retrieve(query, RetrievalConfig(top_n=top_n, source=source))


def get_document(source: str, max_chars: int = 6000) -> str:
    """Return the full text of an ingested document (ordered by chunk index)."""
    chunks = [c for c in get_store().all_chunks() if c.metadata.get("source") == source]
    chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0))
    return "\n".join(c.text for c in chunks)[:max_chars]


def clean_sql(sql: str) -> str:
    sql = sql.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(1).strip()
    return sql.rstrip(";").strip()


def validate_sql(sql: str) -> str:
    sql = clean_sql(sql)
    if not sql:
        raise SQLGuardError("empty query")
    if ";" in sql:
        raise SQLGuardError("multiple statements are not allowed")
    if not re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE):
        raise SQLGuardError("only SELECT queries are allowed")
    if _FORBIDDEN.search(sql):
        raise SQLGuardError("query contains a forbidden keyword")
    if not re.search(r"\blimit\s+\d+", sql, re.IGNORECASE):
        sql = f"{sql} LIMIT 50"
    return sql


def _connect(db_path: Path) -> sqlite3.Connection:
    # open read-only at the SQLite level as a second layer of defence
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    conn.execute("PRAGMA query_only = ON")
    return conn


def get_schema(db_path: Path | None = None) -> str:
    db_path = db_path or get_settings().sql_db_path
    if not Path(db_path).exists():
        return "(database not found)"
    with _connect(Path(db_path)) as conn:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    return "\n".join(r[0] for r in rows)


def sql_query(sql: str, db_path: Path | None = None) -> dict:
    sql = validate_sql(sql)
    db_path = db_path or get_settings().sql_db_path
    conn = _connect(Path(db_path))
    try:
        conn.set_progress_handler(None, 0)
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(50)
    finally:
        conn.close()
    return {"sql": sql, "columns": cols, "rows": [list(r) for r in rows]}


def format_sql_result(result: dict | None) -> str:
    if not result:
        return "(no database query was run)"
    if result.get("error"):
        return f"(query failed: {result['error']})"
    lines = [" | ".join(result["columns"])]
    lines += [" | ".join(str(v) for v in row) for row in result["rows"]]
    return f"SQL: {result['sql']}\n" + "\n".join(lines)
