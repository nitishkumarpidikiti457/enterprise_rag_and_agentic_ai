"""Shared fixtures: fully offline (hash embeddings, lexical reranker, scripted fake LLM)."""

import os

os.environ.update({
    "EMBEDDING_BACKEND": "hash",
    "RERANKER_BACKEND": "lexical",
    "GROQ_API_KEY": "", "GEMINI_API_KEY": "", "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "",
    "REDIS_URL": "",
    "JWT_SECRET": "test-secret-with-at-least-32-bytes!!",
})

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from app.agent import graph as agent_graph  # noqa: E402
from app.api.auth import reset_rate_limits  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.ingestion.embed import HashEmbedder, set_embedder  # noqa: E402
from app.ingestion.pipeline import ingest_path  # noqa: E402
from app.llm.cache import get_cache  # noqa: E402
from app.llm.providers import ExtractiveProvider, LLMProvider, LLMResponse  # noqa: E402
from app.llm.router import LLMRouter, set_router  # noqa: E402
from app.retrieval.rerank import LexicalReranker, set_reranker  # noqa: E402
from app.retrieval.retriever import invalidate_bm25  # noqa: E402
from app.retrieval.vectorstore import ChromaStore, set_store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
get_settings.cache_clear()


class FakeProvider(LLMProvider):
    """Scripted LLM: returns canned text based on which prompt it receives."""

    name = "fake"

    def __init__(self, fail: bool = False) -> None:
        super().__init__("fake-1")
        self.fail = fail
        self.calls: list[str] = []

    async def complete(self, messages, temperature=0.1, max_tokens=1024):
        if self.fail:
            raise RuntimeError("provider down")
        text = "\n".join(m.content for m in messages)
        self.calls.append(text)
        if "Respond with ONLY one word" in text:
            q = text.rsplit("Question:", 1)[-1].lower()
            if "sales" in q and "recall" in q:
                out = "both"
            elif "sales" in q or "units" in q:
                out = "data"
            elif q.strip().startswith("hi"):
                out = "chat"
            else:
                out = "docs"
        elif "read-only SQLite SELECT" in text:
            out = ("SELECT r.sku, p.name, SUM(s.units) AS units FROM recalls r JOIN products p ON p.sku=r.sku "
                   "JOIN sales s ON s.sku=r.sku WHERE s.sale_date BETWEEN '2026-04-01' AND '2026-06-30' "
                   "GROUP BY r.sku ORDER BY units DESC")
        elif "Database results [DB]" in text:
            out = "Recalled SKUs HF-1042, HF-2210 and HF-3307 are listed in the policy [1] and all had Q2 sales [DB]."
        elif "Context:" in text and "[1]" in text:
            out = "Class I recalls must be pulled within 2 hours of notification [1]."
        else:
            out = "Hello!"
        return LLMResponse(out, self.name, self.model, 10, 10)


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    set_embedder(HashEmbedder())
    set_reranker(LexicalReranker())
    s = ChromaStore(path=str(tmp_path_factory.mktemp("chroma")), collection="test_docs")
    set_store(s)
    invalidate_bm25()
    ingest_path(ROOT / "data" / "sample_docs", store=s)
    return s


@pytest.fixture
def fake_llm():
    fake = FakeProvider()
    set_router(LLMRouter({"fake": fake, "extractive": ExtractiveProvider("x")}))
    get_cache().clear()
    agent_graph.set_graph(agent_graph.build_graph())
    reset_rate_limits()
    yield fake
    set_router(None)
