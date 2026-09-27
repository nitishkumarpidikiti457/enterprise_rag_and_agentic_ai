"""Single-shot grounded RAG: retrieve -> prompt -> answer with citations."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.llm import prompts
from app.llm.providers import Message
from app.llm.router import get_router
from app.retrieval.retriever import RetrievalConfig, format_context, retrieve
from app.retrieval.vectorstore import SearchHit

IDK = "I don't know based on the available documents."
_CIT_RE = re.compile(r"\[(\d+)\]")


@dataclass
class RAGAnswer:
    answer: str
    provider: str
    model: str
    citations: list[dict] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)


def build_messages(question: str, hits: list[SearchHit]) -> list[Message]:
    return [
        Message("system", prompts.render("rag_system")),
        Message("user", prompts.render("rag_user", question=question, context=format_context(hits))),
    ]


def extract_citations(answer: str, hits: list[SearchHit]) -> list[dict]:
    used = sorted({int(n) for n in _CIT_RE.findall(answer) if 0 < int(n) <= len(hits)})
    return [
        {
            "ref": n,
            "source": hits[n - 1].metadata.get("source"),
            "page": hits[n - 1].metadata.get("page"),
            "section": hits[n - 1].metadata.get("section"),
            "snippet": hits[n - 1].text[:240],
        }
        for n in used
    ]


async def answer_question(question: str, provider: str | None = None,
                          cfg: RetrievalConfig | None = None) -> RAGAnswer:
    hits = retrieve(question, cfg)
    if not hits:
        return RAGAnswer(IDK, "none", "none")
    resp = await get_router().complete(build_messages(question, hits), task="answer", provider=provider)
    text = resp.text.strip() or IDK
    return RAGAnswer(text, resp.provider, resp.model, extract_citations(text, hits), hits)


async def stream_answer(question: str, provider: str | None = None,
                        cfg: RetrievalConfig | None = None) -> AsyncIterator[dict]:
    hits = retrieve(question, cfg)
    yield {"event": "sources", "data": [{"ref": i, "source": h.citation(), "section": h.metadata.get("section")}
                                        for i, h in enumerate(hits, 1)]}
    if not hits:
        yield {"event": "token", "data": IDK}
        yield {"event": "done", "data": {"citations": []}}
        return
    full = []
    async for prov, delta in get_router().stream(build_messages(question, hits), provider=provider):
        full.append(delta)
        yield {"event": "token", "data": delta, "provider": prov}
    yield {"event": "done", "data": {"citations": extract_citations("".join(full), hits)}}
