"""Re-rankers: local cross-encoder (free, ONNX via fastembed) or a lexical fallback."""

from __future__ import annotations

from typing import Protocol

from app.config import get_settings
from app.ingestion.embed import tokenize
from app.retrieval.vectorstore import SearchHit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]: ...


class NoopReranker:
    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        return hits[:top_n]


class LexicalReranker:
    """Query-term coverage + fused score. Zero-dependency fallback for tests/offline."""

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        q = set(tokenize(query))
        scored = []
        for h in hits:
            toks = set(tokenize(h.text))
            coverage = len(q & toks) / (len(q) or 1)
            scored.append(SearchHit(h.chunk_id, h.text, h.metadata, 0.7 * coverage + 0.3 * h.score))
        return sorted(scored, key=lambda h: h.score, reverse=True)[:top_n]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name)

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        if not hits:
            return []
        scores = list(self._model.rerank(query, [h.text for h in hits]))
        rescored = [SearchHit(h.chunk_id, h.text, h.metadata, float(s)) for h, s in zip(hits, scores, strict=False)]
        return sorted(rescored, key=lambda h: h.score, reverse=True)[:top_n]


_RERANKER: Reranker | None = None


def get_reranker() -> Reranker:
    global _RERANKER
    if _RERANKER is None:
        s = get_settings()
        if s.reranker_backend == "fastembed":
            _RERANKER = CrossEncoderReranker(s.reranker_model)
        elif s.reranker_backend == "lexical":
            _RERANKER = LexicalReranker()
        else:
            _RERANKER = NoopReranker()
    return _RERANKER


def set_reranker(r: Reranker | None) -> None:
    global _RERANKER
    _RERANKER = r
