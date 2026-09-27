"""Hybrid retriever: dense vector search + BM25, fused with Reciprocal Rank Fusion, then re-ranked."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.ingestion.embed import get_embedder, tokenize
from app.observability.logging import metrics
from app.retrieval.rerank import get_reranker
from app.retrieval.vectorstore import SearchHit, get_store

_bm25_lock = threading.Lock()
_bm25: tuple[BM25Okapi, list[SearchHit]] | None = None


def invalidate_bm25() -> None:
    global _bm25
    with _bm25_lock:
        _bm25 = None


def _get_bm25() -> tuple[BM25Okapi, list[SearchHit]] | None:
    global _bm25
    with _bm25_lock:
        if _bm25 is None:
            chunks = get_store().all_chunks()
            if not chunks:
                return None
            _bm25 = (BM25Okapi([tokenize(c.text) for c in chunks]), chunks)
        return _bm25


def rrf_fuse(rankings: list[list[SearchHit]], weights: list[float], k: int = 60) -> list[SearchHit]:
    scores: dict[str, float] = {}
    by_id: dict[str, SearchHit] = {}
    for ranking, w in zip(rankings, weights, strict=False):
        for rank, hit in enumerate(ranking):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + w / (k + rank + 1)
            by_id.setdefault(hit.chunk_id, hit)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [SearchHit(cid, by_id[cid].text, by_id[cid].metadata, s) for cid, s in fused]


@dataclass
class RetrievalConfig:
    top_k: int | None = None
    top_n: int | None = None
    hybrid: bool = True
    rerank: bool = True
    source: str | None = None


def retrieve(query: str, cfg: RetrievalConfig | None = None) -> list[SearchHit]:
    s = get_settings()
    cfg = cfg or RetrievalConfig()
    top_k = cfg.top_k or s.retrieve_top_k
    top_n = cfg.top_n or s.rerank_top_n
    where = {"source": cfg.source} if cfg.source else None

    with metrics.timer("retrieval"):
        dense = get_store().query(get_embedder().embed_query(query), top_k, where)
        candidates = dense
        if cfg.hybrid:
            bm = _get_bm25()
            if bm is not None:
                index, chunks = bm
                scores = index.get_scores(tokenize(query))
                ranked = sorted(zip(chunks, scores, strict=False), key=lambda x: x[1], reverse=True)
                sparse = [
                    SearchHit(c.chunk_id, c.text, c.metadata, float(sc))
                    for c, sc in ranked
                    if sc > 0 and (not cfg.source or c.metadata.get("source") == cfg.source)
                ][:top_k]
                candidates = rrf_fuse([dense, sparse], [s.hybrid_alpha, 1 - s.hybrid_alpha])[:top_k]
        if cfg.rerank:
            return get_reranker().rerank(query, candidates, top_n)
        return candidates[:top_n]


def format_context(hits: list[SearchHit]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {h.citation()}; section: {h.metadata.get('section') or '-'})\n{h.text}"
        for i, h in enumerate(hits, start=1)
    )
