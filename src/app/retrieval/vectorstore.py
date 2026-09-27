"""Vector store interface + ChromaDB (persistent, local, free) implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.config import get_settings
from app.ingestion.chunking import Chunk


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    metadata: dict
    score: float

    def citation(self) -> str:
        page = self.metadata.get("page", -1)
        return f"{self.metadata.get('source')}" + (f", p.{page}" if page and page > 0 else "")


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def query(self, embedding: list[float], k: int, where: dict | None = None) -> list[SearchHit]: ...
    def delete_doc(self, doc_id: str) -> None: ...
    def all_chunks(self) -> list[SearchHit]: ...
    def doc_ids(self) -> set[str]: ...
    def count(self) -> int: ...


class ChromaStore:
    def __init__(self, path: str | None = None, collection: str | None = None, ephemeral: bool = False):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        s = get_settings()
        cfg = ChromaSettings(anonymized_telemetry=False, allow_reset=True)
        self._client = (
            chromadb.EphemeralClient(settings=cfg)
            if ephemeral
            else chromadb.PersistentClient(path=str(path or s.chroma_dir), settings=cfg)
        )
        self._col = self._client.get_or_create_collection(
            collection or s.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        for i in range(0, len(chunks), 500):
            batch = chunks[i : i + 500]
            self._col.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
                embeddings=embeddings[i : i + 500],
            )

    def query(self, embedding: list[float], k: int, where: dict | None = None) -> list[SearchHit]:
        n = self.count()
        if n == 0:
            return []
        res = self._col.query(query_embeddings=[embedding], n_results=min(k, n), where=where)
        return [
            SearchHit(cid, doc, meta, 1.0 - dist)
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=False
            )
        ]

    def delete_doc(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def all_chunks(self) -> list[SearchHit]:
        res = self._col.get(include=["documents", "metadatas"])
        return [SearchHit(i, d, m, 0.0) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"], strict=False)]

    def doc_ids(self) -> set[str]:
        res = self._col.get(include=["metadatas"])
        return {m["doc_id"] for m in res["metadatas"]}

    def count(self) -> int:
        return self._col.count()


_STORE: VectorStore | None = None


def get_store() -> VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = ChromaStore()
    return _STORE


def set_store(store: VectorStore | None) -> None:
    global _STORE
    _STORE = store
