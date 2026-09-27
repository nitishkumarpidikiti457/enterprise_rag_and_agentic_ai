"""Ingestion pipeline: load -> semantic chunk -> embed -> upsert. Idempotent per document.

CLI:  python -m app.ingestion.pipeline data/sample_docs
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.ingestion.chunking import chunk_document
from app.ingestion.embed import Embedder, get_embedder
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from app.observability.logging import log_event, setup_logging
from app.retrieval.vectorstore import VectorStore, get_store

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    source: str
    doc_id: str
    chunks: int
    skipped: bool = False


def ingest_file(path: Path, store: VectorStore | None = None, embedder: Embedder | None = None) -> IngestResult:
    s = get_settings()
    store = store or get_store()
    embedder = embedder or get_embedder()
    doc = load_document(path)
    if doc.doc_id in store.doc_ids():
        return IngestResult(doc.source, doc.doc_id, 0, skipped=True)
    chunks = chunk_document(doc, embedder, s.chunk_max_chars, s.chunk_min_chars, s.semantic_breakpoint_percentile)
    embeddings = embedder.embed_documents([c.text for c in chunks])
    store.upsert(chunks, embeddings)
    # invalidate the BM25 index so hybrid search sees the new chunks
    from app.retrieval.retriever import invalidate_bm25

    invalidate_bm25()
    log_event(logger, "ingested", source=doc.source, doc_id=doc.doc_id, chunks=len(chunks))
    return IngestResult(doc.source, doc.doc_id, len(chunks))


def ingest_path(path: str | Path, **kw) -> list[IngestResult]:
    p = Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)
    return [ingest_file(f, **kw) for f in files]


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument("path", nargs="?", default="data/sample_docs")
    args = parser.parse_args()
    results = ingest_path(args.path)
    new = [r for r in results if not r.skipped]
    print(f"Ingested {len(new)} new docs ({sum(r.chunks for r in new)} chunks); "
          f"skipped {len(results) - len(new)} unchanged.")


if __name__ == "__main__":
    main()
