"""Semantic chunking.

Algorithm (similar in spirit to LangChain's SemanticChunker):
1. Split each block into sentences.
2. Embed a sliding window of sentences and compute cosine distance between neighbours.
3. Break where the distance exceeds the Nth percentile (a topic shift).
4. Enforce max/min chunk sizes; oversize chunks fall back to a recursive character splitter.
Chunks never cross page/section boundaries so citations stay precise.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

from app.ingestion.embed import Embedder
from app.ingestion.loaders import LoadedDocument

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n{2,}|\n(?=[-*•]\s)")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def recursive_split(text: str, max_chars: int, overlap: int = 100) -> list[str]:
    """Fallback splitter: paragraphs -> lines -> sentences -> words."""
    if len(text) <= max_chars:
        return [text]
    for sep in ("\n\n", "\n", ". ", " "):
        parts = text.split(sep)
        if len(parts) == 1:
            continue
        out, cur = [], ""
        for p in parts:
            cand = f"{cur}{sep}{p}" if cur else p
            if len(cand) <= max_chars:
                cur = cand
            else:
                if cur:
                    out.append(cur)
                cur = p
        if cur:
            out.append(cur)
        if all(len(o) <= max_chars for o in out):
            if overlap and len(out) > 1:
                out = [out[0]] + [o_prev[-overlap:] + sep + o for o_prev, o in zip(out, out[1:], strict=False)]
            return out
        return [s for o in out for s in recursive_split(o, max_chars, overlap)]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _cosine_distances(vectors: list[list[float]]) -> list[float]:
    m = np.asarray(vectors, dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    sims = (m[:-1] * m[1:]).sum(axis=1)
    return (1.0 - sims).tolist()


def semantic_groups(
    sentences: list[str],
    embedder: Embedder,
    percentile: float = 85.0,
    window: int = 1,
) -> list[list[str]]:
    if len(sentences) <= 2:
        return [sentences]
    windows = [
        " ".join(sentences[max(0, i - window) : i + window + 1]) for i in range(len(sentences))
    ]
    dists = _cosine_distances(embedder.embed_documents(windows))
    threshold = float(np.percentile(dists, percentile))
    groups, cur = [], [sentences[0]]
    for sent, d in zip(sentences[1:], dists, strict=False):
        if d > threshold:
            groups.append(cur)
            cur = []
        cur.append(sent)
    groups.append(cur)
    return groups


def chunk_document(
    doc: LoadedDocument,
    embedder: Embedder,
    max_chars: int = 1200,
    min_chars: int = 200,
    percentile: float = 85.0,
) -> list[Chunk]:
    pieces: list[tuple[str, dict]] = []
    for block in doc.blocks:
        meta = {"page": block.page, "section": block.section}
        sentences = split_sentences(block.text)
        groups = semantic_groups(sentences, embedder, percentile)
        merged: list[str] = []
        for g in groups:
            text = " ".join(g).strip()
            # merge tiny groups into the previous chunk of the same block
            if merged and (len(text) < min_chars or len(merged[-1]) < min_chars) and len(merged[-1]) + len(text) <= max_chars:
                merged[-1] = f"{merged[-1]} {text}"
            else:
                merged.append(text)
        for text in merged:
            for part in recursive_split(text, max_chars):
                if part.strip():
                    pieces.append((part.strip(), meta))

    chunks = []
    for idx, (text, meta) in enumerate(pieces):
        content_hash = hashlib.sha1(text.encode()).hexdigest()[:12]
        md = {
            "doc_id": doc.doc_id,
            "source": doc.source,
            "chunk_index": idx,
            "content_hash": content_hash,
            "page": meta["page"] if meta["page"] is not None else -1,
            "section": meta["section"] or "",
        }
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}-{idx}-{content_hash}", text=text, metadata=md))
    return chunks
