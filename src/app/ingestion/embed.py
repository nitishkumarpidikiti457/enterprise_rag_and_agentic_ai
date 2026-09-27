"""Embedding backends behind a single interface.

* FastEmbedEmbedder – free local ONNX model (BAAI/bge-small-en-v1.5), no API key, no GPU.
* HashEmbedder      – deterministic hashing-trick embedder for offline tests / CI.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from app.config import get_settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


_STOP = frozenset("a an the of to in on for and or is are be must what how which who when do does can by "
                  "with at as it its this that from any all".split())


def _stem(tok: str) -> str:
    for suf in ("ies", "ing", "ed", "es", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stop-words, light suffix stemming (used by BM25 and the hash embedder)."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class HashEmbedder:
    """Bag-of-words + bigram hashing embedder (L2-normalised). Deterministic, zero deps."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FastEmbedEmbedder:
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        s = get_settings()
        _EMBEDDER = (
            FastEmbedEmbedder(s.embedding_model)
            if s.embedding_backend == "fastembed"
            else HashEmbedder()
        )
    return _EMBEDDER


def set_embedder(embedder: Embedder | None) -> None:
    """Override the global embedder (used by tests)."""
    global _EMBEDDER
    _EMBEDDER = embedder
