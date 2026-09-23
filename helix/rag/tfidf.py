"""Lexical (TF-IDF) retriever backed by scikit-learn.

Provides a transparent, training-free keyword retrieval path that complements
the dense embedding retriever. Reuses the open-source scikit-learn stack
instead of vendoring a vectorizer.
"""
from __future__ import annotations

from typing import List

from ..core.errors import RetrievalError
from ..core.types import Chunk, ScoredChunk

try:  # keep import optional so the package still loads without scikit-learn
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _HAVE_SK = True
except Exception:  # pragma: no cover - exercised only without scikit-learn
    _HAVE_SK = False


class TfidfRetriever:
    name = "tfidf"

    def __init__(self) -> None:
        if not _HAVE_SK:
            raise RetrievalError(
                "scikit-learn is required for TfidfRetriever; install it or use the "
                "embedding-based retriever."
            )
        self._vectorizer = TfidfVectorizer()
        self._matrix = None
        self._chunks: List[Chunk] = []

    def fit(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise RetrievalError("cannot fit TF-IDF on an empty corpus")
        self._chunks = chunks
        self._matrix = self._vectorizer.fit_transform([c.text for c in chunks])

    async def retrieve(self, query: str, top_k: int = 4) -> List[ScoredChunk]:
        if self._matrix is None:
            raise RetrievalError("TfidfRetriever has not been fitted")
        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._matrix)[0]
        order = np.argsort(sims)[::-1][:top_k]
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(sims[i])) for i in order
        ]
