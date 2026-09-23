"""In-memory vector store backed by NumPy cosine similarity.

Zero external dependencies and fast for small/medium corpora; satisfies the
:class:`VectorStore` contract so it can be swapped for a distributed store
later without touching the RAG pipeline.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.errors import RetrievalError
from ..core.types import Chunk, ScoredChunk


class MemoryVectorStore:
    name = "memory"

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._vectors: List[np.ndarray] = []

    async def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        if len(chunks) != len(vectors):
            raise RetrievalError(
                f"chunk/vector count mismatch: {len(chunks)} != {len(vectors)}"
            )
        self._chunks.extend(chunks)
        self._vectors.extend(np.asarray(v, dtype=np.float64) for v in vectors)

    async def search(self, vector: List[float], top_k: int) -> List[ScoredChunk]:
        if not self._vectors:
            return []
        q = np.asarray(vector, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        sims = []
        for v in self._vectors:
            v_norm = np.linalg.norm(v)
            if v_norm == 0 or q_norm == 0:
                sims.append(0.0)
            else:
                sims.append(float(np.dot(q, v) / (q_norm * v_norm)))
        order = np.argsort(sims)[::-1][:top_k]
        return [ScoredChunk(chunk=self._chunks[i], score=float(sims[i])) for i in order]

    async def count(self) -> int:
        return len(self._chunks)
