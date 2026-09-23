"""Deterministic, dependency-free embedding for offline operation / tests.

Builds a token-hash sparse vector that is L2-normalised. Texts sharing tokens
receive higher cosine similarity, which is enough to exercise the full RAG
retrieval pipeline without downloading any model.
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

from .base import BaseEmbeddingProvider


class DeterministicEmbedding(BaseEmbeddingProvider):
    name = "deterministic-embed"

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def _vector(self, text: str):
        dim = self.dimension
        vec = np.zeros(dim, dtype=np.float64)
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            tokens = ["<empty>"]
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h >> 1) & 1 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec

    async def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t).tolist() for t in texts]

    async def close(self) -> None:
        return None
