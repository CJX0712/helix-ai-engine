"""Tests for embedding providers (deterministic + auto-fallback)."""
import asyncio

import numpy as np

from helix.core.errors import ProviderError
from helix.providers.embeddings.deterministic import DeterministicEmbedding
from helix.providers.embeddings.fallback import AutoFallbackEmbedding


def test_deterministic_shape_and_stability():
    async def run():
        e = DeterministicEmbedding(dimension=64)
        v1 = (await e.embed(["hello world"]))[0]
        v2 = (await e.embed(["hello world"]))[0]
        assert len(v1) == 64
        assert np.allclose(v1, v2)

    asyncio.run(run())


def test_deterministic_similarity_rank():
    async def run():
        e = DeterministicEmbedding(dimension=128)
        a, b, c = await e.embed(
            [
                "the cat sat on the mat",
                "the cat lay on the mat",
                "quantum entanglement of photons",
            ]
        )

        def cos(x, y):
            x, y = np.array(x), np.array(y)
            return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

        assert cos(a, b) > cos(a, c)

    asyncio.run(run())


class _BrokenEmbedding:
    name = "broken"
    dimension = 8

    async def embed(self, texts):
        raise ProviderError("down")

    async def close(self):
        return None


def test_embedding_fallback():
    async def run():
        f = AutoFallbackEmbedding(_BrokenEmbedding(), DeterministicEmbedding(8))
        v = (await f.embed(["x"]))[0]
        assert len(v) == 8

    asyncio.run(run())
