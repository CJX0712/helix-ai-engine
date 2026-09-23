"""Tests for the in-memory vector store."""
import asyncio

from helix.core.types import Chunk
from helix.vectorstore.memory import MemoryVectorStore


def test_add_and_search():
    async def run():
        store = MemoryVectorStore()
        chunks = [
            Chunk(id=f"c{i}", document_id="d", text=f"t{i}", index=i)
            for i in range(3)
        ]
        vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        await store.add(chunks, vecs)
        assert await store.count() == 3

        results = await store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].chunk.id == "c0"
        assert abs(results[0].score - 1.0) < 1e-9

    asyncio.run(run())


def test_search_empty_store():
    async def run():
        store = MemoryVectorStore()
        assert await store.search([1.0], top_k=3) == []

    asyncio.run(run())
