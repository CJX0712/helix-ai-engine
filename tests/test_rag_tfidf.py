"""Tests for the scikit-learn TF-IDF retriever."""
import asyncio

from helix.core.types import Chunk
from helix.rag.tfidf import TfidfRetriever


def test_tfidf_retrieve_ranks_relevant():
    async def run():
        chunks = [
            Chunk(id="c0", document_id="d", text="Helix is a modular AI engine", index=0),
            Chunk(id="c1", document_id="d", text="The solar system has planets", index=1),
        ]
        r = TfidfRetriever()
        r.fit(chunks)
        scored = await r.retrieve("AI engine modular", top_k=1)
        assert scored[0].chunk.id == "c0"

    asyncio.run(run())
