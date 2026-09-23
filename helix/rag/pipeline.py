"""RAG pipeline wiring splitter, embedding and vector store together."""
from __future__ import annotations

from typing import List, Optional

from ..core.types import Document, Message, Role, ScoredChunk
from .splitter import RecursiveCharacterTextSplitter

DEFAULT_RAG_SYSTEM_PROMPT = (
    "You are Helix RAG assistant. Answer the question using ONLY the provided "
    "context. If the context does not contain the answer, say you do not know."
)


class RAGPipeline:
    """Ingests documents and answers questions over them.

    Implements the :class:`Retriever` contract via :meth:`retrieve`.
    """

    def __init__(self, embeddings, vectorstore, splitter=None, top_k: int = 4) -> None:
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.splitter = splitter or RecursiveCharacterTextSplitter()
        self.top_k = top_k

    async def ingest(self, documents: List[Document]) -> int:
        """Split, embed and store documents. Returns number of chunks created."""
        chunks = self.splitter.split_documents(documents)
        if not chunks:
            return 0
        vectors = await self.embeddings.embed([c.text for c in chunks])
        await self.vectorstore.add(chunks, vectors)
        return len(chunks)

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> List[ScoredChunk]:
        k = top_k or self.top_k
        query_vec = (await self.embeddings.embed([query]))[0]
        return await self.vectorstore.search(query_vec, k)

    async def answer(self, query: str, llm, system_prompt: Optional[str] = None) -> str:
        scored = await self.retrieve(query)
        context = "\n\n".join(
            f"[{i + 1}] {item.chunk.text}" for i, item in enumerate(scored)
        )
        messages = [
            Message(
                role=Role.SYSTEM,
                content=system_prompt or DEFAULT_RAG_SYSTEM_PROMPT,
            ),
            Message(
                role=Role.USER,
                content=f"Context:\n{context}\n\nQuestion: {query}",
            ),
        ]
        return await llm.generate(messages)
