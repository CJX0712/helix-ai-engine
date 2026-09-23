"""Recursive character text splitter (single responsibility).

Splits long text into overlapping chunks using a hierarchy of separators,
falling back to character-level cutting when no separator fits. Mirrors the
well-known LangChain-style algorithm but dependency-free.
"""
from __future__ import annotations

from typing import List

from ..core.types import Chunk, Document


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 40,
        separators: List[str] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size - 1)
        self.separators = separators or ["\n\n", "\n", "。", ".", " ", ""]

    def split_text(self, text: str) -> List[str]:
        text = text or ""
        return self._split(text, self.separators)

    def _split(self, text: str, seps: List[str]) -> List[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        sep = seps[0] if seps else ""
        if sep == "":
            step = max(1, self.chunk_size - self.chunk_overlap)
            return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

        parts = text.split(sep)
        chunks: List[str] = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) > self.chunk_size and current:
                chunks.append(current)
                current = part
            else:
                current = candidate
        if current:
            chunks.append(current)

        result: List[str] = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size and len(seps) > 1:
                result.extend(self._split(chunk, seps[1:]))
            else:
                result.append(chunk)
        return result

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            pieces = self.split_text(doc.text)
            for i, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        id=f"{doc.id}-{i}",
                        document_id=doc.id,
                        text=piece,
                        index=i,
                        metadata={**doc.metadata},
                    )
                )
        return chunks
