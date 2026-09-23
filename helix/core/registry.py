"""Dependency-injection container that wires the standard Helix stack.

This is the single place that decides which concrete implementation backs each
interface. Swapping a provider (e.g. a hosted embedding service) is a one-line
change here and nowhere else.
"""
from __future__ import annotations

from typing import Dict, Optional

from .config import Settings
from .logging import get_logger
from ..providers.llm.openai_compatible import OpenAICompatibleLLM
from ..providers.llm.mock import MockLLMProvider
from ..providers.llm.fallback import AutoFallbackLLM
from ..providers.embeddings.openai_compatible import OpenAICompatibleEmbedding
from ..providers.embeddings.deterministic import DeterministicEmbedding
from ..providers.embeddings.fallback import AutoFallbackEmbedding
from ..vectorstore.memory import MemoryVectorStore
from ..rag.splitter import RecursiveCharacterTextSplitter
from ..rag.pipeline import RAGPipeline
from ..agents.reactor import ReActAgent
from ..tools.registry import build_default_tools


class Container:
    """Assembles providers, stores, RAG and the agent into one runnable system."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = self._build_llm(settings)
        self.embeddings = self._build_embeddings(settings)
        self.vectorstore = MemoryVectorStore()
        self.splitter = RecursiveCharacterTextSplitter()
        self.tools: Dict[str, object] = build_default_tools()
        self.rag = RAGPipeline(
            self.embeddings, self.vectorstore, self.splitter, settings.top_k
        )
        self.agent = ReActAgent(self.llm, self.tools, settings.max_agent_iterations)
        self._log = get_logger("helix.container")

    def _build_llm(self, s: Settings):
        mode = s.require_mode()
        if mode == "mock":
            return MockLLMProvider()
        real = OpenAICompatibleLLM(s.llm_base_url, s.llm_api_key, s.llm_model)
        if mode == "openai":
            return real
        return AutoFallbackLLM(real, MockLLMProvider())

    def _build_embeddings(self, s: Settings):
        mode = s.require_mode()
        if mode == "mock":
            return DeterministicEmbedding()
        real = OpenAICompatibleEmbedding(
            s.embedding_base_url, s.embedding_api_key, s.embedding_model
        )
        if mode == "openai":
            return real
        return AutoFallbackEmbedding(real, DeterministicEmbedding())

    async def close(self) -> None:
        try:
            await self.llm.close()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            self._log.debug("llm close: %s", exc)
        try:
            await self.embeddings.close()
        except Exception as exc:  # pragma: no cover
            self._log.debug("embeddings close: %s", exc)
