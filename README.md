# Helix AI Engine

<p align="center">
  <a href="https://github.com/CJX0712/helix-ai-engine/actions/workflows/ci.yml"><img src="https://github.com/CJX0712/helix-ai-engine/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/CJX0712/helix-ai-engine/releases"><img src="https://img.shields.io/github/v/release/CJX0712/helix-ai-engine?sort=semver" alt="release"></a>
  <img src="https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-1f6feb" alt="author">
</p>

A modular, provider-agnostic AI orchestration engine: RAG + tool-using agents,
wired through clean interface contracts so every module is independently
verifiable and the whole system is reproducible from a clean environment.

Author: 晨星

## Highlights
- Provider-agnostic LLM & embedding adapters (OpenAI-compatible / Ollama / local deterministic fallback)
- In-memory NumPy vector store + scikit-learn TF-IDF lexical retriever (hybrid retrieval)
- ReAct agent with a model-agnostic textual tool-calling protocol
- FastAPI service + CLI
- Every module has its own pytest suite; an end-to-end test runs the full link offline

## Quick start
```bash
make setup      # create venv and install pinned deps
make test       # run the full test suite
make serve      # start the HTTP API on :8000
```
Or with PowerShell: `./setup.ps1` then `pytest`.

Set `HELIX_PROVIDER_MODE=mock` to run fully offline (deterministic providers); set
`auto` (default) to use a local Ollama model and gracefully fall back if unavailable.

See docs/ARCHITECTURE.md, docs/DEPLOYMENT.md and docs/USAGE.md.
