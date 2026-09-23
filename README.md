# Helix AI Engine

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
