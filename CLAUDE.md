# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is
Enterprise RAG & Agentic AI Platform. It ingests PDF/DOCX/MD into ChromaDB, runs hybrid retrieval with re-ranking, routes to multiple LLMs, and exposes a LangGraph agent (document search + guarded SQL) through a FastAPI service with SSE streaming. See README.md and PLAN.md.

## Commands
- Install: `pip install -e ".[dev]" langgraph-checkpoint-sqlite`
- Tests: `pytest -q`. They run fully offline with a hash embedder, a lexical re-ranker and a scripted fake LLM (see `tests/conftest.py`).
- Lint: `ruff check src tests eval scripts`
- Regenerate demo data: `python scripts/generate_sample_data.py`
- Run the API: `python -m app.ingestion.pipeline && uvicorn app.api.main:app --reload`
- Eval: `python eval/run_eval.py --generation`

## Conventions
- Every feature ships with tests, and tests must not need network access or API keys.
- The project must stay free to run. Paid providers are optional and off without a key.
- Swappable components sit behind small interfaces with get_/set_ singletons: `Embedder`, `VectorStore`, `Reranker`, `LLMRouter`. Tests override them with `set_*`.
- Prompts live in `src/app/llm/prompts/templates.yaml`. Bump `version` when you change one.
- The SQL tool is read-only. Never weaken `validate_sql` or the `mode=ro` connection.
- Answers must cite `[n]` for passages and `[DB]` for database rows.

## Decisions log
- Embeddings: fastembed `BAAI/bge-small-en-v1.5` (local, free). Re-ranker: `Xenova/ms-marco-MiniLM-L-6-v2`.
- Chunking: semantic breakpoints at the 85th percentile, max 1200 chars, min 200 chars.
- Retrieval: top_k 20 → RRF (alpha 0.5) → re-rank to top_n 5.
- Default provider order: groq, gemini, anthropic, openai; `extractive` is the no-key fallback.
