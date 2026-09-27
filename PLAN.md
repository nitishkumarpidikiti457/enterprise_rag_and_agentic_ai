# Enterprise RAG & Agentic AI Platform — Build Plan

Goal: build the project exactly as described on the resume, so every bullet is backed by working code, tests, and measured numbers you can talk through in an interview.

| Resume claim | Where it gets built |
|---|---|
| Semantic chunking + embedding of PDF/DOCX into a vector DB | Phase 1 |
| High-precision search that grounds LLM output | Phase 2 (+ measured in Phase 6) |
| LangGraph agents: multi-step reasoning, tool calling, stateful workflows, multiple data sources | Phase 4 |
| FastAPI: streaming, request validation, token auth | Phase 5 |
| Model-agnostic routing (Gemini, GPT-4, Llama), prompt templating, re-ranking, caching | Phase 3 |

---

## Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Language / env | Python 3.11+, `uv` (or Poetry) | One lockfile, reproducible installs |
| Orchestration | LangChain + LangGraph | LangGraph for the agent, LangChain for loaders/retrievers |
| LLMs | Claude (Anthropic API or AWS Bedrock), Gemini, GPT-4-class, Llama (Ollama locally or Bedrock) | Add Claude to the router — and to the resume bullet |
| Embeddings | Voyage AI, OpenAI `text-embedding-3-small`, or local `sentence-transformers` (`bge-small`) | Claude has no embedding model of its own; keep embeddings behind an interface |
| Vector DB | ChromaDB (local/dev), Pinecone (optional "prod" backend) | Abstract behind a `VectorStore` interface so both work |
| Parsing | `pypdf` / `pymupdf`, `python-docx` (or `unstructured`) | |
| Chunking | LangChain `SemanticChunker` + fallback recursive splitter | Store source, page, section in metadata |
| Re-ranking | Local cross-encoder (`bge-reranker-base`) or Cohere Rerank | |
| Caching | Redis: exact-match cache + optional semantic cache; provider prompt caching (Claude/Gemini) | |
| API | FastAPI, Pydantic v2, SSE streaming (`StreamingResponse` / `sse-starlette`) | |
| Auth | JWT bearer tokens (`pyjwt`) with a `/auth/token` endpoint, or hashed API keys | |
| Structured data source | SQLite/Postgres sample DB (e.g. a small "orders/inventory" dataset) | Gives the agent a second data source |
| Eval / observability | RAGAS or a hand-built eval set, LangSmith (optional), structured logging | Produces the numbers for the resume |
| Packaging / deploy | Docker, docker-compose, GitHub Actions CI, Cloud Run or AWS (ECS/App Runner) | |

---

## Repository layout

```
enterprise-rag-agent/
├── CLAUDE.md                 # instructions for Claude Code (conventions, commands)
├── README.md                 # architecture diagram, setup, demo GIF, results table
├── pyproject.toml
├── docker-compose.yml        # api + chroma + redis
├── Dockerfile
├── .env.example
├── .github/workflows/ci.yml  # lint, type-check, tests
├── data/
│   ├── sample_docs/          # public PDFs/DOCX (policies, manuals, 10-Ks)
│   └── sample.db             # structured data for the SQL tool
├── src/app/
│   ├── config.py             # pydantic-settings
│   ├── ingestion/            # loaders.py, chunking.py, embed.py, pipeline.py
│   ├── retrieval/            # vectorstore.py, retriever.py, rerank.py
│   ├── llm/                  # router.py, providers.py, prompts/, cache.py
│   ├── agent/                # graph.py, state.py, tools/ (vector_search, sql_query, ...)
│   ├── api/                  # main.py, routes/, schemas.py, auth.py, deps.py
│   └── observability/        # logging, tracing, metrics
├── eval/
│   ├── golden_set.jsonl      # 50–100 question/answer/source triples
│   └── run_eval.py
└── tests/                    # unit + integration tests
```

---

## Phases

### Phase 0 — Setup (half a day)
- Create the repo structure above, `pyproject.toml`, `.env.example`, pre-commit (ruff, mypy).
- Write `CLAUDE.md`: project purpose, folder map, how to run tests, "every feature ships with tests", style rules.
- `docker-compose.yml` with ChromaDB and Redis.
- GitHub Actions: ruff + mypy + pytest on every push.
- **Done when:** `docker compose up` works and CI is green on an empty test.

### Phase 1 — Ingestion pipeline (2–3 days)
- Loaders for PDF and DOCX that keep page numbers and headings.
- Semantic chunking (embedding-similarity breakpoints) with a max-size guard and a recursive-splitter fallback.
- Metadata per chunk: `doc_id`, `source`, `page`, `section`, `chunk_index`, content hash (for dedup / re-ingest).
- `Embedder` interface with 2 implementations (API + local).
- `VectorStore` interface; Chroma implementation first.
- CLI: `python -m app.ingestion.pipeline data/sample_docs`.
- **Done when:** 20+ sample docs ingest idempotently; tests cover loaders and chunking.

### Phase 2 — Retrieval + grounded RAG (2 days)
- Retriever: top-k vector search → optional hybrid (BM25 + vector) → cross-encoder re-rank → top-n.
- Answer prompt that forces citations (`[source, page]`) and "I don't know" when context is insufficient.
- Return answer + cited chunks.
- **Done when:** answers cite real sources; a no-answer question returns a refusal, not a hallucination.

### Phase 3 — Model routing layer (2 days)
- `LLMProvider` interface; providers for Claude, Gemini, GPT-4-class, Llama.
- Router strategies: config-based (per task), cost/latency-based (cheap model for classification/rewrites, strong model for final answers), fallback on error/timeout.
- Prompt templates in versioned files (`llm/prompts/*.yaml`).
- Caching: Redis exact-match cache keyed on (model, prompt hash); provider prompt caching for the long system prompt.
- Log per call: model, tokens, latency, cost estimate, cache hit.
- **Done when:** the same query can be served by any provider via a flag, and a provider outage falls back automatically.

### Phase 4 — LangGraph agent (3–4 days)
- State: messages, user intent, retrieved context, tool results, step count.
- Nodes: `classify_intent` → `plan` → `tool_call` (loop) → `synthesize` → `validate` (checks citations; retries once if unsupported).
- Tools: `vector_search` (Phase 2 retriever), `sql_query` (read-only, schema-aware, parameter-guarded), `get_document` (fetch full section), optionally `web_search`.
- Checkpointer for conversation memory (SQLite/Redis saver) keyed by `thread_id`.
- Guardrails: max steps, SQL restricted to `SELECT`, tool timeouts.
- **Done when:** a question needing both documents and the DB ("which products mentioned in the recall policy had sales last quarter?") is answered with both sources cited.

### Phase 5 — FastAPI service (2 days)
- Endpoints: `POST /auth/token`, `POST /ingest` (file upload), `POST /query` (single-shot RAG), `POST /chat` (agent, SSE streaming of tokens + tool-step events), `GET /health`.
- Pydantic request/response schemas with limits (file size/type, query length, `top_k` bounds).
- JWT dependency on all non-health routes; per-user rate limit (Redis).
- OpenAPI docs cleaned up; a tiny Streamlit or HTML client for demos.
- **Done when:** `curl` with a token streams an agent answer; bad input returns clean 4xx errors; integration tests pass.

### Phase 6 — Evaluation & observability (2 days)
- Golden set of 50–100 Q/A/source triples from your sample docs.
- Metrics: retrieval precision@k / recall@k / MRR (with vs. without re-ranking), faithfulness and answer relevance (RAGAS or LLM-as-judge), p50/p95 latency, cost per query per model, cache hit rate.
- Results table in the README — these are the numbers you quote in interviews.
- **Done when:** `python eval/run_eval.py` prints a comparison table across models and retrieval settings.

### Phase 7 — Ship it (1–2 days)
- Multi-stage Dockerfile, deploy to Cloud Run (matches your GCP skills) or AWS.
- README: architecture diagram, setup in <5 commands, demo GIF, results table, design decisions.
- Tag `v1.0.0`; pin the repo on your GitHub profile.

**Total:** roughly 3–4 weeks part-time.

---

## Using Claude to build it

- **Claude Code in the repo** (terminal, desktop, or web): one phase per session. Start each with "Read CLAUDE.md and PLAN.md, implement Phase N, write tests, run them." Review the diff before committing.
- **Keep `CLAUDE.md` current** with decisions made (e.g. "embeddings use Voyage; chunk size 800 tokens") so later sessions stay consistent.
- **Ask for tests first** on tricky pieces (chunking, router fallback, SQL guard).
- **Make sure you can explain every file.** Interviewers will ask why you chose semantic chunking, how the router decides, and how you measured precision.
- **Claude as a model in the product:** add it to the router via the Anthropic API or Bedrock (already on your resume) and update the bullet to "Claude, Gemini, GPT-4, Llama".

---

## Resume follow-up once built
Replace vague phrases with measured results from Phase 6, e.g.:
- "…improving retrieval precision@5 from X% to Y% with cross-encoder re-ranking"
- "…cutting average cost per query by X% and p95 latency by Y ms via routing and caching"
