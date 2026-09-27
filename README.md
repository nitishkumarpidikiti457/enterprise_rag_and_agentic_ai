# Enterprise RAG & Agentic AI Platform

A production-style **Retrieval-Augmented Generation + agent** platform that answers questions over enterprise documents *and* a SQL database, with grounded, cited answers. It is **free to run**: local embeddings and re-ranking, free-tier LLMs (Groq Llama 3.x, Google Gemini), and optional paid providers (Claude, GPT) that stay off unless you add a key.

- **Semantic ingestion** of PDF / DOCX / Markdown: section- and page-aware loaders, embedding-based semantic chunking with a recursive-split fallback, idempotent re-ingest (content-hashed doc IDs), ChromaDB vector store.
- **High-precision retrieval**: dense vectors + BM25, combined with Reciprocal Rank Fusion, then a **cross-encoder re-ranker**. Answers must cite passages, and the model says "I don't know" when the context is insufficient.
- **Model-agnostic LLM router**: Groq (Llama), Gemini, Claude, and GPT behind one interface. Routing is task-based (a fast model for classification, a strong model for answers), with automatic fallback on errors or timeouts, versioned YAML prompt templates, Redis or in-memory response caching, Claude prompt caching, and per-call logging of tokens, latency, and cost.
- **LangGraph agent**: stateful multi-step workflow (`classify → vector_search / sql_query → synthesize → validate`, retrying once if the answer has no citations). Tools read documents and a SQL database; the SQL guard allows only read-only `SELECT`s. Conversation memory is kept per thread in a SQLite checkpointer.
- **FastAPI service**: JWT auth, per-user rate limiting, Pydantic validation, **SSE streaming** of agent steps and tokens, file upload ingestion, `/health` and `/metrics`, and a small web UI.
- **Evaluation harness**: 50-question golden set; hit@1, recall@5, and MRR for dense vs. hybrid vs. re-ranked retrieval; answer accuracy, citation rate, refusal rate, and latency per provider.

## Architecture

```
            ┌───────────── Ingestion ─────────────┐
 PDF/DOCX → │ loaders → semantic chunker → embed  │ → ChromaDB (vectors + metadata)
    MD      └─────────────────────────────────────┘          │
                                                             ▼
 Client ──JWT──► FastAPI ──► LangGraph agent ──► tools: vector_search (dense+BM25 → RRF → cross-encoder)
   ▲  SSE: steps + tokens     │  classify → retrieve / SQL → synthesize → validate (retry)
   └──────────────────────────┘        │                    sql_query (read-only SQLite)
                                       ▼
                       LLM Router: Groq · Gemini · Claude · GPT
                       (task routing · fallback · cache · cost/latency logs)
```

## Quick start (free)

```bash
git clone <this repo> && cd enterprise-rag-agent
python -m venv .venv && source .venv/bin/activate
make install                 # or: pip install -e ".[dev]" langgraph-checkpoint-sqlite
cp .env.example .env         # add free GROQ_API_KEY and/or GEMINI_API_KEY
make run                     # ingests data/sample_docs, then serves http://localhost:8000
```

Open **http://localhost:8000** for the chat UI (demo login `admin` / `admin123`; change it in `.env`) or **/docs** for the OpenAPI UI.

With Docker (adds Redis caching):

```bash
cp .env.example .env && docker compose up --build
```

> **No API keys?** Everything still runs. The router falls back to a built-in *extractive* provider that quotes the best retrieved passage, so ingestion, retrieval, the agent, the API and all tests work offline.

### Free API keys
| Provider | Free key | Model used |
|---|---|---|
| Groq | https://console.groq.com/keys | `llama-3.3-70b-versatile` (answers), `llama-3.1-8b-instant` (classification) |
| Google Gemini | https://aistudio.google.com/apikey | `gemini-2.0-flash` |
| Anthropic Claude *(optional, paid)* | https://console.anthropic.com | `claude-haiku-4-5` |
| OpenAI *(optional, paid)* | https://platform.openai.com | `gpt-4o-mini` |

Model names can be changed in `.env` (`GROQ_MODEL`, `GEMINI_MODEL`, …).

## API

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/token -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}' | jq -r .access_token)

# Single-shot grounded RAG
curl -s -X POST localhost:8000/query -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question":"How quickly must a Class I recalled product be pulled?", "provider":"groq"}'

# Agent chat with SSE streaming (step events + tokens + citations)
curl -N -X POST localhost:8000/chat -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"message":"Which recalled products had the most sales in Q2 2026?","thread_id":"demo"}'

# Upload new documents
curl -X POST localhost:8000/ingest -H "Authorization: Bearer $TOKEN" -F "files=@my_policy.pdf"
```

| Endpoint | Description |
|---|---|
| `POST /auth/token` | Get a JWT |
| `POST /query` | Retrieval + grounded answer with citations (`stream: true` for SSE); tune `top_k`, `top_n`, `hybrid`, `rerank`, `source`, `provider` |
| `POST /chat` | LangGraph agent; memory per `thread_id`; SSE streaming by default |
| `POST /ingest` | Upload PDF / DOCX / TXT / MD (type and size validated) |
| `GET /health` | Status, indexed chunks, available providers |
| `GET /metrics` | Counters, LLM cost, cache hits, p50/p95 latencies |

## Demo data

`scripts/generate_sample_data.py` builds a fictional grocery retailer, "Harbor Fresh Markets":
- **21 policy documents** (PDF, DOCX, and MD): recalls, returns, food safety, HR, loyalty, security, and more.
- **SQLite database** (`data/sample.db`): `products`, `stores`, `sales` (Jan–Jun 2026), `inventory`, `recalls`.
- **Golden eval set** (`eval/golden_set.jsonl`): 50 question/answer/source triples.

Good multi-source question for the agent: *"Which products in the recall policy had sales last quarter?"* It answers from the policy document **[1]** and the sales database **[DB]**.

## Evaluation

```bash
make eval        # python eval/run_eval.py --generation   → writes eval/results.md
```

It compares **dense-only**, **hybrid (dense + BM25, RRF)**, and **hybrid + cross-encoder re-rank** on hit@1, recall@5, MRR@5, and document precision@5. It then scores each configured LLM provider on answer accuracy (LLM-as-judge), citation rate, refusals on unanswerable questions, and p50/p95 latency. Run it with your own keys to get the numbers for your setup. `eval/results.md` in the repo was produced offline with the hash embedder and lexical re-ranker used in CI.

## Project layout

```
src/app/
  config.py              settings (.env)
  ingestion/             loaders, semantic chunking, embeddings, pipeline CLI
  retrieval/             vector store (Chroma), hybrid retriever + RRF, re-rankers
  llm/                   providers (Groq, Gemini, Claude, OpenAI, extractive), router, cache, prompts/
  rag.py                 single-shot grounded RAG
  agent/                 LangGraph state, tools (vector search, guarded SQL), graph
  api/                   FastAPI app, JWT auth + rate limit, schemas, static UI
  observability/         JSON logging, metrics
eval/                    golden set + evaluation harness
scripts/                 demo data generator
tests/                   42 offline tests (no network, no keys)
```

## Design decisions

- **Semantic chunking over fixed windows.** Breakpoints fall at topic shifts (cosine distance above the 85th percentile), and chunks never cross page or section boundaries. That keeps each citation precise.
- **Hybrid retrieval.** BM25 catches exact identifiers like SKUs and policy numbers that dense vectors miss; RRF fuses the two without score calibration; the cross-encoder then orders the final top-n.
- **The router is plain HTTP, with no vendor SDKs.** Each provider fits in about 40 lines, and adding one means writing a single class.
- **Validation node.** Uncited answers get one stricter retry. The SQL tool is guarded three ways: a keyword blocklist, a single-statement check, and a SQLite read-only connection.
- **Everything runs offline** (hash embedder, lexical re-ranker, extractive LLM), so CI is deterministic and free.

## Development

```bash
make test   # pytest (42 tests, offline)
make lint   # ruff
```
