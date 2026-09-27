"""FastAPI service: auth, ingestion, single-shot RAG, streaming agent chat, health & metrics.

Run:  uvicorn app.api.main:app --reload
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import run_agent, stream_agent
from app.api.auth import authenticate, create_token, rate_limited_user
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    TokenRequest,
    TokenResponse,
)
from app.config import get_settings
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.ingestion.pipeline import ingest_file
from app.llm.router import get_router
from app.observability.logging import metrics, setup_logging
from app.rag import answer_question, stream_answer
from app.retrieval.retriever import RetrievalConfig
from app.retrieval.vectorstore import get_store

setup_logging()
app = FastAPI(
    title="Enterprise RAG & Agentic AI Platform",
    version="1.0.0",
    description="Grounded RAG over enterprise documents + a LangGraph agent with SQL tools, "
                "multi-LLM routing (Groq/Llama, Gemini, Claude, GPT) and streaming responses.",
)
_STATIC = Path(__file__).parent / "static"


def _sse(events):
    async def gen():
        try:
            async for e in events:
                yield {"event": e["event"], "data": json.dumps(e["data"], default=str)}
        except Exception as ex:  # noqa: BLE001 - surface errors to the client as an event
            yield {"event": "error", "data": json.dumps({"detail": str(ex)[:300]})}
    return EventSourceResponse(gen())


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_STATIC / "index.html")


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "documents_indexed_chunks": get_store().count(),
            "providers": get_router().available()}


@app.get("/metrics", tags=["ops"])
def get_metrics(_: str = Depends(rate_limited_user)):
    return metrics.snapshot()


@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
def token(req: TokenRequest):
    if not authenticate(req.username, req.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    tok, exp = create_token(req.username)
    return TokenResponse(access_token=tok, expires_in=exp)


@app.post("/ingest", response_model=list[IngestResponse], tags=["documents"])
async def ingest(files: list[UploadFile] = File(...), _: str = Depends(rate_limited_user)):
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    results = []
    upload_dir = Path(tempfile.mkdtemp(prefix="ingest_"))
    for f in files:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                                f"{name}: supported types are {sorted(SUPPORTED_EXTENSIONS)}")
        data = await f.read()
        if len(data) > max_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"{name} exceeds size limit")
        if not data:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"{name} is empty")
        path = upload_dir / name
        path.write_bytes(data)
        try:
            r = ingest_file(path)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"{name}: could not parse ({e})") from e
        results.append(IngestResponse(source=r.source, doc_id=r.doc_id, chunks=r.chunks, skipped=r.skipped))
    return results


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(req: QueryRequest, _: str = Depends(rate_limited_user)):
    cfg = RetrievalConfig(top_k=req.top_k, top_n=req.top_n, hybrid=req.hybrid, rerank=req.rerank,
                          source=req.source)
    metrics.inc("queries")
    if req.stream:
        return _sse(stream_answer(req.question, req.provider, cfg))
    with metrics.timer("query"):
        res = await answer_question(req.question, req.provider, cfg)
    return QueryResponse(answer=res.answer, provider=res.provider, model=res.model, citations=res.citations)


@app.post("/chat", tags=["agent"], response_model=ChatResponse,
          responses={200: {"content": {"text/event-stream": {}}}})
async def chat(req: ChatRequest, user: str = Depends(rate_limited_user)):
    thread = f"{user}:{req.thread_id}"  # threads are isolated per user
    metrics.inc("chats")
    if req.stream:
        return _sse(stream_agent(req.message, thread, req.provider))
    with metrics.timer("chat"):
        state = await run_agent(req.message, thread, req.provider)
    return ChatResponse(answer=state.get("answer", ""), intent=state.get("intent"), model=state.get("model"),
                        citations=state.get("citations", []), steps=state.get("steps", []))
