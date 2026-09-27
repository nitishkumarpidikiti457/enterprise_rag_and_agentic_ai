"""Pydantic request/response models with validation limits."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ProviderName = Literal["groq", "gemini", "anthropic", "openai", "extractive"]


class TokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    provider: ProviderName | None = None
    top_k: int = Field(default=20, ge=1, le=100)
    top_n: int = Field(default=5, ge=1, le=20)
    hybrid: bool = True
    rerank: bool = True
    source: str | None = Field(default=None, max_length=255)
    stream: bool = False

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question is too short")
        return v


class Citation(BaseModel):
    ref: int | str
    source: str | None = None
    page: int | None = None
    section: str | None = None
    snippet: str | None = None
    sql: str | None = None


class QueryResponse(BaseModel):
    answer: str
    provider: str
    model: str
    citations: list[Citation]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str = Field(default="default", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    provider: ProviderName | None = None
    stream: bool = True


class ChatResponse(BaseModel):
    answer: str
    intent: str | None
    model: str | None
    citations: list[Citation]
    steps: list[dict]


class IngestResponse(BaseModel):
    source: str
    doc_id: str
    chunks: int
    skipped: bool
