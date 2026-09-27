"""Central configuration, loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM providers (free tiers by default; paid ones stay off without a key) ---
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    groq_model: str = "llama-3.3-70b-versatile"
    groq_fast_model: str = "llama-3.1-8b-instant"
    gemini_model: str = "gemini-2.0-flash"
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5"

    # Preferred provider order; the router skips providers without keys and falls back on errors.
    llm_provider_order: str = "groq,gemini,anthropic,openai"
    llm_timeout_s: float = 60.0

    # --- Embeddings / re-ranking (local + free) ---
    embedding_backend: str = "fastembed"  # "fastembed" | "hash" (offline/tests)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_backend: str = "fastembed"  # "fastembed" | "none" | "lexical"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # --- Chunking ---
    chunk_max_chars: int = 1200
    chunk_min_chars: int = 200
    semantic_breakpoint_percentile: float = 85.0

    # --- Retrieval ---
    retrieve_top_k: int = 20
    rerank_top_n: int = 5
    hybrid_alpha: float = 0.5  # weight of vector vs BM25 in reciprocal-rank fusion

    # --- Storage ---
    data_dir: Path = Path("data")
    chroma_dir: Path = Path(".chroma")
    collection_name: str = "enterprise_docs"
    sql_db_path: Path = Path("data/sample.db")
    checkpoint_db_path: Path = Path(".state/checkpoints.sqlite")

    # --- Cache ---
    redis_url: str | None = None  # e.g. redis://localhost:6379/0 ; falls back to in-memory
    cache_ttl_s: int = 3600

    # --- API / auth ---
    jwt_secret: str = "change-me-in-.env"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    demo_username: str = "admin"
    demo_password: str = "admin123"
    rate_limit_per_minute: int = 30
    max_upload_mb: int = 20

    # --- Agent ---
    agent_max_steps: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
