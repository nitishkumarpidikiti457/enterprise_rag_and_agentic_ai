"""Model-agnostic LLM router.

* Task-based routing: cheap/fast models for classification & SQL, strong models for answers.
* Explicit override: callers can pin a provider (e.g. ?provider=gemini).
* Automatic fallback: on error/timeout the next available provider is tried.
* Exact-match caching and per-call logging of model, tokens, latency, cost and cache hits.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator

from app.config import get_settings
from app.llm.cache import cache_key, get_cache
from app.llm.providers import LLMProvider, LLMResponse, Message, build_providers
from app.observability.logging import log_event, metrics

logger = logging.getLogger(__name__)

TASK_PREFERENCES: dict[str, list[str]] = {
    "classify": ["groq-fast", "gemini", "groq", "anthropic", "openai"],
    "sql": ["groq", "gemini", "anthropic", "openai"],
    "answer": [],  # empty -> use LLM_PROVIDER_ORDER
}


class LLMRouter:
    def __init__(self, providers: dict[str, LLMProvider] | None = None) -> None:
        self.providers = providers if providers is not None else build_providers()

    def available(self) -> list[str]:
        return [p for p in self.providers if p != "groq-fast"]

    def chain(self, task: str = "answer", provider: str | None = None) -> list[LLMProvider]:
        order = [p.strip() for p in get_settings().llm_provider_order.split(",") if p.strip()]
        prefs = TASK_PREFERENCES.get(task) or order
        extras = [n for n in self.providers if n != "extractive"]  # custom/registered providers
        names = ([provider] if provider else []) + prefs + order + extras + ["extractive"]
        seen, out = set(), []
        for n in names:
            if n in self.providers and n not in seen:
                seen.add(n)
                out.append(self.providers[n])
        return out

    async def complete(self, messages: list[Message], task: str = "answer", provider: str | None = None,
                       temperature: float = 0.1, max_tokens: int = 1024, use_cache: bool = True) -> LLMResponse:
        errors = []
        for p in self.chain(task, provider):
            key = cache_key(f"{p.name}:{p.model}", [(m.role, m.content) for m in messages] + [temperature, max_tokens])
            if use_cache and (hit := get_cache().get(key)):
                metrics.inc("llm_cache_hits")
                d = json.loads(hit)
                log_event(logger, "llm_call", provider=p.name, model=p.model, task=task, cache_hit=True)
                return LLMResponse(d["text"], p.name, p.model, d.get("in", 0), d.get("out", 0))
            start = time.perf_counter()
            try:
                resp = await p.complete(messages, temperature, max_tokens)
            except Exception as e:  # noqa: BLE001 - any provider failure triggers fallback
                errors.append(f"{p.name}: {e}")
                metrics.inc(f"llm_errors_{p.name}")
                log_event(logger, "llm_fallback", provider=p.name, error=str(e)[:200])
                continue
            latency = time.perf_counter() - start
            metrics.observe(f"llm_{p.name}", latency)
            metrics.inc("llm_calls")
            metrics.inc("llm_cost_usd", p.cost(resp))
            log_event(logger, "llm_call", provider=p.name, model=p.model, task=task, cache_hit=False,
                      latency_ms=round(latency * 1000), in_tokens=resp.input_tokens,
                      out_tokens=resp.output_tokens, cost_usd=round(p.cost(resp), 6))
            if use_cache and resp.text:
                get_cache().set(key, json.dumps({"text": resp.text, "in": resp.input_tokens,
                                                 "out": resp.output_tokens}), get_settings().cache_ttl_s)
            return resp
        raise RuntimeError("All LLM providers failed: " + "; ".join(errors))

    async def stream(self, messages: list[Message], task: str = "answer", provider: str | None = None,
                     temperature: float = 0.1, max_tokens: int = 1024) -> AsyncIterator[tuple[str, str]]:
        """Yields (provider_name, text_delta). Falls back only if a provider fails before emitting."""
        errors = []
        for p in self.chain(task, provider):
            emitted = False
            try:
                async for delta in p.stream(messages, temperature, max_tokens):
                    emitted = True
                    yield p.name, delta
                metrics.inc("llm_calls")
                return
            except Exception as e:  # noqa: BLE001
                if emitted:
                    raise
                errors.append(f"{p.name}: {e}")
                log_event(logger, "llm_fallback", provider=p.name, error=str(e)[:200])
        raise RuntimeError("All LLM providers failed: " + "; ".join(errors))


_ROUTER: LLMRouter | None = None


def get_router() -> LLMRouter:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = LLMRouter()
    return _ROUTER


def set_router(r: LLMRouter | None) -> None:
    global _ROUTER
    _ROUTER = r
