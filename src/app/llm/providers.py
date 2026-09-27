"""LLM providers behind one interface, implemented with plain httpx (no vendor SDKs).

Free by default:  Groq (Llama 3.x) and Google Gemini free tiers.
Optional (paid):  Anthropic Claude and OpenAI GPT — enabled only when an API key is set.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from app.config import get_settings


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict = field(default_factory=dict)


class ProviderError(RuntimeError):
    pass


class LLMProvider:
    name: str = "base"
    # rough USD per 1M tokens (input, output) for cost logging; free tiers = 0
    price: tuple[float, float] = (0.0, 0.0)

    def __init__(self, model: str) -> None:
        self.model = model

    async def complete(self, messages: list[Message], temperature: float = 0.1, max_tokens: int = 1024) -> LLMResponse:
        raise NotImplementedError

    async def stream(self, messages: list[Message], temperature: float = 0.1, max_tokens: int = 1024) -> AsyncIterator[str]:
        resp = await self.complete(messages, temperature, max_tokens)
        yield resp.text

    def cost(self, resp: LLMResponse) -> float:
        return (resp.input_tokens * self.price[0] + resp.output_tokens * self.price[1]) / 1e6


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(get_settings().llm_timeout_s, connect=10.0)


async def _sse_lines(resp: httpx.Response) -> AsyncIterator[dict]:
    async for line in resp.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


class OpenAICompatibleProvider(LLMProvider):
    """Works for Groq and OpenAI (same chat-completions wire format)."""

    base_url = ""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model)
        self.api_key = api_key

    def _payload(self, messages, temperature, max_tokens, stream=False) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def complete(self, messages, temperature=0.1, max_tokens=1024) -> LLMResponse:
        async with httpx.AsyncClient(timeout=_timeout()) as c:
            r = await c.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                             json=self._payload(messages, temperature, max_tokens))
        if r.status_code >= 400:
            raise ProviderError(f"{self.name} {r.status_code}: {r.text[:300]}")
        d = r.json()
        usage = d.get("usage", {})
        return LLMResponse(d["choices"][0]["message"]["content"] or "", self.name, self.model,
                           usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), d)

    async def stream(self, messages, temperature=0.1, max_tokens=1024):
        async with httpx.AsyncClient(timeout=_timeout()) as c:
            async with c.stream("POST", f"{self.base_url}/chat/completions", headers=self._headers(),
                                json=self._payload(messages, temperature, max_tokens, stream=True)) as r:
                if r.status_code >= 400:
                    raise ProviderError(f"{self.name} {r.status_code}: {(await r.aread())[:300]!r}")
                async for d in _sse_lines(r):
                    delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield delta


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"
    base_url = "https://api.openai.com/v1"
    price = (0.15, 0.60)


class GeminiProvider(LLMProvider):
    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model)
        self.api_key = api_key

    def _payload(self, messages, temperature, max_tokens) -> dict:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages if m.role != "system"
        ]
        body: dict = {"contents": contents,
                      "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    @staticmethod
    def _text(d: dict) -> str:
        parts = ((d.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts)

    async def complete(self, messages, temperature=0.1, max_tokens=1024) -> LLMResponse:
        url = f"{self.base_url}/models/{self.model}:generateContent"
        async with httpx.AsyncClient(timeout=_timeout()) as c:
            r = await c.post(url, params={"key": self.api_key}, json=self._payload(messages, temperature, max_tokens))
        if r.status_code >= 400:
            raise ProviderError(f"gemini {r.status_code}: {r.text[:300]}")
        d = r.json()
        u = d.get("usageMetadata", {})
        return LLMResponse(self._text(d), self.name, self.model,
                           u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0), d)

    async def stream(self, messages, temperature=0.1, max_tokens=1024):
        url = f"{self.base_url}/models/{self.model}:streamGenerateContent"
        async with httpx.AsyncClient(timeout=_timeout()) as c:
            async with c.stream("POST", url, params={"key": self.api_key, "alt": "sse"},
                                json=self._payload(messages, temperature, max_tokens)) as r:
                if r.status_code >= 400:
                    raise ProviderError(f"gemini {r.status_code}: {(await r.aread())[:300]!r}")
                async for d in _sse_lines(r):
                    t = self._text(d)
                    if t:
                        yield t


class AnthropicProvider(LLMProvider):
    """Claude via the Messages API. Uses prompt caching on the (long) system prompt."""

    name = "anthropic"
    base_url = "https://api.anthropic.com/v1"
    price = (1.0, 5.0)

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model)
        self.api_key = api_key

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

    def _payload(self, messages, temperature, max_tokens, stream=False) -> dict:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages if m.role != "system"],
            "stream": stream,
        }
        if system:
            body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return body

    async def complete(self, messages, temperature=0.1, max_tokens=1024) -> LLMResponse:
        async with httpx.AsyncClient(timeout=_timeout()) as c:
            r = await c.post(f"{self.base_url}/messages", headers=self._headers(),
                             json=self._payload(messages, temperature, max_tokens))
        if r.status_code >= 400:
            raise ProviderError(f"anthropic {r.status_code}: {r.text[:300]}")
        d = r.json()
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage", {})
        return LLMResponse(text, self.name, self.model, u.get("input_tokens", 0), u.get("output_tokens", 0), d)

    async def stream(self, messages, temperature=0.1, max_tokens=1024):
        async with httpx.AsyncClient(timeout=_timeout()) as c:
            async with c.stream("POST", f"{self.base_url}/messages", headers=self._headers(),
                                json=self._payload(messages, temperature, max_tokens, stream=True)) as r:
                if r.status_code >= 400:
                    raise ProviderError(f"anthropic {r.status_code}: {(await r.aread())[:300]!r}")
                async for d in _sse_lines(r):
                    if d.get("type") == "content_block_delta":
                        t = d.get("delta", {}).get("text")
                        if t:
                            yield t


class ExtractiveProvider(LLMProvider):
    """No-key, no-network fallback: answers by quoting the top retrieved passage.

    Lets the whole platform run (and tests pass) with zero API keys.
    """

    name = "extractive"

    async def complete(self, messages, temperature=0.1, max_tokens=1024) -> LLMResponse:
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        if "Respond with ONLY one word" in user or "JSON" in user.split("\n")[0]:
            return LLMResponse("", self.name, self.model)
        start = user.find("[1] (source:")
        if start < 0:
            return LLMResponse("I don't know based on the available documents.", self.name, self.model)
        first = user[start:].split("\n\n", 1)[0]
        body = first.split("\n", 1)[-1].strip()
        return LLMResponse(f"{body[:600]} [1]", self.name, self.model)


def build_providers() -> dict[str, LLMProvider]:
    s = get_settings()
    providers: dict[str, LLMProvider] = {}
    if s.groq_api_key:
        providers["groq"] = GroqProvider(s.groq_model, s.groq_api_key)
        providers["groq-fast"] = GroqProvider(s.groq_fast_model, s.groq_api_key)
    if s.gemini_api_key:
        providers["gemini"] = GeminiProvider(s.gemini_model, s.gemini_api_key)
    if s.anthropic_api_key:
        providers["anthropic"] = AnthropicProvider(s.anthropic_model, s.anthropic_api_key)
    if s.openai_api_key:
        providers["openai"] = OpenAIProvider(s.openai_model, s.openai_api_key)
    providers["extractive"] = ExtractiveProvider("extractive-v1")
    return providers
