import pytest

from app.llm import prompts
from app.llm.providers import ExtractiveProvider, Message
from app.llm.router import LLMRouter
from tests.conftest import FakeProvider


async def test_fallback_when_provider_fails():
    router = LLMRouter({"groq": FakeProvider(fail=True), "gemini": FakeProvider(),
                        "extractive": ExtractiveProvider("x")})
    resp = await router.complete([Message("user", "hi there")], use_cache=False)
    assert resp.provider == "fake"  # gemini's FakeProvider answered after groq failed


async def test_explicit_provider_is_tried_first():
    a, b = FakeProvider(), FakeProvider()
    router = LLMRouter({"groq": a, "gemini": b, "extractive": ExtractiveProvider("x")})
    await router.complete([Message("user", "hi")], provider="gemini", use_cache=False)
    assert b.calls and not a.calls


async def test_cache_hit_skips_provider():
    p = FakeProvider()
    router = LLMRouter({"groq": p, "extractive": ExtractiveProvider("x")})
    msgs = [Message("user", "cache me please")]
    await router.complete(msgs)
    await router.complete(msgs)
    assert len(p.calls) == 1


async def test_all_fail_raises():
    router = LLMRouter({"groq": FakeProvider(fail=True)})
    with pytest.raises(RuntimeError):
        await router.complete([Message("user", "x")], use_cache=False)


async def test_stream_falls_back():
    router = LLMRouter({"groq": FakeProvider(fail=True), "extractive": ExtractiveProvider("x")})
    out = [d async for _, d in router.stream([Message("user", "no context")])]
    assert "don't know" in "".join(out)


def test_prompt_templates_render():
    txt = prompts.render("rag_user", question="Q?", context="[1] ctx")
    assert "Q?" in txt and "[1] ctx" in txt
    assert prompts.version()
