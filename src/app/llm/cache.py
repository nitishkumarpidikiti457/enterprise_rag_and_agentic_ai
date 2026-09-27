"""Exact-match response cache: Redis if configured, otherwise in-process TTL dict."""

from __future__ import annotations

import hashlib
import json
import time

from app.config import get_settings


def cache_key(model: str, payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return "llmcache:" + hashlib.sha256(f"{model}|{raw}".encode()).hexdigest()


class MemoryCache:
    def __init__(self) -> None:
        self._d: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> str | None:
        item = self._d.get(key)
        if not item:
            return None
        exp, val = item
        if exp < time.time():
            self._d.pop(key, None)
            return None
        return val

    def set(self, key: str, value: str, ttl: int) -> None:
        if len(self._d) > 5000:
            self._d.clear()
        self._d[key] = (time.time() + ttl, value)

    def clear(self) -> None:
        self._d.clear()


class RedisCache:
    def __init__(self, url: str) -> None:
        import redis

        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._r.ping()

    def get(self, key: str) -> str | None:
        return self._r.get(key)

    def set(self, key: str, value: str, ttl: int) -> None:
        self._r.set(key, value, ex=ttl)

    def clear(self) -> None:
        for k in self._r.scan_iter("llmcache:*"):
            self._r.delete(k)


_CACHE: MemoryCache | RedisCache | None = None


def get_cache() -> MemoryCache | RedisCache:
    global _CACHE
    if _CACHE is None:
        url = get_settings().redis_url
        try:
            _CACHE = RedisCache(url) if url else MemoryCache()
        except Exception:
            _CACHE = MemoryCache()
    return _CACHE
