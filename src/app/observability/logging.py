"""Structured JSON logging and simple in-process metrics."""

import json
import logging
import sys
import time
from collections import defaultdict
from contextlib import contextmanager


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def log_event(logger: logging.Logger, msg: str, **fields) -> None:
    logger.info(msg, extra={"extra_fields": fields})


class Metrics:
    """Tiny in-memory metrics registry exposed at /metrics."""

    def __init__(self) -> None:
        self.counters: dict[str, float] = defaultdict(float)
        self.latencies: dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, value: float = 1.0) -> None:
        self.counters[name] += value

    def observe(self, name: str, seconds: float) -> None:
        bucket = self.latencies[name]
        bucket.append(seconds)
        if len(bucket) > 1000:
            del bucket[:500]

    @contextmanager
    def timer(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - start)

    def snapshot(self) -> dict:
        def pct(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            s = sorted(values)
            return round(s[min(len(s) - 1, int(p * len(s)))] * 1000, 1)

        return {
            "counters": dict(self.counters),
            "latency_ms": {
                k: {"p50": pct(v, 0.5), "p95": pct(v, 0.95), "n": len(v)}
                for k, v in self.latencies.items()
            },
        }


metrics = Metrics()
