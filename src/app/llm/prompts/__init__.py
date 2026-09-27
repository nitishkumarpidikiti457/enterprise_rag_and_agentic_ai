"""Versioned prompt templates loaded from YAML."""

from functools import lru_cache
from pathlib import Path

import yaml

_PATH = Path(__file__).with_name("templates.yaml")


@lru_cache
def _templates() -> dict:
    return yaml.safe_load(_PATH.read_text())


def render(name: str, **kwargs) -> str:
    return _templates()[name].format(**kwargs)


def version() -> str:
    return str(_templates().get("version", "0"))
