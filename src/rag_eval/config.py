"""Judge settings. This is the ONLY module that reads environment variables.

Every value can be overridden by an explicit argument (the CLI passes its flags here).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ENV_BASE_URL = "RAG_EVAL_JUDGE_BASE_URL"
ENV_API_KEY = "RAG_EVAL_JUDGE_API_KEY"
ENV_MODEL = "RAG_EVAL_JUDGE_MODEL"
ENV_CACHE_PATH = "RAG_EVAL_CACHE_PATH"
ENV_MAX_RPM = "RAG_EVAL_JUDGE_MAX_RPM"
ENV_REASONING_EFFORT = "RAG_EVAL_JUDGE_REASONING_EFFORT"

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_CACHE_PATH = ".rag_eval_cache.sqlite"


@dataclass(frozen=True)
class JudgeSettings:
    """Where the judge LLM lives and where its answers are cached."""

    base_url: str
    api_key: str | None
    model: str
    cache_path: str
    max_rpm: float | None = None  # client-side throttle; None = unlimited
    reasoning_effort: str | None = None  # sent as reasoning_effort when set


def load_settings(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    cache_path: str | None = None,
    max_rpm: float | None = None,
    reasoning_effort: str | None = None,
) -> JudgeSettings:
    """Build settings: explicit argument > environment variable > default.

    There is deliberately no default API key (see docs/adr/0002). For a local Ollama
    server set RAG_EVAL_JUDGE_API_KEY to any non-empty string, e.g. "ollama".
    """
    env = os.environ
    return JudgeSettings(
        base_url=base_url or env.get(ENV_BASE_URL) or DEFAULT_BASE_URL,
        api_key=api_key or env.get(ENV_API_KEY) or None,
        model=model or env.get(ENV_MODEL) or DEFAULT_MODEL,
        cache_path=cache_path or env.get(ENV_CACHE_PATH) or DEFAULT_CACHE_PATH,
        max_rpm=max_rpm or _float_or_none(env.get(ENV_MAX_RPM)),
        reasoning_effort=reasoning_effort or env.get(ENV_REASONING_EFFORT) or None,
    )


def _float_or_none(value: str | None) -> float | None:
    if not value:
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{ENV_MAX_RPM} must be a number, got {value!r}") from exc
    return number if number > 0 else None
