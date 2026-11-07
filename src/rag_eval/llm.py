"""The judge LLM client: any OpenAI-compatible endpoint, temperature 0, JSON out, sqlite cache.

Prompts are read from ``prompts/*.txt`` and filled with ``string.Template`` (``$name``
placeholders), so JSON examples inside prompts need no escaping.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from string import Template
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from rag_eval.config import ENV_API_KEY, JudgeSettings

PROMPTS_DIR = Path(__file__).parent / "prompts"
logger = logging.getLogger(__name__)
JSON_REMINDER = "json_reminder"
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_RETRY_IN = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?([\d.]+)s", re.IGNORECASE)
MAX_BACKOFF_S = 90.0


class JudgeError(RuntimeError):
    """The judge failed for one item (e.g. invalid JSON twice). Recorded per question."""


class JudgeConfigError(JudgeError):
    """The judge cannot work at all (no API key, bad URL, unknown model). Aborts the run."""


class Judge(Protocol):
    """What the metrics need from a judge. ``FakeJudgeClient`` in the tests satisfies it."""

    model: str

    def complete_json(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        schema: type[BaseModel] | None = None,
    ) -> dict: ...


# --- prompts -----------------------------------------------------------------------------


def prompt_path(name: str) -> Path:
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"no judge prompt named {name!r} in {PROMPTS_DIR}")
    return path


def load_prompt(name: str) -> str:
    return prompt_path(name).read_bytes().decode("utf-8")


def prompt_sha256(name: str) -> str:
    return hashlib.sha256(prompt_path(name).read_bytes()).hexdigest()


def prompt_versions(names: list[str] | None = None) -> dict[str, str]:
    """SHA-256 of each prompt file (all prompts if ``names`` is None), sorted by name."""
    if names is None:
        names = [p.stem for p in PROMPTS_DIR.glob("*.txt")]
    return {name: prompt_sha256(name) for name in sorted(names)}


def render_prompt(name: str, variables: dict[str, Any]) -> str:
    """Fill a prompt template. Raises ``KeyError`` if a variable is missing."""
    return Template(load_prompt(name)).substitute({k: str(v) for k, v in variables.items()})


# --- cache -------------------------------------------------------------------------------


def cache_key(model: str, prompt_text: str) -> str:
    return hashlib.sha256((model + prompt_text).encode("utf-8")).hexdigest()


class JudgeCache:
    """A tiny sqlite key-value store: sha256(model + prompt) -> JSON response."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)  # the file is created on first use, not here

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS judge_cache ("
            "key TEXT PRIMARY KEY, model TEXT, response TEXT, created_at TEXT)"
        )
        return conn

    def get(self, key: str) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT response FROM judge_cache WHERE key = ?", (key,)).fetchone()
        finally:
            conn.close()
        return None if row is None else row[0]

    def set(self, key: str, model: str, response: str) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO judge_cache VALUES (?, ?, ?, ?)",
                    (key, model, response, datetime.now(UTC).isoformat()),
                )
        finally:
            conn.close()


# --- client ------------------------------------------------------------------------------


def parse_json_reply(text: str, schema: type[BaseModel] | None = None) -> dict:
    """Parse a judge reply into a dict (tolerating ```json fences) and validate it."""
    stripped = text.strip()
    fenced = _FENCE.match(stripped)
    if fenced:
        stripped = fenced.group(1)
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    if schema is not None:
        data = schema.model_validate(data).model_dump(mode="json")
    return data


def retry_delay_seconds(exc: Exception, attempt: int) -> float:
    """How long to wait before retrying a throttled or transient failure.

    Uses the server's hint when present (a Retry-After header, or Google-style
    "retry in 23.4s" / "retryDelay": "23s" in the error body), else exponential backoff.
    """
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}).get("retry-after") if response is not None else None
    if header:
        try:
            return min(float(header) + 1, MAX_BACKOFF_S)
        except ValueError:
            pass
    text = str(exc)
    for pattern in (_RETRY_IN, _RETRY_DELAY):
        if match := pattern.search(text):
            return min(float(match.group(1)) + 1, MAX_BACKOFF_S)
    return min(2.0 * 2**attempt, MAX_BACKOFF_S)


def is_daily_quota_error(exc: Exception) -> bool:
    """A per-day quota is exhausted: retrying within this run is pointless."""
    text = str(exc).lower().replace("_", "")
    return "perday" in text or "daily" in text and "quota" in text


class JudgeClient:
    """Calls the judge LLM at temperature 0 and returns validated JSON.

    ``completion_fn`` replaces the network call (prompt -> reply text); tests use it to
    count calls without touching the network.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        cache: JudgeCache | None = None,
        json_mode: bool = True,
        timeout: float = 120.0,
        completion_fn: Callable[[str], str] | None = None,
        max_rpm: float | None = None,
        reasoning_effort: str | None = None,
        max_attempts: int = 8,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.cache = cache
        self.json_mode = json_mode
        self.timeout = timeout
        self._completion_fn = completion_fn
        self._client: Any = None
        self.n_network_calls = 0
        self.min_interval_s = 60.0 / max_rpm if max_rpm else 0.0
        self.reasoning_effort = reasoning_effort
        self.max_attempts = max_attempts
        self._sleep = sleep
        self._last_call = 0.0

    @classmethod
    def from_settings(
        cls, settings: JudgeSettings, use_cache: bool = True, json_mode: bool = True
    ) -> JudgeClient:
        return cls(
            model=settings.model,
            base_url=settings.base_url,
            api_key=settings.api_key,
            cache=JudgeCache(settings.cache_path) if use_cache else None,
            json_mode=json_mode,
            max_rpm=settings.max_rpm,
            reasoning_effort=settings.reasoning_effort,
        )

    def complete_json(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        schema: type[BaseModel] | None = None,
    ) -> dict:
        """Render a prompt, ask the judge, parse and validate the JSON reply.

        On a bad reply, retry once with a "valid JSON only" reminder, then raise
        ``JudgeError``. Successful replies are cached under sha256(model + prompt).
        """
        prompt = render_prompt(prompt_name, variables)
        key = cache_key(self.model, prompt)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return json.loads(cached)
        data = self._ask_with_retry(prompt_name, prompt, schema)
        if self.cache is not None:
            self.cache.set(key, self.model, json.dumps(data, sort_keys=True))
        return data

    def _ask_with_retry(
        self, prompt_name: str, prompt: str, schema: type[BaseModel] | None
    ) -> dict:
        reply = self._complete(prompt)
        try:
            return parse_json_reply(reply, schema)
        except (ValueError, ValidationError):
            pass
        reply = self._complete(prompt + load_prompt(JSON_REMINDER))
        try:
            return parse_json_reply(reply, schema)
        except (ValueError, ValidationError) as exc:
            raise JudgeError(
                f"judge returned invalid JSON for prompt {prompt_name!r} twice "
                f"({type(exc).__name__}: {exc}); last reply starts: {reply[:200]!r}"
            ) from exc

    def _complete(self, prompt: str) -> str:
        self.n_network_calls += 1
        if self._completion_fn is not None:
            return self._completion_fn(prompt)
        return self._openai_complete(prompt)

    def _openai_complete(self, prompt: str) -> str:
        import openai

        if not self.api_key:
            raise JudgeConfigError(
                f"no judge API key configured. Set {ENV_API_KEY} (any non-empty string such "
                "as 'ollama' works for a local Ollama server) or pass --judge-api-key. "
                "To skip answer metrics entirely, use --retrieval-only."
            )
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
            )
        extra: dict[str, Any] = {}
        if self.json_mode:
            extra["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort
        try:
            response = self._create_with_backoff(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                **extra,
            )
        except (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.APIConnectionError,
        ) as exc:
            raise JudgeConfigError(
                f"cannot use judge model {self.model!r} at {self.base_url}: {exc}"
            ) from exc
        except openai.OpenAIError as exc:
            raise JudgeError(f"judge request failed: {exc}") from exc
        return response.choices[0].message.content or ""

    def _throttle(self) -> None:
        wait = self._last_call + self.min_interval_s - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last_call = time.monotonic()

    def _create_with_backoff(self, **request: Any) -> Any:
        """Throttle, then retry rate-limit (429), 5xx and timeout errors with backoff.

        A per-day quota error is fatal (JudgeConfigError) so the run stops instead of
        recording a judge error on every remaining question.
        """
        import openai

        transient = (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError)
        for attempt in range(self.max_attempts):
            self._throttle()
            try:
                return self._client.chat.completions.create(**request)
            except transient as exc:
                if isinstance(exc, openai.RateLimitError) and is_daily_quota_error(exc):
                    raise JudgeConfigError(
                        f"judge model {self.model!r} hit its daily quota; re-run later "
                        f"(cached judgements are reused): {exc}"
                    ) from exc
                if attempt == self.max_attempts - 1:
                    raise
                delay = retry_delay_seconds(exc, attempt)
                logger.warning(
                    "judge %s (%s); retry %d/%d in %.0fs",
                    type(exc).__name__,
                    self.model,
                    attempt + 1,
                    self.max_attempts - 1,
                    delay,
                )
                self._sleep(delay)
        raise AssertionError("unreachable")
