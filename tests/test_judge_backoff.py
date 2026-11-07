"""Throttling, rate-limit backoff and reasoning_effort in JudgeClient (docs/adr/0012)."""

import openai
import pytest

try:  # openai>=3 uses httpx2; older versions use httpx
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from rag_eval.config import load_settings
from rag_eval.llm import JudgeClient, JudgeConfigError, JudgeError


def _rate_limit(message: str) -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "http://judge/chat"))
    return openai.RateLimitError(message, response=response, body=None)


class _Reply:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


class FakeOpenAI:
    """Scripted chat.completions.create: items are exceptions to raise or reply texts."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **request):
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Reply(item)


def _client(script, **kwargs):
    sleeps = []
    judge = JudgeClient(model="m", api_key="k", sleep=sleeps.append, **kwargs)
    judge._client = FakeOpenAI(script)
    return judge, sleeps


def test_rate_limit_is_retried_with_server_delay():
    judge, sleeps = _client([_rate_limit("Quota exceeded. Please retry in 12.5s."), '{"ok": true}'])
    assert judge._openai_complete("p") == '{"ok": true}'
    assert sleeps == [13.5]


def test_backoff_without_hint_is_exponential():
    judge, sleeps = _client([_rate_limit("slow down"), _rate_limit("slow down"), "{}"])
    judge._openai_complete("p")
    assert sleeps == [2.0, 4.0]


def test_daily_quota_is_fatal():
    judge, _ = _client([_rate_limit("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")])
    with pytest.raises(JudgeConfigError, match="daily quota"):
        judge._openai_complete("p")


def test_gives_up_after_max_attempts():
    judge, sleeps = _client([_rate_limit("x")] * 3, max_attempts=3)
    with pytest.raises(JudgeError):
        judge._openai_complete("p")
    assert len(sleeps) == 2


def test_reasoning_effort_and_json_mode_are_sent():
    judge, _ = _client(["{}"], reasoning_effort="low")
    judge._openai_complete("p")
    request = judge._client.requests[0]
    assert request["reasoning_effort"] == "low"
    assert request["response_format"] == {"type": "json_object"}


def test_no_reasoning_effort_by_default():
    judge, _ = _client(["{}"])
    judge._openai_complete("p")
    assert "reasoning_effort" not in judge._client.requests[0]


def test_throttle_spaces_calls(monkeypatch):
    judge, sleeps = _client(["{}", "{}"], max_rpm=60)  # one call per second
    clock = iter([100.0, 100.0, 100.2, 101.0])
    monkeypatch.setattr("rag_eval.llm.time.monotonic", lambda: next(clock))
    judge._openai_complete("a")
    judge._openai_complete("b")
    assert sleeps == [pytest.approx(0.8)]


def test_settings_read_env(monkeypatch):
    monkeypatch.setenv("RAG_EVAL_JUDGE_MAX_RPM", "8")
    monkeypatch.setenv("RAG_EVAL_JUDGE_REASONING_EFFORT", "low")
    s = load_settings()
    assert (s.max_rpm, s.reasoning_effort) == (8.0, "low")
    assert load_settings(max_rpm=2).max_rpm == 2


def test_settings_reject_bad_rpm(monkeypatch):
    monkeypatch.setenv("RAG_EVAL_JUDGE_MAX_RPM", "fast")
    with pytest.raises(ValueError, match="RAG_EVAL_JUDGE_MAX_RPM"):
        load_settings()
