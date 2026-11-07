import hashlib

import pytest
from pydantic import BaseModel

from rag_eval import llm
from rag_eval.config import load_settings
from rag_eval.llm import (
    JudgeCache,
    JudgeClient,
    JudgeConfigError,
    JudgeError,
    parse_json_reply,
    prompt_versions,
    render_prompt,
)


class Verdict(BaseModel):
    supported: bool
    evidence_id: str | None


class Scripted:
    """A completion function that returns scripted replies and records prompts."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


VARS = {"claim": "Leave is 25 days.", "context": "[c1] Staff get 25 days of leave."}


def test_render_prompt_fills_variables():
    text = render_prompt("verify_claim", VARS)
    assert "Leave is 25 days." in text and "[c1] Staff get 25 days" in text
    assert "$claim" not in text


def test_render_prompt_missing_variable():
    with pytest.raises(KeyError):
        render_prompt("verify_claim", {"claim": "x"})


def test_prompt_versions_are_file_hashes():
    versions = prompt_versions()
    assert {"extract_claims", "verify_claim", "correctness", "refusal"} <= set(versions)
    raw = (llm.PROMPTS_DIR / "verify_claim.txt").read_bytes()
    assert versions["verify_claim"] == hashlib.sha256(raw).hexdigest()


def test_changing_a_prompt_changes_its_version(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "PROMPTS_DIR", tmp_path)
    (tmp_path / "p.txt").write_text("v1 $x", encoding="utf-8")
    before = prompt_versions(["p"])
    (tmp_path / "p.txt").write_text("v2 $x", encoding="utf-8")
    assert prompt_versions(["p"]) != before


def test_parse_json_reply_handles_fences_and_schema():
    assert parse_json_reply('```json\n{"a": 1}\n```') == {"a": 1}
    reply = '{"supported": "true", "evidence_id": null}'
    assert parse_json_reply(reply, Verdict) == {"supported": True, "evidence_id": None}
    with pytest.raises(ValueError):
        parse_json_reply("[1, 2]")


def test_complete_json_ok():
    fn = Scripted('{"supported": true, "evidence_id": "c1"}')
    client = JudgeClient("m", completion_fn=fn)
    assert client.complete_json("verify_claim", VARS, Verdict) == {
        "supported": True,
        "evidence_id": "c1",
    }
    assert client.n_network_calls == 1


def test_complete_json_retries_once_with_reminder():
    fn = Scripted("Sure! Here you go: supported", '{"supported": false, "evidence_id": null}')
    client = JudgeClient("m", completion_fn=fn)
    assert client.complete_json("verify_claim", VARS, Verdict)["supported"] is False
    assert client.n_network_calls == 2
    assert "valid JSON only" in fn.prompts[1]


def test_schema_mismatch_also_retries_then_raises():
    fn = Scripted('{"verdict": "yes"}')
    client = JudgeClient("m", completion_fn=fn)
    with pytest.raises(JudgeError, match="invalid JSON for prompt 'verify_claim' twice"):
        client.complete_json("verify_claim", VARS, Verdict)
    assert client.n_network_calls == 2


def test_second_identical_call_is_served_from_cache(tmp_path):
    cache = JudgeCache(tmp_path / "cache.sqlite")
    fn = Scripted('{"supported": true, "evidence_id": "c1"}')
    first = JudgeClient("m", cache=cache, completion_fn=fn)
    a = first.complete_json("verify_claim", VARS, Verdict)
    # A brand-new client sharing only the cache file makes zero network calls.
    second = JudgeClient("m", cache=JudgeCache(tmp_path / "cache.sqlite"), completion_fn=fn)
    b = second.complete_json("verify_claim", VARS, Verdict)
    assert a == b
    assert (first.n_network_calls, second.n_network_calls) == (1, 0)
    # A different model is a different cache key.
    third = JudgeClient("other", cache=cache, completion_fn=fn)
    third.complete_json("verify_claim", VARS, Verdict)
    assert third.n_network_calls == 1


def test_invalid_replies_are_not_cached(tmp_path):
    cache = JudgeCache(tmp_path / "cache.sqlite")
    client = JudgeClient("m", cache=cache, completion_fn=Scripted("nope"))
    with pytest.raises(JudgeError):
        client.complete_json("verify_claim", VARS, Verdict)
    client._completion_fn = Scripted('{"supported": true, "evidence_id": "c1"}')
    assert client.complete_json("verify_claim", VARS, Verdict)["supported"] is True


def test_missing_api_key_is_a_config_error():
    client = JudgeClient("m", base_url="http://localhost:1/v1", api_key=None)
    with pytest.raises(JudgeConfigError, match="RAG_EVAL_JUDGE_API_KEY"):
        client.complete_json("verify_claim", VARS, Verdict)


@pytest.mark.llm
def test_real_judge_verifies_a_simple_claim(tmp_path):
    settings = load_settings(cache_path=str(tmp_path / "c.sqlite"))
    client = JudgeClient.from_settings(settings)
    result = client.complete_json("verify_claim", VARS, Verdict)
    assert result["supported"] is True
    assert result["evidence_id"] == "c1"
