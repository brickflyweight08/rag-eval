# 1. Judge: any OpenAI-compatible endpoint, hosted provider by default in practice

Status: accepted (2026-11-07)

## Context

The spec allows "local Ollama or a hosted provider". The developer has no GPU and no
Ollama install, and chose a hosted OpenAI-compatible provider whose key will be added
later. At build time there was no key, so no real LLM call could be made.

## Decision

- `llm.JudgeClient` talks to any OpenAI-compatible `/v1/chat/completions` endpoint via
  the `openai` client, at temperature 0, with `response_format={"type": "json_object"}`.
  `--no-json-mode` turns that off for endpoints that reject it; replies wrapped in
  ```` ```json ```` fences are still parsed.
- The defaults in `config.py` stay as in the spec (Ollama URL, `llama3.1:8b`) so a
  local setup works with no flags; the README also documents a hosted example.
- Every unit test uses `FakeJudgeClient` or an injected `completion_fn`. Real calls exist
  only in tests marked `@pytest.mark.llm`; `tests/conftest.py` skips them when
  `RAG_EVAL_JUDGE_API_KEY` is unset.
- The demo generator (`examples/toy_handbook/systems.py`) uses the same kind of
  endpoint, configured by `RAG_EVAL_DEMO_*`, falling back to `RAG_EVAL_JUDGE_*`.

## Consequences

- No judge prompt has been tried against a real model yet. The first run with a key
  should be followed by `rag-eval calibrate` before trusting groundedness numbers.
- The README's answer-metric results are placeholders until then (see ADR 0008).
