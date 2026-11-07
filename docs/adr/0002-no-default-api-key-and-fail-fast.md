# 2. No default API key; configuration errors abort the run

Status: accepted (2026-11-07)

## Context

The spec shows `RAG_EVAL_JUDGE_API_KEY=ollama` as an example, and says a failing
question must never abort a run. With a hosted provider, a default key of `ollama`
would only surface as an authentication error on every single question.

## Decision

- `config.load_settings()` has **no default API key** (`None`). A real network call
  without a key raises `JudgeConfigError` with a message explaining what to set
  (any non-empty string works for Ollama) and suggesting `--retrieval-only`.
- `JudgeConfigError` (no key, authentication/permission failure, unknown model,
  unreachable endpoint after the client's own retries) **aborts the run**. Everything
  else (`JudgeError`: invalid JSON twice, rate limits, server errors) is recorded in
  `QuestionResult.error` for that question and the run continues. System exceptions
  are always recorded per question, as the spec requires.
- The judge is created lazily: a retrieval-only run never needs a key and never creates
  the cache file. Cache hits need no key either, so a fully cached run is reproducible
  offline.

## Consequences

- A misconfigured judge fails within the first question instead of producing a run
  full of identical errors.
- The demo generator raises a plain `RuntimeError` when it has no key; that is a
  system error, so it is recorded per question (retrieval metrics are still scored).
