# 12. Judge throttling, rate-limit backoff and reasoning effort

**Status:** accepted (2026-11-07)

## Context
The developer's judge runs on a free-tier hosted endpoint (Gemini's OpenAI-compatible API). It has small per-minute and per-day request quotas, and its stronger models "think" by default, which took ~18 s per call versus ~1.4 s with `reasoning_effort: "low"`. Before this change, the client relied on the openai SDK's two quick retries. A burst of 429s would therefore turn into per-question `JudgeError`s, and a run would finish "successfully" with silently missing answer metrics.

## Decision
- Two new optional settings, each with an env var and a CLI flag:
  - `max_rpm` (`RAG_EVAL_JUDGE_MAX_RPM`, `--judge-max-rpm`): a client-side minimum interval between judge calls.
  - `reasoning_effort` (`RAG_EVAL_JUDGE_REASONING_EFFORT`, `--judge-reasoning-effort`): passed through as `reasoning_effort` when set, and omitted otherwise.
- `RateLimitError`, `InternalServerError` and `APITimeoutError` are retried up to 8 attempts. The wait uses the server's hint when present (a `Retry-After` header, "retry in 12.5s" or `retryDelay` in the body), otherwise exponential backoff, capped at 90 s.
- A 429 that names a **per-day** quota raises `JudgeConfigError`, which stops the run (ADR 0002) instead of recording an error on every remaining question. Re-running later reuses the cache.
- The cache key stays sha256(model + prompt) as the spec requires, so it does **not** include `reasoning_effort`. Changing the effort for the same model needs `--no-cache` to take effect.

## Consequences
Nothing changes for endpoints without limits, since both settings default to off. Long runs on free tiers are slower but complete, and a hard daily stop is visible.
