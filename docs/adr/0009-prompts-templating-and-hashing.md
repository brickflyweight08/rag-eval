# 9. Prompt templating, hashing and caching details

Status: accepted (2026-11-07)

## Context

Prompts live in `src/rag_eval/prompts/*.txt` and their SHA-256 goes into
`RunConfig.prompt_versions`. Prompts contain JSON examples, the retry needs a "valid
JSON only" reminder, and a hash of a text file changes if git rewrites line endings.

## Decision

- Templates use `string.Template` (`$claim`, `$context`) so JSON braces need no escaping.
  A missing variable raises `KeyError`.
- The retry reminder is itself a prompt file (`json_reminder.txt`), so no prompt text
  is inlined in code.
- A prompt version is the SHA-256 of the file's **raw bytes**. `.gitattributes` forces
  LF endings so hashes (and `questions_sha256`) are identical on every OS.
- `prompt_versions` lists the prompts the run actually used (claims + verification,
  plus refusal and correctness when enabled); it is `{}` for runs that never called the
  judge, and `judge_model` is then `None`.
- Cache key = sha256(model + rendered prompt). Only replies that parsed and validated
  are cached; the stored value is the validated JSON.
- Schema validation of judge replies (pydantic) counts as a parse failure and triggers
  the single retry.

## Consequences

- Editing a prompt changes both its recorded version and every cache key that uses it,
  so stale judgements are never reused.
