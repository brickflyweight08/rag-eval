# 5. Correctness: numeric score and the judge's reason

Status: accepted (2026-11-07)

## Context

`--correctness` asks the judge for `correct | partial | incorrect` "with a one-line
reason". The report must show correctness as `mean [95% CI]`, which needs numbers.
`QuestionResult` (public API, frozen without the developer's approval) has a
`correctness` field but no field for the reason.

## Decision

- For averaging, correct = 1, partial = 0.5, incorrect = 0.
- The judge is still asked for a reason (it tends to improve the verdict, and it is kept
  in the judge cache) and the reason is written to the progress log, but it is **not
  stored in `results.jsonl`**.
- Correctness is only judged when the question has a `reference_answer`; otherwise it is
  `None`.

## Consequences

- Reasons are lost after the run. Adding `correctness_reason: str | None = None` to
  `QuestionResult` would be backward compatible; it needs the developer's approval
  because the schema is shared with other projects.
