# 10. Retrieval and runner edge cases

Status: accepted (2026-11-07)

## Context

The R2/R5 definitions leave a few cases open.

## Decision

- **Precision@k divides by k**, even if the system returned fewer than k chunks, so
  returning less is not rewarded.
- `retrieved_ids` in results are the de-duplicated top k (what was scored).
- An answerable question with an empty gold set cannot exist (validation rejects it),
  so "empty gold" in the metric functions means unanswerable: every retrieval metric is
  `None`.
- Live mode calls `retrieve(question, k)` for retrieval metrics and `answer(question, k)`
  separately; answer metrics use `output.context`, not the `retrieve` result.
- If `answer` raises `NotImplementedError`, the run becomes retrieval-only from that
  question on (it is normally the first question). Other exceptions in `retrieve` or
  `answer` are recorded as `"<stage>: <ExceptionType>: <message>"`; if both fail, both
  are recorded, joined by `; `.
- Offline mode: a question without a prediction gets the error
  `no prediction for this question` and no metrics; predictions for unknown IDs are
  ignored with a warning; a prediction without `output` is scored for retrieval only.
- Questions given as an in-memory list are hashed as their canonical JSONL
  serialisation, so they will not match the hash of a file with different formatting.
- `run_id` defaults to the `--out` folder name.

## Consequences

- These choices are covered by unit tests in `tests/test_retrieval.py` and
  `tests/test_runner.py`.
