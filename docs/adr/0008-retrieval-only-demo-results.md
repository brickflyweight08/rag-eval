# 8. Demo results are retrieval-only until an LLM key is available

Status: accepted (2026-11-07)

## Context

R8 asks for the demo's results table in the README and a notebook that runs top to
bottom. Answer metrics need both a generator LLM and a judge LLM; no key existed when
the repo was built.

## Decision

- BM25 and dense runs were executed for real with `--retrieval-only` (k = 3). Their
  numbers are in the README; the report and comparison are committed in
  `examples/toy_handbook/results/`, and the run summaries in `runs/*/`.
- Answer-metric columns in the README show "pending LLM key".
- Notebook cells that need an LLM check `RAG_EVAL_JUDGE_API_KEY` and print "Skipped"
  when it is unset, so the notebook runs top to bottom either way. It was executed
  without a key and committed with its outputs.

## Consequences

- After adding a key: run the two full commands from the README, regenerate the report,
  run `rag-eval calibrate`, and replace the placeholders.
