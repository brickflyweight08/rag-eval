# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status and spec

The spec is `IMPLEMENTATION_PLAN.md` in this folder. Section 9 of that file refers to `docs/IMPLEMENTATION_PLAN.md`, but no such path exists. As of this writing, no code exists yet, so the commands and layout below come from the plan and are not yet verified.

- Work **one phase at a time, in order** (R1 → R8). A phase is done only when every acceptance criterion in the plan passes.
- **Ask the developer before changing the public API**: `schema.py` types, the `RAGSystem` protocol in `protocols.py`, and the `__init__.py` exports (`EvalQuestion`, `RetrievedChunk`, `SystemOutput`, `Prediction`, `RAGSystem`, `run_live`, `score_offline`, `load_questions`). Other projects depend on it. Everything else is internal.

## Commands (planned)

```bash
pip install -e ".[demo,dev]"         # demo = rank-bm25 + sentence-transformers; dev = pytest + ruff

rag-eval validate questions.jsonl
rag-eval run --system pkg.module:factory --system-kwargs '{"strategy": "hybrid"}' \
             --questions questions.jsonl --k 5 --out runs/NAME [--correctness] [--retrieval-only] [--no-cache]
rag-eval score --predictions preds.jsonl --questions questions.jsonl --k 5 --out runs/NAME   # offline mode
rag-eval report runs/* --out REPORT.md
rag-eval compare runs/A runs/B
rag-eval calibrate --labels human_labels.jsonl

pytest -m "not llm"                  # default test run; target < 30 s
pytest tests/test_retrieval.py::test_name   # single test
pytest -m llm                        # tests that make real judge calls
ruff check . && ruff format .
grep -ri "arango\|mysql\|restaurant" src/   # must return nothing (R8 acceptance check)
```

The demo systems for the toy demo are `examples.toy_handbook.systems:make_bm25` and `examples.toy_handbook.systems:make_dense`.

## Architecture

There are two ways to evaluate a system. Both write the same outputs to `--out`: `config.json` (`RunConfig`), `results.jsonl` (one `QuestionResult` per line) and `summary.json` (aggregates with CIs, overall and per qtype).

- **Live (`runner.run_live`):** `loading.load_system("module:factory", kwargs)` imports a factory that returns an object satisfying `RAGSystem`. For each question, the runner calls `retrieve(question, k)` to compute retrieval metrics, then `answer(question, k)` to compute answer metrics. Answer metrics are computed on **`output.context`**, the chunks the generator actually saw, not on the `retrieve` result. If `answer` raises `NotImplementedError`, the system is treated as retrieval-only. Any other exception is recorded in `QuestionResult.error`, and the run continues.
- **Offline (`runner.score_offline`):** reads a `Prediction` JSONL file produced by any external system. No plugin is needed.

Metrics are computed per question in `metrics/`, then aggregated by `stats.py`:

- **Retrieval metrics** (hit, recall, precision, reciprocal rank):
  - Pure set arithmetic over gold IDs vs. the top-k retrieved IDs.
  - Duplicate retrieved IDs are de-duplicated (keeping the first occurrence) and a warning is logged.
  - Unanswerable questions get `None` and are excluded from these metrics.
- **Groundedness** takes two judge steps:
  1. Extract the claims from the answer.
  2. Verify each claim against the context chunks only, each prefixed with `[chunk_id]`.

  Groundedness is supported claims ÷ total claims, or `None` if there are no claims. A refusal must yield `None`, not count as ungrounded.
- **Citation precision** reuses the verifier's `evidence_id`s: it is the share of valid citations that served as evidence for some claim. Citations not in `context` are counted as hallucinated citations.
- **Refusal detection** matches a configurable phrase list by default, or uses the LLM with `--refusal-judge`. **Correctness** is optional (`--correctness`) and compares the answer to `reference_answer`.

`stats.py` is implemented by hand with numpy:

- `bootstrap_ci` is used for every reported metric.
- `paired_bootstrap` (B − A over the same questions) backs `compare`.
- `cohen_kappa` backs `calibrate`.
- Seeded randomness must give deterministic results.

`llm.JudgeClient.complete_json(prompt_name, variables)`:

- Runs at temperature 0 and expects JSON. If parsing fails it retries once, then raises an error that the runner records for that question.
- Results are cached in sqlite keyed by `sha256(model + prompt_text)`, so a repeated run makes zero network calls.
- Prompts are loaded from `prompts/*.txt`. Their SHA-256 hashes go into `RunConfig.prompt_versions`, so editing a prompt shows up in the results.

**Reproducibility contract:** re-running the same command with the cache must give byte-identical `results.jsonl`, ignoring timestamps and latency. Keep output ordering and serialization deterministic.

## Rules

- **No domain coupling:** `src/rag_eval/` must never import or mention ArangoDB, MySQL, the restaurant project or any specific dataset. System-specific data goes in `EvalQuestion.metadata`, which rag-eval stores but never reads. Do not create or edit `restaurant-graphrag` code from this repo.
- Never generate gold labels or reference answers with an LLM. Ground truth is the user's job.
- Honest reporting: always show `mean [95% CI]` and the sample size. Never declare a winner when the CI includes zero. `compare` must refuse runs whose `questions_sha256` differ.
- Judge prompts live only in `src/rag_eval/prompts/*.txt`. Never inline them in code.
- Read env vars (`RAG_EVAL_JUDGE_BASE_URL`, `_API_KEY`, `_MODEL`, `RAG_EVAL_CACHE_PATH`) only in `config.py`. Every setting must also be overridable by a CLI flag. The judge is any OpenAI-compatible endpoint (Ollama by default).
- Testing:
  - Unit tests use `FakeJudgeClient` (scripted JSON) and `FakeRAGSystem` (fixed outputs; one variant raises).
  - Metrics are tested against hand-computed values.
  - Real LLM calls happen only under `@pytest.mark.llm`.
- Python 3.11+, pydantic v2, type hints everywhere, ruff-formatted, small readable functions. Run `pytest -m "not llm"` and `ruff check` before finishing a task.
- The toy demo corpus is an **original, fictional** company handbook. That keeps the groundedness test meaningful, because an LLM can't answer from memory.
