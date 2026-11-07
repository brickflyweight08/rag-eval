# Implementation Plan: `rag-eval`

**RAG Evaluation Framework** | Python, Embeddings, Vector Search, LLMs

> **How to use this document (for Claude Code):** This is the reference spec for the `rag-eval` repository. Work **one phase at a time**, in order (R1 → R8). Each phase lists tasks and **acceptance criteria**; a phase is not done until every criterion passes. Before changing the public data types or the `RAGSystem` protocol (Section 4), ask the developer first, because other projects depend on them. Keep code typed, tested, and small enough for a beginner to read.

> **This is one of two separate repositories.** `rag-eval` is a **general-purpose** toolkit for evaluating any retrieval-augmented generation (RAG) system. It must **never import or mention ArangoDB, MySQL, or the restaurant project in its code**. The restaurant project (`restaurant-graphrag`, separate repo and plan) installs `rag-eval` as a dependency and plugs into it through the `RAGSystem` protocol. Do not create or edit `restaurant-graphrag` code from this repo.

---

## 0. Build order across both repos

| Step | Repo | Phases |
|---|---|---|
| 1 | `restaurant-graphrag` | Phases 1–6: working GraphRAG pipeline |
| 2 | `rag-eval` | **Phases R1–R8 (this document)**: standalone toolkit, tested on its own toy demo. Tag `v0.1.0`. |
| 3 | `restaurant-graphrag` | Phase 7: installs `rag-eval`, builds the restaurant benchmark, runs and reports |

`rag-eval` must be fully usable and demonstrable on its own (via the toy demo in R8) before step 3.

---

## 1. Goals

- Measure **retrieval quality** of any RAG system against a benchmark of questions with labelled ground-truth chunk IDs: hit rate@k, recall@k, precision@k, and MRR.
- Measure **answer quality**: groundedness (share of answer claims supported by the retrieved context), citation precision, hallucinated citations, refusal accuracy, and optional answer correctness.
- Make results **trustworthy**: bootstrap confidence intervals, paired comparison between runs, LLM-judge calibration against human labels, and reproducible cached runs.
- Work with any system through a tiny plugin protocol, or offline from a predictions file.

**Non-goals:** a web dashboard, hosted service, or publishing to PyPI. Users install from GitHub.

---

## 2. Tech Stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | Type hints throughout |
| Data types | `pydantic` v2 | All inputs/outputs validated |
| CLI | `typer` | Entry point `rag-eval` |
| LLM judge | Any **OpenAI-compatible** endpoint via the `openai` client | Local Ollama or a hosted provider, configured by env vars |
| LLM cache | `sqlite3` (standard library) | Keyed by (model, prompt hash) |
| Stats | Plain Python + `numpy` | Bootstrap and Cohen's kappa implemented by hand (small and testable) |
| Reports | Markdown tables + `matplotlib` charts | |
| Toy demo only | `rank-bm25`, `sentence-transformers` | Optional extra `[demo]` |
| Tests | `pytest` | |
| Lint/format | `ruff` | |

### 2.1 Configuration (environment variables)

```
RAG_EVAL_JUDGE_BASE_URL=http://localhost:11434/v1
RAG_EVAL_JUDGE_API_KEY=ollama
RAG_EVAL_JUDGE_MODEL=llama3.1:8b      # prefer a stronger model than the system under test
RAG_EVAL_CACHE_PATH=.rag_eval_cache.sqlite
```

Every setting can also be overridden by a CLI flag. Read env vars in one place (`config.py`).

### 2.2 Packaging

```toml
[project]
name = "rag-eval"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["pydantic>=2", "typer", "openai", "numpy", "matplotlib"]

[project.optional-dependencies]
demo = ["rank-bm25", "sentence-transformers"]
dev = ["pytest", "ruff"]

[project.scripts]
rag-eval = "rag_eval.cli:app"
```

Other projects install it with `pip install "rag-eval @ git+https://github.com/<your-username>/rag-eval.git@v0.1.0"`.

---

## 3. Repository Layout

```
rag-eval/
├── CLAUDE.md                     # rules for Claude Code (Section 9)
├── README.md
├── LICENSE                       # MIT
├── pyproject.toml
├── src/rag_eval/
│   ├── __init__.py               # re-exports the public API (Section 4)
│   ├── config.py
│   ├── schema.py                 # EvalQuestion, RetrievedChunk, SystemOutput, RunConfig, QuestionResult
│   ├── protocols.py              # RAGSystem protocol
│   ├── loading.py                # load "module.path:factory" plugins; load/validate JSONL
│   ├── llm.py                    # JudgeClient: OpenAI-compatible, temperature 0, JSON mode, sqlite cache
│   ├── metrics/
│   │   ├── retrieval.py          # hit@k, recall@k, precision@k, reciprocal rank
│   │   ├── groundedness.py       # claim extraction + verification
│   │   ├── citations.py          # citation precision, hallucinated citations
│   │   ├── refusal.py            # refusal detection + accuracy
│   │   └── correctness.py        # optional LLM-judged correctness vs reference
│   ├── stats.py                  # bootstrap CI, paired bootstrap, Cohen's kappa
│   ├── runner.py                 # live runs (plugin) and offline scoring (predictions file)
│   ├── report.py                 # markdown report, charts, failure log
│   ├── calibrate.py              # judge vs human-label agreement
│   ├── prompts/                  # judge prompts as .txt files, versioned
│   └── cli.py
├── examples/
│   └── toy_handbook/
│       ├── corpus.jsonl          # ~30 short passages about a FICTIONAL company
│       ├── questions.jsonl       # ~15 benchmark questions
│       ├── systems.py            # BM25 and dense-embedding demo systems
│       └── human_labels.jsonl    # ~10 hand-labelled answers for calibration demo
├── notebooks/
│   └── 01_quickstart.ipynb
└── tests/
```

---

## 4. Public API (keep stable)

### 4.1 Data types (`schema.py`)

```python
class EvalQuestion(BaseModel):
    id: str
    question: str
    qtype: str = "default"            # free-form category used for per-type breakdowns
    answerable: bool = True
    gold_ids: list[str] = []          # chunk IDs needed to answer; empty if not answerable
    reference_answer: str | None = None
    metadata: dict[str, Any] = {}     # anything system-specific (rag-eval never reads it)

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] = {}

class SystemOutput(BaseModel):
    answer: str
    citations: list[str] = []         # chunk IDs the system cited
    context: list[RetrievedChunk]     # what the generator actually saw
    latency_ms: float | None = None

class Prediction(BaseModel):          # one line of an offline predictions file
    question_id: str
    retrieved: list[RetrievedChunk]   # in rank order
    output: SystemOutput | None = None

class RunConfig(BaseModel):
    run_id: str
    system: str                       # plugin path or "offline:<file>"
    system_kwargs: dict[str, Any] = {}
    k: int
    judge_model: str | None
    questions_path: str
    questions_sha256: str
    prompt_versions: dict[str, str]   # judge prompt file hashes
    rag_eval_version: str
    created_at: str

class QuestionResult(BaseModel):
    question_id: str
    qtype: str
    answerable: bool
    retrieved_ids: list[str]
    hit: bool | None                  # None for unanswerable questions
    recall: float | None
    precision: float | None
    reciprocal_rank: float | None
    answer: str | None
    citations: list[str] = []
    hallucinated_citations: list[str] = []
    claims: list[dict] = []           # {"claim", "supported", "evidence_id"}
    groundedness: float | None = None # None if no factual claims or no answer
    citation_precision: float | None = None
    refused: bool | None = None
    correctness: Literal["correct", "partial", "incorrect"] | None = None
    latency_ms: float | None = None
    error: str | None = None          # system exception, recorded instead of crashing the run
```

### 4.2 Plugin protocol (`protocols.py`)

```python
class RAGSystem(Protocol):
    name: str
    def retrieve(self, question: str, k: int) -> list[RetrievedChunk]: ...
    def answer(self, question: str, k: int) -> SystemOutput: ...
```

- A system may be **retrieval-only**: if `answer` raises `NotImplementedError`, the runner skips answer metrics for that run.
- Systems are loaded from an import path to a **factory**: `--system my_pkg.my_module:make_system --system-kwargs '{"strategy": "hybrid"}'`. The factory is called with the kwargs and must return an object satisfying the protocol.

### 4.3 Public exports (`__init__.py`)

`EvalQuestion`, `RetrievedChunk`, `SystemOutput`, `Prediction`, `RAGSystem`, `run_live`, `score_offline`, `load_questions`. Anything else is internal.

---

## 5. Phases

### Phase R1: Scaffold, schemas, dataset validation

**Tasks**
1. `pyproject.toml`, `ruff` config, `src/` layout, `config.py`, `schema.py`, `protocols.py`.
2. `loading.py`: `load_questions(path)` reads JSONL, validates each line, and reports **all** errors with line numbers (not just the first). Also `load_predictions(path)` and `load_system("module:factory", kwargs)`.
3. Dataset checks: unique IDs; answerable questions have at least one gold ID; unanswerable questions have none; non-empty question text. Warn (don't fail) on qtypes with fewer than 3 questions.
4. CLI: `rag-eval validate questions.jsonl` prints a summary (count per qtype, answerable vs not) and any problems.

**Acceptance criteria**
- Invalid files produce clear, line-numbered errors; valid files print the summary.
- `load_system` gives a helpful error if the module, attribute, or protocol methods are missing.

### Phase R2: Retrieval metrics and statistics

**Definitions** (for a question with gold set `G` and ranked retrieved IDs `R`, top-k):

- **Hit@k** = 1 if `G ∩ R[:k]` is non-empty, else 0
- **Recall@k** = `|G ∩ R[:k]| / |G|`
- **Precision@k** = `|G ∩ R[:k]| / k`
- **Reciprocal rank** = `1 / rank` of the first gold ID in `R[:k]`, or 0 if none
- Unanswerable questions are excluded from retrieval metrics (their fields are `None`).
- Duplicate IDs in `R` are de-duplicated (keeping the first occurrence) before scoring, and a warning is logged.

**`stats.py`**
- `bootstrap_ci(values, n_resamples=2000, alpha=0.05, seed=0)` → (mean, low, high).
- `paired_bootstrap(a, b, ...)` → mean difference `b − a` with a 95% CI, computed over the **same questions**. This is what `compare` uses.
- `cohen_kappa(labels_a, labels_b)` for binary labels.

**Acceptance criteria**
- Unit tests with hand-computed toy cases: empty gold, gold at rank 1, gold at rank k, gold missing, duplicates in `R`.
- Bootstrap results are deterministic for a fixed seed.

### Phase R3: Judge LLM client and cache

**Tasks**
1. `llm.py`: `JudgeClient.complete_json(prompt_name, variables) -> dict`. Temperature 0. Ask for JSON; parse and validate; on failure, retry once with a "return valid JSON only" reminder; then raise a clear error that the runner records per question.
2. Prompts live in `prompts/*.txt`. Each prompt's SHA-256 is recorded in `RunConfig.prompt_versions`, so changing a prompt is visible in results.
3. sqlite cache keyed by `sha256(model + prompt_text)`. A `--no-cache` flag bypasses it.

**Acceptance criteria**
- A second identical run makes zero network calls (verified with a mocked client counting calls).
- Unit tests use a fake client; real calls only in tests marked `@pytest.mark.llm`.

### Phase R4: Answer metrics

**4a. Groundedness** (`groundedness.py`): whether the answer is supported by the **retrieved context**. This is not the same as correctness.

1. **Claim extraction** (judge, JSON): split the answer into atomic factual claims, ignoring hedges, greetings, and refusals. Output `{"claims": ["...", "..."]}`.
2. **Claim verification** (judge, JSON): for each claim, give the judge **only** the context chunks (each prefixed with `[chunk_id]`) and ask for `{"supported": true|false, "evidence_id": "<chunk_id or null>"}`. Instruct the judge to mark a claim unsupported if it relies on outside knowledge.
3. **Groundedness** = supported claims ÷ total claims; `None` when there are zero claims.

**4b. Citations** (`citations.py`)
- **Hallucinated citations:** cited IDs not present in `context`.
- **Citation precision:** share of valid citations whose chunk the verifier used as evidence for at least one claim.

**4c. Refusal** (`refusal.py`)
- Detect refusals with a configurable list of phrases (default: "I don't have enough information", "I don't know", "not in the context"), with an optional `--refusal-judge` flag to use the LLM instead.
- **Refusal accuracy** on unanswerable questions; **false-refusal rate** on answerable ones.

**4d. Correctness** (`correctness.py`, optional, `--correctness` flag): the judge compares the answer to `reference_answer` and returns `correct`, `partial`, or `incorrect`, with a one-line reason.

**Acceptance criteria**
- With a fake judge returning scripted outputs, every metric matches hand-computed values.
- A refusal produces `groundedness=None` and is not counted as ungrounded.

### Phase R5: Runner

**Live mode** (`rag-eval run`):

```bash
rag-eval run --system my_pkg.module:make_system --system-kwargs '{"strategy": "hybrid"}' \
             --questions questions.jsonl --k 5 --out runs/hybrid_k5 [--correctness] [--retrieval-only]
```

For each question: call `retrieve(question, k)` for retrieval metrics, then (unless `--retrieval-only`) call `answer(question, k)` and compute answer metrics on `output.context`. Exceptions from the system are caught and recorded in `QuestionResult.error`; the run continues.

**Offline mode** (`rag-eval score`): score a predictions JSONL produced by any external system, with no plugin needed:

```bash
rag-eval score --predictions preds.jsonl --questions questions.jsonl --k 5 --out runs/external
```

**Outputs** (both modes) in the `--out` folder:
- `config.json` (`RunConfig`), `results.jsonl` (`QuestionResult` per line), `summary.json` (aggregates with CIs, overall and per qtype).

**Acceptance criteria**
- Running the same command twice (with cache) produces byte-identical `results.jsonl` (except timestamps and latency, which are excluded from the comparison).
- A progress bar or per-question log line shows progress; a failed question never aborts the run.

### Phase R6: Reports and comparison

**`rag-eval report runs/* --out REPORT.md`**
- Main table: rows = runs; columns = Hit@k, Recall@k, MRR, Groundedness, Citation precision, Correctness (if run), Refusal accuracy, p50/p95 latency. Each metric shown as `mean [95% CI]`.
- Per-qtype table for each metric.
- Charts saved next to the report (PNG): Hit@k by run; groundedness by qtype.
- Failure log: the 5 worst questions per run (lowest recall, then lowest groundedness), showing retrieved vs gold IDs.
- A header stating the number of questions, the questions file hash, the judge model, and the rag-eval version.

**`rag-eval compare runs/A runs/B`**
- Paired bootstrap difference (B − A) for each metric, overall and per qtype, with a plain-language line, for example: "Hit@5 +12.5 pts [95% CI +2.5, +22.5]". If the CI includes zero, say the difference is not clear at this sample size.
- Refuse to compare runs whose `questions_sha256` differ.

**Acceptance criteria**
- Report renders correctly in GitHub's markdown preview.
- `compare` errors clearly on mismatched question files.

### Phase R7: Judge calibration

**`rag-eval calibrate --labels human_labels.jsonl`**

Label file format, one claim per line:

```json
{"question_id": "q003", "claim": "The dental plan covers two cleanings per year.", "context": [{"chunk_id": "c12", "text": "..."}], "human_supported": true}
```

- Run the judge's verification step on each claim and compare with `human_supported`.
- Report percentage agreement, Cohen's kappa, and every disagreement for inspection.
- Print guidance: agreement below about 80% or kappa below about 0.6 means the groundedness numbers should not be trusted yet (improve the prompt or use a stronger judge).

**Acceptance criteria**
- Kappa matches a hand-computed value on a fixed toy label set.

### Phase R8: Toy demo, quickstart notebook, README, v0.1.0

This makes the repo a complete, standalone project.

1. **Corpus** (`examples/toy_handbook/corpus.jsonl`): about 30 short passages (2–4 sentences each) of an **original, fictional** employee handbook for an invented company (leave policy, equipment, travel, security, benefits). Write it from scratch. Because the company is fictional, an LLM cannot answer from memory, which makes the groundedness test meaningful.
2. **Benchmark** (`questions.jsonl`): about 15 questions with gold chunk IDs, covering lookups, multi-passage questions, near-miss wording, and 3 unanswerable questions.
3. **Demo systems** (`systems.py`): `BM25System` (keyword retrieval with `rank-bm25`) and `DenseSystem` (`sentence-transformers` embeddings with cosine similarity), both with the same simple LLM answer step that cites `[chunk_id]`s. Factories: `make_bm25`, `make_dense`.
4. **Calibration labels** (`human_labels.jsonl`): about 10 hand-labelled claims.
5. **Notebook** `01_quickstart.ipynb`: install → validate → run both systems → report → compare → calibrate, with markdown explanations before each cell.
6. **README**: what problem rag-eval solves, a metric glossary with formulas, a 5-line quickstart, the plugin protocol, offline mode, the demo's results table, a "Used by" section linking to `restaurant-graphrag`, and limitations (LLM-judge bias, small benchmarks, groundedness ≠ correctness).
7. Optional: a GitHub Actions workflow running `pytest -m "not llm"` and `ruff check`.
8. Tag `v0.1.0`.

**Acceptance criteria**
- On a fresh clone: `pip install -e ".[demo,dev]"`, then the commands in the README quickstart, produce a report without errors.
- The notebook runs top to bottom.
- `grep -ri "arango\|mysql\|restaurant" src/` returns nothing.

---

## 6. Testing Strategy

- **Unit:** schema validation, loaders, every metric, stats functions, prompt hashing, refusal detection, report formatting.
- **Fake judge:** a `FakeJudgeClient` returning scripted JSON, used across unit tests.
- **Fake system:** a `FakeRAGSystem` returning fixed chunks and answers, used for runner tests (including one that raises an exception).
- **LLM tests:** real calls only under `@pytest.mark.llm`.
- CI goal: `pytest -m "not llm"` passes in under 30 seconds.

---

## 7. Milestones

| Week | Deliverable |
|---|---|
| 1 | R1–R2: schemas, validation, retrieval metrics, bootstrap |
| 2 | R3–R5: judge client + cache, answer metrics, runner (live + offline) |
| 3 | R6–R8: reports, compare, calibration, toy demo, notebook, README. **Tag v0.1.0.** |

Then switch to `restaurant-graphrag` Phase 7.

---

## 8. Design rules

- **No domain coupling:** nothing in `src/rag_eval/` may reference a specific database, dataset, or project. System-specific data goes in `EvalQuestion.metadata`, which rag-eval stores but never reads.
- **Ground truth is the user's job:** rag-eval never generates gold labels or reference answers with an LLM. The README should say so and explain why.
- **Fairness:** every run records the questions file hash; `compare` refuses mismatched files. The README asks users to freeze their benchmark before looking at results.
- **Honest reporting:** always show confidence intervals and sample size; never print a "winner" when the CI includes zero.

---

## 9. `CLAUDE.md` for this repo

```markdown
# rag-eval: rules for Claude Code
- Spec: docs/IMPLEMENTATION_PLAN.md. Work on one phase at a time (R1 → R8); meet every acceptance criterion.
- This is a GENERAL-PURPOSE toolkit. Never import or mention ArangoDB, MySQL or any specific
  project in src/. The restaurant GraphRAG project is a separate repo that depends on this one.
- Do not change the public API (schema.py, protocols.py, __init__.py exports) without asking.
- Python 3.11, type hints everywhere, ruff-formatted. Keep functions small and readable.
- Judge prompts live in src/rag_eval/prompts/*.txt; never inline them in code.
- Unit tests use FakeJudgeClient and FakeRAGSystem; real LLM calls only under @pytest.mark.llm.
- Run `pytest -m "not llm"` and `ruff check` before finishing a task.
- Never generate gold labels or reference answers with an LLM.
```

---

## 10. Definition of Done

- [ ] `rag-eval validate | run | score | report | compare | calibrate` all work as described.
- [ ] Toy demo reproduces from a fresh clone; quickstart notebook runs top to bottom.
- [ ] Every metric has unit tests against hand-computed values.
- [ ] README has the metric glossary, quickstart, plugin guide, demo results, and limitations.
- [ ] `src/` contains no domain-specific code; tagged `v0.1.0`.
- [ ] Used successfully by `restaurant-graphrag` Phase 7, with a "Used by" link in the README.