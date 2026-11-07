# rag-eval

A small, general-purpose toolkit for **evaluating retrieval-augmented generation (RAG)
systems**: does retrieval find the right chunks, is the answer supported by what was
retrieved, are the citations real, does the system refuse when it should, and is a
change an actual improvement or just noise?

- **Retrieval metrics** against human-labelled gold chunk IDs: Hit@k, Recall@k,
  Precision@k, MRR.
- **Answer metrics** with an LLM judge: groundedness, citation precision, hallucinated
  citations, refusal accuracy, false-refusal rate, optional correctness.
- **Trustworthy numbers**: every metric as `mean [95% bootstrap CI] (n)`, paired
  bootstrap comparisons between runs, judge calibration against human labels, a
  benchmark hash in every run, and a judge cache that makes re-runs reproducible.
- **Works with any system** through a three-member plugin protocol, or offline from a
  predictions file.

## Install

Python 3.11+. Install from GitHub (there is no PyPI release):

```bash
pip install "rag-eval @ git+https://github.com/<your-username>/rag-eval.git@v0.1.0"
# or, from a clone, with the demo systems and dev tools:
pip install -e ".[demo,dev]"
```

## Quickstart (toy demo, no LLM needed)

From a clone of this repo:

```bash
pip install -e ".[demo,dev]"
rag-eval validate examples/toy_handbook/questions.jsonl
rag-eval run --system examples.toy_handbook.systems:make_bm25 --questions examples/toy_handbook/questions.jsonl --k 3 --out runs/bm25_k3 --retrieval-only
rag-eval run --system examples.toy_handbook.systems:make_dense --questions examples/toy_handbook/questions.jsonl --k 3 --out runs/dense_k3 --retrieval-only
rag-eval report runs/bm25_k3 runs/dense_k3 --out runs/REPORT.md && rag-eval compare runs/bm25_k3 runs/dense_k3
```

With an LLM configured (below), drop `--retrieval-only` to get the answer metrics, and
add `--correctness` to grade answers against the reference answers. The notebook
[`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) walks through the same
steps with explanations.

## Configuration

The judge is any **OpenAI-compatible** endpoint, set with environment variables (each
also has a CLI flag such as `--judge-model`; flags win):

```bash
# Local Ollama
RAG_EVAL_JUDGE_BASE_URL=http://localhost:11434/v1
RAG_EVAL_JUDGE_API_KEY=ollama            # any non-empty string for Ollama
RAG_EVAL_JUDGE_MODEL=llama3.1:8b         # prefer a stronger model than the system under test
RAG_EVAL_CACHE_PATH=.rag_eval_cache.sqlite

# Hosted provider (example)
RAG_EVAL_JUDGE_BASE_URL=https://api.your-provider.example/v1
RAG_EVAL_JUDGE_API_KEY=sk-...
RAG_EVAL_JUDGE_MODEL=your-strongest-model

# Optional, for rate-limited (e.g. free-tier) endpoints and "thinking" models
RAG_EVAL_JUDGE_MAX_RPM=8                 # client-side throttle, requests per minute
RAG_EVAL_JUDGE_REASONING_EFFORT=low      # sent as reasoning_effort when set
```

There is no default API key: without one, answer scoring stops with a clear message and
`--retrieval-only` still works. Judge replies are cached in sqlite keyed by
sha256(model + prompt); `--no-cache` bypasses it. If your endpoint rejects
`response_format`, pass `--no-json-mode`.

Rate-limit (429), 5xx and timeout errors are retried with backoff, using the server's
retry hint when it gives one. If a **per-day** quota is exhausted, the run stops with an
error instead of recording a judge failure on every remaining question. Re-run later and
cached judgements are reused. The cache key does not include `reasoning_effort`, so use
`--no-cache` after changing it.

## Commands

| Command | What it does |
|---|---|
| `rag-eval validate questions.jsonl` | Checks every line (with line numbers) and prints counts per qtype |
| `rag-eval run --system pkg.mod:factory --questions Q --k 5 --out runs/X` | Live run through the plugin protocol. Options: `--system-kwargs '{"strategy": "hybrid"}'`, `--retrieval-only`, `--correctness`, `--refusal-judge`, `--refusal-phrase` (repeatable), `--no-cache` |
| `rag-eval score --predictions P --questions Q --k 5 --out runs/X` | Offline run from a predictions file |
| `rag-eval report runs/* --out REPORT.md` | Markdown report with CIs, per-qtype tables, charts and a failure log |
| `rag-eval compare runs/A runs/B` | Paired bootstrap B − A per metric, overall and per qtype |
| `rag-eval calibrate --labels labels.jsonl` | Judge vs human agreement and Cohen's kappa |

Each run folder contains `config.json` (system, k, judge model, questions file SHA-256,
prompt hashes, rag-eval version), `results.jsonl` (one `QuestionResult` per question)
and `summary.json` (mean, 95% CI and n per metric, overall and per qtype).

## Benchmark format

One JSON object per line:

```json
{"id": "q01", "question": "How many days of paid annual leave do I get?", "qtype": "lookup", "answerable": true, "gold_ids": ["leave-01"], "reference_answer": "27 days.", "metadata": {}}
{"id": "q13", "question": "Can I bring my dog to work?", "qtype": "unanswerable", "answerable": false, "gold_ids": []}
```

`metadata` is for anything system-specific; rag-eval stores it but never reads it.

**Ground truth is your job.** rag-eval never generates gold IDs, reference answers or
calibration labels with an LLM: an LLM-written benchmark inherits the blind spots of the
model that wrote it, and then measures systems by how much they agree with that model.
**Freeze your benchmark before you look at results**: every run records the questions
file's SHA-256 and `compare` refuses to compare runs made on different files.

## Metrics

G = gold chunk IDs, R = retrieved IDs in rank order (duplicates removed, first kept).
Full definitions: [docs/glossary.md](docs/glossary.md).

| Metric | Formula / meaning |
|---|---|
| Hit@k | 1 if G ∩ R[:k] is non-empty, else 0 |
| Recall@k | \|G ∩ R[:k]\| / \|G\| |
| Precision@k | \|G ∩ R[:k]\| / k |
| MRR | mean of 1 / rank of the first gold ID in R[:k] (0 if none) |
| Groundedness | supported claims / claims; the judge extracts atomic claims from the answer, then checks each against **only** the context chunks. `None` with no claims (e.g. a refusal) |
| Hallucinated citations | cited IDs not in the context the generator saw |
| Citation precision | valid citations used as evidence by a supported claim / valid citations |
| Refusal accuracy | refused / unanswerable questions |
| False-refusal rate | refused / answerable questions (lower is better) |
| Correctness (optional) | judge vs `reference_answer`: correct 1, partial 0.5, incorrect 0 |

Retrieval metrics skip unanswerable questions. Every aggregate is shown as
`mean [95% bootstrap CI] (n)`. `compare` uses a **paired** bootstrap over the same
questions and says "not clear at this sample size" whenever the CI includes zero; it
never names a winner in that case.

## Plugin protocol

```python
from rag_eval import RetrievedChunk, SystemOutput

class MySystem:
    name = "my-system"

    def retrieve(self, question: str, k: int) -> list[RetrievedChunk]:
        ...  # ranked chunks, each with a chunk_id matching your gold_ids

    def answer(self, question: str, k: int) -> SystemOutput:
        ...  # SystemOutput(answer=..., citations=[chunk ids], context=[chunks the LLM saw])
        # raise NotImplementedError for a retrieval-only system

def make_system(strategy: str = "dense") -> MySystem:  # the factory
    return MySystem()
```

```bash
rag-eval run --system my_pkg.my_module:make_system --system-kwargs '{"strategy": "hybrid"}' \
             --questions questions.jsonl --k 5 --out runs/hybrid_k5
```

The factory is called with the JSON kwargs. Answer metrics are computed on
`output.context`, i.e. what the generator actually saw. A system exception is recorded
for that question and the run continues. The same is available from Python:

```python
from rag_eval import run_live, score_offline, load_questions
artifacts = run_live("my_pkg.my_module:make_system", "questions.jsonl", k=5, out_dir="runs/x")
print(artifacts.summary["overall"]["recall"])
```

Public API: `EvalQuestion`, `RetrievedChunk`, `SystemOutput`, `Prediction`, `RAGSystem`,
`run_live`, `score_offline`, `load_questions`.

## Offline mode

No plugin needed: write one `Prediction` per line and score it.

```json
{"question_id": "q01", "retrieved": [{"chunk_id": "leave-01", "text": "...", "score": 12.3}], "output": {"answer": "27 days [leave-01].", "citations": ["leave-01"], "context": [{"chunk_id": "leave-01", "text": "..."}], "latency_ms": 840}}
```

```bash
rag-eval score --predictions preds.jsonl --questions questions.jsonl --k 5 --out runs/external
```

`output` is optional; without it only retrieval is scored.

## Judge calibration

An LLM judge has biases. Before trusting groundedness numbers, label a sample of claims
yourself (one per line: `question_id`, `claim`, `context`, `human_supported`) and run
`rag-eval calibrate --labels labels.jsonl`. Below about 80% agreement or 0.6 kappa,
improve the prompt or use a stronger judge first.

## Demo results

Toy handbook demo: 30 passages of an original, fictional company handbook, 15 questions
(12 answerable, 3 unanswerable), k = 3. Retrieval numbers are real runs
(`--retrieval-only`); see the full
[report](examples/toy_handbook/results/REPORT_retrieval_only.md) and
[comparison](examples/toy_handbook/results/COMPARE_retrieval_only.md).

| System | Hit@3 | Recall@3 | MRR | Groundedness | Citation precision | Refusal accuracy |
|---|---|---|---|---|---|---|
| BM25 | 0.917 [0.750, 1.000] (n=12) | 0.833 [0.666, 1.000] (n=12) | 0.819 [0.625, 1.000] (n=12) | pending LLM key | pending LLM key | pending LLM key |
| Dense (all-MiniLM-L6-v2) | 1.000 [1.000, 1.000] (n=12) | 0.917 [0.792, 1.000] (n=12) | 0.903 [0.764, 1.000] (n=12) | pending LLM key | pending LLM key | pending LLM key |

Dense retrieval looks better on the paraphrased questions (for example "pedalling to
the office" vs a passage about cycling), but with 12 questions every paired 95% CI
includes zero, so the difference is **not clear at this sample size**. That is the
point of showing CIs.

> The answer-metric columns will be filled in once an LLM key is configured (see
> [ADR 0008](docs/adr/0008-retrieval-only-demo-results.md)). The demo's
> `human_labels.jsonl` is a **Claude-labelled draft that needs human review**, not a set
> of human labels ([ADR 0006](docs/adr/0006-claude-drafted-calibration-labels.md)).

## Used by

- [restaurant-graphrag](https://github.com/<your-username>/restaurant-graphrag): a
  GraphRAG pipeline evaluated through the `RAGSystem` protocol.

## Limitations

- **LLM-judge bias.** Groundedness, citation precision and correctness come from an LLM
  judge, which can be lenient, inconsistent, or biased towards fluent answers. Calibrate
  it against human labels, and prefer a judge stronger than the system under test.
- **Small benchmarks.** With tens of questions, confidence intervals are wide and most
  differences are not clear. Report the CIs; do not over-read point estimates.
- **Groundedness ≠ correctness.** An answer can be fully supported by a retrieved chunk
  that is irrelevant or wrong. Use retrieval metrics and (optionally) correctness too.
- **Single evidence chunk.** The verifier names one supporting chunk per claim, so
  citation precision can undercount answers that correctly cite several supporting
  chunks.
- **Phrase-based refusal detection** is simple; use `--refusal-judge` if your system
  phrases refusals in many ways.

## Development

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[demo,dev]"
pytest -m "not llm"        # unit tests, fake judge and fake systems, < 30 s
pytest -m llm              # real judge calls; skipped unless RAG_EVAL_JUDGE_API_KEY is set
ruff check . && ruff format --check .
```

Design decisions are recorded in [docs/adr/](docs/adr/); terms are defined in
[docs/glossary.md](docs/glossary.md). Licence: MIT.
