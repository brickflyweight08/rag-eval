# Glossary

Terms as rag-eval uses them. Formulas use G = gold chunk IDs, R = retrieved chunk IDs in
rank order (de-duplicated), k = cut-off.

**Benchmark (questions file).** A JSONL file of `EvalQuestion`s. Written and frozen by the
user before looking at results; rag-eval records its SHA-256 in every run.

**Chunk / chunk ID.** A unit of text a system can retrieve, identified by a string ID that
the system and the benchmark agree on.

**Gold IDs (`gold_ids`).** The chunk IDs a human decided are needed to answer a question.
Empty for unanswerable questions. Ground truth is the user's job: rag-eval never
generates it.

**Answerable / unanswerable question.** Unanswerable questions have no answer in the
corpus; the correct behaviour is to refuse. They are excluded from retrieval metrics.

**qtype.** A free-form category per question (for example `lookup`, `multi_passage`) used
for per-type breakdowns. Fewer than 3 questions per qtype triggers a warning.

**Reference answer.** A human-written answer used only by the optional correctness metric.

**Hit@k.** 1 if any gold ID is in R[:k], else 0. "Did retrieval find anything useful?"

**Recall@k.** |G ∩ R[:k]| / |G|. Share of the needed chunks that were retrieved.

**Precision@k.** |G ∩ R[:k]| / k. Share of the k slots filled with gold chunks.

**Reciprocal rank / MRR.** 1 / rank of the first gold ID in R[:k] (0 if none); MRR is its
mean over questions.

**Context.** The chunks the generator actually saw (`SystemOutput.context`). Answer
metrics are computed against it, not against the `retrieve` result.

**Claim.** One atomic factual statement extracted from an answer by the judge.

**Groundedness.** Supported claims / total claims, where "supported" means the context
states or directly implies the claim. `None` when there are no claims (e.g. a refusal).
Not the same as correctness: a grounded answer can still be wrong if the context is.

**Evidence ID.** The chunk the verifier names as the best support for a supported claim.

**Citation.** A chunk ID the system says it used (for example `[leave-01]` in the answer).

**Hallucinated citation.** A cited ID that is not in the context the generator saw.

**Citation precision.** Of the valid (in-context) citations, the share used as evidence for
at least one supported claim. `None` with no valid citations or no claims.

**Refusal.** An answer that declines or says it lacks information, detected by phrase
matching (default) or by the judge (`--refusal-judge`).

**Refusal accuracy.** Share of unanswerable questions the system refused. Higher is better.

**False-refusal rate.** Share of answerable questions the system refused. Lower is better.

**Correctness.** Optional judge verdict against the reference answer: correct (1),
partial (0.5) or incorrect (0).

**Judge.** The LLM that extracts and verifies claims and grades correctness. Any
OpenAI-compatible endpoint, called at temperature 0 with JSON output. Ideally stronger
than the system under test.

**Judge cache.** sqlite file mapping sha256(model + prompt) to the judge's validated JSON
reply, so a repeated run makes no network calls.

**Prompt version.** SHA-256 of a judge prompt file, stored in `RunConfig.prompt_versions`.

**Run.** One evaluation of one system on one benchmark at one k, stored as a folder with
`config.json`, `results.jsonl` and `summary.json`. Live runs call a plugin; offline runs
score a predictions file.

**Plugin / factory.** A `module.path:factory` import path; the factory returns an object
with `name`, `retrieve(question, k)` and `answer(question, k)` (the `RAGSystem` protocol).

**Retrieval-only run.** A run that scores retrieval but not answers (`--retrieval-only`, or
a system whose `answer` raises `NotImplementedError`). Needs no LLM.

**Prediction.** One line of an offline predictions file: the retrieved chunks for a
question and, optionally, the system's output.

**Bootstrap confidence interval.** Resample the per-question values with replacement many
times (2,000 by default, fixed seed), take the mean each time, and report the middle 95%
of those means. Shown as `mean [low, high]`.

**Paired bootstrap.** For two runs on the same questions, bootstrap the per-question
differences B − A. Much tighter than comparing two separate CIs, because question
difficulty cancels out. Used by `compare`.

**"Not clear at this sample size".** What rag-eval says when a difference's 95% CI
includes zero. It never names a winner in that case.

**n.** The number of questions a metric was computed on. Shown next to every number.

**Calibration.** Checking the judge against human labels on the same claims: percentage
agreement and Cohen's kappa. Below about 80% agreement or 0.6 kappa, do not trust
groundedness numbers yet.

**Cohen's kappa.** Agreement corrected for chance: (p_o − p_e) / (1 − p_e), where p_o is
observed agreement and p_e the agreement expected from each rater's label frequencies.

**Freezing a benchmark.** Fixing the questions file before looking at any results, so the
benchmark is not tuned to make one system look good.
