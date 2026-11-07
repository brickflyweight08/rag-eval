"""Run an evaluation: live (a plugin system) or offline (a predictions file).

Both modes write the same three files to ``out_dir``:
``config.json`` (RunConfig), ``results.jsonl`` (one QuestionResult per line) and
``summary.json`` (aggregates with 95% CIs, overall and per qtype).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_eval.aggregate import summarize
from rag_eval.config import load_settings
from rag_eval.llm import Judge, JudgeClient, prompt_versions
from rag_eval.loading import file_sha256, load_predictions, load_questions, load_system
from rag_eval.metrics.answer import AnswerOptions, AnswerScores, score_answer
from rag_eval.metrics.correctness import CORRECTNESS_PROMPT
from rag_eval.metrics.groundedness import EXTRACT_PROMPT, VERIFY_PROMPT
from rag_eval.metrics.refusal import DEFAULT_REFUSAL_PHRASES, REFUSAL_PROMPT
from rag_eval.metrics.retrieval import RetrievalScores, score_retrieval, top_k
from rag_eval.protocols import RAGSystem
from rag_eval.schema import EvalQuestion, Prediction, QuestionResult, RunConfig, SystemOutput
from rag_eval.version import __version__

logger = logging.getLogger(__name__)

NO_RETRIEVAL = RetrievalScores(hit=None, recall=None, precision=None, reciprocal_rank=None)
QuestionsInput = str | Path | Sequence[EvalQuestion]


@dataclass
class RunArtifacts:
    """What a run produced (also written to ``out_dir`` when one is given)."""

    config: RunConfig
    results: list[QuestionResult]
    summary: dict[str, Any]


class _LazyJudge:
    """Builds the default judge (from env settings) only if a judge call is needed."""

    def __init__(self, judge: Judge | None) -> None:
        self._judge = judge
        self.used = False

    def get(self) -> Judge:
        if self._judge is None:
            self._judge = JudgeClient.from_settings(load_settings())
        self.used = True
        return self._judge

    @property
    def model(self) -> str | None:
        return self._judge.model if (self.used and self._judge is not None) else None


# --- helpers -----------------------------------------------------------------------------


def _resolve_questions(questions: QuestionsInput) -> tuple[list[EvalQuestion], str, str]:
    """Returns (questions, path label, sha256)."""
    if isinstance(questions, str | Path):
        return load_questions(questions), str(questions), file_sha256(questions)
    items = list(questions)
    canonical = "".join(q.model_dump_json() + "\n" for q in items)
    return items, "<in-memory>", hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _used_prompts(options: AnswerOptions) -> list[str]:
    names = [EXTRACT_PROMPT, VERIFY_PROMPT]
    if options.refusal_judge:
        names.append(REFUSAL_PROMPT)
    if options.correctness:
        names.append(CORRECTNESS_PROMPT)
    return names


def _error_text(stage: str, exc: BaseException) -> str:
    return f"{stage}: {type(exc).__name__}: {exc}"


def _build_result(
    q: EvalQuestion,
    retrieved_ids: list[str],
    retrieval: RetrievalScores,
    answer: AnswerScores | None,
    latency_ms: float | None,
    errors: list[str],
) -> QuestionResult:
    if answer is not None and answer.error:
        errors = [*errors, answer.error]
    fields: dict[str, Any] = {}
    if answer is not None:
        fields = {
            "citations": answer.citations,
            "hallucinated_citations": answer.hallucinated_citations,
            "claims": answer.claims,
            "groundedness": answer.groundedness,
            "citation_precision": answer.citation_precision,
            "refused": answer.refused,
            "correctness": answer.correctness,
        }
    return QuestionResult(
        question_id=q.id,
        qtype=q.qtype,
        answerable=q.answerable,
        retrieved_ids=retrieved_ids,
        hit=retrieval.hit,
        recall=retrieval.recall,
        precision=retrieval.precision,
        reciprocal_rank=retrieval.reciprocal_rank,
        answer=None if answer is None else answer.answer,
        latency_ms=latency_ms,
        error="; ".join(errors) if errors else None,
        **fields,
    )


def _log_progress(i: int, n: int, r: QuestionResult) -> None:
    parts = [f"[{i}/{n}] {r.question_id}"]
    if r.hit is not None:
        parts.append(f"hit={int(r.hit)} recall={r.recall:.2f}")
    if r.groundedness is not None:
        parts.append(f"groundedness={r.groundedness:.2f}")
    if r.refused:
        parts.append("refused")
    if r.latency_ms is not None:
        parts.append(f"({r.latency_ms:.0f} ms)")
    if r.error:
        logger.warning("%s ERROR %s", " ".join(parts), r.error)
    else:
        logger.info("%s", " ".join(parts))


def write_run(artifacts: RunArtifacts, out_dir: str | Path) -> Path:
    """Write config.json, results.jsonl and summary.json (deterministic, LF line endings)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        artifacts.config.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    lines = "".join(r.model_dump_json() + "\n" for r in artifacts.results)
    (out / "results.jsonl").write_text(lines, encoding="utf-8", newline="\n")
    (out / "summary.json").write_text(
        json.dumps(artifacts.summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return out


def _finish(
    results: list[QuestionResult],
    *,
    run_id: str,
    system: str,
    system_kwargs: dict[str, Any],
    k: int,
    judge: _LazyJudge,
    options: AnswerOptions,
    questions_path: str,
    questions_sha: str,
    out_dir: str | Path | None,
    n_resamples: int,
    seed: int,
) -> RunArtifacts:
    config = RunConfig(
        run_id=run_id,
        system=system,
        # Non-JSON values (possible from Python, e.g. a function) are stored as text.
        system_kwargs=json.loads(json.dumps(system_kwargs, default=repr)),
        k=k,
        judge_model=judge.model,
        questions_path=questions_path,
        questions_sha256=questions_sha,
        prompt_versions=prompt_versions(_used_prompts(options)) if judge.used else {},
        rag_eval_version=__version__,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    artifacts = RunArtifacts(config, results, summarize(results, k, n_resamples, seed))
    if out_dir is not None:
        write_run(artifacts, out_dir)
        logger.info("wrote %s", out_dir)
    return artifacts


def _options(
    correctness: bool, refusal_judge: bool, refusal_phrases: Sequence[str] | None
) -> AnswerOptions:
    return AnswerOptions(
        refusal_phrases=tuple(refusal_phrases or DEFAULT_REFUSAL_PHRASES),
        refusal_judge=refusal_judge,
        correctness=correctness,
    )


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000


# --- live mode ---------------------------------------------------------------------------


def run_live(
    system: RAGSystem | str,
    questions: QuestionsInput,
    *,
    k: int = 5,
    out_dir: str | Path | None = None,
    system_kwargs: dict[str, Any] | None = None,
    judge: Judge | None = None,
    retrieval_only: bool = False,
    correctness: bool = False,
    refusal_judge: bool = False,
    refusal_phrases: Sequence[str] | None = None,
    run_id: str | None = None,
    n_resamples: int = 2000,
    seed: int = 0,
) -> RunArtifacts:
    """Evaluate a live system.

    ``system`` is an object satisfying ``RAGSystem`` or a ``"module.path:factory"`` string
    (called with ``system_kwargs``). For each question the runner calls
    ``retrieve(question, k)`` for retrieval metrics, then (unless ``retrieval_only``)
    ``answer(question, k)`` and scores the answer against ``output.context``. System
    exceptions are recorded per question; the run continues. If ``answer`` raises
    ``NotImplementedError``, answer metrics are skipped for the rest of the run.

    ``judge`` defaults to a ``JudgeClient`` built from the environment, created only
    when an answer actually needs judging.
    """
    system_kwargs = dict(system_kwargs or {})
    if isinstance(system, str):
        system_label, system = system, load_system(system, system_kwargs)
    else:
        system_label = f"object:{type(system).__module__}.{type(system).__qualname__}"
    items, questions_path, questions_sha = _resolve_questions(questions)
    options = _options(correctness, refusal_judge, refusal_phrases)
    lazy_judge = _LazyJudge(judge)
    answering = not retrieval_only
    results: list[QuestionResult] = []
    for i, q in enumerate(items, start=1):
        result, answering = _live_question(system, q, k, answering, lazy_judge, options)
        results.append(result)
        _log_progress(i, len(items), result)
    return _finish(
        results,
        run_id=run_id or (Path(out_dir).name if out_dir else f"{system.name}-k{k}"),
        system=system_label,
        system_kwargs=system_kwargs,
        k=k,
        judge=lazy_judge,
        options=options,
        questions_path=questions_path,
        questions_sha=questions_sha,
        out_dir=out_dir,
        n_resamples=n_resamples,
        seed=seed,
    )


def _live_question(
    system: RAGSystem,
    q: EvalQuestion,
    k: int,
    answering: bool,
    judge: _LazyJudge,
    options: AnswerOptions,
) -> tuple[QuestionResult, bool]:
    errors: list[str] = []
    retrieved_ids: list[str] = []
    retrieval = NO_RETRIEVAL
    latency: float | None = None
    try:
        chunks, latency = _timed(lambda: system.retrieve(q.question, k))
        all_ids = [c.chunk_id for c in chunks]
        retrieved_ids = top_k(all_ids, k)
        retrieval = score_retrieval(q.gold_ids, all_ids, k, q.id)
    except Exception as exc:
        errors.append(_error_text("retrieve", exc))
    answer: AnswerScores | None = None
    if answering:
        output: SystemOutput | None = None
        try:
            output, latency = _timed(lambda: system.answer(q.question, k))
        except NotImplementedError:
            logger.warning("%s.answer is not implemented: retrieval-only run", system.name)
            answering, latency = False, latency
        except Exception as exc:
            errors.append(_error_text("answer", exc))
            latency = None
        if output is not None:
            if output.latency_ms is not None:
                latency = output.latency_ms
            answer = score_answer(judge.get(), q, output, options)
    return _build_result(q, retrieved_ids, retrieval, answer, latency, errors), answering


# --- offline mode ------------------------------------------------------------------------


def score_offline(
    predictions: str | Path | Sequence[Prediction],
    questions: QuestionsInput,
    *,
    k: int = 5,
    out_dir: str | Path | None = None,
    judge: Judge | None = None,
    retrieval_only: bool = False,
    correctness: bool = False,
    refusal_judge: bool = False,
    refusal_phrases: Sequence[str] | None = None,
    run_id: str | None = None,
    n_resamples: int = 2000,
    seed: int = 0,
) -> RunArtifacts:
    """Score predictions produced by any external system (no plugin needed).

    Questions without a prediction get an error and no metrics; predictions for unknown
    question IDs are ignored with a warning. A prediction without ``output`` is scored
    for retrieval only.
    """
    if isinstance(predictions, str | Path):
        system_label, preds = f"offline:{predictions}", load_predictions(predictions)
    else:
        system_label, preds = "offline:<in-memory>", list(predictions)
    items, questions_path, questions_sha = _resolve_questions(questions)
    by_id = {p.question_id: p for p in preds}
    unknown = sorted(set(by_id) - {q.id for q in items})
    if unknown:
        logger.warning("ignoring predictions for unknown question IDs: %s", unknown)
    options = _options(correctness, refusal_judge, refusal_phrases)
    lazy_judge = _LazyJudge(judge)
    results = []
    for i, q in enumerate(items, start=1):
        result = _offline_question(q, by_id.get(q.id), k, retrieval_only, lazy_judge, options)
        results.append(result)
        _log_progress(i, len(items), result)
    return _finish(
        results,
        run_id=run_id or (Path(out_dir).name if out_dir else "offline"),
        system=system_label,
        system_kwargs={},
        k=k,
        judge=lazy_judge,
        options=options,
        questions_path=questions_path,
        questions_sha=questions_sha,
        out_dir=out_dir,
        n_resamples=n_resamples,
        seed=seed,
    )


def _offline_question(
    q: EvalQuestion,
    pred: Prediction | None,
    k: int,
    retrieval_only: bool,
    judge: _LazyJudge,
    options: AnswerOptions,
) -> QuestionResult:
    if pred is None:
        return _build_result(q, [], NO_RETRIEVAL, None, None, ["no prediction for this question"])
    all_ids = [c.chunk_id for c in pred.retrieved]
    retrieval = score_retrieval(q.gold_ids, all_ids, k, q.id)
    answer = None
    latency = None
    if pred.output is not None:
        latency = pred.output.latency_ms
        if not retrieval_only:
            answer = score_answer(judge.get(), q, pred.output, options)
    return _build_result(q, top_k(all_ids, k), retrieval, answer, latency, [])
