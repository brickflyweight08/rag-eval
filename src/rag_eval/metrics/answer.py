"""All answer metrics for one question, computed on ``output.context`` (what the generator saw)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_eval.llm import Judge, JudgeConfigError, JudgeError
from rag_eval.metrics.citations import citation_precision, hallucinated_citations
from rag_eval.metrics.correctness import Verdict, judge_correctness
from rag_eval.metrics.groundedness import score_groundedness
from rag_eval.metrics.refusal import DEFAULT_REFUSAL_PHRASES, is_refusal, judge_refusal
from rag_eval.schema import EvalQuestion, SystemOutput

logger = logging.getLogger(__name__)


@dataclass
class AnswerOptions:
    refusal_phrases: Sequence[str] = DEFAULT_REFUSAL_PHRASES
    refusal_judge: bool = False  # ask the judge instead of matching phrases
    correctness: bool = False  # judge the answer against reference_answer


@dataclass
class AnswerScores:
    answer: str
    citations: list[str] = field(default_factory=list)
    hallucinated_citations: list[str] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    groundedness: float | None = None
    citation_precision: float | None = None
    refused: bool | None = None
    correctness: Verdict | None = None
    error: str | None = None  # judge failure for this question (other metrics kept)


def _judge_metrics(
    scores: AnswerScores,
    judge: Judge,
    question: EvalQuestion,
    output: SystemOutput,
    options: AnswerOptions,
) -> None:
    if options.refusal_judge:
        scores.refused = judge_refusal(judge, question.question, output.answer)
    if not scores.refused:
        # A refusal has no factual claims: groundedness stays None (not "ungrounded").
        grounded = score_groundedness(judge, question.question, output.answer, output.context)
        scores.claims = grounded.claims
        scores.groundedness = grounded.groundedness
        scores.citation_precision = citation_precision(
            output.citations, output.context, grounded.claims
        )
    if options.correctness and question.reference_answer:
        verdict, reason = judge_correctness(
            judge, question.question, output.answer, question.reference_answer
        )
        scores.correctness = verdict
        logger.info("question %s: correctness %s (%s)", question.id, verdict, reason)


def score_answer(
    judge: Judge, question: EvalQuestion, output: SystemOutput, options: AnswerOptions
) -> AnswerScores:
    """Score one answer. A ``JudgeError`` is recorded in ``error``; a ``JudgeConfigError``
    (no key, unreachable endpoint) is raised because every other question would fail too.
    """
    scores = AnswerScores(
        answer=output.answer,
        citations=list(output.citations),
        hallucinated_citations=hallucinated_citations(output.citations, output.context),
    )
    if not options.refusal_judge:
        scores.refused = is_refusal(output.answer, options.refusal_phrases)
    try:
        _judge_metrics(scores, judge, question, output, options)
    except JudgeConfigError:
        raise
    except JudgeError as exc:
        scores.error = f"judge: {exc}"
    return scores
