"""Public data types. Other projects depend on these: do not change them without asking."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class EvalQuestion(BaseModel):
    id: str
    question: str
    qtype: str = "default"  # free-form category used for per-type breakdowns
    answerable: bool = True
    gold_ids: list[str] = []  # chunk IDs needed to answer; empty if not answerable
    reference_answer: str | None = None
    metadata: dict[str, Any] = {}  # anything system-specific (rag-eval never reads it)


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] = {}


class SystemOutput(BaseModel):
    answer: str
    citations: list[str] = []  # chunk IDs the system cited
    context: list[RetrievedChunk]  # what the generator actually saw
    latency_ms: float | None = None


class Prediction(BaseModel):  # one line of an offline predictions file
    question_id: str
    retrieved: list[RetrievedChunk]  # in rank order
    output: SystemOutput | None = None


class RunConfig(BaseModel):
    run_id: str
    system: str  # plugin path or "offline:<file>"
    system_kwargs: dict[str, Any] = {}
    k: int
    judge_model: str | None
    questions_path: str
    questions_sha256: str
    prompt_versions: dict[str, str]  # judge prompt file hashes
    rag_eval_version: str
    created_at: str


class QuestionResult(BaseModel):
    question_id: str
    qtype: str
    answerable: bool
    retrieved_ids: list[str]
    hit: bool | None  # None for unanswerable questions
    recall: float | None
    precision: float | None
    reciprocal_rank: float | None
    answer: str | None
    citations: list[str] = []
    hallucinated_citations: list[str] = []
    claims: list[dict] = []  # {"claim", "supported", "evidence_id"}
    groundedness: float | None = None  # None if no factual claims or no answer
    citation_precision: float | None = None
    refused: bool | None = None
    correctness: Literal["correct", "partial", "incorrect"] | None = None
    latency_ms: float | None = None
    error: str | None = None  # system exception, recorded instead of crashing the run
