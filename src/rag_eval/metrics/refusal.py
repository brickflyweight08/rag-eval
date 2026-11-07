"""Refusal detection and refusal metrics.

- Refusal accuracy: share of UNANSWERABLE questions the system refused (higher is better).
- False-refusal rate: share of ANSWERABLE questions the system refused (lower is better).

Questions with no answer (errors, retrieval-only runs) have ``refused=None`` and are left
out of both rates (see docs/adr/0004).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from rag_eval.llm import Judge

REFUSAL_PROMPT = "refusal"

DEFAULT_REFUSAL_PHRASES: tuple[str, ...] = (
    "I don't have enough information",
    "I don't know",
    "not in the context",
)


class RefusalReply(BaseModel):
    refused: bool


def _normalize(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip().lower()


def is_refusal(answer: str, phrases: Sequence[str] = DEFAULT_REFUSAL_PHRASES) -> bool:
    """True if the answer contains any refusal phrase (case- and apostrophe-insensitive)."""
    normalized = _normalize(answer)
    return any(_normalize(p) in normalized for p in phrases if p.strip())


def judge_refusal(judge: Judge, question: str, answer: str) -> bool:
    """Ask the judge whether the answer is a refusal (``--refusal-judge``)."""
    reply = judge.complete_json(
        REFUSAL_PROMPT, {"question": question, "answer": answer}, RefusalReply
    )
    return bool(reply["refused"])


def _rate(refused: Sequence[bool | None], answerable: Sequence[bool], want: bool) -> float | None:
    picked = [r for r, a in zip(refused, answerable, strict=True) if a == want and r is not None]
    return sum(picked) / len(picked) if picked else None


def refusal_accuracy(refused: Sequence[bool | None], answerable: Sequence[bool]) -> float | None:
    """Share of unanswerable questions that were refused."""
    return _rate(refused, answerable, want=False)


def false_refusal_rate(refused: Sequence[bool | None], answerable: Sequence[bool]) -> float | None:
    """Share of answerable questions that were (wrongly) refused."""
    return _rate(refused, answerable, want=True)
