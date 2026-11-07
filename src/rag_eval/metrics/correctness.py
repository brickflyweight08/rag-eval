"""Optional answer correctness (``--correctness``): the judge compares the answer with the
human-written ``reference_answer`` and returns correct / partial / incorrect plus a reason.

For averaging, correct = 1, partial = 0.5, incorrect = 0 (see docs/adr/0005).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from rag_eval.llm import Judge

CORRECTNESS_PROMPT = "correctness"

Verdict = Literal["correct", "partial", "incorrect"]
CORRECTNESS_SCORES: dict[str, float] = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class CorrectnessReply(BaseModel):
    verdict: Verdict
    reason: str = ""


def judge_correctness(
    judge: Judge, question: str, answer: str, reference_answer: str
) -> tuple[Verdict, str]:
    """Returns ``(verdict, one-line reason)``."""
    reply = judge.complete_json(
        CORRECTNESS_PROMPT,
        {"question": question, "answer": answer, "reference_answer": reference_answer},
        CorrectnessReply,
    )
    return reply["verdict"], reply.get("reason", "")
