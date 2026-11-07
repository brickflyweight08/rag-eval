"""Groundedness: the share of an answer's factual claims supported by the retrieved context.

Two judge steps:
1. ``extract_claims``: split the answer into atomic factual claims.
2. ``verify_claim``: check each claim against ONLY the context chunks, each prefixed
   with ``[chunk_id]``; the judge names the supporting chunk as ``evidence_id``.

Groundedness = supported claims / total claims, or ``None`` when there are no claims.
It is not correctness: an answer can be fully grounded in a wrong or irrelevant chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from rag_eval.llm import Judge
from rag_eval.schema import RetrievedChunk

EXTRACT_PROMPT = "extract_claims"
VERIFY_PROMPT = "verify_claim"


class ClaimsReply(BaseModel):
    claims: list[str]


class VerifyReply(BaseModel):
    supported: bool
    evidence_id: str | None = None


@dataclass
class GroundednessResult:
    claims: list[dict] = field(default_factory=list)  # {"claim", "supported", "evidence_id"}
    groundedness: float | None = None


def format_context(context: list[RetrievedChunk]) -> str:
    """Context as the judge sees it: one ``[chunk_id] text`` block per chunk."""
    return "\n\n".join(f"[{chunk.chunk_id}] {chunk.text}" for chunk in context)


def extract_claims(judge: Judge, question: str, answer: str) -> list[str]:
    reply = judge.complete_json(
        EXTRACT_PROMPT, {"question": question, "answer": answer}, ClaimsReply
    )
    return [c.strip() for c in reply["claims"] if c and c.strip()]


def _clean_evidence_id(raw: str | None, valid_ids: set[str]) -> str | None:
    if raw is None:
        return None
    chunk_id = raw.strip().strip("[]").strip()
    return chunk_id if chunk_id in valid_ids else None


def verify_claim(judge: Judge, claim: str, context: list[RetrievedChunk]) -> dict:
    """Verify one claim. Returns ``{"claim", "supported", "evidence_id"}``.

    ``evidence_id`` is kept only if the claim is supported AND the ID is really in the
    context; anything else becomes ``None`` (see docs/adr/0003).
    """
    if not context:
        return {"claim": claim, "supported": False, "evidence_id": None}
    reply = judge.complete_json(
        VERIFY_PROMPT, {"claim": claim, "context": format_context(context)}, VerifyReply
    )
    supported = bool(reply["supported"])
    valid_ids = {chunk.chunk_id for chunk in context}
    evidence = _clean_evidence_id(reply.get("evidence_id"), valid_ids) if supported else None
    return {"claim": claim, "supported": supported, "evidence_id": evidence}


def groundedness_score(claims: list[dict]) -> float | None:
    """Supported claims / total claims; ``None`` when there are no claims."""
    if not claims:
        return None
    return sum(1 for c in claims if c["supported"]) / len(claims)


def score_groundedness(
    judge: Judge, question: str, answer: str, context: list[RetrievedChunk]
) -> GroundednessResult:
    claims = [verify_claim(judge, c, context) for c in extract_claims(judge, question, answer)]
    return GroundednessResult(claims=claims, groundedness=groundedness_score(claims))
