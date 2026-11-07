"""Citation metrics.

- Hallucinated citations: cited IDs that are not in the context the generator saw.
- Citation precision: share of VALID citations (cited IDs that are in the context) whose
  chunk the verifier used as evidence for at least one supported claim.
  ``None`` when there are no valid citations or no claims (see docs/adr/0003).
"""

from __future__ import annotations

from rag_eval.metrics.retrieval import dedupe
from rag_eval.schema import RetrievedChunk


def _context_ids(context: list[RetrievedChunk]) -> set[str]:
    return {chunk.chunk_id for chunk in context}


def hallucinated_citations(citations: list[str], context: list[RetrievedChunk]) -> list[str]:
    """Distinct cited IDs not present in ``context``, in citation order."""
    ids = _context_ids(context)
    return [c for c in dedupe(citations)[0] if c not in ids]


def valid_citations(citations: list[str], context: list[RetrievedChunk]) -> list[str]:
    """Distinct cited IDs present in ``context``, in citation order."""
    ids = _context_ids(context)
    return [c for c in dedupe(citations)[0] if c in ids]


def citation_precision(
    citations: list[str], context: list[RetrievedChunk], claims: list[dict]
) -> float | None:
    valid = valid_citations(citations, context)
    if not valid or not claims:
        return None
    evidence = {c["evidence_id"] for c in claims if c["supported"] and c["evidence_id"]}
    return sum(1 for chunk_id in valid if chunk_id in evidence) / len(valid)
