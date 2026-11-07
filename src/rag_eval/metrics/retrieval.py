"""Retrieval metrics for one question: hit@k, recall@k, precision@k, reciprocal rank.

For gold set G and ranked retrieved IDs R (de-duplicated, keeping first occurrences):

- Hit@k        = 1 if G and R[:k] share an ID, else 0
- Recall@k     = |G ∩ R[:k]| / |G|
- Precision@k  = |G ∩ R[:k]| / k   (k, not len(R[:k]): returning fewer chunks is not rewarded)
- Reciprocal rank = 1 / rank of the first gold ID in R[:k], or 0 if none

An empty gold set means the question is unanswerable: every metric is ``None``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalScores:
    hit: bool | None
    recall: float | None
    precision: float | None
    reciprocal_rank: float | None


def dedupe(ids: Iterable[str]) -> tuple[list[str], list[str]]:
    """Remove repeated IDs, keeping the first occurrence. Returns (unique, duplicates)."""
    seen: set[str] = set()
    unique: list[str] = []
    duplicates: list[str] = []
    for chunk_id in ids:
        if chunk_id in seen:
            duplicates.append(chunk_id)
        else:
            seen.add(chunk_id)
            unique.append(chunk_id)
    return unique, duplicates


def top_k(retrieved: Sequence[str], k: int) -> list[str]:
    """The first k distinct retrieved IDs."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return dedupe(retrieved)[0][:k]


def _n_found(gold: Iterable[str], retrieved: Sequence[str], k: int) -> int:
    return len(set(gold) & set(top_k(retrieved, k)))


def hit_at_k(gold: Iterable[str], retrieved: Sequence[str], k: int) -> bool | None:
    gold = set(gold)
    return None if not gold else _n_found(gold, retrieved, k) > 0


def recall_at_k(gold: Iterable[str], retrieved: Sequence[str], k: int) -> float | None:
    gold = set(gold)
    return None if not gold else _n_found(gold, retrieved, k) / len(gold)


def precision_at_k(gold: Iterable[str], retrieved: Sequence[str], k: int) -> float | None:
    gold = set(gold)
    return None if not gold else _n_found(gold, retrieved, k) / k


def reciprocal_rank(gold: Iterable[str], retrieved: Sequence[str], k: int) -> float | None:
    gold = set(gold)
    if not gold:
        return None
    for rank, chunk_id in enumerate(top_k(retrieved, k), start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def score_retrieval(
    gold: Iterable[str], retrieved: Sequence[str], k: int, question_id: str = "?"
) -> RetrievalScores:
    """All retrieval metrics for one question (logs a warning on duplicate IDs)."""
    _, duplicates = dedupe(retrieved)
    if duplicates:
        logger.warning(
            "question %s: duplicate retrieved IDs %s were removed before scoring",
            question_id,
            sorted(set(duplicates)),
        )
    gold = list(gold)
    return RetrievalScores(
        hit=hit_at_k(gold, retrieved, k),
        recall=recall_at_k(gold, retrieved, k),
        precision=precision_at_k(gold, retrieved, k),
        reciprocal_rank=reciprocal_rank(gold, retrieved, k),
    )
