import logging

import pytest

from rag_eval.metrics.retrieval import (
    dedupe,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
    top_k,
)


def test_empty_gold_gives_none():
    scores = score_retrieval([], ["a", "b"], k=2)
    assert scores.hit is None
    assert scores.recall is None
    assert scores.precision is None
    assert scores.reciprocal_rank is None


def test_gold_at_rank_1():
    s = score_retrieval(["a"], ["a", "b", "c"], k=3)
    assert s.hit is True
    assert s.recall == 1.0
    assert s.precision == pytest.approx(1 / 3)
    assert s.reciprocal_rank == 1.0


def test_gold_at_rank_k():
    s = score_retrieval(["c", "z"], ["a", "b", "c"], k=3)
    assert s.hit is True
    assert s.recall == 0.5
    assert s.precision == pytest.approx(1 / 3)
    assert s.reciprocal_rank == pytest.approx(1 / 3)


def test_gold_just_beyond_k_is_missed():
    s = score_retrieval(["d"], ["a", "b", "c", "d"], k=3)
    assert s.hit is False
    assert s.recall == 0.0
    assert s.precision == 0.0
    assert s.reciprocal_rank == 0.0


def test_gold_missing():
    s = score_retrieval(["x", "y"], ["a", "b"], k=5)
    assert (s.hit, s.recall, s.precision, s.reciprocal_rank) == (False, 0.0, 0.0, 0.0)


def test_fewer_results_than_k_precision_uses_k():
    assert precision_at_k(["a"], ["a"], k=4) == 0.25


def test_duplicates_are_removed_keeping_first(caplog):
    # Without de-duplication "a" would fill two of the three slots and push "c" out.
    retrieved = ["a", "a", "b", "c"]
    assert dedupe(retrieved) == (["a", "b", "c"], ["a"])
    assert top_k(retrieved, 3) == ["a", "b", "c"]
    with caplog.at_level(logging.WARNING):
        s = score_retrieval(["a", "c"], retrieved, k=3, question_id="q7")
    assert s.recall == 1.0
    assert s.precision == pytest.approx(2 / 3)
    assert s.reciprocal_rank == 1.0
    assert "q7" in caplog.text and "duplicate" in caplog.text


def test_individual_functions_agree():
    gold, retrieved = ["b", "d"], ["a", "b", "c", "d"]
    assert hit_at_k(gold, retrieved, 2) is True
    assert recall_at_k(gold, retrieved, 2) == 0.5
    assert recall_at_k(gold, retrieved, 4) == 1.0
    assert reciprocal_rank(gold, retrieved, 4) == 0.5


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        top_k(["a"], 0)
