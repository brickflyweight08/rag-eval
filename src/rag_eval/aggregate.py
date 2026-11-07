"""Turn per-question results into run-level numbers (mean [95% CI], n), overall and per qtype.

Each metric is defined once in ``METRICS`` as a function from a ``QuestionResult`` to a
number or ``None``. ``None`` means "not applicable or not measured" and is left out of
the mean and of n (see docs/adr/0004).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from rag_eval.metrics.correctness import CORRECTNESS_SCORES
from rag_eval.schema import QuestionResult
from rag_eval.stats import bootstrap_ci, percentile


def _bool(value: bool | None) -> float | None:
    return None if value is None else float(value)


def _hallucinated(r: QuestionResult) -> float | None:
    return None if r.answer is None else float(bool(r.hallucinated_citations))


def _refused_if(answerable: bool) -> Callable[[QuestionResult], float | None]:
    def extract(r: QuestionResult) -> float | None:
        return _bool(r.refused) if r.answerable == answerable else None

    return extract


def _correctness(r: QuestionResult) -> float | None:
    return None if r.correctness is None else CORRECTNESS_SCORES[r.correctness]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str  # "{k}" is replaced by the run's k
    extract: Callable[[QuestionResult], float | None]
    higher_is_better: bool = True

    def title(self, k: int | str) -> str:
        return self.label.format(k=k)


METRICS: list[MetricSpec] = [
    MetricSpec("hit", "Hit@{k}", lambda r: _bool(r.hit)),
    MetricSpec("recall", "Recall@{k}", lambda r: r.recall),
    MetricSpec("precision", "Precision@{k}", lambda r: r.precision),
    MetricSpec("mrr", "MRR", lambda r: r.reciprocal_rank),
    MetricSpec("groundedness", "Groundedness", lambda r: r.groundedness),
    MetricSpec("citation_precision", "Citation precision", lambda r: r.citation_precision),
    MetricSpec("hallucinated_citation_rate", "Hallucinated-citation rate", _hallucinated, False),
    MetricSpec("correctness", "Correctness", _correctness),
    MetricSpec("refusal_accuracy", "Refusal accuracy", _refused_if(answerable=False)),
    MetricSpec("false_refusal_rate", "False-refusal rate", _refused_if(answerable=True), False),
]
METRICS_BY_KEY = {m.key: m for m in METRICS}


def metric_values(results: Sequence[QuestionResult], spec: MetricSpec) -> list[float]:
    """The metric's values for every question where it applies (``None`` dropped)."""
    return [v for v in (spec.extract(r) for r in results) if v is not None]


def describe(values: Sequence[float], n_resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """``{"mean", "low", "high", "n"}``; all ``None`` (and n = 0) for no values."""
    if not values:
        return {"mean": None, "low": None, "high": None, "n": 0}
    mean, low, high = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
    return {"mean": mean, "low": low, "high": high, "n": len(values)}


def latency_summary(results: Sequence[QuestionResult]) -> dict[str, Any]:
    values = [r.latency_ms for r in results if r.latency_ms is not None]
    if not values:
        return {"p50": None, "p95": None, "n": 0}
    return {"p50": percentile(values, 50), "p95": percentile(values, 95), "n": len(values)}


def _metrics_block(
    results: Sequence[QuestionResult], n_resamples: int, seed: int
) -> dict[str, dict[str, Any]]:
    return {spec.key: describe(metric_values(results, spec), n_resamples, seed) for spec in METRICS}


def summarize(
    results: Sequence[QuestionResult], k: int, n_resamples: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """The content of ``summary.json``."""
    qtypes = sorted({r.qtype for r in results})
    per_qtype = {}
    for qtype in qtypes:
        subset = [r for r in results if r.qtype == qtype]
        per_qtype[qtype] = {
            "n_questions": len(subset),
            "metrics": _metrics_block(subset, n_resamples, seed),
        }
    n_answerable = sum(r.answerable for r in results)
    return {
        "k": k,
        "n_questions": len(results),
        "n_answerable": n_answerable,
        "n_unanswerable": len(results) - n_answerable,
        "n_errors": sum(r.error is not None for r in results),
        "n_answered": sum(r.answer is not None for r in results),
        "bootstrap": {"n_resamples": n_resamples, "alpha": 0.05, "seed": seed},
        "overall": _metrics_block(results, n_resamples, seed),
        "latency_ms": latency_summary(results),
        "per_qtype": per_qtype,
    }
