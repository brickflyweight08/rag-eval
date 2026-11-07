"""Small, hand-written statistics: bootstrap CIs, paired bootstrap, Cohen's kappa, percentiles.

Everything is seeded so results are deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _resample_means(values: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    return values[idx].mean(axis=1)


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of ``values`` with a percentile-bootstrap (1 - alpha) confidence interval.

    Returns ``(mean, low, high)``. Raises ``ValueError`` on an empty input.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    means = _resample_means(arr, n_resamples, seed)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(low), float(high)


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean difference ``b - a`` over the SAME questions, with a bootstrap CI.

    ``a[i]`` and ``b[i]`` must be scores for the same question. Resampling questions
    (not runs) keeps the pairing, which makes the interval much tighter than comparing
    two independent CIs.
    """
    if len(a) != len(b):
        raise ValueError(f"paired_bootstrap needs equal lengths, got {len(a)} and {len(b)}")
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return bootstrap_ci(diffs, n_resamples=n_resamples, alpha=alpha, seed=seed)


def cohen_kappa(labels_a: Sequence[bool], labels_b: Sequence[bool]) -> float:
    """Cohen's kappa for two raters giving binary labels.

    kappa = (p_o - p_e) / (1 - p_e), where p_o is the observed agreement and p_e the
    agreement expected by chance from each rater's label frequencies. Returns ``nan``
    when p_e == 1 (both raters always gave the same single label): kappa is undefined.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("cohen_kappa needs two label lists of equal length")
    if not labels_a:
        raise ValueError("cohen_kappa needs at least one pair of labels")
    n = len(labels_a)
    a = [bool(x) for x in labels_a]
    b = [bool(x) for x in labels_b]
    p_o = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa_true, pb_true = sum(a) / n, sum(b) / n
    p_e = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if math.isclose(p_e, 1.0):
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0-100) with linear interpolation. Raises on empty input."""
    if not values:
        raise ValueError("percentile needs at least one value")
    return float(np.percentile(np.asarray(values, dtype=float), q))
