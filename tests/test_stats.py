import math

import pytest

from rag_eval.stats import bootstrap_ci, cohen_kappa, paired_bootstrap, percentile


def test_bootstrap_is_deterministic_for_fixed_seed():
    values = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)
    continuous = [0.13, 0.52, 0.77, 0.91, 0.05, 0.64, 0.38, 0.29]
    assert bootstrap_ci(continuous, seed=3) == bootstrap_ci(continuous, seed=3)
    assert bootstrap_ci(continuous, seed=3) != bootstrap_ci(continuous, seed=4)


def test_bootstrap_mean_and_bounds():
    values = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    mean, low, high = bootstrap_ci(values)
    assert mean == pytest.approx(0.7)
    assert 0.0 <= low < mean < high <= 1.0


def test_bootstrap_constant_values_have_zero_width():
    assert bootstrap_ci([0.5, 0.5, 0.5]) == (0.5, 0.5, 0.5)


def test_bootstrap_single_value():
    assert bootstrap_ci([1.0]) == (1.0, 1.0, 1.0)


def test_bootstrap_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_paired_bootstrap_constant_shift():
    a = [0.1, 0.5, 0.9, 0.3]
    b = [x + 0.25 for x in a]
    diff, low, high = paired_bootstrap(a, b)
    assert diff == pytest.approx(0.25)
    assert low == pytest.approx(0.25) and high == pytest.approx(0.25)


def test_paired_bootstrap_direction_is_b_minus_a():
    diff, _, _ = paired_bootstrap([1, 1, 1, 1], [0, 0, 1, 1])
    assert diff == pytest.approx(-0.5)


def test_paired_bootstrap_needs_same_length():
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0], [1])


def test_cohen_kappa_hand_computed():
    # p_o = 8/10, p_e = 0.6*0.6 + 0.4*0.4 = 0.52, kappa = 0.28 / 0.48
    a = [1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
    b = [1, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    assert cohen_kappa(a, b) == pytest.approx(0.28 / 0.48)


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa([True, False, True], [True, False, True]) == pytest.approx(1.0)
    # Opposite labels: p_o = 0, p_e = 0.5 -> kappa = -1
    assert cohen_kappa([True, False], [False, True]) == pytest.approx(-1.0)


def test_cohen_kappa_undefined_when_all_labels_identical():
    assert math.isnan(cohen_kappa([True, True], [True, True]))


def test_percentile():
    assert percentile([10, 20, 30, 40], 50) == 25.0
    assert percentile([5.0], 95) == 5.0
