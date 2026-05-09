from __future__ import annotations

import pytest

from proofrag.evaluation.statistics import (
    bootstrap_mean_ci,
    paired_binary_comparison,
)


def test_bootstrap_mean_ci_is_deterministic_and_bounded():
    ci = bootstrap_mean_ci([0.0, 1.0, 1.0, 1.0], samples=200, seed=7)

    assert ci.estimate == 0.75
    assert 0.0 <= ci.lower <= ci.estimate
    assert ci.estimate <= ci.upper <= 1.0
    assert ci.confidence == 0.95


def test_bootstrap_mean_ci_empty_values():
    ci = bootstrap_mean_ci([])

    assert ci.estimate == 0.0
    assert ci.lower == 0.0
    assert ci.upper == 0.0


def test_bootstrap_mean_ci_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0], samples=0)
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0], confidence=1.5)


def test_paired_binary_comparison_counts_discordant_pairs():
    comparison = paired_binary_comparison(
        baseline_correct=[True, False, False, True],
        treatment_correct=[True, True, False, False],
    )

    assert comparison.baseline_wins == 1
    assert comparison.treatment_wins == 1
    assert comparison.ties == 2
    assert comparison.total == 4
    assert comparison.treatment_win_rate_delta == 0.0
    assert comparison.exact_p_value == 1.0


def test_paired_binary_comparison_requires_equal_lengths():
    with pytest.raises(ValueError):
        paired_binary_comparison(
            baseline_correct=[True],
            treatment_correct=[True, False],
        )

