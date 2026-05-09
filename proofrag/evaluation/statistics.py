"""
statistics.py — Dependency-light statistical helpers for evaluation.

These helpers are deliberately small and deterministic. They provide the basic
confidence intervals and paired tests needed for ProofRAG-vs-baseline reports
without requiring scipy or numpy in the default install.
"""

from __future__ import annotations

import math
import random

from pydantic import BaseModel


class ConfidenceInterval(BaseModel):
    """A point estimate with lower/upper confidence bounds."""

    estimate: float
    lower: float
    upper: float
    confidence: float


class PairedComparison(BaseModel):
    """Paired binary comparison summary for two methods."""

    baseline_wins: int
    treatment_wins: int
    ties: int
    total: int
    treatment_win_rate_delta: float
    exact_p_value: float


def bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    samples: int = 1000,
    seed: int = 13,
) -> ConfidenceInterval:
    """Return a percentile bootstrap confidence interval for the mean."""

    if not values:
        return ConfidenceInterval(
            estimate=0.0,
            lower=0.0,
            upper=0.0,
            confidence=confidence,
        )
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()

    alpha = 1.0 - confidence
    lower_index = max(0, math.floor((alpha / 2) * (samples - 1)))
    upper_index = min(samples - 1, math.ceil((1 - alpha / 2) * (samples - 1)))
    return ConfidenceInterval(
        estimate=sum(values) / n,
        lower=means[lower_index],
        upper=means[upper_index],
        confidence=confidence,
    )


def paired_binary_comparison(
    *,
    baseline_correct: list[bool],
    treatment_correct: list[bool],
) -> PairedComparison:
    """Compare two methods on paired binary outcomes using McNemar's exact test."""

    if len(baseline_correct) != len(treatment_correct):
        raise ValueError("paired comparisons require equal-length outcome lists")

    baseline_wins = 0
    treatment_wins = 0
    ties = 0
    for baseline, treatment in zip(baseline_correct, treatment_correct):
        if baseline == treatment:
            ties += 1
        elif baseline and not treatment:
            baseline_wins += 1
        else:
            treatment_wins += 1

    total = len(baseline_correct)
    baseline_rate = sum(1 for value in baseline_correct if value) / total if total else 0.0
    treatment_rate = sum(1 for value in treatment_correct if value) / total if total else 0.0

    return PairedComparison(
        baseline_wins=baseline_wins,
        treatment_wins=treatment_wins,
        ties=ties,
        total=total,
        treatment_win_rate_delta=treatment_rate - baseline_rate,
        exact_p_value=_mcnemar_exact_p_value(baseline_wins, treatment_wins),
    )


def _mcnemar_exact_p_value(baseline_wins: int, treatment_wins: int) -> float:
    discordant = baseline_wins + treatment_wins
    if discordant == 0:
        return 1.0

    smaller = min(baseline_wins, treatment_wins)
    cumulative = sum(_binomial_pmf(discordant, k, 0.5) for k in range(smaller + 1))
    return min(1.0, 2.0 * cumulative)


def _binomial_pmf(n: int, k: int, p: float) -> float:
    return math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))

