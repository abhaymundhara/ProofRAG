"""
comparison.py — Automated baseline-vs-ProofRAG comparison helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from proofrag.evaluation.error_analysis import ErrorAnalysisReport, analyze_errors
from proofrag.evaluation.statistics import PairedComparison, paired_binary_comparison


class MethodSummary(BaseModel):
    """Aggregate summary for one evaluated method."""

    method: str
    total: int
    answered: int
    correct: int
    abstained: int
    accuracy: float
    precision_at_answered: float
    unsafe_allow_count: int


class ComparisonReport(BaseModel):
    """MiniRAG-vs-ProofRAG comparison report."""

    baseline: MethodSummary
    proofrag: MethodSummary
    paired_answer_accuracy: PairedComparison
    proofrag_error_analysis: ErrorAnalysisReport


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL result file."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compare_minirag_vs_proofrag(rows: list[dict[str, Any]]) -> ComparisonReport:
    """Compare baseline MiniRAG answers against ProofRAG-gated answers."""

    baseline_correct = [_baseline_correct(row) for row in rows]
    proofrag_correct = [_proofrag_answer_correct(row) for row in rows]

    return ComparisonReport(
        baseline=_summarize_baseline(rows, baseline_correct),
        proofrag=_summarize_proofrag(rows, proofrag_correct),
        paired_answer_accuracy=paired_binary_comparison(
            baseline_correct=baseline_correct,
            treatment_correct=proofrag_correct,
        ),
        proofrag_error_analysis=analyze_errors(rows),
    )


def _summarize_baseline(
    rows: list[dict[str, Any]],
    baseline_correct: list[bool],
) -> MethodSummary:
    total = len(rows)
    correct = sum(1 for value in baseline_correct if value)
    method = str(rows[0].get("baseline_method", "minirag")) if rows else "minirag"
    return MethodSummary(
        method=method,
        total=total,
        answered=total,
        correct=correct,
        abstained=0,
        accuracy=correct / total if total else 0.0,
        precision_at_answered=correct / total if total else 0.0,
        unsafe_allow_count=0,
    )


def _summarize_proofrag(
    rows: list[dict[str, Any]],
    proofrag_correct: list[bool],
) -> MethodSummary:
    total = len(rows)
    answered = sum(1 for row in rows if bool(row.get("model_called", row.get("answer_allowed"))))
    correct = sum(1 for value in proofrag_correct if value)
    unsafe = sum(
        1
        for row in rows
        if bool(row.get("answer_allowed"))
        and row.get("expected_answer_allowed") is False
    )
    return MethodSummary(
        method="minirag+proofrag",
        total=total,
        answered=answered,
        correct=correct,
        abstained=total - answered,
        accuracy=correct / total if total else 0.0,
        precision_at_answered=correct / answered if answered else 0.0,
        unsafe_allow_count=unsafe,
    )


def _baseline_correct(row: dict[str, Any]) -> bool:
    if "baseline_correct" in row:
        return bool(row["baseline_correct"])
    metrics = row.get("baseline_metrics") or {}
    if "correct" in metrics:
        return bool(metrics["correct"])
    return False


def _proofrag_answer_correct(row: dict[str, Any]) -> bool:
    if not bool(row.get("answer_allowed", row.get("actual_answer_allowed", False))):
        return False
    if "correct_when_answered" in row:
        return bool(row["correct_when_answered"])
    if "answer_correct" in row:
        return bool(row["answer_correct"])
    return bool(row.get("expected_answer_allowed", False))

