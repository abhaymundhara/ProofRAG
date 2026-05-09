"""
error_analysis.py — Categorize ProofRAG and baseline evaluation outcomes.
"""

from __future__ import annotations

from pydantic import BaseModel


class ErrorBucket(BaseModel):
    """Count and example IDs for one error-analysis category."""

    label: str
    count: int
    ids: list[str]


class ErrorAnalysisReport(BaseModel):
    """Grouped error-analysis output for a result set."""

    total: int
    buckets: list[ErrorBucket]


def classify_result(row: dict) -> str:
    """Classify one evaluation row into a stable error-analysis bucket."""

    answer_allowed = bool(row.get("answer_allowed", row.get("actual_answer_allowed")))
    model_called = bool(row.get("model_called", answer_allowed))
    correct = bool(row.get("correct_when_answered", row.get("answer_correct", False)))
    contains_gold = bool(row.get("contains_gold_answer_raw", False))
    expected_allowed = row.get("expected_answer_allowed")
    missing_slots = row.get("missing_required_slots") or row.get("missing_slots") or []
    contradiction_count = int(row.get("contradiction_count") or 0)

    if expected_allowed is not None:
        if answer_allowed and not bool(expected_allowed):
            return "unsafe_false_allow"
        if not answer_allowed and bool(expected_allowed):
            return "false_abstain"

    if not answer_allowed:
        if contradiction_count > 0:
            return "blocked_contradiction"
        if missing_slots:
            return "blocked_missing_evidence"
        return "blocked_other"

    if answer_allowed and not model_called:
        return "allowed_but_model_not_called"

    if correct:
        return "correct"

    if contains_gold:
        return "gold_present_but_semantically_wrong"

    return "incorrect"


def analyze_errors(rows: list[dict]) -> ErrorAnalysisReport:
    """Return count buckets with row IDs for deterministic error analysis."""

    grouped: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        label = classify_result(row)
        row_id = str(row.get("id") or row.get("run_id") or index)
        grouped.setdefault(label, []).append(row_id)

    buckets = [
        ErrorBucket(label=label, count=len(ids), ids=ids)
        for label, ids in sorted(grouped.items())
    ]
    return ErrorAnalysisReport(total=len(rows), buckets=buckets)

