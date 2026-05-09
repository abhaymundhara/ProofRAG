"""
tables.py — Markdown table helpers for evaluation reports.
"""

from __future__ import annotations

from collections.abc import Sequence

from proofrag.evaluation.comparison import ComparisonReport
from proofrag.evaluation.metrics import BenchmarkMetrics


def benchmark_metrics_markdown(metrics: BenchmarkMetrics) -> str:
    """Render benchmark metrics as a compact Markdown table."""

    rows = [
        ("Total Questions", str(metrics.total_questions)),
        ("Behavioural Pass Rate", _pct(metrics.behavioural_pass_rate)),
        ("Answer Allowed", str(metrics.answer_allowed_count)),
        ("Abstained", str(metrics.abstained_count)),
        ("Unsafe Allow Rate", _pct(metrics.unsafe_answer_rate)),
        ("Precision@Answered", _pct(metrics.precision_at_answered)),
        ("Mean Coverage", f"{metrics.coverage_score_mean:.2f}"),
    ]
    return _markdown_table(["Metric", "Value"], rows)


def comparison_markdown(report: ComparisonReport) -> str:
    """Render MiniRAG-vs-ProofRAG comparison as Markdown."""

    rows = [
        (
            report.baseline.method,
            str(report.baseline.total),
            str(report.baseline.answered),
            str(report.baseline.correct),
            _pct(report.baseline.accuracy),
            _pct(report.baseline.precision_at_answered),
            str(report.baseline.unsafe_allow_count),
        ),
        (
            report.proofrag.method,
            str(report.proofrag.total),
            str(report.proofrag.answered),
            str(report.proofrag.correct),
            _pct(report.proofrag.accuracy),
            _pct(report.proofrag.precision_at_answered),
            str(report.proofrag.unsafe_allow_count),
        ),
    ]
    return _markdown_table(
        [
            "Method",
            "Total",
            "Answered",
            "Correct",
            "Accuracy",
            "Precision@Answered",
            "Unsafe Allows",
        ],
        rows,
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _pct(value: float) -> str:
    return f"{value:.1%}"
