from __future__ import annotations

from pathlib import Path

import pytest

from proofrag.evaluation.comparison import compare_minirag_vs_proofrag
from proofrag.evaluation.metrics import BenchmarkMetrics
from proofrag.evaluation.plots import bar_chart_svg, write_bar_chart_svg
from proofrag.evaluation.tables import benchmark_metrics_markdown, comparison_markdown


def test_benchmark_metrics_markdown_table():
    metrics = BenchmarkMetrics(
        total_questions=10,
        behavioural_pass_count=9,
        behavioural_pass_rate=0.9,
        answer_allowed_count=6,
        abstained_count=4,
        false_allow_count=1,
        false_abstain_count=0,
        unsafe_answer_rate=0.1,
        coverage_score_mean=0.82,
        evidence_contract_completion_rate=0.82,
        abstention_rate=0.4,
        answered_accuracy=0.83,
        precision_at_answered=0.83,
        answered_correct_count=5,
    )

    table = benchmark_metrics_markdown(metrics)

    assert "| Metric | Value |" in table
    assert "| Unsafe Allow Rate | 10.0% |" in table
    assert "| Precision@Answered | 83.0% |" in table


def test_comparison_markdown_table():
    report = compare_minirag_vs_proofrag(
        [
            {
                "baseline_correct": False,
                "answer_allowed": True,
                "model_called": True,
                "correct_when_answered": True,
            }
        ]
    )

    table = comparison_markdown(report)

    assert "| Method | Total | Answered | Correct |" in table
    assert "minirag+proofrag" in table


def test_bar_chart_svg_and_write(tmp_path: Path):
    svg = bar_chart_svg({"Accuracy": 0.75, "Unsafe": 0.05}, title="Results")

    assert svg.startswith("<svg")
    assert "Results" in svg
    assert "75.0%" in svg

    output_path = write_bar_chart_svg(
        tmp_path / "figures" / "chart.svg",
        {"Accuracy": 1.0},
        title="Chart",
    )
    assert output_path.exists()
    assert "Chart" in output_path.read_text(encoding="utf-8")


def test_bar_chart_rejects_tiny_dimensions():
    with pytest.raises(ValueError):
        bar_chart_svg({"x": 1.0}, title="Tiny", width=100)

