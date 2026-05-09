from __future__ import annotations

import json
from pathlib import Path

from proofrag.evaluation.comparison import (
    compare_minirag_vs_proofrag,
    load_jsonl,
)


def test_compare_minirag_vs_proofrag_summarizes_methods():
    rows = [
        {
            "id": "q1",
            "baseline_method": "minirag",
            "baseline_correct": False,
            "answer_allowed": True,
            "model_called": True,
            "correct_when_answered": True,
            "expected_answer_allowed": True,
        },
        {
            "id": "q2",
            "baseline_method": "minirag",
            "baseline_correct": True,
            "answer_allowed": False,
            "model_called": False,
            "correct_when_answered": False,
            "expected_answer_allowed": False,
            "missing_required_slots": ["answer"],
        },
        {
            "id": "q3",
            "baseline_method": "minirag",
            "baseline_correct": False,
            "answer_allowed": True,
            "model_called": True,
            "correct_when_answered": False,
            "expected_answer_allowed": False,
        },
    ]

    report = compare_minirag_vs_proofrag(rows)

    assert report.baseline.total == 3
    assert report.baseline.correct == 1
    assert report.proofrag.answered == 2
    assert report.proofrag.correct == 1
    assert report.proofrag.unsafe_allow_count == 1
    assert report.paired_answer_accuracy.treatment_wins == 1
    assert report.paired_answer_accuracy.baseline_wins == 1
    buckets = {bucket.label: bucket.count for bucket in report.proofrag_error_analysis.buckets}
    assert buckets["unsafe_false_allow"] == 1


def test_load_jsonl_for_comparison(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a"}),
                "",
                json.dumps({"id": "b"}),
            ]
        ),
        encoding="utf-8",
    )

    assert [row["id"] for row in load_jsonl(path)] == ["a", "b"]

