from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_compare_minirag_proofrag_script_writes_artifacts(tmp_path: Path):
    input_path = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.json"
    table_path = tmp_path / "table.md"
    chart_path = tmp_path / "chart.svg"
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
    ]
    input_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_minirag_proofrag.py",
            "--input",
            str(input_path),
            "--summary-json",
            str(summary_path),
            "--table-md",
            str(table_path),
            "--chart-svg",
            str(chart_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Wrote comparison summary" in result.stdout
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["baseline"]["method"] == "minirag"
    assert summary["proofrag"]["method"] == "minirag+proofrag"
    assert "Precision@Answered" in table_path.read_text(encoding="utf-8")
    chart = chart_path.read_text(encoding="utf-8")
    assert chart.startswith("<svg")
    assert "unsafe allow rate" in chart
