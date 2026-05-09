from __future__ import annotations

import json
from pathlib import Path

from proofrag.human_eval.schema import (
    build_human_eval_item,
    export_human_eval_jsonl,
    load_result_jsonl,
)


def test_build_human_eval_item_from_experiment_row():
    item = build_human_eval_item(
        {
            "id": "q1",
            "question": "Who asked?",
            "baseline_answer": "Sarah",
            "answer": "Tom asked.",
            "gold_answer": "Tom",
            "sufficiency": {
                "answer_allowed": True,
                "coverage_score": 1.0,
                "missing_required_slots": [],
                "contradiction_count": 0,
            },
            "ledger": {
                "records": [
                    {"text": "Tom asked LiHua about the warranty."},
                ]
            },
        }
    )

    assert item.id == "q1"
    assert item.proofrag_abstained is False
    assert item.evidence == ["Tom asked LiHua about the warranty."]
    assert item.metadata["coverage_score"] == 1.0


def test_build_human_eval_item_marks_abstention():
    item = build_human_eval_item(
        {
            "run_id": "run-1",
            "question": "Q?",
            "sufficiency": {"answer_allowed": False},
        }
    )

    assert item.id == "run-1"
    assert item.proofrag_abstained is True


def test_export_and_load_human_eval_jsonl(tmp_path: Path):
    input_path = tmp_path / "results.jsonl"
    output_path = tmp_path / "human_eval.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Q?",
                "sufficiency": {"answer_allowed": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_result_jsonl(input_path)
    items = export_human_eval_jsonl(rows, output_path)

    assert len(items) == 1
    written = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert written["id"] == "q1"
    assert written["proofrag_abstained"] is True

