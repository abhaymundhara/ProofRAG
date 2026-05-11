from __future__ import annotations

import json
from pathlib import Path

from proofrag.evaluation.lihua_augmentation import augment_export_with_lihua_sources


def test_augment_export_with_lihua_sources_appends_ranked_context(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "20260108_1100.txt").write_text(
        "Time: 20260108_11:00\nWolfgangSchulz: The cozy cafe downtown dinner is tonight.",
        encoding="utf-8",
    )
    (data_dir / "20260109_1100.txt").write_text(
        "Time: 20260109_11:00\nAdamSmith: The plumber arrives tomorrow.",
        encoding="utf-8",
    )
    source = tmp_path / "export.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "q1",
                "dataset": "LiHua-World",
                "question": "Where did Wolfgang have dinner downtown?",
                "query_type": "single",
                "gold_answer": "cozy cafe downtown",
                "gold_supporting_sources": ["20260108_11:00"],
                "retrieved_context": [],
                "baseline_answer": "cozy cafe downtown",
                "baseline_method": "minirag",
                "baseline_metrics": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "augmented.jsonl"
    rows = augment_export_with_lihua_sources(
        input_path=source,
        output_path=output,
        data_dir=data_dir,
        top_k=1,
    )

    assert rows == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["retrieved_context"][0]["source_id"] == "20260108_11:00"
    assert row["retrieved_context"][0]["metadata"]["retriever"] == "lihua_bm25"
