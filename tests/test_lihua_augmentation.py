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


def test_augment_export_with_lihua_sources_expands_fitness_queries(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "20261202_1400.txt").write_text(
        (
            "Time: 20261202_14:00\n"
            "WolfgangSchulz: I admire your good body shape.\n"
            "LiHua: You should exercise more often and get a good figure.\n"
            "WolfgangSchulz: I need to hit the gym more. Maybe we can do a "
            "workout together after training?"
        ),
        encoding="utf-8",
    )
    (data_dir / "20260307_1300.txt").write_text(
        (
            "Time: 20260307_13:00\n"
            "YurikoYamamoto: Let's discuss blog post structure and SEO strategy."
        ),
        encoding="utf-8",
    )
    source = tmp_path / "export.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "q1",
                "dataset": "LiHua-World",
                "question": (
                    "Did Li Hua discuss his progress with the fitness plan before "
                    "he shared a blog post about his recent fitness achievements?"
                ),
                "query_type": "multi",
                "gold_answer": "Yes",
                "gold_supporting_sources": ["20261202_14:00"],
                "retrieved_context": [],
                "baseline_answer": "Yes",
                "baseline_method": "minirag",
                "baseline_metrics": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "augmented.jsonl"
    augment_export_with_lihua_sources(
        input_path=source,
        output_path=output,
        data_dir=data_dir,
        top_k=1,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["retrieved_context"][0]["source_id"] == "20261202_14:00"
