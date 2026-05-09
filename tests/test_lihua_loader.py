from __future__ import annotations

from pathlib import Path

from proofrag.evaluation.lihua import (
    load_lihua_qa_csv,
    parse_evidence_ids,
    resolve_lihua_sources,
)


def test_parse_evidence_ids_supports_and_and_commas():
    assert parse_evidence_ids("a<and>b") == ["a", "b"]
    assert parse_evidence_ids("a, b") == ["a", "b"]
    assert parse_evidence_ids("") == []


def test_load_lihua_qa_csv(tmp_path: Path):
    csv_path = tmp_path / "qa.csv"
    csv_path.write_text(
        "ID,Question,Gold Answer,Evidence,Type\n"
        "q1,When?,20260105,20260105_14:00<and>doc2,Single\n"
        "q2,Who?,Tom,doc3,Multi\n",
        encoding="utf-8",
    )

    rows = load_lihua_qa_csv(csv_path, limit=1)

    assert len(rows) == 1
    assert rows[0].id == "q1"
    assert rows[0].question == "When?"
    assert rows[0].evidence_ids == ["20260105_14:00", "doc2"]
    assert rows[0].question_type == "Single"


def test_resolve_lihua_sources_by_filename_and_content(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "20260105_14-00.txt").write_text("check-in", encoding="utf-8")
    (data_dir / "other.txt").write_text("contains evidence-id-7", encoding="utf-8")

    resolution = resolve_lihua_sources(
        evidence_ids=["20260105_14:00", "evidence-id-7", "missing"],
        data_dir=data_dir,
    )

    assert resolution.files["20260105_14:00"] == "20260105_14-00.txt"
    assert resolution.files["evidence-id-7"] == "other.txt"
    assert resolution.missing_source_ids == ["missing"]

