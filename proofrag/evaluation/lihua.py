"""
lihua.py — LiHua-World dataset loading and source resolution helpers.

The full LiHua-World data remains external. This module provides reusable,
tested logic for reading QA CSV files and resolving evidence IDs against an
extracted data directory.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from pydantic import BaseModel, Field


class LiHuaQuestion(BaseModel):
    """One LiHua-World QA row."""

    id: str
    question: str
    gold_answer: str
    evidence_ids: list[str] = Field(default_factory=list)
    question_type: str = ""


class LiHuaSourceResolution(BaseModel):
    """Resolution result for LiHua evidence source files."""

    files: dict[str, str]
    missing_source_ids: list[str]


def parse_evidence_ids(evidence: str) -> list[str]:
    """Parse LiHua evidence fields using `<and>` or comma separators."""

    if not evidence:
        return []
    if "<and>" in evidence:
        return [part.strip() for part in evidence.split("<and>") if part.strip()]
    return [part.strip() for part in evidence.split(",") if part.strip()]


def load_lihua_qa_csv(path: str | Path, *, limit: int | None = None) -> list[LiHuaQuestion]:
    """Load LiHua-World QA rows from a CSV file."""

    rows: list[LiHuaQuestion] = []
    with Path(path).open("r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            question = row.get("Question") or row.get("question") or ""
            gold_answer = row.get("Gold Answer") or row.get("gold_answer") or ""
            evidence = row.get("Evidence") or row.get("evidence") or ""
            question_type = row.get("Type") or row.get("type") or ""
            row_id = row.get("ID") or row.get("id") or f"lihua-{index:05d}"
            rows.append(
                LiHuaQuestion(
                    id=row_id,
                    question=question,
                    gold_answer=gold_answer,
                    evidence_ids=parse_evidence_ids(evidence),
                    question_type=question_type,
                )
            )
    return rows


def resolve_lihua_sources(
    *,
    evidence_ids: list[str],
    data_dir: str | Path,
) -> LiHuaSourceResolution:
    """Resolve evidence IDs to source files under an extracted LiHua data dir."""

    data_path = Path(data_dir)
    all_files = [
        Path(root) / filename
        for root, _, files in os.walk(data_path)
        for filename in files
        if not filename.startswith(".")
    ]

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for evidence_id in evidence_ids:
        match = _find_source_file(evidence_id, all_files)
        if match is None:
            missing.append(evidence_id)
        else:
            resolved[evidence_id] = str(match.relative_to(data_path))
    return LiHuaSourceResolution(files=resolved, missing_source_ids=missing)


def _find_source_file(evidence_id: str, files: list[Path]) -> Path | None:
    variations = [
        evidence_id,
        evidence_id.replace(":", "-"),
        evidence_id.replace(":", "_"),
        evidence_id.replace(":", ""),
    ]
    for path in files:
        if any(variation == path.stem or variation in path.name for variation in variations):
            return path

    for path in files:
        if path.suffix.lower() not in {".txt", ".json", ".csv", ".md"}:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if evidence_id in content:
            return path
    return None

