"""
schema.py — Human evaluation export schema.

The schema is designed for lightweight annotation tools: each JSONL row gives a
human reviewer the question, baseline answer, ProofRAG answer or abstention,
gold answer when available, and the evidence snippets used by ProofRAG.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class HumanEvalItem(BaseModel):
    """One human-evaluation annotation item."""

    id: str
    question: str
    baseline_answer: str = ""
    proofrag_answer: str = ""
    proofrag_abstained: bool
    gold_answer: str = ""
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_human_eval_item(row: dict[str, Any]) -> HumanEvalItem:
    """Normalize one experiment row into a HumanEvalItem."""

    sufficiency = row.get("sufficiency") or row.get("sufficiency_report") or {}
    answer_allowed = bool(
        row.get("answer_allowed", sufficiency.get("answer_allowed", False))
    )
    evidence_records = row.get("evidence_records") or (row.get("ledger") or {}).get("records") or []

    return HumanEvalItem(
        id=str(row.get("id") or row.get("run_id") or ""),
        question=str(row.get("question") or ""),
        baseline_answer=str(row.get("baseline_answer") or ""),
        proofrag_answer=str(row.get("answer") or row.get("proofrag_answer") or ""),
        proofrag_abstained=not answer_allowed,
        gold_answer=str(row.get("gold_answer") or ""),
        evidence=[str(record.get("text") or "") for record in evidence_records],
        metadata={
            "answer_allowed": answer_allowed,
            "coverage_score": sufficiency.get("coverage_score", row.get("coverage_score")),
            "missing_required_slots": sufficiency.get(
                "missing_required_slots",
                row.get("missing_required_slots", []),
            ),
            "contradiction_count": sufficiency.get(
                "contradiction_count",
                row.get("contradiction_count", 0),
            ),
            "baseline_method": row.get("baseline_method"),
        },
    )


def export_human_eval_jsonl(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> list[HumanEvalItem]:
    """Write normalized human-evaluation items to JSONL."""

    items = [build_human_eval_item(row) for row in rows]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")
    return items


def load_result_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load arbitrary result JSONL rows for human-eval export."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

