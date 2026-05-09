from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from proofrag.cli import app
from proofrag.contracts.schema import EvidenceContract, EvidenceSlot
from proofrag.retrieval.bm25 import BM25Retriever


def _contract() -> EvidenceContract:
    return EvidenceContract(
        question="Who asked LiHua about the laptop warranty?",
        query_type="factoid",
        slots=[
            EvidenceSlot(
                slot_id="who_asked",
                description="The person who asked LiHua",
                evidence_type="factual",
            ),
            EvidenceSlot(
                slot_id="warranty_context",
                description="Context about the laptop warranty issue",
                evidence_type="contextual",
            ),
        ],
    )


def test_bm25_retriever_ranks_and_maps_supported_slots(tmp_path: Path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "source_id": "doc-noise",
                    "text": "LiHua discussed campus lunch plans.",
                    "keywords": ["lihua", "lunch"],
                    "supports_slots": [],
                },
                {
                    "source_id": "doc-answer",
                    "text": "Tom asked LiHua about the laptop warranty issue.",
                    "keywords": ["tom", "lihua", "laptop", "warranty", "asked"],
                    "supports_slots": ["who_asked", "warranty_context"],
                },
            ]
        ),
        encoding="utf-8",
    )

    ledger = BM25Retriever(context_path=corpus_path, top_k=1).retrieve(
        question="Who asked LiHua about the laptop warranty?",
        contract=_contract(),
    )

    assert len(ledger.records) == 1
    assert ledger.records[0].source_id == "doc-answer"
    assert set(ledger.records[0].supports_slots) == {
        "who_asked",
        "warranty_context",
    }
    assert ledger.records[0].confidence > 0


def test_bm25_retriever_preserves_contradiction_records(tmp_path: Path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "source_id": "doc-contradiction",
                    "text": "No one asked LiHua about the laptop warranty.",
                    "keywords": ["lihua", "laptop", "warranty", "asked"],
                    "supports_slots": [],
                    "contradicts": ["who_asked"],
                }
            ]
        ),
        encoding="utf-8",
    )

    ledger = BM25Retriever(context_path=corpus_path).retrieve(
        question="Who asked LiHua about the laptop warranty?",
        contract=_contract(),
    )

    assert len(ledger.records) == 1
    assert ledger.records[0].supports_slots == []
    assert ledger.records[0].contradicts == ["who_asked"]


def test_cli_can_select_bm25_retriever(tmp_path: Path):
    output_path = tmp_path / "bm25.jsonl"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ask",
            "--question",
            "Who asked LiHua about the laptop warranty issue?",
            "--config",
            "configs/default.yaml",
            "--retriever",
            "bm25",
            "--output",
            str(output_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert record["summary"]["retriever_backend"] == "bm25"
    assert record["sufficiency"]["answer_allowed"] is True

