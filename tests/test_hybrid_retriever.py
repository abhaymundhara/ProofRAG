from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from proofrag.cli import app
from proofrag.contracts.schema import EvidenceContract, EvidenceSlot
from proofrag.evidence.ledger import EvidenceRecord
from proofrag.retrieval.hybrid import HybridRetriever
from proofrag.retrieval.rerank import ContractAwareReranker


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


def test_contract_aware_reranker_prefers_direct_required_coverage():
    records = [
        EvidenceRecord(
            record_id="background",
            source_id="doc-background",
            text="LiHua discussed laptop purchasing.",
            supports_slots=["warranty_context"],
            confidence=0.99,
            evidence_strength="background",
        ),
        EvidenceRecord(
            record_id="direct",
            source_id="doc-direct",
            text="Tom asked LiHua about the laptop warranty.",
            supports_slots=["who_asked", "warranty_context"],
            confidence=0.40,
            evidence_strength="direct",
        ),
    ]

    reranked = ContractAwareReranker().rerank(
        records=records,
        contract=_contract(),
    )

    assert reranked[0].record_id == "direct"


def test_hybrid_retriever_reranks_bm25_candidates(tmp_path: Path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "source_id": "doc-context",
                    "text": "LiHua laptop warranty warranty warranty support context.",
                    "keywords": ["lihua", "laptop", "warranty", "support"],
                    "supports_slots": ["warranty_context"],
                },
                {
                    "source_id": "doc-answer",
                    "text": "Tom asked LiHua about the laptop warranty.",
                    "keywords": ["tom", "asked", "lihua", "laptop", "warranty"],
                    "supports_slots": ["who_asked", "warranty_context"],
                },
            ]
        ),
        encoding="utf-8",
    )

    ledger = HybridRetriever(
        context_path=corpus_path,
        top_k=1,
        candidate_k=2,
    ).retrieve(
        question="Who asked LiHua about the laptop warranty?",
        contract=_contract(),
    )

    assert len(ledger.records) == 1
    assert ledger.records[0].source_id == "doc-answer"


def test_cli_can_select_hybrid_retriever(tmp_path: Path):
    output_path = tmp_path / "hybrid.jsonl"
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
            "hybrid",
            "--output",
            str(output_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert record["summary"]["retriever_backend"] == "hybrid"
    assert record["sufficiency"]["answer_allowed"] is True

