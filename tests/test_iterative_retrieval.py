from __future__ import annotations

from proofrag.contracts.schema import EvidenceContract, EvidenceSlot
from proofrag.evidence.ledger import EvidenceLedger, EvidenceRecord
from proofrag.retrieval.base import BaseRetriever
from proofrag.retrieval.iterative import ContractGapRetriever


class ScriptedRetriever(BaseRetriever):
    def __init__(self, ledgers: list[EvidenceLedger]) -> None:
        self.ledgers = ledgers
        self.calls: list[str] = []
        self.total_docs = 0

    def retrieve(self, question: str, contract: EvidenceContract) -> EvidenceLedger:
        _ = contract
        self.calls.append(question)
        index = min(len(self.calls) - 1, len(self.ledgers) - 1)
        ledger = self.ledgers[index]
        self.total_docs += len(ledger.records)
        return ledger


def _contract() -> EvidenceContract:
    return EvidenceContract(
        question="Who asked LiHua about the warranty and when?",
        query_type="factoid",
        slots=[
            EvidenceSlot(
                slot_id="who_asked",
                description="The person who asked LiHua",
                evidence_type="factual",
            ),
            EvidenceSlot(
                slot_id="when_asked",
                description="The time when LiHua was asked",
                evidence_type="temporal",
            ),
        ],
    )


def test_iterative_retrieval_expands_query_for_missing_slots():
    first = EvidenceLedger(
        records=[
            EvidenceRecord(
                record_id="r1",
                source_id="doc-1",
                text="Tom asked LiHua about the warranty.",
                supports_slots=["who_asked"],
                confidence=1.0,
            )
        ]
    )
    second = EvidenceLedger(
        records=[
            EvidenceRecord(
                record_id="r2",
                source_id="doc-2",
                text="Tom asked LiHua on Monday.",
                supports_slots=["when_asked"],
                confidence=1.0,
            )
        ]
    )
    retriever = ScriptedRetriever([first, second])

    result = ContractGapRetriever(retriever, max_rounds=2).retrieve(
        question="Who asked LiHua about the warranty and when?",
        contract=_contract(),
    )

    assert result.report.answer_allowed is True
    assert len(result.ledger.records) == 2
    assert len(result.rounds) == 2
    assert "when_asked" in result.rounds[0].missing_required_slots
    assert "The time when LiHua was asked" in retriever.calls[1]


def test_iterative_retrieval_stops_when_first_round_satisfies_contract():
    ledger = EvidenceLedger(
        records=[
            EvidenceRecord(
                record_id="r1",
                source_id="doc-1",
                text="Tom asked LiHua about the warranty on Monday.",
                supports_slots=["who_asked", "when_asked"],
                confidence=1.0,
            )
        ]
    )
    retriever = ScriptedRetriever([ledger])

    result = ContractGapRetriever(retriever, max_rounds=3).retrieve(
        question="Who asked LiHua about the warranty and when?",
        contract=_contract(),
    )

    assert result.report.answer_allowed is True
    assert len(result.rounds) == 1
    assert len(retriever.calls) == 1


def test_iterative_retrieval_deduplicates_repeated_records():
    repeated = EvidenceLedger(
        records=[
            EvidenceRecord(
                record_id="r1",
                source_id="doc-1",
                text="Tom asked LiHua about the warranty.",
                supports_slots=["who_asked"],
                confidence=1.0,
            )
        ]
    )
    retriever = ScriptedRetriever([repeated, repeated])

    result = ContractGapRetriever(retriever, max_rounds=2).retrieve(
        question="Who asked LiHua about the warranty and when?",
        contract=_contract(),
    )

    assert result.report.answer_allowed is False
    assert len(result.ledger.records) == 1
    assert len(result.rounds) == 2

