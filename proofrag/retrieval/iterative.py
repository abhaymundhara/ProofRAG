"""
iterative.py — Contract-gap guided retrieval loop.

The loop is deliberately backend-agnostic: any retriever implementing
``retrieve(question, contract)`` can be retried with a query expanded from the
missing evidence slots reported by the sufficiency scorer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from proofrag.contracts.schema import EvidenceContract
from proofrag.evidence.ledger import EvidenceLedger, EvidenceRecord
from proofrag.evidence.sufficiency import RuleBasedSufficiencyScorer, SufficiencyReport
from proofrag.retrieval.base import BaseRetriever


class RetrievalRound(BaseModel):
    """One retrieval/scoring attempt in an iterative retrieval run."""

    round_index: int
    query: str
    retrieved_records: int
    answer_allowed: bool
    missing_required_slots: list[str] = Field(default_factory=list)


class IterativeRetrievalResult(BaseModel):
    """Final result of contract-gap guided retrieval."""

    ledger: EvidenceLedger
    report: SufficiencyReport
    rounds: list[RetrievalRound]


class ContractGapRetriever:
    """Retry retrieval by expanding the query with missing slot descriptions."""

    def __init__(
        self,
        retriever: BaseRetriever,
        *,
        scorer: RuleBasedSufficiencyScorer | None = None,
        max_rounds: int = 2,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.retriever = retriever
        self.scorer = scorer or RuleBasedSufficiencyScorer()
        self.max_rounds = max_rounds

    def retrieve(
        self,
        *,
        question: str,
        contract: EvidenceContract,
    ) -> IterativeRetrievalResult:
        """Run retrieval until the contract passes or max rounds is reached."""

        query = question
        merged_records: list[EvidenceRecord] = []
        rounds: list[RetrievalRound] = []
        report: SufficiencyReport | None = None

        for round_index in range(1, self.max_rounds + 1):
            ledger = self.retriever.retrieve(question=query, contract=contract)
            merged_records = _merge_records(merged_records, ledger.records)
            merged_ledger = EvidenceLedger(records=merged_records)
            report = self.scorer.score(contract=contract, ledger=merged_ledger)
            rounds.append(
                RetrievalRound(
                    round_index=round_index,
                    query=query,
                    retrieved_records=len(ledger.records),
                    answer_allowed=report.answer_allowed,
                    missing_required_slots=report.missing_required_slots,
                )
            )
            if report.answer_allowed:
                break
            if round_index < self.max_rounds:
                query = _expand_query(question, contract, report.missing_required_slots)

        if report is None:
            report = self.scorer.score(
                contract=contract,
                ledger=EvidenceLedger(records=merged_records),
            )

        return IterativeRetrievalResult(
            ledger=EvidenceLedger(records=merged_records),
            report=report,
            rounds=rounds,
        )


def _expand_query(
    question: str,
    contract: EvidenceContract,
    missing_slot_ids: list[str],
) -> str:
    missing_descriptions = [
        slot.description
        for slot in contract.slots
        if slot.slot_id in missing_slot_ids
    ]
    if not missing_descriptions:
        return question
    return f"{question} Required missing evidence: {'; '.join(missing_descriptions)}"


def _merge_records(
    existing: list[EvidenceRecord],
    incoming: list[EvidenceRecord],
) -> list[EvidenceRecord]:
    seen = {(record.source_id, record.text) for record in existing}
    merged = list(existing)
    for record in incoming:
        key = (record.source_id, record.text)
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)
    return merged

