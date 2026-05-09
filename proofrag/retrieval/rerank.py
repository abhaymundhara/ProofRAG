"""
rerank.py — Deterministic contract-aware reranking.

The reranker is intentionally lightweight. It does not try to replace a
cross-encoder reranker; it provides a reproducible scoring layer that prefers
records most useful for satisfying an EvidenceContract.
"""

from __future__ import annotations

from proofrag.contracts.schema import EvidenceContract
from proofrag.evidence.ledger import EvidenceRecord


class ContractAwareReranker:
    """Rerank evidence records by contract utility and retrieval confidence."""

    def rerank(
        self,
        *,
        records: list[EvidenceRecord],
        contract: EvidenceContract,
        top_k: int | None = None,
    ) -> list[EvidenceRecord]:
        """Return records ordered by deterministic contract-aware score."""

        ranked = sorted(
            records,
            key=lambda record: self.score(record=record, contract=contract),
            reverse=True,
        )
        return ranked[:top_k] if top_k is not None else ranked

    def score(self, *, record: EvidenceRecord, contract: EvidenceContract) -> float:
        """Score one evidence record for a contract."""

        required_slot_ids = {slot.slot_id for slot in contract.required_slots}
        supported_required = required_slot_ids & set(record.supports_slots)
        contradicted_required = required_slot_ids & set(record.contradicts)

        strength_bonus = {
            "direct": 1.0,
            "indirect": 0.35,
            "background": 0.0,
        }.get(record.evidence_strength, 0.0)

        return (
            len(supported_required) * 3.0
            + strength_bonus
            + record.confidence
            + len(record.contradicts) * 0.5
            - len(contradicted_required) * 0.25
        )

