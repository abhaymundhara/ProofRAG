"""
schema.py — Core contract schemas for ProofRAG.

An EvidenceContract declares *what* evidence a question requires before an
answer is permitted.  It is paired at runtime with an EvidenceLedger to
produce a SufficiencyReport that gates generation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceSlot(BaseModel):
    """A single required evidence category within a contract.

    Attributes:
        slot_id:        Unique identifier used to cross-reference records.
        description:    Human-readable description of the expected evidence.
        evidence_type:  Broad category of evidence (e.g. "factual", "temporal").
        required:       Whether the slot must be filled before answering.
        min_sources:    Minimum number of distinct records needed to satisfy
                        the slot.
    """

    slot_id: str
    description: str
    evidence_type: str
    required: bool = True
    min_sources: int = Field(default=1, ge=1)


class EvidenceContract(BaseModel):
    """A declarative contract that specifies the evidence requirements for a
    question.

    Attributes:
        question:                  The natural-language question being answered.
        query_type:                High-level classification of the query
                                   (e.g. "factoid", "causal", "comparative").
        slots:                     Ordered list of evidence slots that must be
                                   (or should be) satisfied.
        must_check_contradictions: When True the scorer must inspect the ledger
                                   for contradicting records.
        strict_mode:               When True any contradiction found in the
                                   ledger causes answer_allowed=False.
    """

    question: str
    query_type: str
    slots: list[EvidenceSlot]
    must_check_contradictions: bool = True
    strict_mode: bool = True

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def required_slots(self) -> list[EvidenceSlot]:
        """Return only the slots marked as required."""
        return [s for s in self.slots if s.required]

    @property
    def slot_ids(self) -> list[str]:
        """Return all slot IDs in declaration order."""
        return [s.slot_id for s in self.slots]
