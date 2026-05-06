"""
ledger.py — EvidenceRecord and EvidenceLedger.

The EvidenceLedger collects all retrieved EvidenceRecords and exposes query
methods used by the SufficiencyScorer and StrictContextPacker.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceRecord(BaseModel):
    """A single piece of evidence retrieved from a source.

    Attributes:
        record_id:         Unique identifier for this record.
        source_id:         Identifier of the document/chunk this record came from.
        text:              The verbatim or summarised text of the evidence.
        supports_slots:    List of slot_ids that this record supports.
        confidence:        Retrieval confidence in [0, 1].
        contradicts:       List of slot_ids that this record contradicts.
        evidence_strength: The qualitative strength of the evidence
                           (direct, indirect, background).
    """

    record_id: str
    source_id: str
    text: str
    supports_slots: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    contradicts: list[str] = Field(default_factory=list)
    evidence_strength: str = Field(default="direct")


class EvidenceLedger(BaseModel):
    """An ordered collection of EvidenceRecords for a single query.

    The ledger is the runtime evidence store that the scoring and packing
    layers consume.  It is intentionally read-only after construction —
    retrieval backends build it, downstream modules only read it.

    Attributes:
        records: All retrieved evidence records.
    """

    records: list[EvidenceRecord] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def covered_slots(self) -> set[str]:
        """Return the set of slot_ids covered by at least one record."""
        slots: set[str] = set()
        for rec in self.records:
            slots.update(rec.supports_slots)
        return slots

    def records_for_slot(self, slot_id: str) -> list[EvidenceRecord]:
        """Return all records that support the given slot_id."""
        return [rec for rec in self.records if slot_id in rec.supports_slots]

    def contradictions(self) -> list[EvidenceRecord]:
        """Return all records that contradict at least one slot."""
        return [rec for rec in self.records if rec.contradicts]
