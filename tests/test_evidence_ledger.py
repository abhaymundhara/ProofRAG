"""
test_evidence_ledger.py — Tests for EvidenceRecord and EvidenceLedger.
"""

from proofrag.evidence.ledger import EvidenceRecord, EvidenceLedger


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _record(
    record_id: str,
    supports: list[str],
    contradicts: list[str] | None = None,
    confidence: float = 0.9,
) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=record_id,
        source_id=f"src-{record_id}",
        text=f"Evidence text for {record_id}.",
        supports_slots=supports,
        confidence=confidence,
        contradicts=contradicts or [],
    )


def _ledger(*records: EvidenceRecord) -> EvidenceLedger:
    return EvidenceLedger(records=list(records))


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceLedger:
    def test_covered_slots_empty(self):
        ledger = _ledger()
        assert ledger.covered_slots() == set()

    def test_covered_slots_returns_correct_set(self):
        ledger = _ledger(
            _record("r1", supports=["slot_a", "slot_b"]),
            _record("r2", supports=["slot_c"]),
        )
        assert ledger.covered_slots() == {"slot_a", "slot_b", "slot_c"}

    def test_covered_slots_deduplicates(self):
        ledger = _ledger(
            _record("r1", supports=["slot_a"]),
            _record("r2", supports=["slot_a"]),
        )
        assert ledger.covered_slots() == {"slot_a"}

    def test_records_for_slot_returns_matching(self):
        r1 = _record("r1", supports=["slot_x"])
        r2 = _record("r2", supports=["slot_y"])
        r3 = _record("r3", supports=["slot_x", "slot_y"])
        ledger = _ledger(r1, r2, r3)

        results = ledger.records_for_slot("slot_x")
        assert len(results) == 2
        ids = {r.record_id for r in results}
        assert ids == {"r1", "r3"}

    def test_records_for_slot_empty(self):
        ledger = _ledger(_record("r1", supports=["slot_a"]))
        assert ledger.records_for_slot("slot_missing") == []

    def test_contradictions_returns_only_contradicting(self):
        r_clean = _record("rc", supports=["s1"])
        r_contra = _record("rx", supports=["s2"], contradicts=["s1"])
        ledger = _ledger(r_clean, r_contra)

        result = ledger.contradictions()
        assert len(result) == 1
        assert result[0].record_id == "rx"

    def test_contradictions_empty_when_none(self):
        ledger = _ledger(
            _record("r1", supports=["s1"]),
            _record("r2", supports=["s2"]),
        )
        assert ledger.contradictions() == []
