"""
test_sufficiency.py — Tests for RuleBasedSufficiencyScorer.

Key scenarios:
  1. All required slots covered → answer_allowed=True
  2. Missing a required slot → answer_allowed=False
  3. Contradiction present in strict_mode → answer_allowed=False
  4. Contradiction present but strict_mode=False → answer_allowed=True
  5. min_sources enforcement
  6. coverage_score computed correctly
"""

import pytest

from proofrag.contracts.schema import EvidenceContract, EvidenceSlot
from proofrag.evidence.ledger import EvidenceRecord, EvidenceLedger
from proofrag.evidence.sufficiency import RuleBasedSufficiencyScorer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _slot(slot_id: str, required: bool = True, min_sources: int = 1) -> EvidenceSlot:
    return EvidenceSlot(
        slot_id=slot_id,
        description=f"Slot {slot_id}",
        evidence_type="factual",
        required=required,
        min_sources=min_sources,
    )


def _contract(
    slots: list[EvidenceSlot],
    strict_mode: bool = True,
) -> EvidenceContract:
    return EvidenceContract(
        question="Test question?",
        query_type="factoid",
        slots=slots,
        must_check_contradictions=True,
        strict_mode=strict_mode,
    )


def _record(
    record_id: str,
    supports: list[str],
    contradicts: list[str] | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=record_id,
        source_id="src",
        text="Some evidence.",
        supports_slots=supports,
        confidence=0.9,
        contradicts=contradicts or [],
    )


scorer = RuleBasedSufficiencyScorer()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRuleBasedSufficiencyScorer:
    def test_all_slots_covered_allows_answer(self):
        contract = _contract([_slot("s1"), _slot("s2")])
        ledger = EvidenceLedger(
            records=[
                _record("r1", supports=["s1"]),
                _record("r2", supports=["s2"]),
            ]
        )
        report = scorer.score(contract, ledger)

        assert report.answer_allowed is True
        assert report.missing_required_slots == []
        assert set(report.covered_slots) == {"s1", "s2"}

    def test_missing_required_slot_blocks_answer(self):
        contract = _contract([_slot("s1"), _slot("s2")])
        ledger = EvidenceLedger(
            records=[_record("r1", supports=["s1"])]  # s2 missing
        )
        report = scorer.score(contract, ledger)

        assert report.answer_allowed is False
        assert "s2" in report.missing_required_slots

    def test_contradiction_in_strict_mode_blocks_answer(self):
        contract = _contract([_slot("s1")], strict_mode=True)
        ledger = EvidenceLedger(
            records=[
                _record("r1", supports=["s1"]),
                _record("r2", supports=[], contradicts=["s1"]),
            ]
        )
        report = scorer.score(contract, ledger)

        assert report.answer_allowed is False
        assert report.contradiction_count == 1

    def test_contradiction_in_non_strict_mode_allows_answer(self):
        contract = _contract([_slot("s1")], strict_mode=False)
        ledger = EvidenceLedger(
            records=[
                _record("r1", supports=["s1"]),
                _record("r2", supports=[], contradicts=["s1"]),
            ]
        )
        report = scorer.score(contract, ledger)

        assert report.answer_allowed is True
        assert report.contradiction_count == 1

    def test_min_sources_not_met_blocks_answer(self):
        contract = _contract([_slot("s1", min_sources=2)])
        ledger = EvidenceLedger(
            records=[_record("r1", supports=["s1"])]  # only 1 source, need 2
        )
        report = scorer.score(contract, ledger)

        assert report.answer_allowed is False
        assert "s1" in report.missing_required_slots

    def test_min_sources_met_allows_answer(self):
        contract = _contract([_slot("s1", min_sources=2)])
        ledger = EvidenceLedger(
            records=[
                _record("r1", supports=["s1"]),
                _record("r2", supports=["s1"]),
            ]
        )
        report = scorer.score(contract, ledger)

        assert report.answer_allowed is True

    def test_coverage_score_partial(self):
        contract = _contract([_slot("s1"), _slot("s2"), _slot("s3")])
        ledger = EvidenceLedger(
            records=[
                _record("r1", supports=["s1"]),
                _record("r2", supports=["s2"]),
                # s3 missing
            ]
        )
        report = scorer.score(contract, ledger)

        assert report.answer_allowed is False
        assert abs(report.coverage_score - 2 / 3) < 0.001

    def test_coverage_score_full(self):
        contract = _contract([_slot("s1"), _slot("s2")])
        ledger = EvidenceLedger(
            records=[
                _record("r1", supports=["s1"]),
                _record("r2", supports=["s2"]),
            ]
        )
        report = scorer.score(contract, ledger)

        assert report.coverage_score == 1.0

    def test_no_required_slots_gives_full_coverage(self):
        contract = _contract([_slot("s1", required=False)])
        ledger = EvidenceLedger(records=[])
        report = scorer.score(contract, ledger)

        assert report.answer_allowed is True
        assert report.coverage_score == 1.0

    def test_report_has_reason_string(self):
        contract = _contract([_slot("s1")])
        ledger = EvidenceLedger(records=[])
        report = scorer.score(contract, ledger)

        assert isinstance(report.reason, str)
        assert len(report.reason) > 0
