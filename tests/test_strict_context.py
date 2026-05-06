"""
test_strict_context.py — Tests for StrictContextPacker.

Verifies that the packed prompt contains all required sections and
accurately reflects the state of the contract, ledger, and report.
"""


from proofrag.contracts.schema import EvidenceContract, EvidenceSlot
from proofrag.evidence.ledger import EvidenceRecord, EvidenceLedger
from proofrag.evidence.sufficiency import RuleBasedSufficiencyScorer, SufficiencyReport
from proofrag.packing.strict_context import StrictContextPacker


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_QUESTION = "Who asked LiHua about the laptop warranty issue?"


def _make_contract(strict_mode: bool = True) -> EvidenceContract:
    return EvidenceContract(
        question=_QUESTION,
        query_type="factoid",
        slots=[
            EvidenceSlot(
                slot_id="who_asked",
                description="Person who asked",
                evidence_type="factual",
                required=True,
            ),
            EvidenceSlot(
                slot_id="warranty_context",
                description="Warranty context",
                evidence_type="contextual",
                required=True,
            ),
        ],
        must_check_contradictions=True,
        strict_mode=strict_mode,
    )


def _make_full_ledger() -> EvidenceLedger:
    return EvidenceLedger(
        records=[
            EvidenceRecord(
                record_id="r1",
                source_id="doc-001",
                text="Tom asked LiHua about the laptop warranty issue.",
                supports_slots=["who_asked"],
                confidence=0.9,
            ),
            EvidenceRecord(
                record_id="r2",
                source_id="doc-002",
                text="The warranty expires in December 2025.",
                supports_slots=["warranty_context"],
                confidence=0.8,
            ),
        ]
    )


def _make_partial_ledger() -> EvidenceLedger:
    """Ledger missing the warranty_context slot."""
    return EvidenceLedger(
        records=[
            EvidenceRecord(
                record_id="r1",
                source_id="doc-001",
                text="Tom asked LiHua about the laptop warranty issue.",
                supports_slots=["who_asked"],
                confidence=0.9,
            ),
        ]
    )


def _make_contradicting_ledger() -> EvidenceLedger:
    return EvidenceLedger(
        records=[
            EvidenceRecord(
                record_id="r1",
                source_id="doc-001",
                text="Tom asked LiHua.",
                supports_slots=["who_asked"],
                confidence=0.9,
            ),
            EvidenceRecord(
                record_id="r2",
                source_id="doc-002",
                text="The warranty context.",
                supports_slots=["warranty_context"],
                confidence=0.8,
            ),
            EvidenceRecord(
                record_id="r3",
                source_id="doc-003",
                text="Actually nobody asked LiHua.",
                supports_slots=[],
                confidence=0.5,
                contradicts=["who_asked"],
            ),
        ]
    )


scorer = RuleBasedSufficiencyScorer()
packer = StrictContextPacker()

_REQUIRED_SECTIONS = [
    "## QUESTION",
    "## REQUIRED EVIDENCE",
    "## EVIDENCE FOUND",
    "## MISSING EVIDENCE",
    "## CONTRADICTIONS",
    "## ANSWER POLICY",
    "## INSTRUCTION",
]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStrictContextPacker:
    def test_all_sections_present_full_coverage(self):
        contract = _make_contract()
        ledger = _make_full_ledger()
        report = scorer.score(contract, ledger)
        prompt = packer.pack(_QUESTION, contract, ledger, report)

        for section in _REQUIRED_SECTIONS:
            assert section in prompt, f"Section '{section}' missing from prompt"

    def test_question_appears_in_prompt(self):
        contract = _make_contract()
        ledger = _make_full_ledger()
        report = scorer.score(contract, ledger)
        prompt = packer.pack(_QUESTION, contract, ledger, report)

        assert _QUESTION in prompt

    def test_missing_evidence_section_lists_absent_slots(self):
        contract = _make_contract()
        ledger = _make_partial_ledger()
        report = scorer.score(contract, ledger)
        prompt = packer.pack(_QUESTION, contract, ledger, report)

        assert "warranty_context" in prompt

    def test_contradictions_section_shows_contradicting_record(self):
        contract = _make_contract()
        ledger = _make_contradicting_ledger()
        report = scorer.score(contract, ledger)
        prompt = packer.pack(_QUESTION, contract, ledger, report)

        assert "r3" in prompt  # contradicting record_id

    def test_answer_policy_blocked_when_missing(self):
        contract = _make_contract()
        ledger = _make_partial_ledger()
        report = scorer.score(contract, ledger)
        prompt = packer.pack(_QUESTION, contract, ledger, report)

        assert "BLOCKED" in prompt

    def test_answer_policy_permitted_when_full(self):
        contract = _make_contract()
        ledger = _make_full_ledger()
        report = scorer.score(contract, ledger)
        prompt = packer.pack(_QUESTION, contract, ledger, report)

        assert "PERMITTED" in prompt

    def test_instruction_tells_model_to_admit_incompleteness(self):
        contract = _make_contract()
        ledger = _make_full_ledger()
        report = scorer.score(contract, ledger)
        prompt = packer.pack(_QUESTION, contract, ledger, report)

        assert "incomplete" in prompt.lower()
        assert "guessing" in prompt.lower()

    def test_all_sections_present_with_contradictions(self):
        contract = _make_contract(strict_mode=True)
        ledger = _make_contradicting_ledger()
        report = scorer.score(contract, ledger)
        prompt = packer.pack(_QUESTION, contract, ledger, report)

        for section in _REQUIRED_SECTIONS:
            assert section in prompt, f"Section '{section}' missing from prompt"
