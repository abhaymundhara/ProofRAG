"""
test_v01_hardening.py — Tests for v0.1 hardening (stricter retrieval and sufficiency).
"""

import json
from proofrag.contracts.schema import EvidenceContract, EvidenceSlot
from proofrag.retrieval.dummy import DummyRetriever
from proofrag.evidence.sufficiency import RuleBasedSufficiencyScorer
from proofrag.evidence.ledger import EvidenceLedger

def test_dummy_retriever_uses_explicit_supports_slots(tmp_path):
    # Create a custom context file
    context_data = [
        {
            "source_id": "doc-001",
            "text": "Some text",
            "keywords": ["irrelevant"],
            "supports_slots": ["who_asked"]
        },
        {
            "source_id": "doc-002",
            "text": "Another text",
            "keywords": ["who_asked"], # Keyword matches but supports_slots is empty
            "supports_slots": []
        }
    ]
    context_file = tmp_path / "context.json"
    context_file.write_text(json.dumps(context_data))
    
    retriever = DummyRetriever(context_path=context_file)
    contract = EvidenceContract(
        question="Who?",
        query_type="factoid",
        slots=[
            EvidenceSlot(slot_id="who_asked", description="The person who asked", evidence_type="factual")
        ]
    )
    
    ledger = retriever.retrieve("Who asked?", contract)
    
    # doc-001 should be included because of explicit supports_slots
    # doc-002 should NOT be included because supports_slots is empty (it has precedence)
    assert len(ledger.records) == 1
    assert ledger.records[0].source_id == "doc-001"
    assert ledger.records[0].supports_slots == ["who_asked"]

def test_dummy_retriever_fallback_to_keywords(tmp_path):
    # Create a custom context file without supports_slots
    context_data = [
        {
            "source_id": "doc-001",
            "text": "Tom asked LiHua",
            "keywords": ["tom", "asked"]
        }
    ]
    context_file = tmp_path / "context.json"
    context_file.write_text(json.dumps(context_data))
    
    retriever = DummyRetriever(context_path=context_file)
    contract = EvidenceContract(
        question="Who asked?",
        query_type="factoid",
        slots=[
            EvidenceSlot(slot_id="who_asked", description="The person who asked", evidence_type="factual")
        ]
    )
    # The description "The person who asked" contains "asked"
    ledger = retriever.retrieve("Who asked?", contract)
    
    assert len(ledger.records) == 1
    assert "who_asked" in ledger.records[0].supports_slots

def test_sufficiency_fails_if_required_evidence_missing():
    # Scenario: only warranty_context is present, but who_asked (required) is missing.
    contract = EvidenceContract(
        question="Q",
        query_type="factoid",
        slots=[
            EvidenceSlot(slot_id="who_asked", description="D1", evidence_type="factual", required=True),
            EvidenceSlot(slot_id="warranty_context", description="D2", evidence_type="factual", required=True)
        ]
    )
    
    # Ledger only has warranty_context
    from proofrag.evidence.ledger import EvidenceRecord
    ledger = EvidenceLedger(records=[
        EvidenceRecord(
            record_id="r1",
            source_id="s1",
            text="T",
            supports_slots=["warranty_context"],
            confidence=0.9
        )
    ])
    
    scorer = RuleBasedSufficiencyScorer()
    report = scorer.score(contract, ledger)
    
    assert report.answer_allowed is False
    assert "who_asked" in report.missing_required_slots
    assert report.coverage_score == 0.5


def test_indirect_evidence_does_not_satisfy_required_slot():
    """
    Verify that if a required slot is supported only by 'indirect' or 'background'
    evidence, the SufficiencyScorer marks it as missing.
    """
    contract = EvidenceContract(
        question="Who asked LiHua about the laptop warranty issue?",
        query_type="factoid",
        slots=[
            EvidenceSlot(
                slot_id="who_asked",
                description="The person who asked",
                evidence_type="factual",
                required=True,
                min_sources=1
            )
        ]
    )

    # Scenario: We have evidence that Tom was at the meeting (indirect/background),
    # but not that he asked (direct).
    from proofrag.evidence.ledger import EvidenceRecord
    ledger = EvidenceLedger(records=[
        EvidenceRecord(
            record_id="rec-1",
            source_id="doc-006",
            text="The meeting on Monday involved Tom and LiHua.",
            supports_slots=["who_asked"],
            confidence=0.75,
            evidence_strength="indirect"
        )
    ])

    scorer = RuleBasedSufficiencyScorer()
    report = scorer.score(contract, ledger)

    assert report.answer_allowed is False
    assert "who_asked" in report.missing_required_slots
    assert report.coverage_score == 0.0
    assert "Missing required slots: who_asked" in report.reason


def test_direct_evidence_satisfies_required_slot():
    """
    Verify that 'direct' evidence (the default) satisfies a required slot.
    """
    contract = EvidenceContract(
        question="Who asked LiHua about the laptop warranty issue?",
        query_type="factoid",
        slots=[
            EvidenceSlot(
                slot_id="who_asked",
                description="The person who asked",
                evidence_type="factual",
                required=True,
                min_sources=1
            )
        ]
    )

    from proofrag.evidence.ledger import EvidenceRecord
    ledger = EvidenceLedger(records=[
        EvidenceRecord(
            record_id="rec-2",
            source_id="doc-002",
            text="Tom asked LiHua about the laptop warranty issue.",
            supports_slots=["who_asked"],
            confidence=0.75,
            evidence_strength="direct"
        )
    ])

    scorer = RuleBasedSufficiencyScorer()
    report = scorer.score(contract, ledger)

    assert report.answer_allowed is True
    assert len(report.missing_required_slots) == 0
    assert report.coverage_score == 1.0
