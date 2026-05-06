"""
test_contract_schema.py — Tests for EvidenceSlot and EvidenceContract.
"""

import pytest
from pydantic import ValidationError

from proofrag.contracts.schema import EvidenceContract, EvidenceSlot


# ─────────────────────────────────────────────────────────────────────────────
# EvidenceSlot
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceSlot:
    def test_defaults(self):
        slot = EvidenceSlot(
            slot_id="s1",
            description="Who did X?",
            evidence_type="factual",
        )
        assert slot.required is True
        assert slot.min_sources == 1

    def test_custom_values(self):
        slot = EvidenceSlot(
            slot_id="s2",
            description="When did it happen?",
            evidence_type="temporal",
            required=False,
            min_sources=3,
        )
        assert slot.required is False
        assert slot.min_sources == 3

    def test_min_sources_must_be_positive(self):
        with pytest.raises(ValidationError):
            EvidenceSlot(
                slot_id="bad",
                description="x",
                evidence_type="factual",
                min_sources=0,
            )


# ─────────────────────────────────────────────────────────────────────────────
# EvidenceContract
# ─────────────────────────────────────────────────────────────────────────────


def _make_contract(**overrides) -> EvidenceContract:
    base = dict(
        question="Who asked LiHua?",
        query_type="factoid",
        slots=[
            EvidenceSlot(
                slot_id="who_asked",
                description="Person who asked",
                evidence_type="factual",
            ),
            EvidenceSlot(
                slot_id="context",
                description="Background context",
                evidence_type="contextual",
                required=False,
            ),
        ],
    )
    base.update(overrides)
    return EvidenceContract(**base)


class TestEvidenceContract:
    def test_validates_correctly(self):
        c = _make_contract()
        assert c.question == "Who asked LiHua?"
        assert len(c.slots) == 2

    def test_defaults(self):
        c = _make_contract()
        assert c.must_check_contradictions is True
        assert c.strict_mode is True

    def test_required_slots_filter(self):
        c = _make_contract()
        # only "who_asked" is required
        assert len(c.required_slots) == 1
        assert c.required_slots[0].slot_id == "who_asked"

    def test_slot_ids_order(self):
        c = _make_contract()
        assert c.slot_ids == ["who_asked", "context"]

    def test_empty_slots_allowed(self):
        c = EvidenceContract(question="Q?", query_type="factoid", slots=[])
        assert c.slots == []

    def test_requires_question(self):
        with pytest.raises(ValidationError):
            EvidenceContract(query_type="factoid", slots=[])  # type: ignore[call-arg]
