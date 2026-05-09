from __future__ import annotations

from proofrag.evidence.contradiction import detect_contradicted_slots
from proofrag.evidence.extraction import infer_evidence_from_text


def test_time_query_infers_direct_time_and_context():
    inference = infer_evidence_from_text(
        question="What time does Li Hua check in?",
        text="LiHua says she will check in around 5:30 PM.",
    )

    assert "time_answer" in inference.supports_slots
    assert "event_context" in inference.supports_slots
    assert inference.evidence_strength == "direct"


def test_placeholder_context_remains_background():
    inference = infer_evidence_from_text(
        question="When was it?",
        text="Dry-run placeholder context. Real MiniRAG retrieval was not executed.",
    )

    assert inference.supports_slots == []
    assert inference.contradicts == []
    assert inference.evidence_strength == "background"


def test_date_can_be_inferred_from_source_id():
    inference = infer_evidence_from_text(
        question="When was the appointment?",
        text="The appointment details were confirmed.",
        source_id="20260105_1400",
    )

    assert "date_or_time_answer" in inference.supports_slots
    assert inference.evidence_strength == "direct"


def test_who_query_detects_direct_asker_and_contradiction():
    inference = infer_evidence_from_text(
        question="Who asked about the warranty?",
        text="No one asked about the warranty.",
        metadata={"contradiction": True},
    )

    assert "who_asked" not in inference.supports_slots
    assert inference.contradicts == ["who_asked"]


def test_explicit_contradicts_metadata_is_preserved():
    assert detect_contradicted_slots(
        question="What is the Wi-Fi password?",
        text="The password is not campus-guest-42.",
        metadata={"contradicts": ["answer"]},
    ) == ["answer"]
