from proofrag.evidence.ledger import EvidenceRecord
from proofrag.generation.strict_verifier import (
    build_compact_answer_prompt,
    build_strict_verifier_prompt,
    is_strict_abstention,
    is_uncertainty_abstention,
    question_kind,
    rank_evidence_records,
)


def _record(record_id: str, text: str) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=record_id,
        source_id=record_id,
        text=text,
        supports_slots=["answer"],
        evidence_strength="direct",
        confidence=1.0,
    )


def test_question_kind_detects_yesno_and_factual():
    assert question_kind("Did Li Hua ask Adam about repairs?") == "yesno"
    assert question_kind("What is the Wi-Fi password?") == "factual"


def test_rank_evidence_records_prefers_query_terms_and_phrases():
    records = [
        _record("background", "Li Hua asked about a cafe recommendation."),
        _record(
            "target",
            "Li Hua asked Adam Smith about basement window curtains and measurements.",
        ),
    ]

    ranked = rank_evidence_records(
        "Did Li Hua ask Adam Smith about basement window curtains?",
        records,
        limit=1,
    )

    assert [record.record_id for record in ranked] == ["target"]


def test_rank_evidence_records_expands_fitness_terms():
    records = [
        _record("blog", "Li Hua helped structure a blog post with SEO ideas."),
        _record(
            "workout",
            "WolfgangSchulz admired Li Hua's good body shape and asked about a workout.",
        ),
    ]

    ranked = rank_evidence_records(
        "Did Li Hua discuss his progress with the fitness plan before the blog post?",
        records,
        limit=1,
    )

    assert [record.record_id for record in ranked] == ["workout"]


def test_build_compact_answer_prompt_uses_source_ids():
    prompt = build_compact_answer_prompt(
        question="Did Li Hua ask Adam about repairs?",
        records=[_record("20260108_11:00", "LiHua asked Adam about repairs.")],
    )

    assert "one short sentence" in prompt
    assert "[20260108_11:00]" in prompt
    assert "strict evidence verifier" not in prompt


def test_build_strict_verifier_prompt_separates_question_types():
    prompt = build_strict_verifier_prompt(
        question="What is the Wi-Fi password?",
        records=[_record("wifi", "AdamSmith: The Wi-Fi password is Family123.")],
    )

    assert "For factual questions" in prompt
    assert "never answer Yes or No" in prompt
    assert "Family123" in prompt


def test_is_strict_abstention():
    assert is_strict_abstention("Insufficient evidence. [1] is not enough.")
    assert not is_strict_abstention("Yes. Evidence [1].")


def test_uncertainty_abstention_detection():
    assert is_uncertainty_abstention("There is no record of Li Hua asking about that.")
    assert is_uncertainty_abstention("Cannot be determined from the provided evidence.")
    assert not is_uncertainty_abstention("No, the event happened after the reminder.")
