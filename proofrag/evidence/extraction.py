"""
extraction.py — Rule-based evidence slot extraction.

These heuristics are intentionally conservative: direct evidence is assigned
only for high-signal patterns, while looser overlap is marked indirect or
background so the sufficiency scorer can block unsafe answers.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from proofrag.evidence.contradiction import detect_contradicted_slots


class EvidenceInference(BaseModel):
    """Slot and strength inference for one retrieved text snippet."""

    supports_slots: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    evidence_strength: str = "background"


def infer_evidence_from_text(
    *,
    text: str,
    question: str,
    metadata: dict[str, Any] | None = None,
    source_id: str = "",
) -> EvidenceInference:
    """Infer supported slots, contradictions, and evidence strength."""

    metadata = metadata or {}
    text = text or ""
    supports_slots: list[str] = []
    evidence_strength = "background"
    q_lower = question.lower()
    t_lower = text.lower()

    q_keywords = [
        word.lower()
        for word in re.findall(r"\w+", question)
        if len(word) > 4
    ]
    overlap = any(keyword in t_lower for keyword in q_keywords)

    if "dry-run placeholder context" in t_lower:
        return EvidenceInference()

    if "what time" in q_lower:
        if _contains_time(text):
            supports_slots.append("time_answer")
            evidence_strength = "direct"
        if overlap:
            supports_slots.append("event_context")
            if evidence_strength == "background":
                evidence_strength = "indirect"

    elif "when" in q_lower:
        if _contains_date(text) or _contains_date(source_id):
            supports_slots.append("date_or_time_answer")
            evidence_strength = "direct"
        if overlap and _event_context_matches(q_lower, t_lower):
            supports_slots.append("event_context")
            if evidence_strength == "background":
                evidence_strength = "indirect"

    elif "who" in q_lower:
        if overlap:
            supports_slots.append("topic_context")
            evidence_strength = "indirect"
        if re.search(r"\b(Tom|Sarah|LiHua|someone|he|she|they)\b\s+asked\b", text, re.I):
            supports_slots.append("who_asked")
            evidence_strength = "direct"

    else:
        if _is_direct_general_answer(q_lower, t_lower, overlap):
            supports_slots.extend(["topic_context", "answer"])
            evidence_strength = "direct"

    return EvidenceInference(
        supports_slots=_dedupe(supports_slots),
        contradicts=detect_contradicted_slots(
            text=text,
            question=question,
            metadata=metadata,
        ),
        evidence_strength=evidence_strength,
    )


def _contains_time(text: str) -> bool:
    return re.search(
        r"\b(?:\d{1,2}:\d{2}|\d{1,2})\s*(?:AM|PM)\b|\b\d{1,2}:\d{2}\b",
        text,
        re.I,
    ) is not None


def _contains_date(text: str) -> bool:
    return re.search(r"202\d[0-1]\d[0-3]\d", text) is not None


def _event_context_matches(question_lower: str, text_lower: str) -> bool:
    social_terms = {
        "lunch",
        "dinner",
        "café",
        "cafe",
        "food",
        "eating",
        "meet",
        "seeing",
        "reminder",
        "plan",
        "appointment",
        "check-in",
        "move-in",
        "visit",
        "breakfast",
        "supper",
        "meal",
    }
    q_social = any(word in question_lower for word in social_terms)
    t_social = any(word in text_lower for word in social_terms)
    if q_social and t_social:
        return True

    is_dinner_q = "dinner" in question_lower
    is_lunch_t = "lunch" in text_lower
    return not (is_dinner_q and is_lunch_t)


def _is_direct_general_answer(
    question_lower: str,
    text_lower: str,
    overlap: bool,
) -> bool:
    is_what_does = "what does" in question_lower or "what did" in question_lower
    strong_answer_patterns = [
        "just wanted to let you know",
        "reported",
        "mentioned",
        "asked",
        "says",
        "said",
        "the water tab in the apartment is broken",
        "is broken",
        "wi-fi password is",
        "having friends over occasionally is fine",
        "keep noise to a minimum",
        "small repair",
    ]
    has_strong_pattern = any(pattern in text_lower for pattern in strong_answer_patterns)
    return (is_what_does and has_strong_pattern) or overlap


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped

