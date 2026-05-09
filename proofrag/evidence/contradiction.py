"""
contradiction.py — Lightweight contradiction detection helpers.

The first version is deterministic and metadata-aware. It avoids model calls so
that the safety gate remains auditable and usable with small local models.
"""

from __future__ import annotations

from typing import Any


def detect_contradicted_slots(
    *,
    text: str,
    question: str,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Return contract slot IDs contradicted by a retrieved text snippet."""

    metadata = metadata or {}
    q_lower = question.lower()
    t_lower = (text or "").lower()
    contradicted: list[str] = []

    if metadata.get("contradiction") is True or "no one asked" in t_lower:
        if "who" in q_lower:
            contradicted.append("who_asked")

    if metadata.get("contradicts"):
        for slot_id in metadata["contradicts"]:
            if isinstance(slot_id, str) and slot_id not in contradicted:
                contradicted.append(slot_id)

    return contradicted

