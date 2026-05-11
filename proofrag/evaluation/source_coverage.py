"""Source coverage helpers for normalized external RAG exports."""

from __future__ import annotations

from typing import Any


def retrieved_source_ids(
    retrieved_context: list[dict[str, Any]],
    evidence_records: list[dict[str, Any]],
) -> set[str]:
    """Return normalized source IDs observed in retrieved context/evidence."""

    ids = {str(context.get("source_id", "")) for context in retrieved_context}
    for record in evidence_records:
        source_id = str(record.get("source_id", ""))
        if source_id:
            ids.add(source_id)
            if "#src" in source_id:
                ids.add(source_id.rsplit("#src", 1)[-1])
    return {source_id for source_id in ids if source_id}


def source_ids_match(left: str, right: str) -> bool:
    """Return whether two LiHua/MiniRAG source IDs denote the same source."""

    return source_id_variants(left) & source_id_variants(right) != set()


def source_id_variants(source_id: str) -> set[str]:
    """Generate delimiter variants used by LiHua/MiniRAG timestamps."""

    text = str(source_id).strip()
    compact = text.replace(":", "").replace("_", "").replace("-", "")
    return {
        text,
        text.replace(":", ""),
        text.replace(":", "_"),
        text.replace(":", "-"),
        compact,
    }


def expected_answer_allowed_from_sources(
    *,
    gold_supporting_sources: list[str],
    retrieved_context: list[dict[str, Any]],
    evidence_records: list[dict[str, Any]],
) -> bool:
    """Infer the fair expected gate from gold-source coverage and contradictions."""

    retrieved_ids = retrieved_source_ids(retrieved_context, evidence_records)
    gold_ids = set(gold_supporting_sources)
    has_all_gold = (
        all(
            any(source_ids_match(gold_id, retrieved_id) for retrieved_id in retrieved_ids)
            for gold_id in gold_ids
        )
        if gold_ids
        else True
    )
    has_contradiction = any(record.get("contradicts") for record in evidence_records)
    return has_all_gold and not has_contradiction
