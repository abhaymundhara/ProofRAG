"""
infer.py — Rule-based query contract inference.
"""

from proofrag.contracts.schema import EvidenceSlot, EvidenceContract

def infer_contract_from_question(question: str) -> EvidenceContract:
    """Infers an EvidenceContract from a question string using rule-based heuristics."""
    q_lower = question.lower()

    if " before " in q_lower or " after " in q_lower:
        slots = [
            EvidenceSlot(
                slot_id="event_a",
                description="Evidence for the first event mentioned in a temporal-order question",
                evidence_type="temporal_event",
                required=True,
                min_sources=1,
            ),
            EvidenceSlot(
                slot_id="event_b",
                description="Evidence for the second event mentioned in a temporal-order question",
                evidence_type="temporal_event",
                required=True,
                min_sources=1,
            ),
        ]
        return EvidenceContract(question=question, query_type="temporal_order_query", slots=slots)
    
    if "what time" in q_lower:
        slots = [
            EvidenceSlot(
                slot_id="time_answer",
                description="The time requested by the question",
                evidence_type="factual",
                required=True,
                min_sources=1
            ),
            EvidenceSlot(
                slot_id="event_context",
                description="Context connecting the requested time to the event in the question",
                evidence_type="context",
                required=True,
                min_sources=1
            )
        ]
        return EvidenceContract(question=question, query_type="time_query", slots=slots)
    
    elif "when" in q_lower or "when was" in q_lower:
        slots = [
            EvidenceSlot(
                slot_id="date_or_time_answer",
                description="The date or time requested by the question",
                evidence_type="factual",
                required=True,
                min_sources=1
            ),
            EvidenceSlot(
                slot_id="event_context",
                description="Context connecting the requested date/time to the event in the question",
                evidence_type="context",
                required=True,
                min_sources=1
            )
        ]
        return EvidenceContract(question=question, query_type="temporal_query", slots=slots)
    
    elif "who" in q_lower:
        slots = [
            EvidenceSlot(
                slot_id="who_asked",
                description="The person who initiated the request",
                evidence_type="actor",
                required=True,
                min_sources=1
            ),
            EvidenceSlot(
                slot_id="topic_context",
                description="Context about the topic",
                evidence_type="context",
                required=True,
                min_sources=1
            )
        ]
        return EvidenceContract(question=question, query_type="actor_query", slots=slots)
    
    else:
        slots = [
            EvidenceSlot(
                slot_id="answer",
                description="The answer requested by the question",
                evidence_type="factual",
                required=True,
                min_sources=1
            ),
            EvidenceSlot(
                slot_id="topic_context",
                description="Context supporting the answer",
                evidence_type="context",
                required=True,
                min_sources=1
            )
        ]
        return EvidenceContract(question=question, query_type="general_query", slots=slots)
